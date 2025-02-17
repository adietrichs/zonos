# Copyright (c) 2024, Tri Dao, Albert Gu.

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from flash_attn import flash_attn_with_kvcache
from zonos.flash.rotary import RotaryEmbedding


def _update_kv_cache(kv, inference_params, layer_idx):
    """kv: (batch_size, seqlen, 2, nheads, head_dim) or (batch_size, 1, 2, nheads, head_dim)"""
    # Pre-allocate memory for key-values for inference.
    num_heads, head_dim = kv.shape[-2:]
    assert layer_idx in inference_params.key_value_memory_dict
    kv_cache, _ = inference_params.key_value_memory_dict[layer_idx]
    # Adjust key and value for inference
    batch_start = inference_params.batch_size_offset
    batch_end = batch_start + kv.shape[0]
    sequence_start = inference_params.seqlen_offset
    sequence_end = sequence_start + kv.shape[1]
    assert batch_end <= kv_cache.shape[0]
    assert sequence_end <= kv_cache.shape[1]
    assert kv_cache is not None
    kv_cache[batch_start:batch_end, sequence_start:sequence_end, ...] = kv
    return kv_cache[batch_start:batch_end, :sequence_end, ...]


class MHA(nn.Module):
    """Multi-head self-attention and cross-attention"""

    def __init__(
        self,
        embed_dim,
        num_heads,
        num_heads_kv=None,
        head_dim=None,  # If None, use embed_dim // num_heads
        mlp_dim=0,
        qkv_proj_bias=True,
        out_proj_bias=True,
        softmax_scale=None,
        causal=False,
        layer_idx=None,
        d_conv=0,
        rotary_emb_dim=0,
        rotary_emb_base=10000.0,
        rotary_emb_interleaved=False,
        device=None,
        dtype=None,
    ) -> None:
        """
        num_heads_kv: can be used to toggle MQA / GQA. If None, use num_heads.
        return_residual: whether to return the input x along with the output. This is for
            performance reason: for post-norm architecture, returning the input allows us
            to fuse the backward of nn.Linear with the residual connection.
        """
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.embed_dim = embed_dim
        self.layer_idx = layer_idx
        assert d_conv == 0
        self.rotary_emb_dim = rotary_emb_dim
        self.softmax_scale = softmax_scale
        self.causal = causal

        self.num_heads = num_heads
        self.num_heads_kv = num_heads_kv if num_heads_kv is not None else num_heads
        assert (
            self.num_heads % self.num_heads_kv == 0
        ), "num_heads must be divisible by num_heads_kv"
        if head_dim is None:
            assert self.embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        self.head_dim = head_dim if head_dim is not None else self.embed_dim // num_heads
        self.mlp_dim = math.ceil(mlp_dim / 256) * 256
        qkv_dim = self.head_dim * (self.num_heads + 2 * self.num_heads_kv)
        out_dim = self.head_dim * self.num_heads

        if self.rotary_emb_dim > 0:
            assert RotaryEmbedding is not None, "rotary requires flash_attn to be installed"
            self.rotary_emb = RotaryEmbedding(
                self.rotary_emb_dim,
                base=rotary_emb_base,
                interleaved=rotary_emb_interleaved,
                device=device,
            )

        self.in_proj = nn.Linear(embed_dim, qkv_dim + self.mlp_dim, bias=qkv_proj_bias, **factory_kwargs)
        self.out_proj = nn.Linear(out_dim + self.mlp_dim // 2, embed_dim, bias=out_proj_bias, **factory_kwargs)

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None):
        dtype = self.out_proj.weight.dtype if dtype is None else dtype
        device = self.out_proj.weight.device
        conv_state = None
        kv_cache = torch.empty(
            batch_size, max_seqlen, 2, self.num_heads_kv, self.head_dim, dtype=dtype, device=device,
        )
        return kv_cache, conv_state

    def _update_kv_cache(self, kv, inference_params):
        """kv: (batch_size, seqlen, 2, nheads, head_dim) or (batch_size, 1, 2, nheads, head_dim)"""
        assert self.layer_idx is not None, "Generation requires layer_idx in the constructor"
        return _update_kv_cache(kv, inference_params, self.layer_idx)

    def _update_kvcache_attention(self, q, kv, inference_params):
        """Write kv to inference_params, then do attention"""
        if (
            inference_params.seqlen_offset == 0
            or flash_attn_with_kvcache is None
        ):
            # TODO: this only uses seqlen_offset and not lengths_per_sample.
            kv = self._update_kv_cache(kv, inference_params)
            k, v = kv.unbind(dim=-3)
            k = torch.repeat_interleave(k, dim=2, repeats=self.num_heads // self.num_heads_kv)
            v = torch.repeat_interleave(v, dim=2, repeats=self.num_heads // self.num_heads_kv)
            return F.scaled_dot_product_attention(
                q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), is_causal=self.causal, scale=self.softmax_scale
            ).transpose(1, 2)
        else:
            batch = q.shape[0]
            kv_cache, _ = inference_params.key_value_memory_dict[self.layer_idx]
            kv_cache = kv_cache[:batch]
            cache_seqlens = (
                inference_params.lengths_per_sample[:batch]
                if inference_params.lengths_per_sample is not None
                else inference_params.seqlen_offset
            )
            return flash_attn_with_kvcache(
                q,
                kv_cache[:, :, 0],
                kv_cache[:, :, 1],
                kv[:, :, 0],
                kv[:, :, 1],
                cache_seqlens=cache_seqlens,
                softmax_scale=self.softmax_scale,
                causal=self.causal,
            )

    def forward(self, x, inference_params=None):
        """
        Arguments:
            x: (batch, seqlen, hidden_dim) (where hidden_dim = num heads * head dim) if
                cu_seqlens is None and max_seqlen is None, else (total, hidden_dim) where total
                is the is the sum of the sequence lengths in the batch.
            inference_params: for generation. Adapted from Megatron-LM (and Apex)
            https://github.com/NVIDIA/apex/blob/3ff1a10f72ec07067c4e44759442329804ac5162/apex/transformer/testing/standalone_transformer_lm.py#L470
        """
        if inference_params is not None and self.layer_idx not in inference_params.key_value_memory_dict:
            inference_params.key_value_memory_dict[self.layer_idx] = self.allocate_inference_cache(
                x.shape[0], inference_params.max_seqlen, dtype=x.dtype
            )
        seqlen_offset = (
            0
            if inference_params is None
            else (
                inference_params.lengths_per_sample
                if inference_params.lengths_per_sample is not None
                else inference_params.seqlen_offset
            )
        )
        rotary_max_seqlen = inference_params.max_seqlen if inference_params is not None else None
        qkv = self.in_proj(x)
        if self.mlp_dim > 0:
            qkv, x_mlp = qkv.split([qkv.shape[-1] - self.mlp_dim, self.mlp_dim], dim=-1)
            x_mlp_up, x_mlp_gate = x_mlp.chunk(2, dim=-1)
            x_mlp = x_mlp_up * F.silu(x_mlp_gate)
        q, kv = qkv.split([self.num_heads * self.head_dim, self.num_heads_kv * 2 * self.head_dim], dim=-1)
        q = rearrange(q, "... (h d) -> ... h d", d=self.head_dim)
        kv = rearrange(kv, "... (two hkv d) -> ... two hkv d", two=2, d=self.head_dim)
        if self.rotary_emb_dim > 0:
            q, kv = self.rotary_emb(
                q, kv, seqlen_offset=seqlen_offset, max_seqlen=rotary_max_seqlen
            )
        if inference_params is None:
            k, v = kv.unbind(dim=-3)
            k = torch.repeat_interleave(k, dim=2, repeats=self.num_heads // self.num_heads_kv)
            v = torch.repeat_interleave(v, dim=2, repeats=self.num_heads // self.num_heads_kv)
            context = F.scaled_dot_product_attention(
                q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), is_causal=self.causal, scale=self.softmax_scale
            ).transpose(1, 2)
        else:
            context = self._update_kvcache_attention(q, kv, inference_params)
        context = rearrange(context, "... h d -> ... (h d)")
        if self.mlp_dim > 0:
            context = torch.cat([context, x_mlp], dim=-1)
        out = self.out_proj(context)
        return out
