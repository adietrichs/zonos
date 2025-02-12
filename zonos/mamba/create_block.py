from functools import partial

import torch.nn as nn

from zonos.mamba.mha import MHA
from zonos.mamba.mlp import GatedMLP
from zonos.mamba.block import Block


def create_block(
    d_model,
    d_intermediate,
    ssm_cfg=None,
    attn_layer_idx=None,
    attn_cfg=None,
    norm_epsilon=1e-5,
    rms_norm=False,
    residual_in_fp32=False,
    fused_add_norm=False,
    layer_idx=None,
    device=None,
    dtype=None,
):
    assert d_intermediate > 0
    assert not rms_norm
    if attn_layer_idx is None:
        attn_layer_idx = []
    assert layer_idx in attn_layer_idx
    if attn_cfg is None:
        attn_cfg = {}
    factory_kwargs = {"device": device, "dtype": dtype}
    mixer_cls = partial(MHA, layer_idx=layer_idx, **attn_cfg, **factory_kwargs)
    norm_cls = partial(
        nn.LayerNorm, eps=norm_epsilon, **factory_kwargs
    )
    mlp_cls = partial(
        GatedMLP, hidden_features=d_intermediate, out_features=d_model, **factory_kwargs
    )
    block = Block(
        d_model,
        mixer_cls,
        mlp_cls,
        norm_cls=norm_cls,
        fused_add_norm=fused_add_norm,
        residual_in_fp32=residual_in_fp32,
    )
    block.layer_idx = layer_idx
    return block
