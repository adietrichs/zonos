import torch
import torch.nn as nn
from zonos.mamba.create_block import create_block
from zonos.mamba.inference_params import InferenceParams

from zonos.config import BackboneConfig


class ZonosBackbone(nn.Module):
    def __init__(self, config: BackboneConfig):
        super().__init__()
        self.config = config

        self.layers = nn.ModuleList(
            [
                create_block(
                    d_model=config.d_model,
                    d_intermediate=config.d_intermediate
                    if (i not in config.attn_layer_idx)
                    else config.attn_mlp_d_intermediate,
                    ssm_cfg=config.ssm_cfg,
                    layer_idx=i,
                    attn_layer_idx=config.attn_layer_idx,
                    attn_cfg=config.attn_cfg,
                    norm_epsilon=config.norm_epsilon,
                    residual_in_fp32=config.residual_in_fp32,
                    fused_add_norm=False,
                    rms_norm=config.rms_norm,
                )
                for i in range(config.n_layer)
            ]
        )

        self.norm_f = nn.LayerNorm(config.d_model, eps=config.norm_epsilon)

    def forward(self, hidden_states: torch.Tensor, inference_params: InferenceParams | None = None):
        residual = None
        for layer in self.layers:
            hidden_states, residual = layer(hidden_states, residual, inference_params)

        residual = (hidden_states + residual) if residual is not None else hidden_states
        hidden_states = self.norm_f(residual.to(dtype=self.norm_f.weight.dtype))
        if self.config.residual_in_fp32:
            residual = residual.to(torch.float32)

        return hidden_states
