# SPDX-License-Identifier: Apache-2.0
# RMSNorm and rotary math adapted from the user's seg_pixelumm/model.py.
import torch
from torch import nn
from torch.nn import functional as F

class RMSNorm(nn.Module):
    def __init__(self, dim, eps):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        normalized = x.float() * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + self.eps)
        return self.weight * normalized.to(x.dtype)


def _rotate_axis(x, positions, theta):
    dim = x.shape[-1]
    inv_freq = theta ** (-torch.arange(0, dim, 2, device=x.device, dtype=torch.float32) / dim)
    angles = positions.float()[:, None] * inv_freq[None, :]
    angles = torch.cat((angles, angles), dim=-1)
    first, second = x.chunk(2, dim=-1)
    rotated = torch.cat((-second, first), dim=-1)
    return x * angles.cos().to(x.dtype) + rotated * angles.sin().to(x.dtype)


class SpatialAttention(nn.Module):
    """PixelSeg UND GQA/MRoPE, with explicit positions for camera and view tokens.

    Temporal position stays zero, as in PixelSeg. Views share spatial coordinates;
    they are not interpreted as temporal tubelets.
    """
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.q_proj = nn.Linear(cfg.hidden_size, cfg.num_attention_heads * cfg.head_dim, bias=False)
        self.k_proj = nn.Linear(cfg.hidden_size, cfg.num_key_value_heads * cfg.head_dim, bias=False)
        self.v_proj = nn.Linear(cfg.hidden_size, cfg.num_key_value_heads * cfg.head_dim, bias=False)
        self.o_proj = nn.Linear(cfg.num_attention_heads * cfg.head_dim, cfg.hidden_size, bias=False)
        self.q_norm_t = RMSNorm(cfg.head_dim // 2, cfg.rms_norm_eps)
        self.q_norm_hw = RMSNorm(cfg.head_dim // 2, cfg.rms_norm_eps)
        self.k_norm_t = RMSNorm(cfg.head_dim // 2, cfg.rms_norm_eps)
        self.k_norm_hw = RMSNorm(cfg.head_dim // 2, cfg.rms_norm_eps)

    def _position(self, x, norm_t, norm_hw, positions):
        temporal, spatial = x.chunk(2, dim=-1)
        height, width = norm_hw(spatial).chunk(2, dim=-1)
        return torch.cat((norm_t(temporal),
                          _rotate_axis(height, positions[:, 0], self.cfg.rope_theta_hw),
                          _rotate_axis(width, positions[:, 1], self.cfg.rope_theta_hw)), dim=-1)

    def forward(self, x, positions):
        b, n, _ = x.shape
        def split(projection, heads):
            return projection(x).reshape(b, n, heads, self.cfg.head_dim).transpose(1, 2)
        q = self._position(split(self.q_proj, self.cfg.num_attention_heads), self.q_norm_t, self.q_norm_hw, positions)
        k = self._position(split(self.k_proj, self.cfg.num_key_value_heads), self.k_norm_t, self.k_norm_hw, positions)
        v = split(self.v_proj, self.cfg.num_key_value_heads)
        result = F.scaled_dot_product_attention(q, k, v, is_causal=False, enable_gqa=True)
        return self.o_proj(result.transpose(1, 2).reshape(b, n, -1))


class UNDLayer(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.input_layernorm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.self_attn = SpatialAttention(cfg)
        self.post_attention_layernorm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.gate_proj = nn.Linear(cfg.hidden_size, cfg.intermediate_size, bias=False)
        self.up_proj = nn.Linear(cfg.hidden_size, cfg.intermediate_size, bias=False)
        self.down_proj = nn.Linear(cfg.intermediate_size, cfg.hidden_size, bias=False)

    def forward(self, x, positions):
        x = x + self.self_attn(self.input_layernorm(x), positions)
        z = self.post_attention_layernorm(x)
        return x + self.down_proj(F.silu(self.gate_proj(z)) * self.up_proj(z))
