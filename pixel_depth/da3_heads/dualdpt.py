# SPDX-License-Identifier: Apache-2.0
"""Dual depth/ray DPT head used by the DA3-style model.

The depth branch predicts depth plus confidence.  The auxiliary branch
predicts a six-channel per-pixel ray (origin + direction) plus confidence.
The two branches share feature projection but have independent fusion stacks,
matching the design used by the public DA3 implementation.
"""

from typing import List, Sequence, Tuple

import torch
from torch import nn
from addict import Dict

from .dpt import _make_fusion_block, _make_scratch
from .head_utils import create_uv_grid, custom_interpolate, position_grid_to_embed


class DualDPT(nn.Module):
    def __init__(
        self,
        dim_in: int,
        *,
        patch_size: int = 14,
        output_dim: int = 2,
        activation: str = "exp",
        conf_activation: str = "expp1",
        features: int = 256,
        out_channels: Sequence[int] = (256, 512, 1024, 1024),
        pos_embed: bool = True,
        down_ratio: int = 1,
        head_names: Tuple[str, str] = ("depth", "ray"),
    ) -> None:
        super().__init__()
        if output_dim != 2:
            raise ValueError("DualDPT expects output_dim=2 for value + confidence")
        self.patch_size = patch_size
        self.activation = activation
        self.conf_activation = conf_activation
        self.pos_embed = pos_embed
        self.down_ratio = down_ratio
        self.head_main, self.head_aux = head_names
        self.intermediate_layer_idx = (0, 1, 2, 3)

        self.norm = nn.LayerNorm(dim_in)
        self.projects = nn.ModuleList(
            [nn.Conv2d(dim_in, oc, kernel_size=1) for oc in out_channels]
        )
        self.resize_layers = nn.ModuleList([
            nn.ConvTranspose2d(out_channels[0], out_channels[0], 4, 4),
            nn.ConvTranspose2d(out_channels[1], out_channels[1], 2, 2),
            nn.Identity(),
            nn.Conv2d(out_channels[3], out_channels[3], 3, 2, 1),
        ])

        self.scratch = _make_scratch(list(out_channels), features, expand=False)

        # Main/depth fusion.
        self.scratch.refinenet1 = _make_fusion_block(features)
        self.scratch.refinenet2 = _make_fusion_block(features)
        self.scratch.refinenet3 = _make_fusion_block(features)
        self.scratch.refinenet4 = _make_fusion_block(features, has_residual=False)
        self.scratch.output_conv1 = nn.Conv2d(features, features // 2, 3, 1, 1)
        self.scratch.output_conv2 = nn.Sequential(
            nn.Conv2d(features // 2, 32, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, output_dim, 1),
        )

        # Independent ray fusion.  The final seven channels are six ray
        # values and one confidence value.
        self.scratch.refinenet1_aux = _make_fusion_block(features)
        self.scratch.refinenet2_aux = _make_fusion_block(features)
        self.scratch.refinenet3_aux = _make_fusion_block(features)
        self.scratch.refinenet4_aux = _make_fusion_block(features, has_residual=False)
        self.scratch.output_conv2_aux = nn.Sequential(
            nn.Conv2d(features, 32, 3, 1, 1),
            nn.GroupNorm(1, 32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 7, 1),
        )

    def forward(self, feats: List[torch.Tensor], H: int, W: int,
                patch_start_idx: int, chunk_size: int = 8, **kwargs) -> Dict:
        B, S, N, C = feats[0][0].shape
        flat = [feat[0].reshape(B * S, N, C) for feat in feats]
        if chunk_size is None or chunk_size >= B * S:
            result = self._forward_impl(flat, H, W, patch_start_idx)
        else:
            chunks = []
            for start in range(0, B * S, chunk_size):
                stop = min(start + chunk_size, B * S)
                chunks.append(self._forward_impl(
                    [feat[start:stop] for feat in flat], H, W, patch_start_idx))
            result = {key: torch.cat([item[key] for item in chunks], dim=0)
                      for key in chunks[0]}
        return Dict({key: value.reshape(B, S, *value.shape[1:])
                     for key, value in result.items()})

    def _forward_impl(self, feats, H, W, patch_start_idx):
        B, _, C = feats[0].shape
        ph, pw = H // self.patch_size, W // self.patch_size
        resized = []
        for stage, take in enumerate(self.intermediate_layer_idx):
            x = self.norm(feats[take][:, patch_start_idx:])
            x = x.permute(0, 2, 1).contiguous().reshape(B, C, ph, pw)
            x = self.projects[stage](x)
            if self.pos_embed:
                x = self._add_pos_embed(x, W, H)
            resized.append(self.resize_layers[stage](x))

        depth_fused, ray_fused = self._fuse(resized)
        out_h, out_w = int(ph * self.patch_size / self.down_ratio), int(pw * self.patch_size / self.down_ratio)

        depth_fused = custom_interpolate(depth_fused, (out_h, out_w), mode="bilinear", align_corners=True)
        ray_fused = custom_interpolate(ray_fused, (out_h, out_w), mode="bilinear", align_corners=True)
        if self.pos_embed:
            depth_fused = self._add_pos_embed(depth_fused, W, H)
            ray_fused = self._add_pos_embed(ray_fused, W, H)

        depth_logits = self.scratch.output_conv2(self.scratch.output_conv1(depth_fused))
        depth_logits = depth_logits.permute(0, 2, 3, 1)
        depth = self._activation(depth_logits[..., :-1], self.activation).squeeze(-1)
        depth_conf = self._activation(depth_logits[..., -1], self.conf_activation)

        ray_logits = self.scratch.output_conv2_aux(ray_fused).permute(0, 2, 3, 1)
        ray = ray_logits[..., :-1]
        ray_conf = self._activation(ray_logits[..., -1], self.conf_activation)
        return {self.head_main: depth, f"{self.head_main}_conf": depth_conf,
                self.head_aux: ray, f"{self.head_aux}_conf": ray_conf}

    def _fuse(self, features):
        l1, l2, l3, l4 = features
        l1 = self.scratch.layer1_rn(l1)
        l2 = self.scratch.layer2_rn(l2)
        l3 = self.scratch.layer3_rn(l3)
        l4 = self.scratch.layer4_rn(l4)

        main = self.scratch.refinenet4(l4, size=l3.shape[-2:])
        ray = self.scratch.refinenet4_aux(l4, size=l3.shape[-2:])
        main = self.scratch.refinenet3(main, l3, size=l2.shape[-2:])
        ray = self.scratch.refinenet3_aux(ray, l3, size=l2.shape[-2:])
        main = self.scratch.refinenet2(main, l2, size=l1.shape[-2:])
        ray = self.scratch.refinenet2_aux(ray, l2, size=l1.shape[-2:])
        main = self.scratch.refinenet1(main, l1)
        ray = self.scratch.refinenet1_aux(ray, l1)
        return main, ray

    def _add_pos_embed(self, x, W, H, ratio=0.1):
        pw, ph = x.shape[-1], x.shape[-2]
        pe = create_uv_grid(pw, ph, aspect_ratio=W / H, dtype=x.dtype, device=x.device)
        pe = position_grid_to_embed(pe, x.shape[1]) * ratio
        return x + pe.permute(2, 0, 1)[None]

    @staticmethod
    def _activation(x, activation):
        if activation == "exp":
            return torch.exp(x)
        if activation == "expp1":
            return torch.exp(x) + 1
        if activation == "relu":
            return torch.relu(x)
        if activation == "softplus":
            return torch.nn.functional.softplus(x)
        return x
