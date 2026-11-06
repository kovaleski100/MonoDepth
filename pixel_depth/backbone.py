# SPDX-License-Identifier: Apache-2.0
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint
from .layers import RMSNorm, UNDLayer


class PixelDepthBackbone(nn.Module):
    """PixelSeg UND blocks + DA3-style local/global attention and camera tokens."""
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.patch_embed = nn.Linear(cfg.patch_size ** 2 * cfg.in_channels, cfg.hidden_size)
        self.layers = nn.ModuleList(UNDLayer(cfg) for _ in range(cfg.num_layers))
        self.norm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.camera_token = nn.Parameter(torch.empty(1, 2, 1, cfg.hidden_size))
        nn.init.normal_(self.camera_token, std=0.02)

    def forward(self, images, cam_token=None, export_feat_layers=()):
        b, s, c, h, w = images.shape
        p = self.cfg.patch_size
        gh, gw = h // p, w // p
        if h % p or w % p:
            raise ValueError('Backbone requires padded image dimensions divisible by patch_size')
        exports = tuple(export_feat_layers or ())
        if len(set(exports)) != len(exports) or any(i < 0 or i >= len(self.layers) for i in exports):
            raise ValueError('Invalid or repeated export feature indices')
        patches = images.reshape(b, s, c, gh, p, gw, p)
        patches = patches.permute(0, 1, 3, 5, 4, 6, 2).reshape(b, s, gh * gw, -1)
        x = self.patch_embed(patches)
        cameras = torch.cat((self.camera_token[:, :1].expand(b, 1, -1, -1),
                             self.camera_token[:, 1:].expand(b, s - 1, -1, -1)), dim=1)
        if cam_token is not None:
            if cam_token.shape != (b, s, self.cfg.hidden_size):
                raise ValueError('cam_token must be [B, views, hidden_size]')
            cameras = cameras + cam_token.unsqueeze(2)
        x = torch.cat((cameras, x), dim=2)
        rows = torch.arange(gh, device=x.device).repeat_interleave(gw) + 1
        cols = torch.arange(gw, device=x.device).repeat(gh) + 1
        pos = torch.cat((torch.zeros(1, 2, device=x.device), torch.stack((rows, cols), dim=-1)), dim=0)
        n = x.shape[2]
        outputs, auxiliary = {}, {}
        local_x = x
        for i, layer in enumerate(self.layers):
            is_global = self.cfg.alt_start != -1 and i >= self.cfg.alt_start and i % 2 == 1
            z = x.reshape(b, s * n, -1) if is_global else x.reshape(b * s, n, -1)
            positions = pos.repeat(s, 1) if is_global else pos
            if self.cfg.gradient_checkpointing and self.training and torch.is_grad_enabled():
                z = checkpoint(layer, z, positions, use_reentrant=False)
            else:
                z = layer(z, positions)
            x = z.reshape(b, s, n, -1)
            if not is_global:
                local_x = x
            if i in self.cfg.out_layers:
                normalized = self.norm(x)
                features = torch.cat((local_x, normalized), dim=-1) if self.cfg.cat_local_global else normalized
                outputs[i] = (features[:, :, 1:], features[:, :, 0])
            if i in exports:
                auxiliary[i] = self.norm(x[:, :, 1:])
        return tuple(outputs[i] for i in self.cfg.out_layers), {i: auxiliary[i] for i in exports}
