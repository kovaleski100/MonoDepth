# SPDX-License-Identifier: Apache-2.0
from dataclasses import asdict
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint
from .config import PixelDepthConfig
from .model import PixelDepth
from .da3_encoder.vision_transformer import DinoVisionTransformer


SIZES = {
    'tiny': dict(hidden_size=32, intermediate_size=64, num_layers=4, num_attention_heads=4,
                 num_key_value_heads=2, head_dim=8, out_layers=(0, 1, 2, 3),
                 decoder_features=16, decoder_channels=(16, 24, 32, 48)),
    'small': dict(hidden_size=384, intermediate_size=1536, num_layers=12, num_attention_heads=6,
                  num_key_value_heads=3, head_dim=64, out_layers=(5, 7, 9, 11),
                  decoder_features=64, decoder_channels=(48, 96, 192, 384)),
    'large': dict(hidden_size=1024, intermediate_size=4096, num_layers=24, num_attention_heads=16,
                  num_key_value_heads=8, head_dim=64, out_layers=(4, 11, 17, 23),
                  decoder_features=256, decoder_channels=(256, 512, 1024, 1024)),
}


class DA3MonoBackbone(nn.Module):
    """Unmodified DA3 ViT computations in a monocular, scratch-training adapter.

    Large reproduces DA3-Mono-Large's encoder settings. Small/tiny are scaled
    monocular ablations, not official released checkpoints. No pretrained weights.
    """
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.encoder = DinoVisionTransformer(
            img_size=518, patch_size=cfg.patch_size, embed_dim=cfg.hidden_size,
            depth=cfg.num_layers, num_heads=cfg.num_attention_heads, mlp_ratio=4,
            alt_start=-1, qknorm_start=-1, rope_start=-1, cat_token=False,
        )
        if cfg.gradient_checkpointing:
            for block in self.encoder.blocks:
                original = block.forward
                # Capture each bound method; checkpoint only when gradients are enabled.
                def wrapped(x, *args, _fn=original, **kwargs):
                    if self.training and torch.is_grad_enabled():
                        return checkpoint(_fn, x, *args, use_reentrant=False, **kwargs)
                    return _fn(x, *args, **kwargs)
                block.forward = wrapped

    def forward(self, images, cam_token=None, export_feat_layers=()):
        if cam_token is not None:
            raise ValueError('This controlled monocular baseline does not accept camera conditioning')
        layers = tuple(export_feat_layers or ())
        feats, aux = self.encoder.get_intermediate_layers(images, self.cfg.out_layers, export_feat_layers=layers)
        return feats, dict(zip(sorted(layers), aux))


def build_comparison_model(architecture, size='small', gradient_checkpointing=False):
    cfg = PixelDepthConfig(**SIZES[size], patch_size=14, alt_start=-1,
                           cat_local_global=False, predict_cameras=False,
                           depth_confidence=False, predict_rays=False,
                           gradient_checkpointing=gradient_checkpointing)
    # Same constructor and seed give both runs exactly the same initial DPT weights.
    model = PixelDepth(cfg)
    if architecture == 'da3':
        model.backbone = DA3MonoBackbone(cfg)
    elif architecture != 'pixel_depth':
        raise ValueError(f'Unknown architecture: {architecture}')
    return model, asdict(cfg)


def load_comparison_checkpoint(path, device='cpu'):
    saved = torch.load(path, map_location='cpu', weights_only=True)
    if saved.get('format') != 'pixel-depth-comparison-v1':
        raise ValueError('Expected a comparison checkpoint')
    signature = saved['signature']
    model, _ = build_comparison_model(signature['architecture'], signature['size'])
    model.load_state_dict(saved['model'])
    return model.to(device).eval()
