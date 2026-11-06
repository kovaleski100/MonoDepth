# SPDX-License-Identifier: Apache-2.0
from dataclasses import dataclass


@dataclass(frozen=True)
class PixelDepthConfig:
    in_channels: int = 3
    patch_size: int = 16
    hidden_size: int = 384
    intermediate_size: int = 1536
    num_layers: int = 12
    num_attention_heads: int = 8
    num_key_value_heads: int = 4
    head_dim: int = 48
    out_layers: tuple[int, ...] = (5, 7, 9, 11)
    alt_start: int = 4
    cat_local_global: bool = True
    predict_cameras: bool = True
    depth_confidence: bool = True
    predict_rays: bool = True
    decoder_features: int = 64
    decoder_channels: tuple[int, ...] = (48, 96, 192, 384)
    rms_norm_eps: float = 1e-6
    rope_theta_hw: float = 10000.0
    gradient_checkpointing: bool = False

    def __post_init__(self):
        object.__setattr__(self, 'out_layers', tuple(self.out_layers))
        object.__setattr__(self, 'decoder_channels', tuple(self.decoder_channels))
        for name in ('in_channels', 'patch_size', 'hidden_size', 'intermediate_size',
                     'num_layers', 'num_attention_heads', 'num_key_value_heads',
                     'head_dim', 'decoder_features'):
            if type(getattr(self, name)) is not int or getattr(self, name) <= 0:
                raise ValueError(f'{name} must be a positive integer')
        if self.num_attention_heads % self.num_key_value_heads or self.head_dim % 8:
            raise ValueError('GQA requires divisible head counts; head_dim must be divisible by 8')
        if len(self.out_layers) != 4 or tuple(sorted(set(self.out_layers))) != self.out_layers:
            raise ValueError('out_layers must contain four distinct increasing layer indices')
        if not 0 <= self.out_layers[0] <= self.out_layers[-1] < self.num_layers:
            raise ValueError('out_layers outside the backbone')
        if self.alt_start != -1 and not 1 <= self.alt_start < self.num_layers:
            raise ValueError('alt_start must be -1 or between 1 and num_layers-1')
        if len(self.decoder_channels) != 4 or any(c <= 0 for c in self.decoder_channels):
            raise ValueError('decoder_channels must contain four positive widths')
        if self.rms_norm_eps <= 0 or self.rope_theta_hw <= 0:
            raise ValueError('normalization epsilon and RoPE theta must be positive')

    @property
    def feature_dim(self):
        return self.hidden_size * (2 if self.cat_local_global else 1)
