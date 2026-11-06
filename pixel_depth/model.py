# SPDX-License-Identifier: Apache-2.0
from dataclasses import asdict
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F
from .da3_heads.dpt import DPT
from .da3_heads.dualdpt import DualDPT
from .da3_heads.cam_dec import CameraDec
from .da3_heads.transform import (
    extri_intri_to_pose_encoding, pose_encoding_to_extri_intri,
)
from .config import PixelDepthConfig
from .backbone import PixelDepthBackbone


class PixelDepth(nn.Module):
    """Trainable PixelSeg/DA3 hybrid. Input is an externally normalized tensor.

    Accepts [B,C,H,W] or [B,S,C,H,W]. Outputs always retain the view axis.
    Depth is positive but its scale depends on the training supervision.
    Camera extrinsics use DA3's world-to-camera convention; view zero is reference.
    """
    def __init__(self, config=None):
        super().__init__()
        self.config = config or PixelDepthConfig()
        cfg = self.config
        self.backbone = PixelDepthBackbone(cfg)
        if cfg.predict_rays:
            # Full DA3-style dense target: depth (+ confidence) and a
            # six-channel ray map (+ confidence).
            self.head = DualDPT(
                dim_in=cfg.feature_dim,
                patch_size=cfg.patch_size,
                output_dim=2,
                features=cfg.decoder_features,
                out_channels=cfg.decoder_channels,
            )
        else:
            self.head = DPT(
                dim_in=cfg.feature_dim,
                patch_size=cfg.patch_size,
                output_dim=2 if cfg.depth_confidence else 1,
                features=cfg.decoder_features,
                out_channels=cfg.decoder_channels,
                use_sky_head=False,
            )
        if cfg.predict_cameras:
            self.camera_encoder = nn.Sequential(nn.Linear(9, cfg.hidden_size), nn.SiLU(),
                                                nn.Linear(cfg.hidden_size, cfg.hidden_size))
            self.camera_decoder = CameraDec(dim_in=cfg.feature_dim)
            # Start with identity quaternion and a reasonable field of view.
            nn.init.zeros_(self.camera_decoder.fc_qvec.weight)
            with torch.no_grad():
                self.camera_decoder.fc_qvec.bias.copy_(torch.tensor([0., 0., 0., 1.]))
                self.camera_decoder.fc_fov[0].bias.fill_(1.)

    def forward(self, images, extrinsics=None, intrinsics=None, export_feat_layers=()):
        if images.ndim == 4:
            images = images.unsqueeze(1)
        if images.ndim != 5 or images.shape[2] != self.config.in_channels:
            raise ValueError('images must be [B,S,C,H,W] or [B,C,H,W] with configured channels')
        if not images.is_floating_point() or min(images.shape) <= 0:
            raise ValueError('images must be non-empty floating point tensors')
        b, s, c, h, w = images.shape
        if (extrinsics is None) != (intrinsics is None):
            raise ValueError('Provide both extrinsics and intrinsics')
        camera = None
        if extrinsics is not None:
            if not self.config.predict_cameras:
                raise ValueError('Camera conditioning requires predict_cameras=True')
            if extrinsics.shape not in ((b, s, 3, 4), (b, s, 4, 4)) or intrinsics.shape != (b, s, 3, 3):
                raise ValueError('Invalid camera shapes: expected [B,S,3/4,4] and [B,S,3,3]')
            with torch.autocast(device_type=images.device.type, enabled=False):
                homogeneous = torch.eye(4, device=images.device).expand(b, s, 4, 4).clone()
                homogeneous[..., :3, :] = extrinsics[..., :3, :].float()
                pose = extri_intri_to_pose_encoding(torch.linalg.inv(homogeneous), intrinsics.float(), (h, w))
                camera = self.camera_encoder(pose)
        p = self.config.patch_size
        flat = F.pad(images.reshape(b * s, c, h, w), (0, (-w) % p, 0, (-h) % p), mode='replicate')
        hp, wp = flat.shape[-2:]
        padded = flat.reshape(b, s, c, hp, wp)
        feats, aux = self.backbone(padded, cam_token=camera, export_feat_layers=export_feat_layers)
        # DPT and camera geometry run in float32 even under mixed precision.
        with torch.autocast(device_type=images.device.type, enabled=False):
            feats32 = tuple((f.float(), t.float()) for f, t in feats)
            output = dict(self.head(feats32, hp, wp, patch_start_idx=0))
            for key in ('depth', 'depth_conf'):
                if key in output:
                    value = output[key]
                    if value.ndim == 5 and value.shape[-1] == 1:
                        value = value.squeeze(-1)
                    output[key] = value[..., :h, :w]
            if 'ray' in output:
                # DualDPT stores rays channel-last: [B,S,H,W,6].
                output['ray'] = output['ray'][..., :h, :w, :]
            if 'ray_conf' in output:
                output['ray_conf'] = output['ray_conf'][..., :h, :w]
            if self.config.predict_cameras:
                pose = self.camera_decoder(feats32[-1][1])
                c2w, calibration = pose_encoding_to_extri_intri(pose, (h, w))
                homogeneous = torch.eye(4, device=images.device).expand(b, s, 4, 4).clone()
                homogeneous[..., :3, :] = c2w
                output['pose_encoding'] = pose
                output['extrinsics'] = torch.linalg.inv(homogeneous)[..., :3, :]
                output['intrinsics'] = calibration
        output['aux'] = {i: f.reshape(b, s, hp // p, wp // p, -1) for i, f in aux.items()}
        return output

    def save(self, path, **metadata):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({'format': 'pixel-depth-v2', 'config': asdict(self.config),
                    'model': self.state_dict(), 'metadata': metadata}, path)

    @classmethod
    def load(cls, path, map_location='cpu'):
        payload = torch.load(path, map_location=map_location, weights_only=True)
        if payload.get('format') not in ('pixel-depth-v1', 'pixel-depth-v2'):
            raise ValueError('Expected a Pixel-Depth checkpoint, not a DA3/PixelUMM checkpoint')
        config = dict(payload['config'])
        # v1 checkpoints predate the DualDPT ray branch.
        if payload.get('format') == 'pixel-depth-v1' and 'predict_rays' not in config:
            config['predict_rays'] = False
        model = cls(PixelDepthConfig(**config))
        model.load_state_dict(payload['model'], strict=True)
        return model.to(map_location)

    def load_pixelseg_backbone(self, state_dict):
        """Transfer a raw SegPixelUMM state dict (or backbone.state_dict()).

        Exact shape matching is required. Segmentation decoder weights are excluded.
        Only the new camera token is allowed to be absent; errors precede mutation.
        """
        if any(k.startswith('backbone.') for k in state_dict):
            weights = {k[len('backbone.'):]: v for k, v in state_dict.items() if k.startswith('backbone.')}
        else:
            weights = dict(state_dict)
        expected = self.backbone.state_dict()
        required = set(expected) - {'camera_token'}
        missing = required - set(weights)
        unexpected = set(weights) - set(expected)
        mismatched = [k for k in set(weights) & set(expected) if weights[k].shape != expected[k].shape]
        if missing or unexpected or mismatched:
            raise ValueError(f'Incompatible PixelSeg backbone: missing={sorted(missing)}, unexpected={sorted(unexpected)}, shapes={mismatched}')
        return self.backbone.load_state_dict(weights, strict=False)
