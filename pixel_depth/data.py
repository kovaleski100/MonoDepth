# SPDX-License-Identifier: Apache-2.0
import csv
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset


def _rays_from_cameras(extrinsics, intrinsics, height, width):
    """Build world-space rays in channel-last layout [S,H,W,6]."""
    if extrinsics.ndim != 3 or intrinsics.shape != (extrinsics.shape[0], 3, 3):
        raise ValueError('extrinsics/intrinsics have incompatible shapes')
    if extrinsics.shape[1:] not in ((3, 4), (4, 4)):
        raise ValueError('extrinsics must be [S,3,4] or [S,4,4]')
    s = extrinsics.shape[0]
    homogeneous = torch.eye(4, dtype=extrinsics.dtype, device=extrinsics.device).expand(s, 4, 4).clone()
    homogeneous[:, :3, :] = extrinsics[:, :3, :]
    c2w = torch.linalg.inv(homogeneous)
    rotation = c2w[:, :3, :3]
    origin = c2w[:, :3, 3].view(s, 1, 1, 3).expand(s, height, width, 3)
    v, u = torch.meshgrid(
        torch.arange(height, dtype=intrinsics.dtype, device=intrinsics.device),
        torch.arange(width, dtype=intrinsics.dtype, device=intrinsics.device),
        indexing='ij',
    )
    pixels = torch.stack((u, v, torch.ones_like(u)), dim=-1).view(1, height * width, 3)
    directions_camera = torch.linalg.solve(intrinsics, pixels.transpose(1, 2)).transpose(1, 2)
    directions_world = torch.bmm(directions_camera, rotation.transpose(1, 2))
    directions_world = directions_world.view(s, height, width, 3)
    return torch.cat((origin, directions_world), dim=-1)


class DepthDataset(Dataset):
    """CSV with an `sample` column pointing to NPZ samples relative to the CSV.

    image: [C,H,W] or [S,C,H,W], already normalized consistently for train/eval.
    depth: [H,W] or [S,H,W], positive metric or consistently scaled depth.
    Optional valid, teacher_depth, ray [S,H,W,6] or [S,6,H,W],
    extrinsics [S,3/4,4], intrinsics [S,3,3].
    """
    def __init__(self, manifest):
        manifest = Path(manifest)
        with manifest.open(newline='') as stream:
            reader = csv.DictReader(stream)
            if 'sample' not in (reader.fieldnames or []):
                raise ValueError('CSV must contain a sample column')
            self.samples = [(manifest.parent / row['sample']).resolve() for row in reader]
        if not self.samples:
            raise ValueError('Empty manifest')

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        with np.load(self.samples[index], allow_pickle=False) as sample:
            result = {k: torch.from_numpy(np.array(sample[k], copy=True)).float()
                      for k in ('image', 'depth', 'valid', 'teacher_depth',
                                'teacher_valid', 'ray', 'ray_valid',
                                'extrinsics', 'intrinsics')
                      if k in sample}
        if not {'image', 'depth'} <= result.keys():
            raise ValueError('Each NPZ needs image and depth arrays')
        if result['image'].ndim == 3:
            result['image'] = result['image'].unsqueeze(0)
        if result['depth'].ndim == 2:
            result['depth'] = result['depth'].unsqueeze(0)
        for key in ('valid', 'teacher_depth', 'teacher_valid', 'ray_valid'):
            if key in result and result[key].ndim == 2:
                result[key] = result[key].unsqueeze(0)
        if 'ray' in result:
            ray = result['ray']
            if ray.ndim == 3 and ray.shape[-1] == 6:
                ray = ray.unsqueeze(0)
            elif ray.ndim == 3 and ray.shape[0] == 6:
                ray = ray.permute(1, 2, 0).unsqueeze(0)
            elif ray.ndim == 4 and ray.shape[-1] == 6:
                pass
            elif ray.ndim == 4 and ray.shape[1] == 6:
                ray = ray.permute(0, 2, 3, 1)
            else:
                raise ValueError('ray must be [H,W,6], [S,H,W,6], [6,H,W] or [S,6,H,W]')
            result['ray'] = ray
        if result['image'].ndim != 4 or result['depth'].shape != result['image'].shape[:1] + result['image'].shape[-2:]:
            raise ValueError('Image/depth view counts or spatial dimensions do not match')
        views, height, width = result['image'].shape[0], result['image'].shape[-2], result['image'].shape[-1]
        for key in ('valid', 'teacher_depth', 'teacher_valid', 'ray_valid'):
            if key in result and result[key].shape != (views, height, width):
                raise ValueError(f'{key} must match depth shape [views,height,width]')
        if 'ray' in result and result['ray'].shape != (views, height, width, 6):
            raise ValueError('ray must match image views and spatial dimensions')
        if not torch.isfinite(result['image']).all():
            raise ValueError('Input image has NaN/Inf values')
        if ('extrinsics' in result) != ('intrinsics' in result):
            raise ValueError('Camera annotations require both extrinsics and intrinsics')
        if 'extrinsics' in result and 'ray' not in result:
            result['ray'] = _rays_from_cameras(
                result['extrinsics'], result['intrinsics'], height, width)
            if 'ray_valid' not in result:
                result['ray_valid'] = result.get(
                    'valid', torch.ones(views, height, width, dtype=torch.bool)
                ).bool()
        return result
