#!/usr/bin/env python3
"""Attach DA3 teacher pseudo-depth to an NPZ manifest.

The public DA3 monocular checkpoint can be used as a practical teacher.  For
an exact reproduction, pass the teacher checkpoint used by the experiment.
The script keeps the original NPZ fields and adds ``teacher_depth`` and
``teacher_valid``.  Relative teacher depth is robustly scale/shift aligned to
the valid metric depth already stored in the sample.
"""

import argparse
import csv
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


def _to_rgb(image, mean, std):
    image = np.asarray(image, dtype=np.float32)
    if image.ndim == 3:
        image = image[None]
    if image.ndim != 4 or image.shape[1] != 3:
        raise ValueError('Teacher generation currently requires RGB [C,H,W] or [S,3,H,W]')
    mean = np.asarray(mean, dtype=np.float32)[None, :, None, None]
    std = np.asarray(std, dtype=np.float32)[None, :, None, None]
    rgb = np.clip(image * std + mean, 0., 1.)
    # DA3's input processor forwards NumPy images to PIL, which requires
    # uint8 for HWC RGB arrays.  The dataset stores normalized float32 CHW.
    return [
        (np.transpose(frame, (1, 2, 0)) * 255.0 + 0.5).clip(0, 255).astype(np.uint8)
        for frame in rgb
    ]


def _fit_scale_shift(relative, target, valid):
    mask = valid & np.isfinite(relative) & np.isfinite(target) & (target > 0)
    result = np.full_like(relative, np.nan, dtype=np.float32)
    if mask.sum() < 2:
        result[mask] = relative[mask]
        return result, mask
    x, y = relative[mask].astype(np.float64), target[mask].astype(np.float64)
    inliers = np.ones_like(x, dtype=bool)
    for _ in range(4):
        A = np.stack((x[inliers], np.ones(inliers.sum())), axis=1)
        scale, shift = np.linalg.lstsq(A, y[inliers], rcond=None)[0]
        residual = scale * x + shift - y
        center = np.median(residual[inliers])
        deviation = np.mean(np.abs(residual[inliers] - center))
        inliers = np.abs(residual - center) <= max(2.5 * deviation, 1e-6)
        if inliers.sum() < 2:
            break
    scale = max(float(scale), 1e-6)
    result = scale * relative + float(shift)
    result[~np.isfinite(result)] = np.nan
    result[~valid] = np.nan
    return result.astype(np.float32), mask


def _resize_depth(depth, height, width):
    tensor = torch.from_numpy(np.asarray(depth, dtype=np.float32))[None, None]
    return F.interpolate(tensor, (height, width), mode='bilinear', align_corners=False)[0, 0].numpy()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', required=True, help='CSV with a sample column')
    parser.add_argument('--output-manifest', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--model', default='depth-anything/DA3MONO-LARGE',
                        help='Hugging Face/local DA3 teacher checkpoint')
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--process-res', type=int, default=504)
    parser.add_argument('--mean', type=float, nargs=3, default=[.485, .456, .406])
    parser.add_argument('--std', type=float, nargs=3, default=[.229, .224, .225])
    args = parser.parse_args()

    # The local DA3 source tree exposes the API class from ``api.py``;
    # the namespace package itself does not re-export it.
    from depth_anything_3.api import DepthAnything3

    device = torch.device(args.device)
    teacher = DepthAnything3.from_pretrained(args.model).to(device).eval()
    manifest = Path(args.manifest).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_manifest = Path(args.output_manifest).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    with manifest.open(newline='') as stream:
        rows = list(csv.DictReader(stream))
    if not rows or 'sample' not in rows[0]:
        raise ValueError('Manifest must contain a sample column')

    for row in rows:
        source = (manifest.parent / row['sample']).resolve()
        with np.load(source, allow_pickle=False) as sample:
            fields = {key: np.array(sample[key], copy=True) for key in sample.files}
        image = fields['image']
        depth = fields['depth']
        valid = fields.get('valid', np.isfinite(depth) & (depth > 0))
        if image.ndim == 3:
            image = image[None]
            depth = depth[None]
            valid = valid[None]
            single = True
        else:
            single = False
        rgb_views = _to_rgb(image, args.mean, args.std)
        teacher_depth = []
        teacher_valid = []
        for rgb, target, target_valid in zip(rgb_views, depth, valid):
            with torch.inference_mode():
                prediction = teacher.inference([rgb], process_res=args.process_res)
            relative = _resize_depth(prediction.depth[0], target.shape[-2], target.shape[-1])
            aligned, valid_mask = _fit_scale_shift(relative, target, target_valid.astype(bool))
            teacher_depth.append(aligned)
            teacher_valid.append(valid_mask)
        fields['teacher_depth'] = np.stack(teacher_depth).astype(np.float32)
        fields['teacher_valid'] = np.stack(teacher_valid).astype(np.bool_)
        if single:
            fields['teacher_depth'] = fields['teacher_depth'][0]
            fields['teacher_valid'] = fields['teacher_valid'][0]
        destination = output_dir / Path(row['sample']).name
        destination.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(destination, **fields)
        row['sample'] = str(destination.relative_to(output_manifest.parent))

    output_manifest.parent.mkdir(parents=True, exist_ok=True)
    with output_manifest.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=['sample'])
        writer.writeheader()
        writer.writerows({'sample': row['sample']} for row in rows)


if __name__ == '__main__':
    main()
