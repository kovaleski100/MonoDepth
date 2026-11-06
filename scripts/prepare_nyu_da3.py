#!/usr/bin/env python3
"""Convert the cached NYUv2 MAT into Pixel-Depth DA3-style NPZ manifests.

NYUv2 is monocular and does not contain multiview camera poses.  This
converter therefore stores the fixed Kinect camera ray map and leaves camera
pose supervision disabled.  It is suitable for running the depth+ray+teacher
pipeline, but it is not a multiview reproduction of the official DA3 data
protocol.
"""

import argparse
import csv
import json
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn.functional as F


# Common NYUv2 Kinect intrinsics at the original 640x480 image resolution.
FX = 5.8262448167737955e2
FY = 5.826910327098863e2
CX = 3.1304475870804731e2
CY = 2.3844389626620386e2


def _resize(array, height, width, mode):
    tensor = torch.from_numpy(np.asarray(array, dtype=np.float32))[None, None]
    return F.interpolate(tensor, (height, width), mode=mode)[0, 0].numpy()


def _ray_map(height, width):
    sx, sy = width / 640.0, height / 480.0
    k = np.array([[FX * sx, 0.0, CX * sx],
                  [0.0, FY * sy, CY * sy],
                  [0.0, 0.0, 1.0]], dtype=np.float32)
    v, u = np.meshgrid(np.arange(height, dtype=np.float32),
                        np.arange(width, dtype=np.float32), indexing='ij')
    pixels = np.stack((u, v, np.ones_like(u)), axis=-1)
    directions = pixels @ np.linalg.inv(k).T
    origins = np.zeros_like(directions)
    return np.concatenate((origins, directions), axis=-1).astype(np.float32)


def _write_manifest(path, rows):
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=['sample'])
        writer.writeheader()
        writer.writerows({'sample': row} for row in rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--labeled-mat', required=True)
    parser.add_argument('--split-json', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--height', type=int, default=224)
    parser.add_argument('--width', type=int, default=294)
    args = parser.parse_args()

    output = Path(args.output_dir).resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError(f'Output directory must be new and empty: {output}')
    (output / 'npz').mkdir(parents=True, exist_ok=True)
    split = json.loads(Path(args.split_json).read_text())
    ray = _ray_map(args.height, args.width)
    mean = np.asarray([.485, .456, .406], dtype=np.float32)[:, None, None]
    std = np.asarray([.229, .224, .225], dtype=np.float32)[:, None, None]
    rows = {name: [] for name in ('train', 'val', 'test')}

    with h5py.File(args.labeled_mat, 'r') as source:
        if source['images'].shape[1:] != (3, 640, 480):
            raise ValueError(f'Unexpected images shape: {source["images"].shape}')
        for name in ('train', 'val', 'test'):
            for index in split[name]:
                image = np.asarray(source['images'][index]).transpose(0, 2, 1).astype(np.float32) / 255.0
                image = F.interpolate(
                    torch.from_numpy(image)[None],
                    (args.height, args.width),
                    mode='bilinear',
                    align_corners=False,
                )[0].numpy()
                image = (image - mean) / std

                depth = np.asarray(source['depths'][index]).T.astype(np.float32)
                valid = np.zeros_like(depth, dtype=bool)
                valid[45:471, 41:601] = True
                valid &= np.isfinite(depth) & (depth > 1e-3) & (depth < 10.0)
                depth = _resize(depth, args.height, args.width, 'nearest').astype(np.float32)
                valid = _resize(valid.astype(np.float32), args.height, args.width, 'nearest') > 0.5

                filename = f'{int(index):04d}.npz'
                np.savez_compressed(
                    output / 'npz' / filename,
                    image=image.astype(np.float32),
                    depth=depth,
                    valid=valid,
                    ray=ray,
                    ray_valid=valid,
                )
                rows[name].append(f'npz/{filename}')

    for name, values in rows.items():
        _write_manifest(output / f'{name}.csv', values)
        print(f'{name}: {len(values)} samples', flush=True)
    print(f'Created NYUv2 DA3-style dataset at {output}', flush=True)


if __name__ == '__main__':
    main()
