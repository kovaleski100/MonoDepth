# SPDX-License-Identifier: Apache-2.0
"""NYUv2 official labeled MAT, streamed download and scene-disjoint holdout."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import time
from urllib.request import Request, urlopen
import h5py
import numpy as np
from scipy.io import loadmat
import torch
from torch.nn import functional as F
from torch.utils.data import Dataset

LABELED_URL = 'https://horatio.cs.nyu.edu/mit/silberman/nyu_depth_v2/nyu_depth_v2_labeled.mat'
SPLITS_URL = 'https://raw.githubusercontent.com/cleinc/bts/master/utils/splits.mat'
LABELED_BYTES = 2972037809
SPLITS_SHA256 = '6e081404491a8bfba2f066beaa9713ef6046f9007aa5f799f2a350506a580cee'


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def download(url, destination, expected_bytes=None):
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if expected_bytes and destination.stat().st_size != expected_bytes:
            raise ValueError(f'Unexpected file size: {destination}; move it aside to download again')
        return destination
    part = destination.with_suffix(destination.suffix + '.part')
    for attempt in range(4):
        offset = part.stat().st_size if part.exists() else 0
        if expected_bytes and offset == expected_bytes:
            part.replace(destination)
            return destination
        free = shutil.disk_usage(destination.parent).free
        if expected_bytes and free < expected_bytes - offset + 256 * 1024**2:
            raise OSError(f'Insufficient space for download in {destination.parent}')
        try:
            request = Request(url, headers={'User-Agent': 'Pixel-Depth/0.1', **({'Range': f'bytes={offset}-'} if offset else {})})
            with urlopen(request, timeout=120) as response:
                status = getattr(response, 'status', None)
                if offset and status == 206:
                    content_range = response.headers.get('Content-Range', '')
                    if not content_range.startswith(f'bytes {offset}-'):
                        raise ValueError(f'Unexpected Content-Range: {content_range}')
                    mode = 'ab'
                else:
                    mode, offset = 'wb', 0  # Server ignored Range: restart instead of corrupt append.
                received = offset
                last_print = time.monotonic()
                with part.open(mode) as stream:
                    while True:
                        block = response.read(4 * 1024 * 1024)
                        if not block:
                            break
                        stream.write(block)
                        received += len(block)
                        if time.monotonic() - last_print > 10:
                            print(f'Download {destination.name}: {received / 1024**3:.2f} GiB', flush=True)
                            last_print = time.monotonic()
                length = response.headers.get('Content-Length')
                if length and received != offset + int(length):
                    raise OSError('Download ended before Content-Length')
            if expected_bytes and part.stat().st_size != expected_bytes:
                raise OSError(f'Incomplete file: {part.stat().st_size} / {expected_bytes} bytes')
            part.replace(destination)
            return destination
        except (OSError, TimeoutError) as error:
            if attempt == 3:
                raise
            print(f'Download interrupted: {error}; retry {attempt+1}/3', flush=True)
            time.sleep(2)
    raise RuntimeError('Download failed')


def _scene_names(file, count):
    refs = np.asarray(file['scenes']).reshape(-1)
    if len(refs) != count:
        raise ValueError('Unexpected scenes metadata')
    return [''.join(chr(int(c)) for c in np.asarray(file[ref]).reshape(-1)) for ref in refs]


def prepare_split(labeled, splits, output, seed=42, val_fraction=.1, strict_counts=True):
    output = Path(output)
    if strict_counts and sha256(splits) != SPLITS_SHA256:
        raise ValueError('Unexpected splits.mat checksum; use the verified official split file')
    official = loadmat(splits)
    train = np.asarray(official['trainNdxs']).reshape(-1).astype(int) - 1
    test = np.asarray(official['testNdxs']).reshape(-1).astype(int) - 1
    with h5py.File(labeled, 'r') as file:
        count = file['images'].shape[0]
        if file['images'].shape[1:] != (3, 640, 480) or file['depths'].shape != (count, 640, 480):
            raise ValueError('Unexpected NYUv2 MAT layout; expected MATLAB v7.3 axis ordering')
        scenes = _scene_names(file, count)
    if strict_counts and (count, len(train), len(test)) != (1449, 795, 654):
        raise ValueError('Official labeled NYUv2 must have 1449 samples, 795 train and 654 test')
    if len(set(train)) != len(train) or len(set(test)) != len(test) or set(train) & set(test):
        raise ValueError('Duplicated/overlapping official indices')
    if set(train) | set(test) != set(range(count)):
        raise ValueError('Splits do not partition all samples')
    train_scenes = sorted({scenes[i] for i in train})
    if set(train_scenes) & {scenes[i] for i in test}:
        raise ValueError('Train/test scene overlap; refusing a leaking protocol')
    if not 0 < val_fraction < 1 or len(train_scenes) < 2:
        raise ValueError('Need at least two training scenes and val_fraction in (0,1)')
    rng = np.random.default_rng(seed)
    val_count = min(len(train_scenes)-1, max(1, round(len(train_scenes)*val_fraction)))
    validation_scenes = set(rng.permutation(train_scenes)[:val_count])
    validation = [int(i) for i in train if scenes[i] in validation_scenes]
    training = [int(i) for i in train if scenes[i] not in validation_scenes]
    result = {'version': 1, 'seed': seed, 'val_fraction': val_fraction,
              'labeled_sha256': sha256(labeled), 'splits_sha256': sha256(splits),
              'train': training, 'val': validation, 'test': test.tolist(),
              'scene_counts': {name: len({scenes[i] for i in ids}) for name, ids in
                               (('train', training), ('val', validation), ('test', test))}}
    if output.exists():
        if json.loads(output.read_text()) != result:
            raise ValueError('Cached split belongs to different data/seed/protocol; use a new work directory')
    else:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, indent=2)+'\n')
    return result


class NYUDataset(Dataset):
    def __init__(self, labeled, indices, image_size=(224, 294), training=False):
        self.labeled = str(labeled)
        self.indices = list(indices)
        self.image_size = tuple(image_size)
        self.training = training
        self._file = None
        self._pid = None

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        if self._file is None or self._pid != os.getpid():
            if self._file is not None:
                self._file.close()
            self._file = h5py.File(self.labeled, 'r')
            self._pid = os.getpid()
        i = self.indices[index]
        image = torch.from_numpy(np.asarray(self._file['images'][i]).transpose(0, 2, 1).copy()).float() / 255.
        depth = torch.from_numpy(np.asarray(self._file['depths'][i]).T.copy()).float()
        # NYU evaluation crop at original resolution, before any resize.
        mask = torch.zeros_like(depth, dtype=torch.bool)
        mask[45:471, 41:601] = True
        mask &= torch.isfinite(depth) & (depth > 1e-3) & (depth < 10.)
        image = F.interpolate(image[None], self.image_size, mode='bilinear', align_corners=False)[0]
        # ImageNet normalization, identical for both backbones.
        mean = image.new_tensor([.485, .456, .406])[:, None, None]
        std = image.new_tensor([.229, .224, .225])[:, None, None]
        image = (image-mean)/std
        if self.training:
            depth = F.interpolate(depth[None, None], self.image_size, mode='nearest')[0, 0]
            mask = F.interpolate(mask[None, None].float(), self.image_size, mode='nearest')[0, 0].bool()
        return {'image': image, 'depth': depth.unsqueeze(0), 'valid': mask.unsqueeze(0), 'index': i}

    def __getstate__(self):
        state = self.__dict__.copy()
        state['_file'], state['_pid'] = None, None
        return state
