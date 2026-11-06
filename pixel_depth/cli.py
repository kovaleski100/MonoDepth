# SPDX-License-Identifier: Apache-2.0
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import random
import numpy as np
import torch
from torch.utils.data import DataLoader
from . import PixelDepth, PixelDepthConfig
from .architecture import write_architecture_svg
from .data import DepthDataset
from .losses import (
    depth_loss, depth_metrics, da3_loss,
    adaptive_berhu_loss, adaptive_berhu_loss_RMSE,
)


def _device(name):
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu') if name == 'auto' else torch.device(name)


def _seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _prediction(model, batch, condition):
    cameras = {k: batch[k] for k in ('extrinsics', 'intrinsics')} if condition else {}
    return model(batch['image'], **cameras)


def _loss(args, prediction, batch):
    """Build the selected objective and reject incomplete DA3 supervision."""
    if args.loss == 'berhu_rmse':
        return adaptive_berhu_loss_RMSE(prediction, batch['depth'], batch.get('valid'))
    if args.loss == 'berhu':
        return adaptive_berhu_loss(prediction, batch['depth'], batch.get('valid'))
    if args.ray_weight > 0 and 'ray' not in batch:
        raise ValueError('ray_weight > 0 requires ray in every training NPZ')
    if args.teacher_weight > 0 and 'teacher_depth' not in batch:
        raise ValueError('teacher_weight > 0 requires teacher_depth in every training NPZ')
    camera = None
    if args.camera_weight > 0:
        if 'extrinsics' not in batch or 'intrinsics' not in batch:
            raise ValueError('camera_weight > 0 requires extrinsics and intrinsics')
        camera = (batch['extrinsics'], batch['intrinsics'], batch['image'].shape[-2:])
    return da3_loss(
        prediction, batch['depth'], batch.get('valid'),
        ray=batch.get('ray'), ray_valid=batch.get('ray_valid'),
        teacher_depth=batch.get('teacher_depth'),
        teacher_valid=batch.get('teacher_valid'),
        ray_weight=args.ray_weight, teacher_weight=args.teacher_weight,
        camera=camera, camera_weight=args.camera_weight,
    )


def _evaluate(model, loader, device, condition=False):
    model.eval()
    sums = {'abs_rel': 0., 'squared_error': 0., 'delta1': 0.,
            'delta2': 0., 'delta3': 0., 'valid_pixels': 0}
    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            metrics = depth_metrics(_prediction(model, batch, condition), batch['depth'], batch.get('valid'))
            n = metrics['valid_pixels']
            sums['abs_rel'] += metrics['abs_rel'] * n
            sums['squared_error'] += metrics['rmse'] ** 2 * n
            sums['delta1'] += metrics['delta1'] * n
            sums['delta2'] += metrics['delta2'] * n
            sums['delta3'] += metrics['delta3'] * n
            sums['valid_pixels'] += n
    n = sums['valid_pixels']
    return {'abs_rel': sums['abs_rel']/n, 'rmse': (sums['squared_error']/n)**.5,
            'delta1': sums['delta1']/n, 'delta2': sums['delta2']/n, 'delta3': sums['delta3']/n, 'valid_pixels': n}


def train(args):
    _seed(args.seed)
    device = _device(args.device)
    config = PixelDepthConfig(**json.loads(Path(args.config).read_text()))
    if args.full_da3:
        if not config.predict_rays:
            raise ValueError('--full-da3 requires predict_rays=true in the config')
        args.loss = 'da3'
        args.ray_weight = 1.0
        args.teacher_weight = 1.0
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    # Never silently overwrite a previous experiment.
    if any(output.iterdir()):
        raise ValueError('Output directory must be empty; choose a new experiment directory')
    model = PixelDepth(config).to(device)
    if args.pixelseg_checkpoint:
        payload = torch.load(args.pixelseg_checkpoint, map_location='cpu', weights_only=True)
        model.load_pixelseg_backbone(payload.get('model', payload))
    dataset = DepthDataset(args.train_manifest)
    if args.condition_cameras and not config.predict_cameras:
        raise ValueError('Camera conditioning requires a camera-enabled config')
    has_cameras = 'extrinsics' in dataset[0]
    if args.condition_cameras and not has_cameras:
        raise ValueError('Camera conditioning requires camera annotations')
    if config.predict_cameras and not has_cameras:
        print('Camera decoder has no pose supervision; camera predictions will remain untrained.')
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers)
    val = DataLoader(DepthDataset(args.val_manifest), batch_size=1) if args.val_manifest else None
    if val and set(dataset.samples) & set(val.dataset.samples):
        raise ValueError('Train/validation manifests share sample paths')
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    run = vars(args).copy()
    run.pop('func', None)
    run['model'] = asdict(config)
    run['parameters'] = sum(p.numel() for p in model.parameters())
    (output / 'run.json').write_text(json.dumps(run, indent=2))
    for epoch in range(args.epochs):
        model.train()
        running = 0.
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            optimizer.zero_grad(set_to_none=True)
            prediction = _prediction(model, batch, args.condition_cameras)
            loss = _loss(args, prediction, batch)
            if not torch.isfinite(loss):
                raise FloatingPointError('Nonfinite training loss')
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
            optimizer.step()
            running += loss.item()
        record = {'epoch': epoch + 1, 'loss': running / len(loader)}
        if val:
            record['validation'] = _evaluate(model, val, device, args.condition_cameras)
        print(json.dumps(record), flush=True)
        with (output / 'metrics.jsonl').open('a') as stream:
            stream.write(json.dumps(record) + '\n')
        model.save(output / 'last.pt', epoch=epoch+1, seed=args.seed,
                   camera_supervised=bool(config.predict_cameras and has_cameras))


def evaluate(args):
    device = _device(args.device)
    model = PixelDepth.load(args.checkpoint, device).eval()
    metrics = _evaluate(model, DataLoader(DepthDataset(args.manifest), batch_size=1), device, args.condition_cameras)
    print(json.dumps(metrics, indent=2))
    if args.output:
        path = Path(args.output)
        if path.exists():
            raise FileExistsError(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(metrics, indent=2))


def infer(args):
    device = _device(args.device)
    model = PixelDepth.load(args.checkpoint, device).eval()
    with np.load(args.input, allow_pickle=False) as sample:
        image = torch.from_numpy(np.array(sample['image'], copy=True)).float()
        if image.ndim == 3:
            image = image.unsqueeze(0)
        # NPZ image is [S,C,H,W]; one scene is one batch element.
        image = image.unsqueeze(0).to(device)
        cameras = {}
        if args.condition_cameras:
            cameras = {k: torch.from_numpy(np.array(sample[k], copy=True)).float().unsqueeze(0).to(device)
                       for k in ('extrinsics', 'intrinsics')}
    with torch.no_grad():
        result = model(image, **cameras)
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('wb') as stream:
        np.savez_compressed(stream, **{k: v[0].cpu().numpy() for k, v in result.items() if torch.is_tensor(v)})


def architecture(args):
    config = PixelDepthConfig(**json.loads(Path(args.config).read_text())) if args.config else PixelDepthConfig()
    output = write_architecture_svg(config, args.output)
    print(f'Architecture diagram written to {output}')


def main():
    parser = argparse.ArgumentParser(description='Pixel-Depth: DA3-style depth/ray training')
    sub = parser.add_subparsers(dest='command', required=True)
    t = sub.add_parser('train')
    t.add_argument('--config', required=True)
    t.add_argument('--train-manifest', required=True)
    t.add_argument('--val-manifest')
    t.add_argument('--output', required=True)
    t.add_argument('--epochs', type=int, default=20)
    t.add_argument('--batch-size', type=int, default=1)
    t.add_argument('--workers', type=int, default=0)
    t.add_argument('--lr', type=float, default=1e-4)
    t.add_argument('--weight-decay', type=float, default=0.01)
    t.add_argument('--seed', type=int, default=42)
    t.add_argument('--loss', choices=('da3', 'berhu_rmse', 'berhu'), default='da3')
    t.add_argument('--full-da3', action='store_true',
                   help='Require and train with ray + offline DA3 teacher targets')
    t.add_argument('--ray-weight', type=float, default=0.)
    t.add_argument('--teacher-weight', type=float, default=0.)
    t.add_argument('--camera-weight', type=float, default=0.)
    t.add_argument('--pixelseg-checkpoint')
    t.set_defaults(func=train)
    e = sub.add_parser('evaluate')
    e.add_argument('--checkpoint', required=True)
    e.add_argument('--manifest', required=True)
    e.add_argument('--output')
    e.set_defaults(func=evaluate)
    i = sub.add_parser('infer')
    i.add_argument('--checkpoint', required=True)
    i.add_argument('--input', required=True)
    i.add_argument('--output', required=True)
    i.set_defaults(func=infer)
    a = sub.add_parser('architecture', help='Generate a simplified SVG architecture diagram')
    a.add_argument('--config')
    a.add_argument('--output', required=True)
    a.set_defaults(func=architecture)
    for p in (t, e, i):
        p.add_argument('--device', default='auto')
        p.add_argument('--condition-cameras', action='store_true')
    args = parser.parse_args()
    if args.command == 'train' and (args.epochs <= 0 or args.batch_size <= 0 or args.lr <= 0):
        parser.error('epochs, batch-size and lr must be positive')
    if args.command == 'train' and min(args.ray_weight, args.teacher_weight, args.camera_weight) < 0:
        parser.error('ray-weight, teacher-weight and camera-weight must be nonnegative')
    args.func(args)


if __name__ == '__main__':
    main()
