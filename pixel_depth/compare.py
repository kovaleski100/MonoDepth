# SPDX-License-Identifier: Apache-2.0
"""Download NYUv2, run two controlled scratch trainings, evaluate and compare."""
import argparse
import csv
import gc
import hashlib
import json
import math
from pathlib import Path
import shutil
import time
import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from .comparison_models import build_comparison_model
from .losses import depth_loss
from .nyu import LABELED_URL, SPLITS_URL, LABELED_BYTES, NYUDataset, download, prepare_split


PROTOCOL = 'nyuv2-monocular-scratch-v1'


def atomic_save(payload, path):
    path = Path(path)
    temporary = path.with_suffix(path.suffix+'.tmp')
    torch.save(payload, temporary)
    temporary.replace(path)


def sync(device):
    if device.type == 'cuda':
        torch.cuda.synchronize(device)


def seed_all(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def loader(dataset, args, epoch=None, seed=0):
    generator = torch.Generator().manual_seed(seed + (epoch or 0))
    return DataLoader(dataset, batch_size=args.batch_size if epoch is not None else 1,
                      shuffle=epoch is not None, generator=generator,
                      num_workers=args.workers, pin_memory=args.device.startswith('cuda'),
                      persistent_workers=False)


def _metric_values(pred, target):
    delta = torch.maximum(pred/target, target/pred)
    return {'abs_rel': ((pred-target).abs()/target).mean().item(),
            'rmse': (pred-target).square().mean().sqrt().item(),
            'delta1': (delta < 1.25).float().mean().item()}


@torch.no_grad()
def evaluate(model, dataset, args, device, amp_dtype=None):
    model.eval()
    sums = {}
    count = 0
    for batch in loader(dataset, args):
        image = batch['image'].to(device)
        with torch.autocast(device.type, enabled=amp_dtype is not None, dtype=amp_dtype):
            prediction = model(image)['depth']
        target = batch['depth'].to(device)
        mask = batch['valid'].to(device)
        prediction = F.interpolate(prediction.float(), target.shape[-2:], mode='bilinear', align_corners=False)
        for p, t, valid in zip(prediction, target, mask):
            p, t = p[valid], t[valid]
            if not len(p):
                continue
            if not torch.isfinite(p).all():
                raise FloatingPointError('Nonfinite evaluation prediction')
            # Raw metrics and independently median-aligned metrics are both reported.
            raw = p.clamp(1e-3, 10.)
            aligned = (p * (t.median()/p.median().clamp_min(1e-6))).clamp(1e-3, 10.)
            for prefix, values in (('raw', _metric_values(raw, t)), ('median_aligned', _metric_values(aligned, t))):
                for key, value in values.items():
                    name = f'{prefix}_{key}'
                    sums[name] = sums.get(name, 0.) + value
            count += 1
    if not count:
        raise ValueError('Empty evaluation set')
    return {**{key: value/count for key, value in sums.items()}, 'images': count}


def _amp(args, device):
    if args.amp == 'off':
        return None
    if device.type != 'cuda':
        raise ValueError('--amp requires CUDA; use --amp off for CPU')
    if args.amp == 'bf16' and not torch.cuda.is_bf16_supported():
        raise ValueError('GPU does not support bf16; use --amp fp16 or --amp off')
    return torch.bfloat16 if args.amp == 'bf16' else torch.float16


def _signature(args, architecture, seed, split):
    return {'protocol': PROTOCOL, 'architecture': architecture, 'size': args.size,
            'initialization': 'scratch', 'seed': seed, 'split': split,
            'image_size': [args.height, args.width], 'epochs': args.epochs,
            'batch_size': args.batch_size, 'accumulate': args.accumulate,
            'lr': args.lr, 'weight_decay': args.weight_decay, 'amp': args.amp,
            'gradient_checkpointing': args.gradient_checkpointing}


def train_one(args, architecture, seed, split, labeled, root):
    device = torch.device(args.device)
    amp_dtype = _amp(args, device)
    seed_all(seed)
    model, config = build_comparison_model(architecture, args.size, args.gradient_checkpointing)
    model.to(device)
    parameters = sum(p.numel() for p in model.parameters())
    output = root/f'{architecture}-seed{seed}'
    output.mkdir(parents=True, exist_ok=True)
    signature = _signature(args, architecture, seed, split)
    last, best = output/'last.pt', output/'best.pt'
    if (output/'run.json').exists():
        previous = json.loads((output/'run.json').read_text())
        if previous['signature'] != signature:
            raise ValueError(f'Configuration differs from existing run {output}; use a new --work-dir')
        if not args.resume:
            raise FileExistsError(f'Existing run {output}; use --resume or a new --work-dir')
    elif any(output.iterdir()):
        raise ValueError(f'Unexpected files in {output}; use a new --work-dir')
    metadata = {'signature': signature, 'model_config': config, 'parameters': parameters,
                'torch': str(torch.__version__), 'device': str(device),
                'gpu': torch.cuda.get_device_name(device) if device.type == 'cuda' else None,
                'head': 'identical DPT initialization',
                'baseline_note': 'DA3 monocular encoder; small is a scaled ablation, large matches DA3-Mono-Large encoder settings'}
    (output/'run.json').write_text(json.dumps(metadata, indent=2)+'\n')
    if (output/'test.json').exists() and args.resume:
        print(f'Already evaluated: {output}', flush=True)
        return json.loads((output/'test.json').read_text())
    estimated_checkpoint = parameters * 16  # last: weights+Adam, best: weights
    free = shutil.disk_usage(output).free
    if free < estimated_checkpoint + 512*1024**2:
        raise OSError(f'Need approximately {estimated_checkpoint/1024**3:.1f} GiB for checkpoints in {output}')
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.amp.GradScaler('cuda', enabled=amp_dtype == torch.float16)
    start, best_score, steps, cumulative_seconds = 0, math.inf, 0, 0.
    peak = 0
    if last.exists() and args.resume:
        saved = torch.load(last, map_location='cpu', weights_only=True)
        if saved['signature'] != signature:
            raise ValueError('Checkpoint signature mismatch')
        model.load_state_dict(saved['model'])
        optimizer.load_state_dict(saved['optimizer'])
        scheduler.load_state_dict(saved['scheduler'])
        scaler.load_state_dict(saved['scaler'])
        start, best_score, steps = saved['epoch'], saved['best_score'], saved['steps']
        cumulative_seconds = saved['training_seconds']
        peak = saved.get('peak_cuda_allocated_bytes', 0)
        torch.set_rng_state(saved['rng'])
        if device.type == 'cuda' and saved['cuda_rng']:
            torch.cuda.set_rng_state_all(saved['cuda_rng'])
        # A crash after logging but before checkpoint publication can leave a stale line.
        history = output/'metrics.jsonl'
        if history.exists():
            rows = [json.loads(line) for line in history.read_text().splitlines() if line.strip()]
            history.write_text(''.join(json.dumps(row)+'\n' for row in rows if row['epoch'] <= start))
    elif args.resume and best.exists():
        raise ValueError('Missing last.pt; cannot resume optimizer from best.pt')
    training = NYUDataset(labeled, split['train'], (args.height, args.width), training=True)
    validation = NYUDataset(labeled, split['val'], (args.height, args.width))
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(device)
    print(f'Training {architecture}, seed={seed}, parameters={parameters:,}, epoch={start+1}/{args.epochs}', flush=True)
    for epoch in range(start, args.epochs):
        model.train()
        batches = loader(training, args, epoch=epoch, seed=seed)
        optimizer.zero_grad(set_to_none=True)
        total, seen = 0., 0
        sync(device)
        began = time.perf_counter()
        for index, batch in enumerate(batches):
            image = batch['image'].to(device, non_blocking=True)
            target = batch['depth'].to(device, non_blocking=True)
            valid = batch['valid'].to(device, non_blocking=True)
            with torch.autocast(device.type, enabled=amp_dtype is not None, dtype=amp_dtype):
                prediction = model(image)
                loss = depth_loss(prediction, target, valid)
            if not torch.isfinite(loss):
                raise FloatingPointError(f'Nonfinite loss in {architecture}; try --amp off or a smaller --lr')
            # Correctly normalize the final partial accumulation window.
            window_start = (index//args.accumulate)*args.accumulate
            window_size = min(args.accumulate, len(batches)-window_start)
            scaler.scale(loss/window_size).backward()
            total += loss.item()*len(image)
            seen += len(image)
            if (index+1) % args.accumulate == 0 or index+1 == len(batches):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                steps += 1
            if (index+1) % 25 == 0:
                print(f'{architecture} epoch {epoch+1} batch {index+1}/{len(batches)} loss={total/seen:.5f}', flush=True)
        sync(device)
        seconds = time.perf_counter()-began
        cumulative_seconds += seconds
        metrics = evaluate(model, validation, args, device, amp_dtype)
        scheduler.step()
        if device.type == 'cuda':
            peak = max(peak, torch.cuda.max_memory_allocated(device))
        improved = metrics['raw_abs_rel'] < best_score
        if improved:
            best_score = metrics['raw_abs_rel']
            atomic_save({'format': 'pixel-depth-comparison-v1', 'signature': signature,
                         'model': model.state_dict(), 'epoch': epoch+1}, best)
        record = {'epoch': epoch+1, 'loss': total/seen, 'validation': metrics,
                  'training_seconds': seconds, 'optimizer_steps': steps, 'next_lr': scheduler.get_last_lr()[0]}
        atomic_save({'format': 'pixel-depth-comparison-v1', 'signature': signature,
                     'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
                     'scheduler': scheduler.state_dict(), 'scaler': scaler.state_dict(),
                     'epoch': epoch+1, 'best_score': best_score, 'steps': steps,
                     'training_seconds': cumulative_seconds, 'peak_cuda_allocated_bytes': peak,
                     'rng': torch.get_rng_state(),
                     'cuda_rng': torch.cuda.get_rng_state_all() if device.type == 'cuda' else []}, last)
        with (output/'metrics.jsonl').open('a') as stream:
            stream.write(json.dumps(record)+'\n')
        print(json.dumps({'model': architecture, **record}), flush=True)
    if not best.exists():
        raise ValueError('No best checkpoint produced')
    selected = torch.load(best, map_location='cpu', weights_only=True)
    model.load_state_dict(selected['model'])
    test = NYUDataset(labeled, split['test'], (args.height, args.width))
    sync(device)
    began = time.perf_counter()
    metrics = evaluate(model, test, args, device, amp_dtype)
    sync(device)
    result = {'architecture': architecture, 'seed': seed, 'parameters': parameters,
              'best_epoch': selected['epoch'], 'training_seconds': cumulative_seconds,
              'test_seconds': time.perf_counter()-began, 'peak_cuda_allocated_bytes': peak,
              'optimizer_steps': steps, **metrics}
    (output/'test.json').write_text(json.dumps(result, indent=2)+'\n')
    del model, optimizer
    gc.collect()
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    return result


def summarize(results, output):
    output = Path(output)
    with (output/'comparison.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)
    summary = {}
    for arch in ('da3', 'pixel_depth'):
        rows = [row for row in results if row['architecture'] == arch]
        summary[arch] = {}
        for key in rows[0]:
            if key not in ('architecture', 'seed'):
                values = [row[key] for row in rows]
                summary[arch][key] = {'mean': float(np.mean(values)),
                                     'std': float(np.std(values, ddof=1)) if len(values)>1 else None}
    base = summary['da3']['raw_abs_rel']['mean']
    candidate = summary['pixel_depth']['raw_abs_rel']['mean']
    summary['pixel_depth_abs_rel_improvement_percent'] = 100*(base-candidate)/base if base else None
    seeds = sorted({row['seed'] for row in results})
    paired = []
    for seed in seeds:
        pair = {row['architecture']: row for row in results if row['seed'] == seed}
        if set(pair) != {'da3', 'pixel_depth'}:
            raise ValueError(f'Missing architecture for seed {seed}')
        baseline, candidate = pair['da3']['raw_abs_rel'], pair['pixel_depth']['raw_abs_rel']
        paired.append({'seed': seed, 'abs_rel_improvement_percent': 100*(baseline-candidate)/baseline if baseline else None})
    summary['seeds'] = seeds
    summary['paired_abs_rel_improvement'] = paired
    (output/'comparison.json').write_text(json.dumps(summary, indent=2)+'\n')
    report = ['# DA3 vs Pixel-Depth', '', f'Seeds: {seeds}. Média ± desvio padrão amostral entre seeds.', '',
              '| Métrica | DA3 | Pixel-Depth |', '| --- | --- | --- |']
    metrics = [('raw_abs_rel', 'AbsRel bruto ↓'), ('raw_rmse', 'RMSE bruto (m) ↓'),
               ('raw_delta1', 'δ₁ bruto ↑'), ('median_aligned_abs_rel', 'AbsRel alinhado ↓'),
               ('parameters', 'Parâmetros'), ('training_seconds', 'Treino (s)'),
               ('peak_cuda_allocated_bytes', 'Pico CUDA alocado (bytes)')]
    for key, label in metrics:
        values=[]
        for architecture in ('da3', 'pixel_depth'):
            item=summary[architecture][key]
            values.append(f"{item['mean']:.5f}" + (f" ± {item['std']:.5f}" if item['std'] is not None else ' (uma seed)'))
        report.append(f'| {label} | {values[0]} | {values[1]} |')
    report += ['', 'Melhoria percentual do Pixel-Depth em AbsRel bruto: '
               + str(summary['pixel_depth_abs_rel_improvement_percent']) + ' (positivo = menor erro).', '',
               'Ablação monocular com inicialização do zero. Não equivale ao DA3 oficial pré-treinado.',
               'As seeds não eliminam a incerteza da seleção de cenas deste split.']
    (output/'comparison.md').write_text('\n'.join(report)+'\n', encoding='utf-8')
    print('\nmodel           seed   AbsRel(raw)  RMSE(raw)  delta1(raw)  params', flush=True)
    for row in results:
        print(f"{row['architecture']:<15} {row['seed']:<6} {row['raw_abs_rel']:.5f}      {row['raw_rmse']:.5f}    {row['raw_delta1']:.5f}      {row['parameters']:,}")
    print('\nMédia ± desvio entre seeds (AbsRel bruto):', flush=True)
    for arch in ('da3', 'pixel_depth'):
        value = summary[arch]['raw_abs_rel']
        print(f"{arch}: {value['mean']:.5f}, std={value['std']}", flush=True)
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--work-dir', required=True, help='Dataset cache and experiment results directory')
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--size', choices=('small', 'large', 'tiny'), default='small')
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch-size', type=int, default=2)
    parser.add_argument('--accumulate', type=int, default=4)
    parser.add_argument('--workers', type=int, default=0)
    parser.add_argument('--height', type=int, default=224)
    parser.add_argument('--width', type=int, default=294)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight-decay', type=float, default=.01)
    parser.add_argument('--seeds', type=int, nargs='+', default=[42, 43, 44])
    parser.add_argument('--split-seed', type=int, default=42)
    parser.add_argument('--val-fraction', type=float, default=.1)
    parser.add_argument('--amp', choices=('off', 'bf16', 'fp16'), default='off')
    parser.add_argument('--gradient-checkpointing', action='store_true')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--visual-samples', type=int, default=8, help='Fixed test examples per seed; 0 disables PNG/HTML export')
    parser.add_argument('--visuals-only', action='store_true', help='Export existing best checkpoints without training')
    parser.add_argument('--prepare-only', action='store_true')
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--labeled-mat', help='Existing official NYUv2 labeled MAT; skip its download')
    parser.add_argument('--splits-mat', help='Existing official splits MAT; skip its download')
    args = parser.parse_args(argv)
    for name in ('epochs', 'batch_size', 'accumulate', 'height', 'width'):
        if getattr(args, name) <= 0:
            parser.error(f'{name} must be positive')
    if args.height % 14 or args.width % 14:
        parser.error('height/width must be multiples of 14 to avoid asymmetric padding effects')
    if args.workers < 0 or args.lr <= 0 or args.weight_decay < 0 or not 0 < args.val_fraction < 1:
        parser.error('Invalid workers/lr/weight-decay/val-fraction')
    if args.visual_samples < 0:
        parser.error('visual-samples must be nonnegative')
    if args.visuals_only and args.prepare_only:
        parser.error('visuals-only and prepare-only cannot be combined')
    if args.visuals_only and args.visual_samples == 0:
        parser.error('visuals-only requires visual-samples > 0')
    if len(set(args.seeds)) != len(args.seeds):
        parser.error('Seeds must be distinct')
    device = torch.device(args.device)
    if device.type == 'cuda' and not torch.cuda.is_available():
        parser.error('CUDA is unavailable; install the matching PyTorch build or use --device cpu')
    _amp(args, device)
    plan = {'protocol': PROTOCOL, 'dataset': 'NYUv2 labeled, official train/test + scene holdout',
            'models': ['da3 monocular '+args.size, 'pixel_depth monocular '+args.size],
            'initialization': 'both scratch; identical DPT initial weights',
            'image_size': [args.height, args.width], 'effective_batch': args.batch_size*args.accumulate,
            'epochs': args.epochs, 'seeds': args.seeds, 'device': args.device,
            'visual_samples': args.visual_samples, 'visuals_only': args.visuals_only,
            'evaluation': 'original-resolution NYU crop; per-image mean; raw and median-aligned; best raw val AbsRel',
            'not_official_reproduction': 'Controlled backbone ablation; no large-scale teacher/student pretraining'}
    print(json.dumps(plan, indent=2), flush=True)
    if args.dry_run:
        return
    root = Path(args.work_dir).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    checkpoint_budget = 64*1024**2 if args.visuals_only else {'tiny': 256*1024**2, 'small': 2*1024**3, 'large': 20*1024**3}[args.size]*len(args.seeds)
    dataset_budget = 0 if args.labeled_mat or (root/'downloads/nyu_depth_v2_labeled.mat').exists() else LABELED_BYTES
    if shutil.disk_usage(root).free < checkpoint_budget + dataset_budget:
        raise OSError(f'Need approximately {(checkpoint_budget+dataset_budget)/1024**3:.1f} GiB free in {root}; select a larger disk')
    (root/('visual_plan.json' if args.visuals_only else 'plan.json')).write_text(json.dumps(plan, indent=2)+'\n')
    labeled = Path(args.labeled_mat).expanduser().resolve() if args.labeled_mat else download(
        LABELED_URL, root/'downloads/nyu_depth_v2_labeled.mat', LABELED_BYTES)
    splits = Path(args.splits_mat).expanduser().resolve() if args.splits_mat else download(
        SPLITS_URL, root/'downloads/splits.mat')
    split = prepare_split(labeled, splits, root/'split.json', args.split_seed, args.val_fraction)
    print({name: len(split[name]) for name in ('train', 'val', 'test')}, flush=True)
    if args.prepare_only:
        return
    runs = root/'runs'
    if args.visuals_only:
        from .visual_comparison import export_visual_comparison
        export_visual_comparison(labeled, split, runs, root/'visuals', args.seeds,
                                 args.visual_samples, args.device, args.split_seed)
        return
    results = []
    for seed in args.seeds:
        for architecture in ('da3', 'pixel_depth'):
            results.append(train_one(args, architecture, seed, split, labeled, runs))
    summarize(results, root)
    if args.visual_samples:
        from .visual_comparison import export_visual_comparison
        export_visual_comparison(labeled, split, runs, root/'visuals', args.seeds,
                                 args.visual_samples, args.device, args.split_seed)
        print(f"Gallery: {root/'visuals/index.html'}", flush=True)


if __name__ == '__main__':
    main()
