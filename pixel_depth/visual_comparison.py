# SPDX-License-Identifier: Apache-2.0
"""Fixed test examples, shared metric color scales, and a local HTML gallery."""
import html
import json
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
import torch
from torch.nn import functional as F
from .comparison_models import load_comparison_checkpoint
from .nyu import NYUDataset


def _colors(values, valid, maximum):
    # Same fixed scale for every architecture, scene and seed; invalid pixels black.
    anchors = np.array([[0,0,4], [66,10,104], [147,38,103], [221,81,58], [252,165,10], [252,255,164]], dtype=float)
    position = np.clip(np.nan_to_num(values, nan=0., posinf=maximum, neginf=0.)/maximum, 0., 1.)*(len(anchors)-1)
    index = np.minimum(position.astype(int), len(anchors)-2)
    rgb = anchors[index]*(1-(position-index))[...,None] + anchors[index+1]*(position-index)[...,None]
    rgb[~valid] = 0
    return Image.fromarray(rgb.astype('uint8'))


def _board(rgb, target, da3, pixel, valid, title):
    h,w = target.shape
    label = 28
    board = Image.new('RGB', (4*w, 2*(h+label)+label), 'white')
    draw = ImageDraw.Draw(board)
    draw.text((8,8), title, fill='black')
    panels = [
        (Image.fromarray(rgb), 'RGB original'),
        (_colors(target,valid,10.), 'Ground truth (0-10 m)'),
        (_colors(da3,valid,10.), 'DA3 depth (0-10 m)'),
        (_colors(pixel,valid,10.), 'Pixel-Depth depth (0-10 m)'),
        (None, 'Same color scale; black = invalid'),
        (Image.fromarray(np.repeat((valid*255).astype('uint8')[...,None],3,axis=-1)), 'Valid evaluation mask'),
        (_colors(np.abs(da3-target),valid,2.), 'DA3 absolute error (0-2 m; saturated)'),
        (_colors(np.abs(pixel-target),valid,2.), 'Pixel-Depth absolute error (0-2 m; saturated)'),
    ]
    for i,(picture,text) in enumerate(panels):
        x,y = (i%4)*w, label+(i//4)*(h+label)
        draw.text((x+8,y+8),text,fill='black')
        if picture is not None:
            board.paste(picture,(x,y+label))
        else:
            # Color legend for the shared depth scale, low -> high.
            gradient = np.tile(np.linspace(0,10,w,dtype='float32'),(40,1))
            board.paste(_colors(gradient,np.ones_like(gradient,dtype=bool),10.),(x,y+label+20))
            draw.text((x+8,y+label+68),'Depth: 0 m -> 10 m',fill='black')
            draw.text((x+8,y+label+90),'Error: 0 m -> 2 m (same palette)',fill='black')
            draw.text((x+8,y+label+112),'Raw metric predictions; no median alignment',fill='black')
    return board


@torch.no_grad()
def export_visual_comparison(labeled, split, run_root, output, seeds, count=8, device='cpu', selection_seed=42):
    if count <= 0:
        return None
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    selected = np.random.default_rng(selection_seed).choice(split['test'],size=min(count,len(split['test'])),replace=False).tolist()
    records = []
    for seed in seeds:
        paths = {arch: Path(run_root)/f'{arch}-seed{seed}'/'best.pt' for arch in ('da3','pixel_depth')}
        signatures, epochs = {}, {}
        for arch,path in paths.items():
            if not path.exists():
                raise FileNotFoundError(f'Missing {path}; choose --seeds for completed pairs or finish their training')
            payload = torch.load(path,map_location='cpu',weights_only=True)
            signatures[arch] = payload['signature']
            epochs[arch] = payload['epoch']
            del payload
        first = signatures['da3']
        for arch,signature in signatures.items():
            if signature['seed'] != seed or signature['architecture'] != arch:
                raise ValueError('Checkpoint architecture/seed mismatch')
            for key in ('image_size','split','size','protocol'):
                if signature[key] != first[key]:
                    raise ValueError(f'Checkpoints have different {key}; cannot compare visuals')
            if signature['split'] != split:
                raise ValueError('Checkpoint and requested dataset splits differ')
        dataset = NYUDataset(labeled,selected,first['image_size'])
        # Hold one model on the GPU at a time, matching the sequential training budget.
        predictions = {}
        for arch,path in paths.items():
            model = load_comparison_checkpoint(path,device)
            predictions[arch] = []
            mode = signatures[arch].get('amp', 'off')
            amp_dtype = {'bf16': torch.bfloat16, 'fp16': torch.float16}.get(mode) if torch.device(device).type == 'cuda' else None
            for i in range(len(dataset)):
                sample = dataset[i]
                with torch.autocast(torch.device(device).type, enabled=amp_dtype is not None, dtype=amp_dtype):
                    pred = model(sample['image'][None].to(device))['depth'].float()
                pred = F.interpolate(pred,sample['depth'].shape[-2:],mode='bilinear',align_corners=False)
                predictions[arch].append(pred[0,0].cpu().numpy())
            del model
            if torch.device(device).type == 'cuda':
                torch.cuda.empty_cache()
        seed_dir = output/f'seed{seed}'
        seed_dir.mkdir(exist_ok=True)
        for i,index in enumerate(selected):
            sample = dataset[i]
            target,valid = sample['depth'][0].numpy(),sample['valid'][0].numpy()
            rgb = np.asarray(dataset._file['images'][index]).transpose(2,1,0).copy()
            a,b = predictions['da3'][i],predictions['pixel_depth'][i]
            if not np.isfinite(a).all() or not np.isfinite(b).all():
                raise FloatingPointError('Nonfinite prediction during visual export')
            # Match raw metric clipping. Retain the common evaluation mask.
            a,b = a.clip(.001,10.),b.clip(.001,10.)
            relative = f'seed{seed}/sample_{index:04d}.png'
            _board(rgb,target,a,b,valid,f'NYUv2 index={index} (zero-based), seed={seed}, best epochs: DA3={epochs["da3"]}, Pixel-Depth={epochs["pixel_depth"]}').save(output/relative)
            records.append({'seed':seed,'sample_index':index,'image':relative,
                            'da3_epoch':epochs['da3'],'pixel_depth_epoch':epochs['pixel_depth'],
                            'da3_abs_rel':float((np.abs(a[valid]-target[valid])/target[valid]).mean()),
                            'pixel_depth_abs_rel':float((np.abs(b[valid]-target[valid])/target[valid]).mean())})
        print(f'Visual comparisons exported: seed={seed}, samples={len(selected)}',flush=True)
    manifest = {'selection_seed':selection_seed,'test_indices':selected,'scale':'raw depth, clipped to [0.001,10] meters; no median alignment',
                'inference_precision':'checkpoint AMP mode on CUDA; float32 on CPU',
                'depth_color_range_m':[0,10],'error_color_range_m':[0,2], 'examples':records}
    (output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    cards=[]
    for row in records:
        caption=f"Seed {row['seed']} / sample {row['sample_index']} — AbsRel: DA3 {row['da3_abs_rel']:.4f}, Pixel-Depth {row['pixel_depth_abs_rel']:.4f}"
        cards.append(f'<figure><figcaption>{html.escape(caption)}</figcaption><a href="{row["image"]}"><img loading="lazy" src="{row["image"]}" alt="{html.escape(caption)}"></a></figure>')
    (output/'index.html').write_text('<!doctype html><html lang="pt"><meta charset="utf-8"><title>Pixel-Depth comparison</title>'
        '<style>body{font:16px sans-serif;margin:24px;background:#eee}figure{margin:24px 0}img{width:100%;height:auto}figcaption{padding:8px;background:white}</style>'
        '<h1>DA3 vs Pixel-Depth</h1><p>Exemplos fixos do teste, iguais entre seeds. Escala de profundidade: 0–10 m; erro: 0–2 m. Preto = inválido. Sem alinhamento por mediana.</p>'
        '<p>Exemplos ilustrativos; a avaliação quantitativa usa todo o teste. Não selecione hiperparâmetros olhando estes exemplos.</p>'
        + ''.join(cards)+'</html>',encoding='utf-8')
    return manifest
