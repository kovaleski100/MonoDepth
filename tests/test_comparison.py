from argparse import Namespace
import json
from pathlib import Path
import h5py
import numpy as np
import pytest
from scipy.io import savemat
import torch
from pixel_depth import depth_loss
from pixel_depth.comparison_models import build_comparison_model, load_comparison_checkpoint
from pixel_depth.compare import train_one, summarize, main
from pixel_depth.nyu import NYUDataset, prepare_split, download


@pytest.fixture
def nyu_fixture(tmp_path):
    mat = tmp_path/'nyu.mat'
    count = 8
    with h5py.File(mat, 'w') as file:
        images = file.create_dataset('images', (count, 3, 640, 480), dtype='u1', chunks=(1,3,640,480), compression='gzip')
        depths = file.create_dataset('depths', (count,640,480), dtype='f4', chunks=(1,640,480), compression='gzip')
        scenes = file.create_dataset('scenes', (1,count), dtype=h5py.ref_dtype)
        for i in range(count):
            images[i] = np.full((3,640,480), 30+i*20, dtype='uint8')
            depths[i] = np.full((640,480), 1+i*.1, dtype='float32')
            name = f'room_{i//2}'
            cell = file.create_dataset(f'name{i}', data=np.array([ord(c) for c in name],dtype='uint16')[:,None])
            scenes[0,i] = cell.ref
    splits = tmp_path/'splits.mat'
    savemat(splits, {'trainNdxs': np.arange(1,5)[:,None], 'testNdxs': np.arange(5,9)[:,None]})
    split = prepare_split(mat, splits, tmp_path/'split.json', val_fraction=.5, strict_counts=False)
    return mat, splits, split


def test_shared_head_and_original_backbone_gradient():
    torch.manual_seed(42)
    baseline, _ = build_comparison_model('da3', 'tiny', True)
    torch.manual_seed(42)
    pixel, _ = build_comparison_model('pixel_depth', 'tiny', True)
    for key, value in baseline.head.state_dict().items():
        torch.testing.assert_close(value, pixel.head.state_dict()[key], rtol=0, atol=0)
    for model in (baseline,pixel):
        out = model(torch.randn(1,3,28,42))
        depth_loss(out, torch.ones(1,1,28,42)).backward()
        assert all(p.grad is not None for p in model.backbone.parameters() if p.requires_grad)


def test_mat_orientation_scene_split_and_original_resolution(nyu_fixture):
    mat, _, split = nyu_fixture
    assert len(split['train']) == len(split['val']) == 2
    assert len(split['test']) == 4
    assert not set(split['train']) & set(split['val'])
    assert {i//2 for i in split['train']}.isdisjoint({i//2 for i in split['val']})
    train = NYUDataset(mat,split['train'],(28,42),True)[0]
    test = NYUDataset(mat,split['test'],(28,42))[0]
    assert train['image'].shape == (3,28,42)
    assert train['depth'].shape == (1,28,42)
    assert test['depth'].shape == (1,480,640)
    assert test['valid'].sum() == (471-45)*(601-41)


def test_two_trainings_summary_resume_and_load(tmp_path,nyu_fixture):
    mat, _, split = nyu_fixture
    args = Namespace(device='cpu',amp='off',size='tiny',gradient_checkpointing=True,
                     epochs=1,batch_size=1,accumulate=2,workers=0,height=28,width=42,
                     lr=1e-4,weight_decay=.01,resume=False)
    root=tmp_path/'runs'
    rows=[train_one(args,arch,42,split,mat,root) for arch in ('da3','pixel_depth')]
    summary=summarize(rows,tmp_path)
    assert rows[0]['optimizer_steps']==rows[1]['optimizer_steps']==1
    assert 'pixel_depth_abs_rel_improvement_percent' in summary
    model=load_comparison_checkpoint(root/'da3-seed42/best.pt')
    assert model(torch.zeros(1,3,28,42))['depth'].shape==(1,1,28,42)
    args.resume=True
    assert train_one(args,'da3',42,split,mat,root)==rows[0]
    # Exercise restoring full optimizer/RNG and evaluating from a finished last.pt.
    (root/'da3-seed42/test.json').unlink()
    restored=train_one(args,'da3',42,split,mat,root)
    assert restored['raw_abs_rel']==rows[0]['raw_abs_rel']


def test_download_resumes_or_restarts_without_corruption(tmp_path,monkeypatch):
    import io
    from pixel_depth import nyu
    data=b'valid dataset bytes'
    target=tmp_path/'download.mat'
    target.with_suffix('.mat.part').write_bytes(data[:5])
    class Response(io.BytesIO):
        status=200  # Server ignores Range: full payload must replace partial data.
        headers={'Content-Length': str(len(data))}
    monkeypatch.setattr(nyu,'urlopen',lambda *a,**k: Response(data))
    download('https://example.test/data',target,len(data))
    assert target.read_bytes()==data
    assert not target.with_suffix('.mat.part').exists()


def test_dry_run_does_not_create_workdir(tmp_path):
    root=tmp_path/'not-created'
    main(['--work-dir',str(root),'--device','cpu','--dry-run'])
    assert not root.exists()


def test_interrupted_training_resumes_same_weights(tmp_path,nyu_fixture,monkeypatch):
    from pixel_depth import compare
    mat, _, split=nyu_fixture
    args=Namespace(device='cpu',amp='off',size='tiny',gradient_checkpointing=False,
                   epochs=2,batch_size=1,accumulate=2,workers=0,height=28,width=42,
                   lr=1e-4,weight_decay=.01,resume=False)
    uninterrupted=tmp_path/'uninterrupted'
    interrupted=tmp_path/'interrupted'
    compare.train_one(args,'da3',42,split,mat,uninterrupted)
    evaluate_original=compare.evaluate
    calls=0
    def fail_second_validation(*a,**k):
        nonlocal calls
        calls+=1
        if calls==2:
            raise RuntimeError('Simulated interruption')
        return evaluate_original(*a,**k)
    monkeypatch.setattr(compare,'evaluate',fail_second_validation)
    with pytest.raises(RuntimeError,match='Simulated'):
        compare.train_one(args,'da3',42,split,mat,interrupted)
    monkeypatch.setattr(compare,'evaluate',evaluate_original)
    args.resume=True
    compare.train_one(args,'da3',42,split,mat,interrupted)
    a=torch.load(uninterrupted/'da3-seed42/last.pt',weights_only=True)
    b=torch.load(interrupted/'da3-seed42/last.pt',weights_only=True)
    assert a['steps']==b['steps']==2
    for key,value in a['model'].items():
        torch.testing.assert_close(value,b['model'][key],rtol=0,atol=0)


def test_visual_gallery_uses_same_test_samples_and_shared_scales(tmp_path, nyu_fixture):
    from PIL import Image
    from pixel_depth.compare import atomic_save
    from pixel_depth.visual_comparison import export_visual_comparison
    mat, _, split = nyu_fixture
    root=tmp_path/'runs'
    for seed in (42,43):
        for arch in ('da3','pixel_depth'):
            torch.manual_seed(seed)
            model,_=build_comparison_model(arch,'tiny')
            path=root/f'{arch}-seed{seed}'/'best.pt'
            path.parent.mkdir(parents=True)
            signature={'seed':seed,'architecture':arch,'size':'tiny','image_size':[28,42],
                       'split':split,'protocol':'nyuv2-monocular-scratch-v1','amp':'off'}
            atomic_save({'format':'pixel-depth-comparison-v1','signature':signature,
                         'model':model.state_dict(),'epoch':1},path)
    gallery=tmp_path/'visuals'
    manifest=export_visual_comparison(mat,split,root,gallery,[42,43],count=2)
    assert len(manifest['examples'])==4
    assert [r['sample_index'] for r in manifest['examples'][:2]]==[r['sample_index'] for r in manifest['examples'][2:]]
    assert set(manifest['test_indices']) <= set(split['test'])
    assert manifest['depth_color_range_m']==[0,10]
    assert manifest['error_color_range_m']==[0,2]
    for row in manifest['examples']:
        with Image.open(gallery/row['image']) as img:
            assert img.size==(2560,1044)
    assert (gallery/'index.html').exists()
    assert len(json.loads((gallery/'manifest.json').read_text())['examples'])==4
    assert export_visual_comparison(mat,split,root,gallery,[42],count=0) is None


def test_three_seeds_summary_has_sample_standard_deviation(tmp_path):
    rows=[]
    for seed,a,b in ((42,.29,.30),(43,.31,.28),(44,.30,.29)):
        for arch,value in (('da3',a),('pixel_depth',b)):
            rows.append(dict(architecture=arch,seed=seed,raw_abs_rel=value,raw_rmse=1.,
                             raw_delta1=.5,median_aligned_abs_rel=.2,parameters=10,
                             training_seconds=2.,peak_cuda_allocated_bytes=0))
    report=summarize(rows,tmp_path)
    assert report['seeds']==[42,43,44]
    assert report['da3']['raw_abs_rel']['mean']==pytest.approx(.30)
    assert report['da3']['raw_abs_rel']['std']==pytest.approx(.01)
    assert len(report['paired_abs_rel_improvement'])==3
    assert (tmp_path/'comparison.md').exists()
