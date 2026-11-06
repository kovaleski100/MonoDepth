import csv
import json
import sys
from dataclasses import asdict
import numpy as np
from pixel_depth.cli import main
from test_model import tiny


def test_train_evaluate_infer_smoke(tmp_path, monkeypatch):
    cfg = tiny(predict_cameras=False, depth_confidence=False)
    config = tmp_path / 'config.json'
    config.write_text(json.dumps(asdict(cfg)))
    for partition in ('train', 'val'):
        np.savez(tmp_path / f'{partition}.npz', image=np.zeros((3, 17, 29), dtype='float32'),
                 depth=np.ones((17, 29), dtype='float32'))
        with (tmp_path / f'{partition}.csv').open('w', newline='') as stream:
            writer = csv.writer(stream)
            writer.writerow(['sample'])
            writer.writerow([f'{partition}.npz'])
    def run(*args):
        monkeypatch.setattr(sys, 'argv', ['pixel-depth', *map(str, args)])
        main()
    output = tmp_path / 'run'
    run('train', '--config', config, '--train-manifest', tmp_path/'train.csv',
        '--val-manifest', tmp_path/'val.csv', '--output', output, '--epochs', '1', '--device', 'cpu')
    assert (output/'last.pt').exists()
    assert 'validation' in json.loads((output/'metrics.jsonl').read_text())
    report = tmp_path/'report.json'
    run('evaluate', '--checkpoint', output/'last.pt', '--manifest', tmp_path/'val.csv', '--output', report)
    assert json.loads(report.read_text())['valid_pixels'] == 17*29
    result = tmp_path/'prediction.npz'
    run('infer', '--checkpoint', output/'last.pt', '--input', tmp_path/'val.npz', '--output', result)
    with np.load(result) as predictions:
        assert predictions['depth'].shape == (1, 17, 29)
