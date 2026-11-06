from dataclasses import replace
import torch
import pytest
from pixel_depth import PixelDepth, PixelDepthConfig, depth_loss
from pixel_depth.losses import camera_loss


torch.set_num_threads(1)


def tiny(**changes):
    cfg = PixelDepthConfig(hidden_size=32, intermediate_size=64, num_layers=4,
                           num_attention_heads=4, num_key_value_heads=2, head_dim=8,
                           out_layers=(0, 1, 2, 3), alt_start=2,
                           decoder_features=16, decoder_channels=(16, 24, 32, 48))
    return replace(cfg, **changes)


def test_multiview_depth_camera_and_padding_backward():
    torch.manual_seed(5)
    model = PixelDepth(tiny())
    images = torch.randn(2, 2, 3, 31, 45)
    out = model(images, export_feat_layers=(3, 0))
    assert out['depth'].shape == out['depth_conf'].shape == (2, 2, 31, 45)
    assert out['extrinsics'].shape == (2, 2, 3, 4)
    assert out['intrinsics'].shape == (2, 2, 3, 3)
    assert list(out['aux']) == [3, 0]
    assert out['aux'][3].shape == (2, 2, 2, 3, 32)
    assert (out['depth'] > 0).all()
    assert all(torch.isfinite(v).all() for v in out.values() if torch.is_tensor(v))
    ext = torch.eye(4).expand(2, 2, 4, 4).clone()
    intr = torch.tensor([[45., 0., 22.5], [0., 31., 15.5], [0., 0., 1.]]).expand(2, 2, 3, 3)
    loss = depth_loss(out, torch.ones_like(out['depth'])) + camera_loss(out, ext, intr, (31, 45))
    loss.backward()
    assert model.backbone.patch_embed.weight.grad.abs().sum() > 0
    assert model.backbone.layers[-1].self_attn.q_proj.weight.grad.abs().sum() > 0
    assert model.camera_decoder.fc_t.weight.grad.abs().sum() > 0


def test_other_view_affects_reference_only_with_global_attention():
    torch.manual_seed(7)
    images = torch.randn(1, 2, 3, 32, 48)
    changed = images.clone()
    changed[:, 1] += 3
    for alt, should_change in ((-1, False), (2, True)):
        model = PixelDepth(tiny(alt_start=alt)).eval()
        with torch.no_grad():
            a = model.backbone(images)[0][-1][0][:, 0]
            b = model.backbone(changed)[0][-1][0][:, 0]
        assert (not torch.allclose(a, b)) == should_change


def test_conditioning_and_checkpoint_roundtrip(tmp_path):
    model = PixelDepth(tiny()).eval()
    image = torch.randn(1, 2, 3, 32, 32)
    ext = torch.eye(4).expand(1, 2, 4, 4).clone()
    intr = torch.tensor([[32., 0., 16.], [0., 32., 16.], [0., 0., 1.]]).expand(1, 2, 3, 3)
    out = model(image, ext, intr)
    loss = out['depth'].mean()
    loss.backward()
    assert model.camera_encoder[0].weight.grad.abs().sum() > 0
    path = tmp_path / 'weights.pt'
    model.save(path)
    restored = PixelDepth.load(path).eval()
    torch.testing.assert_close(out['depth'], restored(image, ext, intr)['depth'], rtol=0, atol=0)


def test_checkpointed_gradients_match():
    torch.manual_seed(8)
    a = PixelDepth(tiny())
    b = PixelDepth(tiny(gradient_checkpointing=True))
    b.load_state_dict(a.state_dict())
    image = torch.randn(1, 2, 3, 32, 32)
    aa, bb = a(image)['depth'], b(image)['depth']
    torch.testing.assert_close(aa, bb)
    aa.mean().backward()
    bb.mean().backward()
    for (name, ap), (_, bp) in zip(a.named_parameters(), b.named_parameters()):
        if ap.grad is not None:
            torch.testing.assert_close(ap.grad, bp.grad, msg=name)


def test_mono_invalid_targets_and_pixelseg_transfer():
    model = PixelDepth(tiny(alt_start=-1, cat_local_global=False,
                            predict_cameras=False, depth_confidence=False))
    image = torch.randn(1, 3, 17, 29)
    out = model(image)
    assert out['depth'].shape == (1, 1, 17, 29)
    assert 'extrinsics' not in out
    target = torch.ones_like(out['depth'])
    target[..., 0, 0] = float('nan')
    target[..., 0, 1] = 0
    loss = depth_loss(out, target)
    assert torch.isfinite(loss)
    loss.backward()
    with pytest.raises(ValueError, match='No valid'):
        depth_loss(out, torch.zeros_like(target))
    weights = {'backbone.' + k: v for k, v in model.backbone.state_dict().items() if k != 'camera_token'}
    report = model.load_pixelseg_backbone(weights)
    assert report.missing_keys == ['camera_token']
    bad = dict(weights)
    bad['backbone.patch_embed.weight'] = torch.zeros(3, 3)
    with pytest.raises(ValueError, match='Incompatible'):
        model.load_pixelseg_backbone(bad)


def test_patch14_backbone_without_segmentation_decoder_constraint():
    model = PixelDepth(tiny(patch_size=14))
    assert model(torch.randn(1, 1, 3, 29, 41))['depth'].shape == (1, 1, 29, 41)
