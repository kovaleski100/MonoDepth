# SPDX-License-Identifier: Apache-2.0
import torch


def _depth_mask(target, valid=None):
    mask = torch.isfinite(target) & (target > 0)
    if valid is not None:
        if valid.shape != target.shape:
            raise ValueError('valid mask shape must match target')
        mask &= valid.bool()
    if not mask.any():
        raise ValueError('No valid depth supervision')
    return mask

def adaptive_berhu_loss_RMSE(
    prediction, target, valid=None, alpha=0.2, rmse_weight=1.0
):
    """Mean per-image BerHu + weighted RMSE for inputs [B,S,H,W]."""
    depth = prediction['depth']

    if depth.shape != target.shape:
        raise ValueError('Target and predicted depth shapes must match')
    if not 0 < alpha <= 1:
        raise ValueError('alpha must be in (0,1]')
    if rmse_weight < 0:
        raise ValueError('rmse_weight must be nonnegative')

    mask = _depth_mask(target, valid)

    losses = []

    for predicted, expected, selected in zip(
        depth.flatten(0, 1),
        target.flatten(0, 1),
        mask.flatten(0, 1),
    ):
        if not selected.any():
            continue

        residual = predicted[selected] - expected[selected]
        error = residual.abs()

        c = alpha * error.detach().max()
        c = c.clamp_min(torch.finfo(error.dtype).eps)

        penalty = torch.where(
            error <= c,
            error,
            (error.square() + c.square()) / (2 * c),
        )

        berhu = penalty.mean()

        # Equivale a sqrt(mean(residual²)), com gradiente
        # bem definido pelo PyTorch quando todos os erros são zero.
        rmse = torch.linalg.vector_norm(residual) / (residual.numel() ** 0.5)

        losses.append(berhu + rmse_weight * rmse)

    return torch.stack(losses).mean()
def adaptive_berhu_loss(prediction, target, valid=None, alpha=0.2):
    """Per-image adaptive BerHu depth loss.

    Inputs [B,S,H,W]. Invalid, nonpositive and nonfinite targets
    are excluded. The threshold is calculated per image.
    """
    depth = prediction['depth']

    if depth.shape != target.shape:
        raise ValueError('Target and predicted depth shapes must match')
    if not 0 < alpha <= 1:
        raise ValueError('alpha must be in (0,1]')

    mask = _depth_mask(target, valid)

    losses = []

    for predicted, expected, selected in zip(
        depth.flatten(0, 1),
        target.flatten(0, 1),
        mask.flatten(0, 1),
    ):
        if not selected.any():
            continue

        # Seleciona antes da subtração para excluir targets NaN/Inf.
        error = (predicted[selected] - expected[selected]).abs()

        c = alpha * error.detach().max()
        c = c.clamp_min(torch.finfo(error.dtype).eps)

        penalty = torch.where(
            error <= c,
            error,
            (error.square() + c.square()) / (2 * c),
        )

        losses.append(penalty.mean())

    return torch.stack(losses).mean()

def depth_loss(prediction, target, valid=None, scale_invariant_weight=0.5):
    """Per-image log-depth loss, with optional confidence likelihood (DA3-style).

    Inputs [B,S,H,W]. Invalid, nonpositive and nonfinite targets are excluded.
    Confidence requires supervision here; it is not calibrated uncertainty.
    """
    depth = prediction['depth']
    if depth.shape != target.shape:
        raise ValueError('Target and predicted depth shapes must match')
    if not 0 <= scale_invariant_weight <= 1:
        raise ValueError('scale_invariant_weight must be in [0,1]')
    mask = _depth_mask(target, valid)
    safe_target = torch.where(mask, target, torch.ones_like(target))
    residual = torch.log(depth.clamp_min(1e-6)) - torch.log(safe_target)
    losses = []
    for error, selected in zip(residual.flatten(0, 1), mask.flatten(0, 1)):
        e = error[selected]
        if e.numel():
            losses.append(e.square().mean() - scale_invariant_weight * e.mean().square())
    loss = torch.stack(losses).mean()
    if 'depth_conf' in prediction:
        confidence = prediction['depth_conf'][mask].clamp_min(1e-6)
        loss = loss + 0.1 * (confidence * residual[mask].abs() - confidence.log()).mean()
    return loss


@torch.no_grad()
def depth_metrics(prediction, target, valid=None):
    """Raw-scale metrics, without median alignment or hidden rescaling."""
    mask = torch.isfinite(target) & (target > 0)
    if valid is not None:
        mask &= valid.bool()
    if not mask.any():
        raise ValueError('No valid depth supervision')
    p, t = prediction['depth'][mask].clamp_min(1e-6), target[mask]
    ratio = torch.maximum(p / t, t / p)
    return {'abs_rel': ((p-t).abs()/t).mean().item(),
            'rmse': (p-t).square().mean().sqrt().item(),
            'delta1': (ratio < 1.25).float().mean().item(), 'valid_pixels': int(mask.sum()),
            'delta2': (ratio < (1.25*1.25)).float().mean().item(), 'valid_pixels': int(mask.sum()),
            'delta3': (ratio < (1.25*1.25*1.25)).float().mean().item(), 'valid_pixels': int(mask.sum())}


def camera_loss(prediction, extrinsics, intrinsics, image_size):
    """Supervise predicted world-to-camera transforms and normalized calibration.

    Ground-truth poses must already use the first-view coordinate frame and the
    same translation units as depth. No pose alignment is performed implicitly.
    """
    if 'extrinsics' not in prediction:
        raise ValueError('Camera supervision requires predict_cameras=True')
    h, w = image_size
    pred = prediction['extrinsics']
    gt = extrinsics[..., :3, :].float()
    rotation = (pred[..., :3, :3] - gt[..., :3, :3]).square().mean()
    translation = (pred[..., :3, 3] - gt[..., :3, 3]).abs().mean()
    scale = intrinsics.new_tensor([[w, w, w], [h, h, h], [1, 1, 1]])
    calibration = ((prediction['intrinsics'] - intrinsics) / scale).square().mean()
    return rotation + translation + calibration


def ray_loss(prediction, target, valid=None):
    """L1 supervision for DA3's six-channel per-pixel ray map.

    ``target`` and the model output use channel-last layout ``[B,S,H,W,6]``.
    A ray is valid only when all six values are finite and its corresponding
    depth pixel is valid.
    """
    if 'ray' not in prediction:
        raise ValueError('Ray supervision requires predict_rays=True')
    pred = prediction['ray']
    if pred.shape != target.shape or target.ndim != 5 or target.shape[-1] != 6:
        raise ValueError('ray target must match prediction with shape [B,S,H,W,6]')
    mask = torch.isfinite(target).all(dim=-1)
    if valid is not None:
        if valid.shape != mask.shape:
            raise ValueError('ray valid mask must have shape [B,S,H,W]')
        mask &= valid.bool()
    if not mask.any():
        raise ValueError('No valid ray supervision')
    error = (pred - target).abs()
    loss = error[mask].mean()
    if 'ray_conf' in prediction:
        confidence = prediction['ray_conf'][mask].clamp_min(1e-6)
        loss = loss + 0.1 * (confidence * error.detach().mean(dim=-1)[mask] - confidence.log()).mean()
    return loss


def da3_loss(prediction, depth, valid=None, *, ray=None, ray_valid=None,
             teacher_depth=None, teacher_valid=None, ray_weight=1.0,
             teacher_weight=1.0, camera=None, camera_weight=0.0):
    """Combined DA3-style objective.

    The depth target is mandatory.  Ray and teacher targets are optional at
    the API level, but enabling their weights without supplying the targets is
    an error in the training CLI.  Teacher depth is expected to be generated
    offline by Depth-Anything-3-Teacher and stored in the sample manifest.
    """
    loss = depth_loss(prediction, depth, valid)
    if ray is not None:
        loss = loss + ray_weight * ray_loss(prediction, ray, ray_valid if ray_valid is not None else valid)
    if teacher_depth is not None:
        loss = loss + teacher_weight * depth_loss(prediction, teacher_depth, teacher_valid)
    if camera is not None:
        extrinsics, intrinsics, image_size = camera
        loss = loss + camera_weight * camera_loss(prediction, extrinsics, intrinsics, image_size)
    return loss
