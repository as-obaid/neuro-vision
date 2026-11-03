"""Grad-CAM for the 3D CNNs: where in the volume one output's logit finds its evidence."""

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from neurovision.data.labels import ANEURYSM_PRESENT, TARGETS


def grad_cam(
    model: nn.Module,
    x: torch.Tensor,
    target: int = TARGETS.index(ANEURYSM_PRESENT),
    layer: nn.Module | None = None,
) -> np.ndarray:
    """CAM of output `target` for one volume `x` `(1, 1, D, H, W)`: float32 `(D, H, W)` in [0, 1].

    `layer` defaults to `model.layer4`, the last residual stage of both the MONAI ResNets and
    torchvision's r3d_18. Each of its channels is weighted by the mean gradient of the logit,
    the weighted sum is clipped at 0 and resized trilinearly to the input. A CAM with no positive
    evidence is all zeros. The model's mode and parameter gradients are left as they were.
    """
    if x.shape[0] != 1:
        raise ValueError(f"grad_cam takes one volume, not a batch of {x.shape[0]}")
    stage = model.layer4 if layer is None else layer
    assert isinstance(stage, nn.Module)
    saved: list[torch.Tensor] = []
    handle = stage.register_forward_hook(lambda _m, _i, out: saved.append(out))
    was_training = model.training
    model.eval()
    device = next(model.parameters()).device
    try:
        with torch.enable_grad(), torch.autocast(device.type, enabled=False):
            volume = x.to(device=device, dtype=torch.float32).requires_grad_(True)
            logit = model(volume)[0, target]
            (activation,) = saved
            (grad,) = torch.autograd.grad(logit, activation)
    finally:
        handle.remove()
        model.train(was_training)
    weights = grad.mean(dim=(2, 3, 4), keepdim=True)
    cam = F.relu((weights * activation.detach()).sum(dim=1, keepdim=True))
    cam = F.interpolate(cam, size=tuple(x.shape[2:]), mode="trilinear", align_corners=False)
    out = cam[0, 0].cpu().numpy().astype(np.float32)
    out -= out.min()
    peak = float(out.max())
    return out / peak if peak > 0 else np.zeros_like(out)
