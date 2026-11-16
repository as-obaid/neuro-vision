"""Vessel pretraining network: the r3d_18 encoder plus a U-Net decoder to 27 classes per voxel.

The encoder keeps r3d_18's layer names (stem, layer1..layer4), so `encoder_state` hands its
weights straight to the classifier (`build_model(..., init="vessel")`).
"""

from collections.abc import Mapping
from typing import cast

import torch
import torch.nn.functional as F
from torch import nn

from neurovision.data.vessels import N_SEG_CLASSES

ENCODER = ("stem.", "layer1.", "layer2.", "layer3.", "layer4.")
DECODER_CHANNELS = (256, 128, 64, 32)  # at layer3, layer2, layer1 and full size


def _block(cin: int, cout: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv3d(cin, cout, 3, padding=1, bias=False),
        nn.BatchNorm3d(cout),
        nn.ReLU(inplace=True),
        nn.Conv3d(cout, cout, 3, padding=1, bias=False),
        nn.BatchNorm3d(cout),
        nn.ReLU(inplace=True),
    )


class R3DUNet(nn.Module):
    def __init__(self, base: nn.Module | None = None, n_classes: int = N_SEG_CLASSES) -> None:
        super().__init__()
        if base is None:
            from neurovision.models.resnet3d import build_model

            base = build_model("r3d18")
        self.stem = cast(nn.Module, base.stem)
        self.layer1 = cast(nn.Module, base.layer1)
        self.layer2 = cast(nn.Module, base.layer2)
        self.layer3 = cast(nn.Module, base.layer3)
        self.layer4 = cast(nn.Module, base.layer4)
        c3, c2, c1, c0 = DECODER_CHANNELS
        self.up3 = _block(512 + 256, c3)
        self.up2 = _block(c3 + 128, c2)
        self.up1 = _block(c2 + 64, c1)
        self.up0 = _block(c1 + 1, c0)  # full size; the raw image is the last skip
        self.seg = nn.Conv3d(c0, n_classes, 1)

    @staticmethod
    def _up(x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, size=skip.shape[2:], mode="trilinear", align_corners=False)
        return torch.cat([x, skip], dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decode(x)[0]

    def decode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """(class logits at full size, up1 features at layer1's grid (64 channels; half size
        in-plane for r3d_18), layer4 output)."""
        s1 = self.layer1(self.stem(x))
        s2 = self.layer2(s1)
        s3 = self.layer3(s2)
        bottom = self.layer4(s3)
        d = self.up3(self._up(bottom, s3))
        d = self.up2(self._up(d, s2))
        d1 = self.up1(self._up(d, s1))
        return self.seg(self.up0(self._up(d1, x))), d1, bottom


def encoder_state(model: nn.Module) -> dict[str, torch.Tensor]:
    return {k: v.detach().clone() for k, v in model.state_dict().items() if k.startswith(ENCODER)}


def load_encoder(model: nn.Module, state: Mapping[str, torch.Tensor]) -> None:
    """The pretrained encoder into an r3d18 classifier; its fc and aux heads stay random."""
    missing, unexpected = model.load_state_dict(state, strict=False)
    wrong = {k for k in missing if k.startswith(ENCODER)} | set(unexpected)
    if wrong:
        raise ValueError(f"vessel encoder does not fit the model: {sorted(wrong)}")
