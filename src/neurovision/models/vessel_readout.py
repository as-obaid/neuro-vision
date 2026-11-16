"""Full version (1st-place style): the step-1 U-Net keeps predicting the 27 vessel classes, and
each artery's predicted mask reads that artery's logit from the decoder.

For artery k the soft mask is p(artery k) + p(aneurysm on artery k), from the U-Net's softmax and
averaged down to the up1 grid (64 channels, half size in-plane). Its logit is a per-artery linear
map of (the mask-weighted mean of the up1 features, the highest p(aneurysm on k) anywhere).
Aneurysm Present reads (the mean of the up1 features over the union of the 13 masks, the mean of
layer4, the highest aneurysm probability of any class). The masks are detached where they weight
the features, so the classification loss does not redraw them; the aneurysm scores are not.
"""

import math

import torch
import torch.nn.functional as F
from torch import nn

from neurovision.data.labels import LOCATIONS
from neurovision.models.unet import DECODER_CHANNELS, R3DUNet

N_ARTERIES = len(LOCATIONS)
READOUT_CHANNELS = DECODER_CHANNELS[2]  # up1
BOTTOM_CHANNELS = 512  # r3d_18 layer4
EPS = 1e-6


def masked_average(weights: torch.Tensor, features: torch.Tensor) -> torch.Tensor:
    """(n, k, *grid) weights and (n, c, *grid) features -> (n, k, c) weighted means."""
    w = weights.flatten(2)
    f = features.flatten(2)
    return torch.einsum("nkv,ncv->nkc", w, f) / (w.sum(-1, keepdim=True) + EPS)


class Readout(nn.Module):
    """13 artery logits and Aneurysm Present, in `TARGETS` order, from the U-Net's outputs."""

    def __init__(self, channels: int = READOUT_CHANNELS, bottom: int = BOTTOM_CHANNELS) -> None:
        super().__init__()
        bound = 1 / math.sqrt(channels + 1)  # nn.Linear's default range
        self.artery_weight = nn.Parameter(
            torch.empty(N_ARTERIES, channels + 1).uniform_(-bound, bound)
        )
        self.artery_bias = nn.Parameter(torch.empty(N_ARTERIES).uniform_(-bound, bound))
        self.ap = nn.Linear(channels + bottom + 1, 1)

    def forward(
        self, probs: torch.Tensor, features: torch.Tensor, bottom: torch.Tensor
    ) -> torch.Tensor:
        """`probs` (n, 27, *full) softmax; `features` (n, c, *grid); `bottom` (n, b, *coarse)."""
        arteries = probs[:, 1 : 1 + N_ARTERIES] + probs[:, 1 + N_ARTERIES :]
        grid = (features.shape[2], features.shape[3], features.shape[4])
        masks = F.adaptive_avg_pool3d(arteries.detach(), grid)
        union = masks.sum(1, keepdim=True).clamp(max=1.0)
        pooled = masked_average(masks, features)  # (n, 13, c)
        aneurysm = probs[:, 1 + N_ARTERIES :].flatten(2).amax(-1)  # (n, 13)
        per_artery = torch.cat([pooled, aneurysm[..., None]], dim=-1)
        arteries_out = (per_artery * self.artery_weight).sum(-1) + self.artery_bias
        present = torch.cat(
            [
                masked_average(union, features)[:, 0],
                bottom.mean((2, 3, 4)),
                aneurysm.amax(-1, keepdim=True),
            ],
            dim=-1,
        )
        return torch.cat([arteries_out, self.ap(present)], dim=-1)


class VesselReadout(nn.Module):
    """`unet` (R3DUNet, the step-1 weights) plus `fc` (`Readout`, the only new layer).

    `forward` returns the 14 logits, so prediction, EMA and scoring treat it like any model;
    `forward_with_seg` also returns the 27-class logits for the segmentation loss.
    """

    def __init__(self, unet: R3DUNet | None = None) -> None:
        super().__init__()
        self.unet = unet if unet is not None else R3DUNet()
        self.fc = Readout()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward_with_seg(x)[0]

    def forward_with_seg(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        seg, features, bottom = self.unet.decode(x)
        # float32 for the softmax and the pooling sums over millions of voxels
        with torch.autocast(x.device.type, enabled=False):
            probs = torch.softmax(seg.float(), dim=1)
            logits = self.fc(probs, features.float(), bottom.float())
        return logits, seg
