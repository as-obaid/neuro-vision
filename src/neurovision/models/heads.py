"""What every backbone shares: the 14-way output and the location-map head."""

import torch
import torch.nn.functional as F
from torch import nn

from neurovision.data.points import MAP_STRIDE

N_CLASSES = 14
MAP_CHANNELS = 32


def map_head(channels: int) -> nn.Sequential:
    """3x3x3 conv to 32 channels, BatchNorm, ReLU, then a 1x1x1 conv to one logit per cell."""
    return nn.Sequential(
        nn.Conv3d(channels, MAP_CHANNELS, 3, padding=1, bias=False),
        nn.BatchNorm3d(MAP_CHANNELS),
        nn.ReLU(inplace=True),
        nn.Conv3d(MAP_CHANNELS, 1, 1),
    )


def map_grid(x: torch.Tensor) -> tuple[int, int, int]:
    """The map target's grid for an input batch: r3d_18's layer3 cells, (D/4, H/8, W/8)."""
    d, h, w = x.shape[2:]
    return d // MAP_STRIDE[0], h // MAP_STRIDE[1], w // MAP_STRIDE[2]


def to_grid(features: torch.Tensor, grid: tuple[int, int, int]) -> torch.Tensor:
    """(n, c, d, h, w) features resampled to `grid`; halving an axis averages each pair."""
    if tuple(features.shape[2:]) == grid:
        return features
    return F.interpolate(features, size=grid, mode="trilinear", align_corners=False)


class MapModel(nn.Module):
    """A model that can carry the location-map head (`aux`, built only when asked for).

    `forward` returns the 14 logits, so prediction, EMA and scoring treat it like any model.
    `forward_with_map` also returns map logits on the layer3 grid, (n, 1, D/4, H/8, W/8).
    """

    aux: nn.Module | None

    def _features(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """(14 logits, channels-first features the map head reads)."""
        raise NotImplementedError

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._features(x)[0]

    def forward_with_map(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if self.aux is None:
            raise ValueError("this model was built without the map head (aux_map=False)")
        logits, mid = self._features(x)
        return logits, self.aux(to_grid(mid, map_grid(x)))
