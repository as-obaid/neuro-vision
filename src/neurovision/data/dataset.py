"""Cached-volume dataset for the CNN arms."""

from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from monai.utils import convert_to_tensor
from torch.utils.data import Dataset

from neurovision.data.augment import IMAGE, MAP
from neurovision.data.labels import ANEURYSM_PRESENT, LR_SWAP, TARGETS
from neurovision.data.points import ball_map, pool_map

LR_AXIS = 0  # RAS+ cache: array axis 0 runs patient left to right (checked on localizers)
AP = TARGETS.index(ANEURYSM_PRESENT)


def load_index(cache_dir: Path) -> pd.DataFrame:
    table = pd.read_csv(cache_dir / "index.csv", dtype={"reason": str})
    table["reason"] = table["reason"].fillna("")
    return table.set_index("SeriesInstanceUID")


class CacheDataset(Dataset[tuple[torch.Tensor, ...]]):
    """Cached volumes and 14 targets; with `points`, also the location-map target.

    Without `points` an item is (volume, targets). With `points` (series -> (k, 3) model-input
    voxels inside the cube) an item is (volume, targets, map, keep): the ball map of the expert
    points max-pooled to the layer3 grid, and keep = 0 for a positive with no point (its map is
    unknown), else 1. A negative's map is empty. `transform` then takes and returns a dict with
    "image" and "map" (`build_augment(..., with_map=True)`).
    """

    def __init__(
        self,
        cache_dir: Path,
        uids: list[str],
        targets: np.ndarray,
        transform: Callable[..., Any] | None,
        flip_lr: float = 0.0,
        points: Mapping[str, np.ndarray] | None = None,
    ) -> None:
        self.cache_dir = cache_dir
        self.uids = uids
        self.targets = targets.astype(np.float32)
        self.transform = transform
        self.flip_lr = flip_lr
        self.flip_seed = 0
        self.points = points

    def __len__(self) -> int:
        return len(self.uids)

    def set_flip_seed(self, seed: int) -> None:
        self.flip_seed = seed

    def _flips(self, i: int) -> bool:
        """Per-sample draw from `(flip_seed, i)`: independent of loader order and worker count."""
        return self.flip_lr > 0 and bool(
            np.random.default_rng([self.flip_seed, i]).random() < self.flip_lr
        )

    def __getitem__(self, i: int) -> tuple[torch.Tensor, ...]:
        volume = np.load(self.cache_dir / f"{self.uids[i]}.npy").astype(np.float32)
        target = self.targets[i]
        flipped = self._flips(i)
        if flipped:
            volume = np.ascontiguousarray(np.flip(volume, axis=LR_AXIS))
            target = target[list(LR_SWAP)]
        x = torch.from_numpy(volume).unsqueeze(0)
        y = torch.from_numpy(np.ascontiguousarray(target))
        if self.points is None:
            if self.transform is not None:
                # MONAI returns MetaTensor, which the default collate and the AMP path handle badly
                x = convert_to_tensor(self.transform(x), track_meta=False)
            return x, y
        volume_map, keep = self._map(i, volume.shape, flipped)
        m = torch.from_numpy(volume_map).unsqueeze(0)
        if self.transform is not None:
            out = self.transform({IMAGE: x, MAP: m})
            x = convert_to_tensor(out[IMAGE], track_meta=False)
            m = convert_to_tensor(out[MAP], track_meta=False)
        return x, y, pool_map(m), torch.tensor(keep, dtype=torch.float32)

    def _map(self, i: int, shape: tuple[int, ...], flipped: bool) -> tuple[np.ndarray, float]:
        """Ball map of series i's expert points in the (possibly flipped) volume, and keep."""
        assert self.points is not None
        if len(set(shape)) != 1:
            raise ValueError(f"the location map needs a cube, got {shape}")
        size = shape[0]
        empty = np.zeros(shape, dtype=np.uint8)
        if self.targets[i][AP] < 0.5:
            return empty, 1.0
        q = self.points.get(self.uids[i])
        if q is None or not len(q):
            return empty, 0.0
        if flipped:
            q = q.copy()
            q[:, LR_AXIS] = size - 1 - q[:, LR_AXIS]
        return ball_map(q, size), 1.0
