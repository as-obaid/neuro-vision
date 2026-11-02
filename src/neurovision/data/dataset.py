"""Cached-volume dataset for the CNN arms."""

from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from monai.utils import convert_to_tensor
from torch.utils.data import Dataset

from neurovision.data.labels import LR_SWAP

LR_AXIS = 0  # RAS+ cache: array axis 0 runs patient left to right (checked on localizers)


def load_index(cache_dir: Path) -> pd.DataFrame:
    table = pd.read_csv(cache_dir / "index.csv", dtype={"reason": str})
    table["reason"] = table["reason"].fillna("")
    return table.set_index("SeriesInstanceUID")


class CacheDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    def __init__(
        self,
        cache_dir: Path,
        uids: list[str],
        targets: np.ndarray,
        transform: Callable[[torch.Tensor], torch.Tensor] | None,
        flip_lr: float = 0.0,
    ) -> None:
        self.cache_dir = cache_dir
        self.uids = uids
        self.targets = targets.astype(np.float32)
        self.transform = transform
        self.flip_lr = flip_lr
        self.flip_seed = 0

    def __len__(self) -> int:
        return len(self.uids)

    def set_flip_seed(self, seed: int) -> None:
        self.flip_seed = seed

    def _flips(self, i: int) -> bool:
        """Per-sample draw from `(flip_seed, i)`: independent of loader order and worker count."""
        return self.flip_lr > 0 and bool(
            np.random.default_rng([self.flip_seed, i]).random() < self.flip_lr
        )

    def __getitem__(self, i: int) -> tuple[torch.Tensor, torch.Tensor]:
        volume = np.load(self.cache_dir / f"{self.uids[i]}.npy").astype(np.float32)
        target = self.targets[i]
        if self._flips(i):
            volume = np.ascontiguousarray(np.flip(volume, axis=LR_AXIS))
            target = target[list(LR_SWAP)]
        x = torch.from_numpy(volume).unsqueeze(0)
        if self.transform is not None:
            # MONAI returns MetaTensor, which the default collate and the AMP path handle badly
            x = convert_to_tensor(self.transform(x), track_meta=False)
        return x, torch.from_numpy(np.ascontiguousarray(target))
