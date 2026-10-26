"""Cached-volume dataset for the CNN arms."""

from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from monai.utils import convert_to_tensor
from torch.utils.data import Dataset


def load_index(cache_dir: Path) -> pd.DataFrame:
    return pd.read_csv(cache_dir / "index.csv", index_col="SeriesInstanceUID")


class CacheDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    def __init__(
        self,
        cache_dir: Path,
        uids: list[str],
        targets: np.ndarray,
        transform: Callable[[torch.Tensor], torch.Tensor] | None,
    ) -> None:
        self.cache_dir = cache_dir
        self.uids = uids
        self.targets = targets.astype(np.float32)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.uids)

    def __getitem__(self, i: int) -> tuple[torch.Tensor, torch.Tensor]:
        volume = np.load(self.cache_dir / f"{self.uids[i]}.npy").astype(np.float32)
        x = torch.from_numpy(volume).unsqueeze(0)
        if self.transform is not None:
            # MONAI returns MetaTensor, which the default collate and the AMP path handle badly
            x = convert_to_tensor(self.transform(x), track_meta=False)
        return x, torch.from_numpy(self.targets[i])
