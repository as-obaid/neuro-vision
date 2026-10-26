from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from neurovision.config import AugmentConfig
from neurovision.data.augment import build_augment
from neurovision.data.dataset import CacheDataset, load_index

SIZE = 32
UIDS = ["s1", "s2", "s3"]


@pytest.fixture
def cache(tmp_path: Path) -> Path:
    rng = np.random.default_rng(0)
    for uid in UIDS:
        np.save(tmp_path / f"{uid}.npy", rng.normal(size=(SIZE,) * 3).astype(np.float16))
    pd.DataFrame(
        {
            "SeriesInstanceUID": [*UIDS, "s4"],
            "status": ["ok"] * 3 + ["excluded"],
            "reason": ["", "", "", "geometry"],
        }
    ).to_csv(tmp_path / "index.csv", index=False)
    return tmp_path


def _cfg(prob: float) -> AugmentConfig:
    return AugmentConfig(
        prob=prob,
        noise_std=0.1,
        gamma=(0.7, 1.5),
        rotate_deg=10,
        scale=0.1,
        lowres_zoom=(0.5, 1.0),
    )


def test_load_index_is_indexed_by_uid(cache: Path) -> None:
    index = load_index(cache)
    assert index.index.name == "SeriesInstanceUID"
    assert index.loc["s4", "status"] == "excluded"
    assert len(index) == 4


def test_dataset_items(cache: Path) -> None:
    targets = np.arange(3 * 14, dtype=np.float64).reshape(3, 14)
    ds = CacheDataset(cache, UIDS, targets, None)
    assert len(ds) == 3
    x, y = ds[1]
    assert x.shape == (1, SIZE, SIZE, SIZE)
    assert x.dtype == torch.float32
    assert y.dtype == torch.float32
    assert y.shape == (14,)
    np.testing.assert_array_equal(y.numpy(), targets[1].astype(np.float32))
    stored = np.load(cache / "s2.npy").astype(np.float32)
    np.testing.assert_array_equal(x[0].numpy(), stored)


def test_dataset_applies_transform(cache: Path) -> None:
    ds = CacheDataset(cache, UIDS, np.zeros((3, 14)), lambda x: x * 2)
    plain = CacheDataset(cache, UIDS, np.zeros((3, 14)), None)
    torch.testing.assert_close(ds[0][0], plain[0][0] * 2)


def test_augment_keeps_shape_and_dtype() -> None:
    out = build_augment(_cfg(1.0))(torch.randn(1, SIZE, SIZE, SIZE))
    assert out.shape == (1, SIZE, SIZE, SIZE)
    assert out.dtype == torch.float32
    assert isinstance(out, torch.Tensor)


def test_augment_prob_zero_is_identity() -> None:
    x = torch.randn(1, SIZE, SIZE, SIZE)
    torch.testing.assert_close(build_augment(_cfg(0.0))(x.clone()), x)


def test_augment_prob_one_changes_volume() -> None:
    x = torch.randn(1, SIZE, SIZE, SIZE)
    assert not torch.allclose(build_augment(_cfg(1.0))(x.clone()), x)


def test_augment_is_repeatable_with_seed() -> None:
    x = torch.randn(1, SIZE, SIZE, SIZE)
    aug = build_augment(_cfg(1.0))
    aug.set_random_state(seed=3)
    a = aug(x.clone())
    aug.set_random_state(seed=3)
    b = aug(x.clone())
    torch.testing.assert_close(a, b)
