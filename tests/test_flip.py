from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from neurovision.config import AugmentConfig
from neurovision.data.dataset import LR_AXIS, CacheDataset
from neurovision.data.labels import LR_SWAP, TARGETS
from neurovision.train.loop import train_loader

SIZE = 8
UIDS = ["s1", "s2", "s3", "s4"]
PAIRS = [
    ("Left Infraclinoid Internal Carotid Artery", "Right Infraclinoid Internal Carotid Artery"),
    ("Left Supraclinoid Internal Carotid Artery", "Right Supraclinoid Internal Carotid Artery"),
    ("Left Middle Cerebral Artery", "Right Middle Cerebral Artery"),
    ("Left Anterior Cerebral Artery", "Right Anterior Cerebral Artery"),
    ("Left Posterior Communicating Artery", "Right Posterior Communicating Artery"),
]


@pytest.fixture
def cache(tmp_path: Path) -> Path:
    rng = np.random.default_rng(0)
    for uid in UIDS:
        np.save(tmp_path / f"{uid}.npy", rng.normal(size=(SIZE,) * 3).astype(np.float16))
    return tmp_path


def _targets() -> np.ndarray:
    return np.arange(len(UIDS) * 14, dtype=np.float32).reshape(len(UIDS), 14)


def test_swap_exchanges_exactly_the_five_pairs() -> None:
    expected = list(range(14))
    for left, right in PAIRS:
        i, j = TARGETS.index(left), TARGETS.index(right)
        expected[i], expected[j] = j, i
    assert list(LR_SWAP) == expected
    assert sum(k != v for k, v in enumerate(LR_SWAP)) == 10


def test_swap_is_its_own_inverse() -> None:
    swap = np.array(LR_SWAP)
    np.testing.assert_array_equal(swap[swap], np.arange(14))


def test_lr_axis_is_the_first_volume_axis() -> None:
    assert LR_AXIS == 0


def test_flipped_sample_is_mirrored_with_permuted_targets(cache: Path) -> None:
    ds = CacheDataset(cache, UIDS, _targets(), None, flip_lr=1.0)
    plain = CacheDataset(cache, UIDS, _targets(), None)
    x, y = ds[2]
    px, py = plain[2]
    np.testing.assert_array_equal(x[0].numpy(), np.flip(px[0].numpy(), axis=LR_AXIS))
    np.testing.assert_array_equal(y.numpy(), py.numpy()[list(LR_SWAP)])
    assert x.is_contiguous()


def test_flip_zero_leaves_samples_untouched(cache: Path) -> None:
    ds = CacheDataset(cache, UIDS, _targets(), None, flip_lr=0.0)
    plain = CacheDataset(cache, UIDS, _targets(), None)
    for i in range(len(UIDS)):
        torch.testing.assert_close(ds[i][0], plain[i][0])
        torch.testing.assert_close(ds[i][1], plain[i][1])


def test_validation_dataset_never_flips(cache: Path) -> None:
    val = CacheDataset(cache, UIDS, _targets(), None)
    assert val.flip_lr == 0.0
    plain = CacheDataset(cache, UIDS, _targets(), None)
    for i in range(len(UIDS)):
        torch.testing.assert_close(val[i][0], plain[i][0])


def test_flip_is_reproducible_from_seed_and_mixes(cache: Path) -> None:
    ds = CacheDataset(cache, UIDS, _targets(), None, flip_lr=0.5)
    cfg = SimpleNamespace(batch_size=2, num_workers=0)

    def flags(seed: int, epoch: int) -> list[bool]:
        train_loader(ds, cfg, seed, epoch, False)
        return [bool(ds[i][1][0] != _targets()[i][0]) for i in range(len(UIDS))]

    assert flags(7, 0) == flags(7, 0)
    seen = {tuple(flags(s, e)) for s in range(5) for e in range(3)}
    assert len(seen) > 1
    assert any(any(f) for f in seen)
    assert any(not all(f) for f in seen)


def test_config_default_is_zero_and_bounded() -> None:
    base = dict(
        prob=0.2, noise_std=0.1, gamma=(0.7, 1.5), rotate_deg=10, scale=0.1, lowres_zoom=(0.5, 1.0)
    )
    assert AugmentConfig(**base).flip_lr == 0.0  # type: ignore[arg-type]
    for bad in (-0.1, 1.1):
        with pytest.raises(ValueError):
            AugmentConfig(**base, flip_lr=bad)  # type: ignore[arg-type]
