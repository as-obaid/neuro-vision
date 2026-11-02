"""Synthetic store with a 32-cube cache for the B2 training tests."""

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

from neurovision.data.dataset import CacheDataset
from neurovision.data.labels import TARGETS

SIZE = 32
UIDS = [f"s{i:04d}" for i in range(12)]
EXCLUDED = ("s0003", "s0010")
REASONS = {"s0003": "error: unreadable", "s0010": "geometry"}
FOLDS = pd.Series([i % 5 for i in range(12)], index=pd.Index(UIDS, name="SeriesInstanceUID"))


def cnn_labels() -> pd.DataFrame:
    """Even series positive, odd negative, in every target."""
    return pd.DataFrame(
        {t: [int(i % 2 == 0) for i in range(12)] for t in TARGETS}, index=FOLDS.index, dtype="int8"
    )


def write_store(root: Path, cache_name: str = "b2-128") -> None:
    """Write folds and cache under `root`.

    Fold 0 validates on s0000 (positive) and s0005 (negative), with s0010 excluded; it trains on
    eight series, so no batch of two leaves a single-volume remainder.
    """
    (root / "folds").mkdir(parents=True)
    FOLDS.rename("fold").to_csv(root / "folds" / "folds.csv")
    sha = hashlib.sha256((root / "folds" / "folds.csv").read_bytes()).hexdigest()
    (root / "folds" / "folds.yaml").write_text(f"folds_sha256: {sha}\n")
    cache = root / "cache" / cache_name
    cache.mkdir(parents=True)
    rng = np.random.default_rng(0)
    for uid in UIDS:
        np.save(cache / f"{uid}.npy", rng.normal(size=(SIZE,) * 3).astype(np.float16))
    pd.DataFrame(
        {
            "SeriesInstanceUID": UIDS,
            "status": ["excluded" if u in EXCLUDED else "ok" for u in UIDS],
            "reason": [REASONS.get(u, "") for u in UIDS],
        }
    ).to_csv(cache / "index.csv", index=False)


def write_config(
    path: Path,
    *,
    epochs: int = 2,
    num_workers: int = 0,
    prob: float = 0.2,
    early_stop_patience: int | None = None,
    loss_weights: str = "uniform",
    pos_weight: str = "null",
    init: str = "random",
    freeze_epochs: int = 0,
    backbone_lr_scale: float = 1.0,
    input_size: int = 128,
) -> Path:
    patience = "null" if early_stop_patience is None else early_stop_patience
    path.write_text(
        "kind: cnn\nseed: 0\nlabels: corrected\n"
        f"train: {{epochs: {epochs}, batch_size: 2, accum_steps: 1, lr: 1.0e-3,"
        f" weight_decay: 0.01, warmup_epochs: 1, ema_decay: 0.9, loss_weights: {loss_weights},"
        f" num_workers: {num_workers}, early_stop_patience: {patience},"
        f" pos_weight: {pos_weight}, init: {init}, freeze_epochs: {freeze_epochs},"
        f" backbone_lr_scale: {backbone_lr_scale}, input_size: {input_size}}}\n"
        f"augment: {{prob: {prob}, noise_std: 0.1, gamma: [0.7, 1.5], rotate_deg: 10,"
        " scale: 0.1, lowres_zoom: [0.5, 1.0]}\n"
    )
    return path


def fold0_datasets(
    root: Path, train_transform: object = None
) -> tuple[CacheDataset, CacheDataset, list[str]]:
    labels = cnn_labels()
    ok = [u for u in UIDS if u not in EXCLUDED]
    train = [u for u in ok if FOLDS[u] != 0]
    val = [u for u in ok if FOLDS[u] == 0]
    cache = root / "cache" / "b2-128"
    targets = lambda uids: labels.loc[uids, list(TARGETS)].to_numpy()  # noqa: E731
    return (
        CacheDataset(cache, train, targets(train), train_transform),  # type: ignore[arg-type]
        CacheDataset(cache, val, targets(val), None),
        [u for u in UIDS if FOLDS[u] == 0],
    )
