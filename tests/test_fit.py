from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from monai.transforms import Randomizable

from neurovision.config import load_config
from neurovision.data.labels import TARGETS
from neurovision.train import loop
from neurovision.train.loop import fit, train_loader
from tests.cnn_fakes import fold0_datasets, write_config, write_store

PRIOR = np.full(14, 0.3)


class Stamp(Randomizable):
    """Replaces the volume with one random draw, so a batch shows which stream made it."""

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return torch.full_like(x, float(self.R.random()))


@pytest.fixture(autouse=True)
def wandb_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("WANDB_PROJECT", "test-project")
    monkeypatch.setenv("WANDB_SILENT", "true")


def _fit(tmp_path: Path, run: str, epochs: int = 2) -> Path:
    if not (tmp_path / "store").exists():
        write_store(tmp_path / "store")
    config = load_config(write_config(tmp_path / "b2.yaml", epochs=epochs))
    train_ds, val_ds, val_uids = fold0_datasets(tmp_path / "store")
    run_dir = tmp_path / run
    run_dir.mkdir()
    fit(config, 0, run_dir, train_ds, val_ds, val_uids, PRIOR)
    return run_dir


def _fit_again(tmp_path: Path, run: str) -> None:
    train_ds, val_ds, val_uids = fold0_datasets(tmp_path / "store")
    fit(load_config(tmp_path / "b2.yaml"), 0, tmp_path / run, train_ds, val_ds, val_uids, PRIOR)


def test_workers_and_epochs_draw_different_augmentation(tmp_path: Path) -> None:
    write_store(tmp_path)
    train_ds, _, _ = fold0_datasets(tmp_path, Stamp())
    config = load_config(write_config(tmp_path / "b2.yaml", num_workers=2))
    assert config.train is not None

    def draws(epoch: int) -> list[float]:
        loader = train_loader(train_ds, config.train, config.seed, epoch, pin_memory=False)
        return [float(x[i, 0, 0, 0, 0]) for x, _ in loader for i in range(len(x))]

    first, second = draws(0), draws(1)
    assert len(set(first)) == len(first)
    assert not set(first) & set(second)
    assert draws(0) == first


def test_resume_reaches_the_same_weights(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    straight = _fit(tmp_path, "straight")

    calls = []
    real = loop.train_one_epoch

    def stop_in_epoch_two(*args, **kwargs):
        calls.append(1)
        if len(calls) == 2:
            raise KeyboardInterrupt
        return real(*args, **kwargs)

    monkeypatch.setattr(loop, "train_one_epoch", stop_in_epoch_two)
    with pytest.raises(KeyboardInterrupt):
        _fit(tmp_path, "resumed")
    monkeypatch.setattr(loop, "train_one_epoch", real)
    _fit_again(tmp_path, "resumed")

    a = torch.load(straight / "last.pt", weights_only=True)
    b = torch.load(tmp_path / "resumed" / "last.pt", weights_only=True)
    assert a["epoch"] == b["epoch"] == 1
    for key, value in a["model"].items():
        torch.testing.assert_close(b["model"][key], value, atol=1e-6, rtol=0)
    history = pd.read_csv(tmp_path / "resumed" / "history.csv")
    assert list(history["epoch"]) == [0, 1]


def test_run_directory_contents(tmp_path: Path) -> None:
    run = _fit(tmp_path, "run")
    history = pd.read_csv(run / "history.csv")
    assert list(history.columns) == [
        "epoch", "train_loss", "val_loss", "val_score", "lr", "train_seconds", "val_seconds",
    ]  # fmt: skip
    assert len(history) == 2
    oof = pd.read_csv(run / "oof.csv", index_col="SeriesInstanceUID")
    assert list(oof.columns) == list(TARGETS)
    assert sorted(oof.index) == ["s0000", "s0005", "s0010"]
    np.testing.assert_allclose(oof.loc["s0010"].to_numpy(), PRIOR)
    meta = pd.read_json(run / "meta.json", typ="series")
    assert {"commit", "train_seconds_per_epoch", "val_seconds_per_epoch"} <= set(meta.index)
    assert meta["peak_gpu_memory_mib"] is None
    assert len(list((run / "wandb").glob("offline-run-*"))) == 1
    best = torch.load(run / "best.pt", weights_only=True)
    assert "n_averaged" not in best
    assert any(k.endswith("running_mean") for k in best)


def test_finished_run_adds_no_epochs(tmp_path: Path) -> None:
    run = _fit(tmp_path, "run")
    before = (run / "history.csv").read_text()
    _fit_again(tmp_path, "run")
    assert (run / "history.csv").read_text() == before
