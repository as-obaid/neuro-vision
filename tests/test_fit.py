import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
import wandb
from monai.transforms import Randomizable

from neurovision.config import load_config
from neurovision.data.labels import TARGETS
from neurovision.train import loop
from neurovision.train.loop import fit, stalled, train_loader, val_loader
from neurovision.train.losses import label_weights, positive_weights
from tests.cnn_fakes import fold0_datasets, write_config, write_store

PRIOR = np.full(14, 0.3)
COUNTS = {"train_used": 8}


class Stamp(Randomizable):
    """Replaces the volume with one random draw, so a batch shows which stream made it."""

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return torch.full_like(x, float(self.R.random()))


@pytest.fixture(autouse=True)
def wandb_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("WANDB_PROJECT", "test-project")
    monkeypatch.setenv("WANDB_SILENT", "true")


def _fit(
    tmp_path: Path,
    run: str,
    epochs: int = 2,
    early_stop_patience: int | None = None,
    loss_weights: str = "uniform",
    pos_weight: str = "null",
) -> Path:
    if not (tmp_path / "store").exists():
        write_store(tmp_path / "store")
    path = write_config(
        tmp_path / "b2.yaml",
        epochs=epochs,
        early_stop_patience=early_stop_patience,
        loss_weights=loss_weights,
        pos_weight=pos_weight,
    )
    config = load_config(path)
    train_ds, val_ds, val_uids = fold0_datasets(tmp_path / "store")
    run_dir = tmp_path / run
    run_dir.mkdir()
    fit(config, 0, run_dir, train_ds, val_ds, val_uids, PRIOR, COUNTS)
    return run_dir


def _fit_again(tmp_path: Path, run: str) -> None:
    train_ds, val_ds, val_uids = fold0_datasets(tmp_path / "store")
    fit(
        load_config(tmp_path / "b2.yaml"),
        0,
        tmp_path / run,
        train_ds,
        val_ds,
        val_uids,
        PRIOR,
        COUNTS,
    )


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
    for key, value in a["ema"].items():
        torch.testing.assert_close(b["ema"][key], value, atol=1e-6, rtol=0)
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


def test_finished_run_changes_nothing(tmp_path: Path) -> None:
    run = _fit(tmp_path, "run")
    names = ["history.csv", "oof.csv", "best.pt", "meta.json"]
    before = {n: (run / n).read_bytes() for n in names}
    _fit_again(tmp_path, "run")
    assert {n: (run / n).read_bytes() for n in names} == before


def _interrupt_in_epoch_two(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = []
    real = loop.train_one_epoch

    def stop(*args, **kwargs):
        calls.append(1)
        if len(calls) == 2:
            raise KeyboardInterrupt
        return real(*args, **kwargs)

    monkeypatch.setattr(loop, "train_one_epoch", stop)


def test_resumed_meta_keeps_checkpointed_peak_and_all_epoch_times(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    real = loop.train_one_epoch
    monkeypatch.setattr(loop, "peak_memory_mib", lambda device: 500.0)
    _interrupt_in_epoch_two(monkeypatch)
    with pytest.raises(KeyboardInterrupt):
        _fit(tmp_path, "run")
    monkeypatch.setattr(loop, "train_one_epoch", real)
    monkeypatch.setattr(loop, "peak_memory_mib", lambda device: 300.0)
    _fit_again(tmp_path, "run")
    meta = json.loads((tmp_path / "run" / "meta.json").read_text())
    history = pd.read_csv(tmp_path / "run" / "history.csv")
    assert meta["peak_gpu_memory_mib"] == 500.0
    assert meta["train_seconds_per_epoch"] == pytest.approx(history["train_seconds"].mean())
    assert meta["val_seconds_per_epoch"] == pytest.approx(history["val_seconds"].mean())


def test_checkpoint_without_a_peak_loads_as_none(tmp_path: Path) -> None:
    run = _fit(tmp_path, "run")
    state = torch.load(run / "last.pt", weights_only=True)
    del state["peak_mib"]
    torch.save(state, run / "last.pt")
    (run / "meta.json").unlink()
    _fit_again(tmp_path, "run")
    assert json.loads((run / "meta.json").read_text())["peak_gpu_memory_mib"] is None


def test_outputs_failure_after_the_last_checkpoint_recovers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    straight = _fit(tmp_path, "straight")
    real = loop._write_outputs
    calls = []

    def fail_once(*args, **kwargs):
        calls.append(1)
        if len(calls) == 1:
            raise RuntimeError("killed")
        return real(*args, **kwargs)

    monkeypatch.setattr(loop, "_write_outputs", fail_once)
    with pytest.raises(RuntimeError, match="killed"):
        _fit(tmp_path, "run")
    assert not (tmp_path / "run" / "meta.json").exists()
    _fit_again(tmp_path, "run")
    run = tmp_path / "run"
    assert (run / "meta.json").exists()
    pd.testing.assert_frame_equal(
        pd.read_csv(run / "oof.csv"), pd.read_csv(straight / "oof.csv"), atol=1e-6, rtol=0
    )
    assert len(pd.read_csv(run / "history.csv")) == 2


@pytest.mark.parametrize("name", ["oof.csv", "meta.json"])
def test_torn_output_write_leaves_no_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, name: str
) -> None:
    real = Path.write_text

    def torn(self: Path, text: str, *args, **kwargs):
        real(self, text[:5], *args, **kwargs)
        if self.name == name + ".tmp":
            raise RuntimeError("disk full")
        return real(self, text, *args, **kwargs)

    real_csv = pd.DataFrame.to_csv

    def torn_csv(self, path=None, *args, **kwargs):
        real_csv(self, path, *args, **kwargs)
        if str(path).endswith("oof.csv.tmp"):
            raise RuntimeError("disk full")

    monkeypatch.setattr(pd.DataFrame, "to_csv", torn_csv)
    monkeypatch.setattr(Path, "write_text", torn)
    with pytest.raises(RuntimeError, match="disk full"):
        _fit(tmp_path, "run")
    assert not (tmp_path / "run" / name).exists()


def test_run_meta_is_taken_before_the_first_epoch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    real = loop.train_one_epoch
    first_epoch_start: list[datetime] = []

    def stamped(*args, **kwargs):
        first_epoch_start.append(datetime.now(UTC))
        return real(*args, **kwargs)

    monkeypatch.setattr(loop, "train_one_epoch", stamped)
    run = _fit(tmp_path, "run")
    started = datetime.fromisoformat(json.loads((run / "meta.json").read_text())["started_at"])
    assert started <= first_epoch_start[0]


def test_meta_carries_the_counts(tmp_path: Path) -> None:
    run = _fit(tmp_path, "run")
    assert json.loads((run / "meta.json").read_text())["counts"] == COUNTS


def test_failed_best_write_leaves_no_best_and_finishes_wandb(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    real = torch.save

    def torn_save(obj, path):
        real(obj, path)
        if str(path).endswith("best.pt.tmp"):
            raise RuntimeError("disk full")

    monkeypatch.setattr(torch, "save", torn_save)
    with pytest.raises(RuntimeError, match="disk full"):
        _fit(tmp_path, "run")
    assert not (tmp_path / "run" / "best.pt").exists()
    assert wandb.run is None


@pytest.mark.parametrize(("workers", "persistent"), [(0, False), (2, True)])
def test_validation_loader_keeps_workers_alive(
    tmp_path: Path, workers: int, persistent: bool
) -> None:
    write_store(tmp_path)
    _, val_ds, _ = fold0_datasets(tmp_path)
    cfg = load_config(write_config(tmp_path / "b2.yaml", num_workers=workers)).train
    assert cfg is not None
    assert val_loader(val_ds, cfg, False).persistent_workers is persistent


@pytest.mark.parametrize(
    ("losses", "patience", "expected"),
    [
        ([], 2, False),
        ([0.3, 0.4, 0.5], None, False),
        ([0.3, 0.4], 2, False),
        ([0.3, 0.4, 0.5], 2, True),
        ([0.3, 0.4, 0.2, 0.5], 2, False),
        ([0.3, 0.3, 0.3], 2, True),  # a tie is not an improvement
    ],
)
def test_stalled(losses: list[float], patience: int | None, expected: bool) -> None:
    assert stalled(losses, patience) is expected


def _rising_val_loss(monkeypatch: pytest.MonkeyPatch) -> None:
    real = loop.evaluate
    calls = []

    def rising(*args, **kwargs):
        probs, _ = real(*args, **kwargs)
        calls.append(1)
        return probs, float(len(calls))

    monkeypatch.setattr(loop, "evaluate", rising)


def test_early_stopping_ends_training_and_writes_outputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _rising_val_loss(monkeypatch)
    run = _fit(tmp_path, "run", epochs=4, early_stop_patience=1)
    assert len(pd.read_csv(run / "history.csv")) == 2
    assert (run / "meta.json").exists()


def test_resume_after_early_stop_trains_no_more(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _rising_val_loss(monkeypatch)
    real = loop._write_outputs
    calls = []

    def fail_once(*args, **kwargs):
        calls.append(1)
        if len(calls) == 1:
            raise RuntimeError("killed")
        return real(*args, **kwargs)

    monkeypatch.setattr(loop, "_write_outputs", fail_once)
    with pytest.raises(RuntimeError, match="killed"):
        _fit(tmp_path, "run", epochs=4, early_stop_patience=1)
    _fit_again(tmp_path, "run")
    assert len(pd.read_csv(tmp_path / "run" / "history.csv")) == 2
    assert (tmp_path / "run" / "meta.json").exists()


def test_fit_trains_and_validates_with_the_configured_weights(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen = []
    real_bce, real_evaluate = loop.bce, loop.evaluate

    def spy_bce(logits, targets, weights=None, pos_weight=None):
        seen.append(("train", weights))
        return real_bce(logits, targets, weights, pos_weight)

    def spy_evaluate(model, loader, device, weights=None, pos_weight=None):
        seen.append(("val", weights))
        return real_evaluate(model, loader, device, weights, pos_weight)

    monkeypatch.setattr(loop, "bce", spy_bce)
    monkeypatch.setattr(loop, "evaluate", spy_evaluate)
    _fit(tmp_path, "run", epochs=2, loss_weights="competition")
    expected = label_weights("competition")
    stages = {stage for stage, w in seen if w is not None and torch.equal(w.cpu(), expected)}
    assert stages == {"train", "val"}


def test_fit_uses_positive_weights_from_the_training_split(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen = []
    real_bce, real_evaluate = loop.bce, loop.evaluate

    def spy_bce(logits, targets, weights=None, pos_weight=None):
        seen.append(("train", pos_weight))
        return real_bce(logits, targets, weights, pos_weight)

    def spy_evaluate(model, loader, device, weights=None, pos_weight=None):
        seen.append(("val", pos_weight))
        return real_evaluate(model, loader, device, weights, pos_weight)

    monkeypatch.setattr(loop, "bce", spy_bce)
    monkeypatch.setattr(loop, "evaluate", spy_evaluate)
    _fit(tmp_path, "run", epochs=2, pos_weight="sqrt")
    train_ds, _, _ = fold0_datasets(tmp_path / "store")
    expected = positive_weights(train_ds.targets, "sqrt")
    assert expected is not None
    stages = {s for s, w in seen if w is not None and torch.allclose(w.cpu(), expected)}
    assert stages == {"train", "val"}
