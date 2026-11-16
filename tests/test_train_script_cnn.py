import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from neurovision.data.labels import TARGETS
from tests.cnn_fakes import EXCLUDED, FOLDS, cnn_labels, write_config, write_store
from tests.script_loader import load_script


@pytest.fixture
def setup(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    train, score = load_script("train"), load_script("score_oof")
    for module in (train, score):
        monkeypatch.setattr(module, "load_labels", lambda store, source: cnn_labels())
    monkeypatch.setenv("NV_STORE", str(tmp_path / "store"))
    monkeypatch.setenv("WANDB_PROJECT", "test-project")
    monkeypatch.setenv("WANDB_SILENT", "true")
    write_store(tmp_path / "store")
    return train, score, tmp_path / "store", str(write_config(tmp_path / "b2.yaml"))


def test_fold_zero_writes_the_run_directory(setup) -> None:
    train, score, store, config = setup
    train.main([config, "--fold", "0"])
    run = store / "runs" / "b2" / "fold0"
    assert {p.name for p in run.iterdir()} >= {
        "oof.csv", "history.csv", "best.pt", "last.pt", "meta.json", "wandb",
    }  # fmt: skip
    assert len(pd.read_csv(run / "history.csv")) == 2
    assert "commit" in json.loads((run / "meta.json").read_text())

    oof = pd.read_csv(run / "oof.csv", index_col="SeriesInstanceUID")
    val = list(FOLDS.index[FOLDS == 0])
    assert sorted(oof.index) == val
    prior = cnn_labels()[FOLDS != 0][list(TARGETS)].to_numpy().mean(axis=0)
    for uid in set(val) & set(EXCLUDED):
        np.testing.assert_allclose(oof.loc[uid].to_numpy(), prior)
    assert not np.allclose(oof.loc["s0000"].to_numpy(), prior)

    score.main([config])
    result = json.loads((store / "runs" / "b2" / "score.json").read_text())
    assert result["folds"] == [0]
    assert result["n"] == len(val)


def test_second_run_resumes_and_adds_no_epochs(setup) -> None:
    train, _, store, config = setup
    train.main([config, "--fold", "0"])
    run = store / "runs" / "b2" / "fold0"
    history, weights = (run / "history.csv").read_text(), torch.load(run / "last.pt")
    train.main([config, "--fold", "0"])
    assert (run / "history.csv").read_text() == history
    assert torch.equal(torch.load(run / "last.pt")["model"]["fc.bias"], weights["model"]["fc.bias"])


def test_cnn_requires_a_fold_that_exists(setup) -> None:
    train, _, store, config = setup
    for argv in ([config], [config, "--fold", "9"]):
        with pytest.raises(SystemExit):
            train.main(argv)
    assert not (store / "runs").exists()


def test_series_missing_from_the_index_are_refused(setup) -> None:
    train, _, store, config = setup
    index_path = store / "cache" / "b2-128" / "index.csv"
    index = pd.read_csv(index_path)
    index.iloc[1:].to_csv(index_path, index=False)
    with pytest.raises(SystemExit, match=r"1 of 12 series have no row in cache/b2-128/index.csv"):
        train.main([config, "--fold", "0"])
    assert not (store / "runs").exists()


def test_input_size_selects_the_cache_folder(setup, tmp_path: Path) -> None:
    train, _, store, _ = setup
    config = str(write_config(tmp_path / "b2_160.yaml", input_size=160))
    (store / "cache" / "b2-128").rename(store / "cache" / "b2-160")
    index_path = store / "cache" / "b2-160" / "index.csv"
    index = pd.read_csv(index_path)
    index.iloc[1:].to_csv(index_path, index=False)
    with pytest.raises(SystemExit, match=r"cache/b2-160/index.csv.*--size 160"):
        train.main([config, "--fold", "0"])
    index.to_csv(index_path, index=False)
    train.main([config, "--fold", "0"])
    assert (store / "runs" / "b2_160" / "fold0" / "oof.csv").exists()


def test_resample_mm_selects_the_mm_cache_folder(setup, tmp_path: Path) -> None:
    train, _, store, _ = setup
    config = str(write_config(tmp_path / "b2_mm.yaml", input_size=160, resample="mm"))
    (store / "cache" / "b2-128").rename(store / "cache" / "b2mm-160")
    index_path = store / "cache" / "b2mm-160" / "index.csv"
    index = pd.read_csv(index_path)
    index.iloc[1:].to_csv(index_path, index=False)
    with pytest.raises(SystemExit, match=r"cache/b2mm-160/index.csv.*--size 160 --resample mm"):
        train.main([config, "--fold", "0"])
    index.to_csv(index_path, index=False)
    train.main([config, "--fold", "0"])
    assert (store / "runs" / "b2_mm" / "fold0" / "oof.csv").exists()


def test_counts_are_printed_and_stored(setup, capsys: pytest.CaptureFixture[str]) -> None:
    train, _, store, config = setup
    train.main([config, "--fold", "0"])
    counts = json.loads((store / "runs" / "b2" / "fold0" / "meta.json").read_text())["counts"]
    assert counts == {
        "train_used": 8,
        "train_excluded": 1,
        "validation_scored": 2,
        "validation_prior_filled": 1,
        "index_errors": 1,
    }
    out = capsys.readouterr().out
    assert "train 8 used, 1 excluded; validation 2 scored, 1 prior-filled; 1 index errors" in out
    assert "s00" not in out
