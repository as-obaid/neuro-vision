import json
from pathlib import Path

import pandas as pd
import pytest

from neurovision.data.folds import make_folds
from neurovision.data.labels import TARGETS
from tests.label_fakes import fake_labels
from tests.script_loader import load_script


@pytest.fixture
def setup(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    train, score = load_script("train"), load_script("score_oof")
    labels = fake_labels()
    folds = make_folds(labels, 5, 0, None)
    # Equal positive counts in every fold give every fold the same prior, so B0 ties everywhere.
    rank = folds.groupby(folds).cumcount().to_numpy()
    for i, target in enumerate(TARGETS):
        labels[target] = ((rank + i) % 10 == 0).astype("int8")
    for module in (train, score):
        monkeypatch.setattr(module, "load_labels", lambda store, source: labels)
    monkeypatch.setenv("NV_STORE", str(tmp_path))
    (tmp_path / "folds").mkdir()
    folds.to_csv(tmp_path / "folds" / "folds.csv")
    configs = tmp_path / "configs"
    configs.mkdir()
    (configs / "b0.yaml").write_text("kind: prior\nseed: 0\nlabels: corrected\n")
    (configs / "b1.yaml").write_text(
        "kind: gbm\nseed: 0\nlabels: corrected\n"
        "gbm: {max_iter: 10, learning_rate: 0.1, max_depth: 2}\n"
    )
    (configs / "b2.yaml").write_text(
        "kind: cnn\nseed: 0\nlabels: corrected\n"
        "train: {epochs: 3, batch_size: 2, accum_steps: 1, lr: 1.0e-4, weight_decay: 0.01,\n"
        "        warmup_epochs: 1, ema_decay: 0.999, loss_weights: uniform, num_workers: 0}\n"
        "augment: {prob: 0.2, noise_std: 0.1, gamma: [0.7, 1.5], rotate_deg: 10, scale: 0.1,\n"
        "          lowres_zoom: [0.5, 1.0]}\n"
    )
    return train, score, tmp_path, labels


def test_b0_scores_exactly_one_half(setup) -> None:
    train, score, store, _ = setup
    config = str(store / "configs" / "b0.yaml")
    train.main([config])
    for k in range(5):
        run = store / "runs" / "b0" / f"fold{k}"
        oof = pd.read_csv(run / "oof.csv")
        assert list(oof.columns) == ["SeriesInstanceUID", *TARGETS]
        assert "commit" in json.loads((run / "meta.json").read_text())
    score.main([config])
    result = json.loads((store / "runs" / "b0" / "score.json").read_text())
    assert result["score"] == 0.5
    assert result["folds"] == [0, 1, 2, 3, 4]
    assert result["n"] == 200
    assert set(result["aucs"]) == set(TARGETS)
    assert {"config", "commit"} <= result.keys()


def test_b1_scores_between_zero_and_one(setup) -> None:
    train, score, store, _ = setup
    config = str(store / "configs" / "b1.yaml")
    train.main([config])
    score.main([config])
    result = json.loads((store / "runs" / "b1" / "score.json").read_text())
    assert 0 <= result["score"] <= 1


def test_fold_flag_refused_for_baselines(setup) -> None:
    train, _, store, _ = setup
    with pytest.raises(SystemExit):
        train.main([str(store / "configs" / "b0.yaml"), "--fold", "1"])
    assert not (store / "runs").exists()


def test_cnn_kind_exits_not_built(setup) -> None:
    train, _, store, _ = setup
    with pytest.raises(SystemExit, match="kind cnn: not built yet"):
        train.main([str(store / "configs" / "b2.yaml"), "--fold", "0"])


def _trained(setup) -> tuple[object, Path, str]:
    train, score, store, _ = setup
    config = str(store / "configs" / "b0.yaml")
    train.main([config])
    return score, store / "runs" / "b0", config


def test_scorer_refuses_missing_series(setup) -> None:
    score, runs, config = _trained(setup)
    path = runs / "fold2" / "oof.csv"
    pd.read_csv(path).iloc[1:].to_csv(path, index=False)
    with pytest.raises(SystemExit, match=r"fold 2: 1 series missing"):
        score.main([config])  # type: ignore[attr-defined]


def test_scorer_refuses_duplicate_series(setup) -> None:
    score, runs, config = _trained(setup)
    path = runs / "fold2" / "oof.csv"
    oof = pd.read_csv(path)
    oof.iloc[0] = oof.iloc[1]
    oof.to_csv(path, index=False)
    with pytest.raises(SystemExit, match=r"fold 2: duplicate series"):
        score.main([config])  # type: ignore[attr-defined]


def test_scorer_refuses_extra_series(setup) -> None:
    score, runs, config = _trained(setup)
    path = runs / "fold2" / "oof.csv"
    oof = pd.read_csv(path)
    extra = pd.read_csv(runs / "fold3" / "oof.csv").iloc[[0]]
    pd.concat([oof, extra]).to_csv(path, index=False)
    with pytest.raises(SystemExit, match=r"fold 2: 0 series missing, 1 not in this fold"):
        score.main([config])  # type: ignore[attr-defined]
