import numpy as np
import pandas as pd
import pytest

from neurovision.data.folds import load_folds, make_folds
from neurovision.data.labels import ANEURYSM_PRESENT
from tests.label_fakes import fake_labels


def fake_groups(index: pd.Index, seed: int = 1) -> pd.Series:
    rng = np.random.default_rng(seed)
    ids = np.concatenate([np.full(rng.integers(1, 5), g) for g in range(len(index))])[: len(index)]
    return pd.Series(ids, index=index)


def test_every_series_in_exactly_one_fold_in_labels_order() -> None:
    labels = fake_labels()
    folds = make_folds(labels, 5, 0, None)
    assert folds.index.equals(labels.index)
    assert sorted(folds.unique()) == [0, 1, 2, 3, 4]


def test_no_group_split_across_folds() -> None:
    labels = fake_labels()
    groups = fake_groups(labels.index)
    folds = make_folds(labels, 5, 0, groups)
    assert (folds.groupby(groups).nunique() == 1).all()
    assert sorted(folds.unique()) == [0, 1, 2, 3, 4]


@pytest.mark.parametrize("use_groups", [False, True])
def test_seed_determines_folds(use_groups: bool) -> None:
    labels = fake_labels()
    groups = fake_groups(labels.index) if use_groups else None
    first = make_folds(labels, 5, 0, groups)
    assert first.equals(make_folds(labels, 5, 0, groups))
    assert not first.equals(make_folds(labels, 5, 1, groups))


@pytest.mark.parametrize("use_groups", [False, True])
def test_positive_rate_per_fold_near_overall(use_groups: bool) -> None:
    labels = fake_labels()
    groups = fake_groups(labels.index) if use_groups else None
    folds = make_folds(labels, 5, 0, groups)
    rates = labels[ANEURYSM_PRESENT].groupby(folds).mean()
    assert (rates - labels[ANEURYSM_PRESENT].mean()).abs().max() < 0.10


def test_load_folds_reads_store_csv(tmp_path) -> None:
    (tmp_path / "folds").mkdir()
    (tmp_path / "folds" / "folds.csv").write_text("SeriesInstanceUID,fold\na,0\nb,3\n")
    folds = load_folds(tmp_path)
    assert folds.to_dict() == {"a": 0, "b": 3}
    assert folds.index.name == "SeriesInstanceUID"
