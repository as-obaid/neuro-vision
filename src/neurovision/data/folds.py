"""Stratified (and optionally grouped) cross-validation folds."""

from pathlib import Path

import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold

from neurovision.data.labels import ANEURYSM_PRESENT


def make_folds(
    labels: pd.DataFrame, n_folds: int, seed: int, groups: pd.Series | None
) -> pd.Series:
    strata = labels[ANEURYSM_PRESENT].astype(str) + "_" + labels["Modality"].astype(str)
    folds = pd.Series(-1, index=labels.index, name="fold")
    if groups is None:
        splits = StratifiedKFold(n_folds, shuffle=True, random_state=seed).split(labels, strata)
    else:
        splitter = StratifiedGroupKFold(n_folds, shuffle=True, random_state=seed)
        splits = splitter.split(labels, strata, groups.loc[labels.index])
    for fold, (_, val_idx) in enumerate(splits):
        folds.iloc[val_idx] = fold
    return folds


def load_folds(store: Path) -> pd.Series:
    frame = pd.read_csv(store / "folds" / "folds.csv", index_col="SeriesInstanceUID")
    return frame["fold"]
