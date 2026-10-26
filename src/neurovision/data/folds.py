"""Stratified (and optionally grouped) cross-validation folds."""

import hashlib
from pathlib import Path

import pandas as pd
import yaml
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
    folds_csv = store / "folds" / "folds.csv"
    manifest = store / "folds" / "folds.yaml"
    if not manifest.exists():
        raise ValueError(f"{manifest} is missing; cannot verify folds.csv")
    expected = yaml.safe_load(manifest.read_text())["folds_sha256"]
    if hashlib.sha256(folds_csv.read_bytes()).hexdigest() != expected:
        raise ValueError(f"{folds_csv} does not match the sha256 in folds.yaml")
    return pd.read_csv(folds_csv, index_col="SeriesInstanceUID")["fold"]
