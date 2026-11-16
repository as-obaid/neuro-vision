"""AUCs within a subset of series, such as one scan type, where some labels may have one class."""

from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from neurovision.data.labels import LOCATIONS, TARGETS


def _auc(y: np.ndarray, p: np.ndarray) -> float | None:
    return float(roc_auc_score(y, p)) if 0 < y.sum() < len(y) else None


def summarize_subset(y_true: np.ndarray, y_prob: np.ndarray) -> dict[str, Any]:
    """Counts and AUCs; a label with one class is left out. `score` needs all 14 labels."""
    ap = _auc(y_true[:, -1], y_prob[:, -1])
    locations = [_auc(y_true[:, k], y_prob[:, k]) for k in range(len(LOCATIONS))]
    defined = [a for a in locations if a is not None]
    score = None
    if ap is not None and len(defined) == len(LOCATIONS):
        score = (len(LOCATIONS) * ap + sum(defined)) / (2 * len(LOCATIONS))
    return {
        "n": len(y_true),
        "positives": int(y_true[:, -1].sum()),
        "ap_auc": ap,
        "location_auc": float(np.mean(defined)) if defined else None,
        "location_columns": len(defined),
        "score": score,
    }


def by_modality(labels: pd.DataFrame, probs: pd.DataFrame) -> dict[str, dict[str, Any]]:
    """`summarize_subset` within each scan type (`Modality` column of `labels`)."""
    missing = labels.index.difference(probs.index)
    if len(missing):
        raise ValueError(f"{len(missing)} labelled series are missing from probs")
    out = {}
    for modality, frame in labels.groupby("Modality"):
        y = frame[list(TARGETS)].to_numpy()
        p = probs.loc[frame.index, list(TARGETS)].to_numpy()
        out[str(modality)] = summarize_subset(y, p)
    return out
