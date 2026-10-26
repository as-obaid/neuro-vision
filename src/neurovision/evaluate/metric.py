"""Competition metric: weighted mean of 14 column AUCs (Aneurysm Present 13, others 1)."""

import numpy as np
from sklearn.metrics import roc_auc_score

from neurovision.data.labels import ANEURYSM_PRESENT, LOCATIONS, TARGETS


def label_aucs(y_true: np.ndarray, y_prob: np.ndarray) -> dict[str, float]:
    if y_true.shape != y_prob.shape or y_true.ndim != 2 or y_true.shape[1] != len(TARGETS):
        raise ValueError(
            f"expected two arrays of shape (n, {len(TARGETS)}), "
            f"got shapes {y_true.shape} and {y_prob.shape}"
        )
    # Ruling: a single-class column raises ValueError. Kaggle's code returns NaN for it under
    # scikit-learn 1.9.1.
    if any(len(np.unique(column)) < 2 for column in y_true.T):
        raise ValueError("every column of y_true must contain both classes")
    aucs = roc_auc_score(y_true, y_prob, average=None)
    return {name: float(auc) for name, auc in zip(TARGETS, aucs, strict=True)}


def score(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    aucs = label_aucs(y_true, y_prob)
    return (len(LOCATIONS) * aucs[ANEURYSM_PRESENT] + sum(aucs[name] for name in LOCATIONS)) / (
        2 * len(LOCATIONS)
    )
