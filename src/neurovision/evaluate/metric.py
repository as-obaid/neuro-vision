"""Competition metric: weighted mean of 14 column AUCs (Aneurysm Present 13, others 1)."""

import numpy as np
from sklearn.metrics import roc_auc_score

from neurovision.data.labels import TARGETS


def label_aucs(y_true: np.ndarray, y_prob: np.ndarray) -> dict[str, float]:
    if y_true.shape != y_prob.shape or y_true.ndim != 2 or y_true.shape[1] != len(TARGETS):
        raise ValueError(
            f"expected two arrays of shape (n, {len(TARGETS)}), "
            f"got shapes {y_true.shape} and {y_prob.shape}"
        )
    # Like the official code, a column with a single class is an error, not a skipped column.
    if any(len(np.unique(column)) < 2 for column in y_true.T):
        raise ValueError("every column of y_true must contain both classes")
    aucs = roc_auc_score(y_true, y_prob, average=None)
    return {name: float(auc) for name, auc in zip(TARGETS, aucs, strict=True)}


def score(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    aucs = list(label_aucs(y_true, y_prob).values())
    return (13 * aucs[13] + sum(aucs[:13])) / 26
