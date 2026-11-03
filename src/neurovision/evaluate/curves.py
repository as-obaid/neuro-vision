"""Aneurysm Present beyond AUC: decisions at a threshold, and calibration."""

from typing import Any

import numpy as np
from sklearn.metrics import roc_curve


def _ratio(num: int, den: int) -> float | None:
    return num / den if den else None


def operating_point(y_true: np.ndarray, y_prob: np.ndarray, threshold: float) -> dict[str, Any]:
    """Confusion counts and rates when a probability >= threshold counts as positive."""
    pred = y_prob >= threshold
    truth = y_true.astype(bool)
    tp, fp = int((pred & truth).sum()), int((pred & ~truth).sum())
    tn, fn = int((~pred & ~truth).sum()), int((~pred & truth).sum())
    return {
        "threshold": float(threshold),
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "sensitivity": _ratio(tp, tp + fn),
        "specificity": _ratio(tn, tn + fp),
        "ppv": _ratio(tp, tp + fp),
        "npv": _ratio(tn, tn + fn),
        "f1": _ratio(2 * tp, 2 * tp + fp + fn),
    }


def operating_points(
    y_true: np.ndarray, y_prob: np.ndarray, sensitivity: float = 0.90
) -> dict[str, dict[str, Any]]:
    """The Youden-J threshold, and the highest threshold whose sensitivity reaches the target."""
    fpr, tpr, thresholds = roc_curve(y_true, y_prob, drop_intermediate=False)
    youden = thresholds[int(np.argmax(tpr - fpr))]
    reach = thresholds[int(np.argmax(tpr >= sensitivity))]
    return {
        "youden": operating_point(y_true, y_prob, youden),
        f"sensitivity_{sensitivity:.2f}": operating_point(y_true, y_prob, reach),
    }


def calibration(y_true: np.ndarray, y_prob: np.ndarray, bins: int = 10) -> dict[str, Any]:
    """Equal-count bins of the sorted predictions; ECE weighs each bin's gap by its size."""
    order = np.argsort(y_prob, kind="stable")
    rows = []
    ece = 0.0
    for idx in np.array_split(order, bins):
        if not len(idx):
            continue
        mean_pred, observed = float(y_prob[idx].mean()), float(y_true[idx].mean())
        rows.append({"mean_predicted": mean_pred, "observed": observed, "count": len(idx)})
        ece += len(idx) / len(y_true) * abs(mean_pred - observed)
    return {
        "bins": rows,
        "ece": ece,
        "brier": float(np.mean((y_prob - y_true) ** 2)),
    }
