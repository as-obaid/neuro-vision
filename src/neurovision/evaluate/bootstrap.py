"""Paired bootstrap over series: how much of a difference between two models is sampling noise."""

from typing import Any

import numpy as np

from neurovision.data.labels import ANEURYSM_PRESENT, LOCATIONS
from neurovision.evaluate.metric import label_aucs

METRICS: tuple[str, ...] = ("score", "ap_auc", "location_auc")


def summarize(y_true: np.ndarray, y_prob: np.ndarray) -> np.ndarray:
    """Competition score, Aneurysm Present AUC and mean location AUC, in METRICS order."""
    aucs = label_aucs(y_true, y_prob)
    ap = aucs[ANEURYSM_PRESENT]
    locations = [aucs[name] for name in LOCATIONS]
    score = (len(LOCATIONS) * ap + sum(locations)) / (2 * len(LOCATIONS))
    return np.array([score, ap, float(np.mean(locations))])


def paired_bootstrap(
    y_true: np.ndarray, p_a: np.ndarray, p_b: np.ndarray, n: int = 2000, seed: int = 0
) -> dict[str, Any]:
    """Resample series with replacement; each metric of model a minus model b, with a 95% CI.

    A resample in which any column lacks one class has no AUC for it and is skipped; `used` counts
    the rest. `p_le_0` is the share of used resamples in which a is not better than b.
    """
    if not y_true.shape == p_a.shape == p_b.shape:
        raise ValueError(f"shapes differ: {y_true.shape}, {p_a.shape}, {p_b.shape}")
    rng = np.random.default_rng(seed)
    rows = len(y_true)
    diffs = []
    for _ in range(n):
        i = rng.integers(0, rows, rows)
        positives = y_true[i].sum(axis=0)
        if ((positives == 0) | (positives == rows)).any():
            continue
        diffs.append(summarize(y_true[i], p_a[i]) - summarize(y_true[i], p_b[i]))
    if not diffs:
        raise ValueError("no resample has both classes in every column")
    d = np.array(diffs)
    point = summarize(y_true, p_a) - summarize(y_true, p_b)
    result: dict[str, Any] = {"resamples": n, "used": len(d), "seed": seed}
    for k, metric in enumerate(METRICS):
        result[metric] = {
            "difference": float(point[k]),
            "mean": float(d[:, k].mean()),
            "lo": float(np.percentile(d[:, k], 2.5)),
            "hi": float(np.percentile(d[:, k], 97.5)),
            "p_le_0": float((d[:, k] <= 0).mean()),
        }
    return result
