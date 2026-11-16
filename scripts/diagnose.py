"""Diagnose a config's out-of-fold predictions against a reference config.

Writes runs/<config>/diagnostics.json: pooled and per-fold scores, AUCs per scan type, a paired
bootstrap (config minus reference), and Aneurysm Present operating points and calibration. Every
value is an aggregate; no series ID is written.
"""

import argparse
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from neurovision.config import load_config
from neurovision.data.folds import load_folds
from neurovision.data.labels import TARGETS, load_labels
from neurovision.evaluate.bootstrap import METRICS, paired_bootstrap, summarize
from neurovision.evaluate.curves import calibration, operating_points
from neurovision.evaluate.subgroups import by_modality, summarize_subset
from neurovision.paths import store


def pooled_oof(root: Path, name: str, folds: pd.Series) -> tuple[pd.DataFrame, list[str]]:
    """Every fold's oof.csv, with the same checks as scripts/score_oof.py, and the commits."""
    run_root = root / "runs" / name
    found = sorted((int(p.parent.name[4:]), p) for p in run_root.glob("fold*/oof.csv"))
    if not found:
        raise SystemExit(f"no oof.csv under {run_root}")
    parts, commits = [], []
    for k, path in found:
        oof = pd.read_csv(path, index_col="SeriesInstanceUID")
        expected = folds.index[folds == k]
        if oof.index.has_duplicates:
            raise SystemExit(f"{name} fold {k}: duplicate series")
        missing, extra = len(expected.difference(oof.index)), len(oof.index.difference(expected))
        if missing or extra:
            raise SystemExit(f"{name} fold {k}: {missing} series missing, {extra} not in this fold")
        parts.append(oof[list(TARGETS)])
        commits.append(json.loads((path.parent / "meta.json").read_text())["commit"])
    return pd.concat(parts), list(dict.fromkeys(commits))


def describe(labels: pd.DataFrame, probs: pd.DataFrame, folds: pd.Series) -> dict[str, Any]:
    y, p = labels[list(TARGETS)].to_numpy(), probs.to_numpy()
    per_fold = {
        str(k): summarize_subset(y[(folds == k).to_numpy()], p[(folds == k).to_numpy()])
        for k in sorted(folds.unique())
    }
    scores = [f["score"] for f in per_fold.values() if f["score"] is not None]
    spread: dict[str, Any] = {"folds": len(scores), "mean": None, "sd": None}
    if len(scores) > 1:
        spread.update(mean=float(np.mean(scores)), sd=float(np.std(scores, ddof=1)))
    return {
        "pooled": dict(zip(METRICS, map(float, summarize(y, p)), strict=True)),
        "per_fold": per_fold,
        "per_fold_score": spread,
        "by_modality": by_modality(labels, probs),
        "aneurysm_present": {
            **operating_points(y[:, -1], p[:, -1]),
            "calibration": calibration(y[:, -1], p[:, -1]),
        },
    }


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("--against", type=Path, required=True, help="reference config")
    args = parser.parse_args(argv)
    config, against = load_config(args.config), load_config(args.against)

    root = store()
    all_folds = load_folds(root)
    probs, commits = pooled_oof(root, config.name, all_folds)
    ref, ref_commits = pooled_oof(root, against.name, all_folds)
    if not probs.index.sort_values().equals(ref.index.sort_values()):
        raise SystemExit(f"{config.name} and {against.name} cover different series")
    ref = ref.loc[probs.index]
    labels = load_labels(root, config.labels).loc[probs.index]
    folds = all_folds.loc[probs.index]

    mine, theirs = describe(labels, probs, folds), describe(labels, ref, folds)
    sections = ("pooled", "per_fold", "per_fold_score", "by_modality", "aneurysm_present")
    y = labels[list(TARGETS)].to_numpy()
    result: dict[str, Any] = {
        "config": config.name,
        "against": against.name,
        "n": len(probs),
        "positives": int(y[:, -1].sum()),
        "commits": {config.name: commits, against.name: ref_commits},
        **{s: {config.name: mine[s], against.name: theirs[s]} for s in sections},
        "bootstrap": paired_bootstrap(y, probs.to_numpy(), ref.to_numpy()),
    }
    out = root / "runs" / config.name / "diagnostics.json"
    out.write_text(json.dumps(result, indent=2))
    boot = result["bootstrap"]
    print(f"{config.name} minus {against.name}, {boot['used']} of {boot['resamples']} resamples")
    for metric in METRICS:
        b = boot[metric]
        print(f"  {metric}: {b['difference']:+.4f} (95% CI {b['lo']:+.4f} to {b['hi']:+.4f})")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
