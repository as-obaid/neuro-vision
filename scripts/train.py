"""Train a baseline (prior or gbm) on every fold and write its out-of-fold predictions."""

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

import pandas as pd

from neurovision.baselines.b0_prior import predict_prior
from neurovision.baselines.b1_gbm import predict_gbm
from neurovision.config import load_config
from neurovision.data.folds import load_folds
from neurovision.data.labels import TARGETS, load_labels
from neurovision.paths import store
from neurovision.track.meta import run_meta


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("--fold", type=int)
    args = parser.parse_args(argv)
    config = load_config(args.config)
    if config.kind == "cnn":
        raise SystemExit("kind cnn: not built yet")
    if args.fold is not None:
        raise SystemExit(f"kind {config.kind} trains all folds; --fold is not allowed")

    root = store()
    folds = load_folds(root)
    labels = load_labels(root, config.labels).loc[folds.index]
    for k in sorted(folds.unique()):
        train, val = labels[folds != k], labels[folds == k]
        if config.gbm is None:
            prob = predict_prior(train, val)
        else:
            prob = predict_gbm(train, val, config.gbm, config.seed)
        run_dir = root / "runs" / config.name / f"fold{k}"
        run_dir.mkdir(parents=True, exist_ok=True)
        oof = pd.DataFrame(prob, index=val.index, columns=list(TARGETS))
        oof.to_csv(run_dir / "oof.csv")
        (run_dir / "meta.json").write_text(json.dumps(run_meta(config), indent=2))
        print(f"fold {k}: {len(val)} validation series")


if __name__ == "__main__":
    main()
