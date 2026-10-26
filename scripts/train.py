"""Train a config and write out-of-fold predictions: prior and gbm on every fold, cnn on one."""

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

import pandas as pd
from monai.transforms import Compose

from neurovision.baselines.b0_prior import predict_prior
from neurovision.baselines.b1_gbm import predict_gbm
from neurovision.config import Config, load_config
from neurovision.data.augment import build_augment
from neurovision.data.dataset import CacheDataset, load_index
from neurovision.data.folds import load_folds
from neurovision.data.labels import TARGETS, load_labels
from neurovision.data.preprocess import ERROR_PREFIX
from neurovision.paths import atomic_write, fold_dir, store
from neurovision.track.meta import run_meta
from neurovision.train.loop import fit


def _train_cnn(config: Config, fold: int) -> None:
    root = store()
    folds = load_folds(root)
    if fold not in set(folds):
        raise SystemExit(f"fold {fold} is not in folds.csv")
    labels = load_labels(root, config.labels).loc[folds.index, list(TARGETS)]
    cache = root / "cache" / "b2-128"
    index = load_index(cache)
    missing = len(folds.index.difference(index.index))
    if missing:
        raise SystemExit(
            f"{missing} of {len(folds)} series have no row in cache/b2-128/index.csv; "
            "rerun slurm/preprocess.sbatch"
        )
    ok = set(index.index[index["status"] == "ok"])

    def dataset(frame: pd.DataFrame, transform: Compose | None) -> CacheDataset:
        frame = frame[frame.index.isin(ok)]
        return CacheDataset(cache, list(frame.index), frame.to_numpy(), transform)

    train, val = labels[folds != fold], labels[folds == fold]
    assert config.augment is not None
    train_ds, val_ds = dataset(train, build_augment(config.augment)), dataset(val, None)
    counts = {
        "train_used": len(train_ds),
        "train_excluded": len(train) - len(train_ds),
        "validation_scored": len(val_ds),
        "validation_prior_filled": len(val) - len(val_ds),
        "index_errors": int(index["reason"].str.startswith(ERROR_PREFIX).sum()),
    }
    print(
        f"train {counts['train_used']} used, {counts['train_excluded']} excluded; "
        f"validation {counts['validation_scored']} scored, "
        f"{counts['validation_prior_filled']} prior-filled; "
        f"{counts['index_errors']} index errors"
    )
    run_dir = fold_dir(root, config.name, fold)
    run_dir.mkdir(parents=True, exist_ok=True)
    fit(
        config,
        fold,
        run_dir,
        train_ds,
        val_ds,
        list(val.index),
        train.to_numpy().mean(axis=0),
        counts,
    )


def _train_baseline(config: Config) -> None:
    root = store()
    folds = load_folds(root)
    labels = load_labels(root, config.labels).loc[folds.index]
    meta = json.dumps(run_meta(config), indent=2)
    for k in sorted(folds.unique()):
        train, val = labels[folds != k], labels[folds == k]
        if config.gbm is None:
            prob = predict_prior(train, val)
        else:
            prob = predict_gbm(train, val, config.gbm, config.seed)
        run_dir = fold_dir(root, config.name, k)
        run_dir.mkdir(parents=True, exist_ok=True)
        oof = pd.DataFrame(prob, index=val.index, columns=list(TARGETS))
        atomic_write(run_dir / "oof.csv", oof.to_csv)
        atomic_write(run_dir / "meta.json", lambda tmp: tmp.write_text(meta))
        print(f"fold {k}: {len(val)} validation series")


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("--fold", type=int)
    args = parser.parse_args(argv)
    config = load_config(args.config)
    if config.kind == "cnn":
        if args.fold is None:
            raise SystemExit("kind cnn trains one fold; --fold is required")
        _train_cnn(config, args.fold)
    elif args.fold is not None:
        raise SystemExit(f"kind {config.kind} trains all folds; --fold is not allowed")
    else:
        _train_baseline(config)


if __name__ == "__main__":
    main()
