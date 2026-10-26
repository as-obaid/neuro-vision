"""Score the out-of-fold predictions of a config and write runs/<name>/score.json."""

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

import pandas as pd

from neurovision.config import load_config
from neurovision.data.folds import load_folds
from neurovision.data.labels import TARGETS, load_labels
from neurovision.evaluate.metric import label_aucs, score
from neurovision.paths import store


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    config = load_config(parser.parse_args(argv).config)

    root = store()
    folds = load_folds(root)
    run_root = root / "runs" / config.name
    files = sorted(run_root.glob("fold*/oof.csv"), key=lambda p: int(p.parent.name[4:]))
    if not files:
        raise SystemExit(f"no oof.csv under {run_root}")
    parts = []
    commits: list[str] = []
    for path in files:
        k = int(path.parent.name[4:])
        oof = pd.read_csv(path, index_col="SeriesInstanceUID")
        expected = folds.index[folds == k]
        if oof.index.has_duplicates:
            raise SystemExit(f"fold {k}: duplicate series")
        missing, extra = len(expected.difference(oof.index)), len(oof.index.difference(expected))
        if missing or extra:
            raise SystemExit(f"fold {k}: {missing} series missing, {extra} not in this fold")
        parts.append(oof[list(TARGETS)])
        meta = json.loads((path.parent / "meta.json").read_text())
        if meta["commit"] not in commits:
            commits.append(meta["commit"])
    oof = pd.concat(parts)

    y_true = load_labels(root, config.labels).loc[oof.index, list(TARGETS)].to_numpy()
    y_prob = oof.to_numpy()
    aucs = label_aucs(y_true, y_prob)
    result = {
        "config": config.name,
        "folds": [int(p.parent.name[4:]) for p in files],
        "n": len(oof),
        "score": score(y_true, y_prob),
        "aucs": aucs,
        "commit": commits[0] if len(commits) == 1 else commits,
    }
    (run_root / "score.json").write_text(json.dumps(result, indent=2))
    print(f"score {result['score']:.4f}")
    for name, auc in aucs.items():
        print(f"  {auc:.4f}  {name}")


if __name__ == "__main__":
    main()
