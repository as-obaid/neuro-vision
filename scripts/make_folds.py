"""Write folds/folds.csv and folds/folds.yaml in the store (5 folds, seed 0, corrected labels)."""

import argparse
import hashlib
from collections.abc import Sequence
from datetime import UTC, datetime
from typing import Final, Literal

import pandas as pd
import yaml

from neurovision.data.folds import make_folds
from neurovision.data.labels import ANEURYSM_PRESENT, load_labels
from neurovision.paths import store

N_FOLDS = 5
SEED = 0
LABEL_SOURCE: Final[Literal["corrected"]] = "corrected"


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--group-key", required=True, help="headers.csv column, or 'none'")
    key = parser.parse_args(argv).group_key
    root = store()
    folds_csv = root / "folds" / "folds.csv"
    if folds_csv.exists():
        raise SystemExit(f"{folds_csv} exists; refusing to overwrite")

    labels = load_labels(root, LABEL_SOURCE)
    groups = None
    if key != "none":
        # Read as text: IDs like "00123" or "NA" must survive; an empty cell is missing.
        headers = pd.read_csv(
            root / "labels" / "headers.csv",
            index_col="SeriesInstanceUID",
            dtype=str,
            keep_default_na=False,
        )
        column = headers[key].reindex(labels.index)
        # A series without a value is its own group.
        own = pd.Series("single:" + labels.index.astype(str), index=labels.index)
        groups = column.where(column.notna() & (column != ""), own)
    folds = make_folds(labels, N_FOLDS, SEED, groups)

    folds_csv.parent.mkdir(parents=True, exist_ok=True)
    folds.rename("fold").to_csv(folds_csv)
    manifest = {
        "n_folds": N_FOLDS,
        "seed": SEED,
        "group_key": key,
        "labels": LABEL_SOURCE,
        "folds_sha256": hashlib.sha256(folds_csv.read_bytes()).hexdigest(),
        "date": datetime.now(UTC).date().isoformat(),
    }
    (folds_csv.parent / "folds.yaml").write_text(yaml.safe_dump(manifest, sort_keys=False))

    print("series per fold:", folds.value_counts().sort_index().to_dict())
    print(
        "aneurysm rate per fold:", labels[ANEURYSM_PRESENT].groupby(folds).mean().round(3).to_dict()
    )
    print("modality per fold:")
    print(pd.crosstab(folds, labels["Modality"]).to_string())
    sizes = (groups if groups is not None else pd.Series(range(len(labels)))).value_counts()
    print(f"groups: {len(sizes)}, largest: {sizes.max()}")


if __name__ == "__main__":
    main()
