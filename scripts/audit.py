"""Training-label audit (counts only) and, with --verify, a check of the unzipped series."""

import argparse
import json
import os
from collections.abc import Sequence
from pathlib import Path

import pandas as pd

from neurovision.data.labels import TARGETS, load_labels
from neurovision.paths import store

AGE_BANDS = (("<40", 0, 40), ("40-59", 40, 60), ("60-79", 60, 80), (">=80", 80, 10**6))


def table(header: list[str], rows: list[list[object]]) -> str:
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join(":--" for _ in header) + "|"]
    lines += ["| " + " | ".join(str(c) for c in row) + " |" for row in rows]
    return "\n".join(lines)


def summarise(labels: pd.DataFrame) -> dict[str, object]:
    modalities = sorted(labels["Modality"].dropna().unique())
    age = labels["PatientAge"].dropna()
    positives = {}
    for target in TARGETS:
        by_modality = labels.groupby("Modality")[target].sum()
        positives[target] = {m: int(by_modality.get(m, 0)) for m in modalities}
        positives[target]["total"] = int(labels[target].sum())
    return {
        "n_series": len(labels),
        "modality": {m: int(n) for m, n in labels["Modality"].value_counts().items()},
        "positives": positives,
        "age": {
            "count": len(age),
            "mean": float(age.mean()),
            "median": float(age.median()),
            "min": int(age.min()),
            "max": int(age.max()),
            "bands": {name: int(((age >= lo) & (age < hi)).sum()) for name, lo, hi in AGE_BANDS},
        },
        "sex": {s: int(n) for s, n in labels["PatientSex"].value_counts().items()},
        "missing": {c: int(n) for c, n in labels.isna().sum().items()},
    }


def print_tables(summary: dict[str, object]) -> None:
    modality: dict[str, int] = summary["modality"]  # type: ignore[assignment]
    positives: dict[str, dict[str, int]] = summary["positives"]  # type: ignore[assignment]
    age: dict[str, object] = summary["age"]  # type: ignore[assignment]
    sex: dict[str, int] = summary["sex"]  # type: ignore[assignment]
    missing: dict[str, int] = summary["missing"]  # type: ignore[assignment]
    print(f"Series: {summary['n_series']}\n")
    print(table(["Modality", "Series"], [[m, n] for m, n in modality.items()]), "\n")
    cols = list(next(iter(positives.values())))
    print(table(["Target", *cols], [[t, *row.values()] for t, row in positives.items()]), "\n")
    stats = {
        k: (round(v, 1) if isinstance(v, float) else v) for k, v in age.items() if k != "bands"
    }
    print(table(["Age statistic", "Value"], [[k, v] for k, v in stats.items()]), "\n")
    print(table(["Age band", "Series"], [[b, n] for b, n in age["bands"].items()]), "\n")  # type: ignore[attr-defined]
    print(table(["Sex", "Series"], [[s, n] for s, n in sex.items()]), "\n")
    print(table(["Column", "Missing"], [[c, n] for c, n in missing.items()]))


def count_dcm(folder: Path) -> int:
    with os.scandir(folder) as entries:
        return sum(1 for e in entries if e.name.endswith(".dcm"))


def verify(root: Path, uids: set[str]) -> int:
    series = root / "raw" / "series"
    present = missing = empty = dcm_total = single = 0
    for uid in sorted(uids):
        folder = series / uid
        if not folder.is_dir():
            missing += 1
            continue
        n = count_dcm(folder)
        present += 1
        empty += n == 0
        single += n == 1
        dcm_total += n
    with os.scandir(series) as entries:
        extra = sum(1 for e in entries if e.is_dir() and e.name not in uids)
    with os.scandir(root / "raw" / "segmentations") as entries:
        segmentations = sum(1 for e in entries if e.is_file())
    print(f"present: {present}\nmissing: {missing}\nempty: {empty}\ndcm files: {dcm_total}")
    print(f"single-file series: {single}\nnot in train.csv: {extra}")
    print(f"segmentation files: {segmentations}")
    return int(missing > 0 or empty > 0)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify", action="store_true", help="check raw/series after the unzip")
    args = parser.parse_args(argv)
    root = store()
    labels = load_labels(root, "original")
    if args.verify:
        if not (root / "raw" / ".complete").is_file():
            raise SystemExit(f"{root / 'raw' / '.complete'} is missing: the unzip has not finished")
        return verify(root, set(labels.index))
    summary = summarise(labels)
    print_tables(summary)
    (root / "logs").mkdir(exist_ok=True)
    (root / "logs" / "audit.json").write_text(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
