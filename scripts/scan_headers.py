"""Scan the first DICOM header of every training series for the patient-grouping decision.

Writes $NV_STORE/labels/headers.csv and prints counts only.
"""

import argparse
import os
from collections.abc import Sequence
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pandas as pd
import pydicom
from pydicom.multival import MultiValue
from pydicom.sequence import Sequence as DicomSequence

from neurovision.data.dicom import TEST_TAGS
from neurovision.paths import store

CANDIDATES = ("PatientID", "StudyInstanceUID", "FrameOfReferenceUID", "StudyDate")
EXTRA_TAGS = tuple(tag for tag in CANDIDATES if tag not in TEST_TAGS)
MAX_AGE_RANGE = 2


def _text(value: object) -> str:
    if isinstance(value, DicomSequence):  # per-frame groups: the frame count, not the content
        return str(len(value))
    if isinstance(value, MultiValue):
        return "\\".join(str(v) for v in value)
    return "" if value is None else str(value)


def scan_series(series_dir: Path) -> dict[str, str]:
    files = sorted(series_dir.glob("*.dcm"))
    ds = pydicom.dcmread(files[0], specific_tags=[*TEST_TAGS, *EXTRA_TAGS])
    row = {"SeriesInstanceUID": series_dir.name, "n_files": str(len(files))}
    for tag in (*TEST_TAGS, *EXTRA_TAGS):
        row[tag] = _text(ds.get(tag, ds.file_meta.get(tag)))
    return row


def summarise(headers: pd.DataFrame, train: pd.DataFrame, tag: str) -> str:
    frame = headers[["SeriesInstanceUID", tag]].merge(
        train[["SeriesInstanceUID", "PatientAge", "PatientSex"]], on="SeriesInstanceUID"
    )
    frame = frame[frame[tag] != ""]
    sizes = frame.groupby(tag).size()
    shared = sizes[sizes >= 2].index
    consistent = 0
    for _, group in frame[frame[tag].isin(shared)].groupby(tag):
        ages = pd.to_numeric(group["PatientAge"], errors="coerce")
        if group["PatientSex"].nunique() == 1 and ages.max() - ages.min() <= MAX_AGE_RANGE:
            consistent += 1
    return (
        f"{tag}: non-missing {len(frame)}, distinct {len(sizes)}, "
        f"series per value max {sizes.max()} mean {sizes.mean():.2f}, "
        f"shared by 2+ series {len(shared)}, consistent sex and age {consistent}"
    )


def main(argv: Sequence[str] | None = None) -> None:
    argparse.ArgumentParser(description=__doc__).parse_args(argv)
    root = store()
    train = pd.read_csv(root / "raw" / "train.csv")
    known = set(train["SeriesInstanceUID"])
    dirs = sorted(d for d in (root / "raw" / "series").iterdir() if d.name in known)
    workers = int(os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count() or 1))
    with ProcessPoolExecutor(max_workers=workers) as pool:
        rows = list(pool.map(scan_series, dirs, chunksize=8))
    headers = pd.DataFrame(rows, dtype=str)
    out = root / "labels" / "headers.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    headers.to_csv(out, index=False)
    print(f"series scanned {len(headers)} of {len(known)} in train.csv")
    for tag in CANDIDATES:
        print(summarise(headers, train, tag))


if __name__ == "__main__":
    main()
