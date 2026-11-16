"""Expert aneurysm points in the voxels of a 1 mm head-crop cache, for the location-map head.

Usage (on a compute node): PYTHONPATH=src python scripts/build_points.py [--size 160]

Reads `raw/train_localizers.csv`, the DICOM headers of every cached series that has a point, and
each series' box start in `cache/b2mm-{size}/index.csv`. Writes `labels/points-b2mm-{size}.csv`
(SeriesInstanceUID, location, x, y, z: model-input voxels on RAS+ axes, possibly outside the
cube) and prints counts only.
"""

import argparse
import os
from collections.abc import Sequence
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from neurovision.data.dataset import load_index
from neurovision.data.points import POINT_COLUMNS, inside, points_file, series_points
from neurovision.data.preprocess import cache_name
from neurovision.paths import atomic_write, store

OFFSETS = ("offset_mm_x", "offset_mm_y", "offset_mm_z")  # cache index: box start, mm
COLUMNS = ["SeriesInstanceUID", "location", *POINT_COLUMNS]

Job = tuple[Path, tuple[float, ...], pd.DataFrame]


def _one(job: Job) -> list[tuple[str, np.ndarray]]:
    series_dir, offset, rows = job
    return series_points(series_dir, offset, rows)


def build(root: Path, size: int, workers: int) -> pd.DataFrame:
    """One row per localizer point of a cached ("ok") series, in that cache's voxels."""
    localizers = pd.read_csv(root / "raw" / "train_localizers.csv")
    index = load_index(root / "cache" / cache_name(size, "mm"))
    ok = index[index["status"] == "ok"]
    uids = sorted(set(localizers["SeriesInstanceUID"]) & set(ok.index))
    groups = dict(tuple(localizers.groupby("SeriesInstanceUID")))
    jobs: list[Job] = [
        (
            root / "raw" / "series" / uid,
            tuple(float(v) for v in ok.loc[uid, list(OFFSETS)]),
            groups[uid],
        )
        for uid in uids
    ]
    with ProcessPoolExecutor(workers) as pool:
        mapped = list(pool.map(_one, jobs, chunksize=4))
    records = [
        {"SeriesInstanceUID": uid, "location": artery, "x": q[0], "y": q[1], "z": q[2]}
        for uid, found in zip(uids, mapped, strict=True)
        for artery, q in found
    ]
    return pd.DataFrame(records, columns=COLUMNS)


def summary(table: pd.DataFrame, localizers: int, size: int) -> dict[str, int]:
    within = np.array([inside(q, size, 0.0) for q in table[list(POINT_COLUMNS)].to_numpy()])
    return {
        "localizer_rows": localizers,
        "points_mapped": len(table),
        "points_inside": int(within.sum()),
        "series_mapped": int(table["SeriesInstanceUID"].nunique()),
        "series_with_a_point_inside": int(table.loc[within, "SeriesInstanceUID"].nunique())
        if len(table)
        else 0,
    }


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--size", type=int, default=160, help="edge of the b2mm cache cube")
    args = parser.parse_args(argv)
    root = store()
    workers = int(os.environ.get("SLURM_CPUS_PER_TASK", "4"))
    table = build(root, args.size, workers)
    path = points_file(root, cache_name(args.size, "mm"))
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write(path, lambda tmp: table.to_csv(tmp, index=False))
    rows = len(pd.read_csv(root / "raw" / "train_localizers.csv"))
    for key, value in summary(table, rows, args.size).items():
        print(f"{key}: {value}")
    print(f"wrote {path.relative_to(root)}")


if __name__ == "__main__":
    main()
