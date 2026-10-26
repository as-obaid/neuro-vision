"""Cache every training series as a 128-cubed volume under $NV_STORE/cache/b2-128.

Resumable: series already in index.csv are skipped. Prints counts and timings only.
"""

import argparse
import csv
import os
import time
from collections import Counter
from collections.abc import Sequence
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from neurovision.data.preprocess import GeometryError, preprocess_with_geometry
from neurovision.paths import store

COLUMNS = (
    "SeriesInstanceUID",
    "status",
    "reason",
    "geometry_source",
    "n_slices",
    "spacing_x",
    "spacing_y",
    "spacing_z",
    "size_x",
    "size_y",
    "size_z",
    "seconds",
)


def _one_thread() -> None:
    torch.set_num_threads(1)  # one process per CPU


def process(uid: str, series_dir: Path, cache: Path) -> dict[str, object]:
    row: dict[str, object] = {"SeriesInstanceUID": uid}
    start = time.perf_counter()
    try:
        volume, geometry = preprocess_with_geometry(series_dir)
    except GeometryError as error:
        row.update(status="excluded", reason=str(error))
    except Exception as error:  # one bad series must not stop a 10-hour run
        row.update(status="excluded", reason=f"error: {type(error).__name__}")
    else:
        tmp = cache / f"{uid}.npy.tmp"
        with tmp.open("wb") as handle:
            np.save(handle, volume)
        os.replace(tmp, cache / f"{uid}.npy")
        row.update(status="ok", reason="", geometry_source=geometry.source)
        row.update(n_slices=geometry.size[2])
        for axis, name in enumerate("xyz"):
            row[f"spacing_{name}"] = round(geometry.spacing[axis], 4)
            row[f"size_{name}"] = geometry.size[axis]
    row["seconds"] = round(time.perf_counter() - start, 2)
    return row


def main(argv: Sequence[str] | None = None) -> None:
    argparse.ArgumentParser(description=__doc__).parse_args(argv)
    root = store()
    cache = root / "cache" / "b2-128"
    cache.mkdir(parents=True, exist_ok=True)
    index = cache / "index.csv"
    done = set(pd.read_csv(index, dtype=str)["SeriesInstanceUID"]) if index.exists() else set()
    uids = pd.read_csv(root / "raw" / "train.csv", usecols=["SeriesInstanceUID"], dtype=str)
    todo = [u for u in uids["SeriesInstanceUID"] if u not in done]
    workers = int(os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count() or 1))
    with (
        index.open("a", newline="") as handle,
        ProcessPoolExecutor(workers, None, _one_thread) as pool,
    ):
        writer = csv.DictWriter(handle, COLUMNS)
        if not done:
            writer.writeheader()
        futures = [pool.submit(process, u, root / "raw" / "series" / u, cache) for u in todo]
        for future in as_completed(futures):
            writer.writerow(future.result())
            handle.flush()
    table = pd.read_csv(index, dtype={"reason": str}).fillna({"reason": ""})
    reasons = Counter(table.loc[table["status"] == "excluded", "reason"])
    print(f"ok {(table['status'] == 'ok').sum()}, excluded {sum(reasons.values())}")
    for reason, count in sorted(reasons.items()):
        print(f"  excluded {reason}: {count}")
    print(f"seconds total {table['seconds'].sum():.0f}, mean {table['seconds'].mean():.2f}")


if __name__ == "__main__":
    main()
