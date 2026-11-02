"""Cache every training series as a cubic volume under $NV_STORE/cache/b2-{size}.

--size sets the edge length (default 128); each size has its own folder and index.csv.

Resumable: series already in index.csv are skipped, except rows whose reason starts with
"error:" (transient failures), which are retried. Prints counts and timings only.
"""

import argparse
import csv
import os
import time
from collections import Counter, deque
from collections.abc import Sequence
from concurrent.futures import FIRST_COMPLETED, Future, ProcessPoolExecutor, wait
from concurrent.futures.process import BrokenProcessPool
from pathlib import Path
from typing import TextIO

import numpy as np
import pandas as pd
import torch

from neurovision.data.dataset import load_index
from neurovision.data.preprocess import (
    ERROR_PREFIX,
    SIZE,
    GeometryError,
    preprocess_with_geometry,
)
from neurovision.paths import atomic_write, require_unzipped, store

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


def _save_npy(path: Path, volume: np.ndarray) -> None:
    with path.open("wb") as handle:  # a handle keeps the name: np.save(path) would add ".npy"
        np.save(handle, volume)


def process(uid: str, series_dir: Path, cache: Path, size: int = SIZE) -> dict[str, object]:
    row: dict[str, object] = {"SeriesInstanceUID": uid}
    start = time.perf_counter()
    try:
        volume, geometry = preprocess_with_geometry(series_dir, size)
    except GeometryError as error:
        row.update(status="excluded", reason=str(error))
    except Exception as error:  # one bad series must not stop a 10-hour run
        row.update(status="excluded", reason=f"{ERROR_PREFIX} {type(error).__name__}")
    else:
        atomic_write(cache / f"{uid}.npy", lambda tmp: _save_npy(tmp, volume))
        row.update(status="ok", reason="", geometry_source=geometry.source)
        row.update(n_slices=geometry.size[2])
        for axis, name in enumerate("xyz"):
            row[f"spacing_{name}"] = round(geometry.spacing[axis], 4)
            row[f"size_{name}"] = geometry.size[axis]
    row["seconds"] = round(time.perf_counter() - start, 2)
    return row


def _load_index(index: Path) -> list[dict[str, str]]:
    """Rewrite the index as header plus complete, final rows; return the rows kept."""
    text = index.read_text() if index.exists() else ""
    lines = text.splitlines(keepends=True)
    if lines and not lines[-1].endswith("\n"):
        lines.pop()  # a row cut short by a kill
    kept = [
        row
        for row in csv.DictReader(lines)
        if not row["reason"].startswith(ERROR_PREFIX) and row["status"] in ("ok", "excluded")
    ]

    def write(tmp: Path) -> None:
        with tmp.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, COLUMNS)
            writer.writeheader()
            writer.writerows(kept)

    atomic_write(index, write)
    return kept


def _died(uid: str) -> dict[str, object]:
    return {"SeriesInstanceUID": uid, "status": "excluded", "reason": f"{ERROR_PREFIX} worker died"}


def _run(
    todo: Sequence[str], root: Path, cache: Path, workers: int, handle: TextIO, size: int = SIZE
) -> None:
    # At most `workers` series in flight, so a dead worker taints few series; they are retried
    # on the next run.
    writer = csv.DictWriter(handle, COLUMNS)

    def record(row: dict[str, object]) -> None:
        writer.writerow(row)
        handle.flush()

    pending = deque(todo)
    while pending:
        inflight: dict[Future[dict[str, object]], str] = {}
        with ProcessPoolExecutor(workers, None, _one_thread) as pool:
            try:
                while pending or inflight:
                    while pending and len(inflight) < workers:
                        uid = pending.popleft()
                        future = pool.submit(
                            process, uid, root / "raw" / "series" / uid, cache, size
                        )
                        inflight[future] = uid
                    finished, _ = wait(inflight, return_when=FIRST_COMPLETED)
                    for future in finished:
                        record(future.result())
                        del inflight[future]
            except BrokenProcessPool:
                for future, uid in inflight.items():
                    broken = not future.done() or future.exception() is not None
                    record(_died(uid) if broken else future.result())


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--size", type=int, default=SIZE, help="edge length in voxels")
    size = parser.parse_args(argv).size
    root = store()
    require_unzipped(root)
    cache = root / "cache" / f"b2-{size}"
    cache.mkdir(parents=True, exist_ok=True)
    index = cache / "index.csv"
    done = {row["SeriesInstanceUID"] for row in _load_index(index)}
    uids = pd.read_csv(root / "raw" / "train.csv", usecols=["SeriesInstanceUID"], dtype=str)
    todo = [u for u in uids["SeriesInstanceUID"] if u not in done]
    workers = int(os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count() or 1))
    with index.open("a", newline="") as handle:
        _run(todo, root, cache, workers, handle, size)
    table = load_index(cache)
    reasons = Counter(table.loc[table["status"] == "excluded", "reason"])
    print(f"ok {(table['status'] == 'ok').sum()}, excluded {sum(reasons.values())}")
    for reason, count in sorted(reasons.items()):
        print(f"  excluded {reason}: {count}")
    print(f"seconds total {table['seconds'].sum():.0f}, mean {table['seconds'].mean():.2f}")


if __name__ == "__main__":
    main()
