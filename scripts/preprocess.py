"""Cache every training series as a cubic volume under $NV_STORE/cache/b2-{size}.

--size sets the edge length (default 128); each size has its own folder and index.csv.
--resample mm writes 1 mm voxels cropped to a box on the head under cache/b2mm-{size}; its index
also records the kept head fraction, the box offset and the scan extent (mm, RAS+ axes).
--shard K/N takes every N-th series from the K-th and logs to index-KofN.csv, so N jobs can run at
once; the next run without --shard merges the shard files into index.csv (start it after all end).

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

from neurovision.data.preprocess import (
    ERROR_PREFIX,
    SIZE,
    GeometryError,
    Resample,
    cache_name,
    preprocess_with_crop,
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
CROP_COLUMNS = (
    "kept_fraction",
    *(f"offset_mm_{a}" for a in "xyz"),
    *(f"extent_mm_{a}" for a in "xyz"),
)


def _one_thread() -> None:
    torch.set_num_threads(1)  # one process per CPU


def _save_npy(path: Path, volume: np.ndarray) -> None:
    with path.open("wb") as handle:  # a handle keeps the name: np.save(path) would add ".npy"
        np.save(handle, volume)


def _columns(resample: Resample) -> tuple[str, ...]:
    return COLUMNS if resample == "cube" else (*COLUMNS, *CROP_COLUMNS)


def process(
    uid: str, series_dir: Path, cache: Path, size: int = SIZE, resample: Resample = "cube"
) -> dict[str, object]:
    row: dict[str, object] = {"SeriesInstanceUID": uid}
    start = time.perf_counter()
    try:
        volume, geometry, crop = preprocess_with_crop(series_dir, size, resample)
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
        if crop is not None:
            row["kept_fraction"] = round(crop.kept_fraction, 4)
            for axis, name in enumerate("xyz"):
                row[f"offset_mm_{name}"] = round(crop.offset_mm[axis], 1)
                row[f"extent_mm_{name}"] = round(crop.extent_mm[axis], 1)
    row["seconds"] = round(time.perf_counter() - start, 2)
    return row


def _final_rows(index: Path) -> list[dict[str, str]]:
    """Complete rows that are final (not an "error:" retry) in an index file."""
    text = index.read_text() if index.exists() else ""
    lines = text.splitlines(keepends=True)
    if lines and not lines[-1].endswith("\n"):
        lines.pop()  # a row cut short by a kill
    return [
        row
        for row in csv.DictReader(lines)
        if not row["reason"].startswith(ERROR_PREFIX) and row["status"] in ("ok", "excluded")
    ]


def _write_index(index: Path, rows: Sequence[dict[str, str]], columns: Sequence[str]) -> None:
    def write(tmp: Path) -> None:
        with tmp.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, columns)
            writer.writeheader()
            writer.writerows(rows)

    atomic_write(index, write)


def _load_index(index: Path, columns: Sequence[str] = COLUMNS) -> list[dict[str, str]]:
    """Rewrite the index as header plus complete, final rows, after merging any shard files."""
    kept = _final_rows(index)
    shards = sorted(index.parent.glob("index-*of*.csv"))
    seen = {row["SeriesInstanceUID"] for row in kept}
    for shard in shards:
        new = [row for row in _final_rows(shard) if row["SeriesInstanceUID"] not in seen]
        kept += new
        seen |= {row["SeriesInstanceUID"] for row in new}
    _write_index(index, kept, columns)
    for shard in shards:
        shard.unlink()
    return kept


def _shard(text: str) -> tuple[int, int]:
    k, _, n = text.partition("/")
    if not (k.isdigit() and n.isdigit() and int(k) < int(n)):
        raise argparse.ArgumentTypeError(f"--shard is K/N with 0 <= K < N, got {text!r}")
    return int(k), int(n)


def _died(uid: str) -> dict[str, object]:
    return {"SeriesInstanceUID": uid, "status": "excluded", "reason": f"{ERROR_PREFIX} worker died"}


def _run(
    todo: Sequence[str],
    root: Path,
    cache: Path,
    workers: int,
    handle: TextIO,
    size: int = SIZE,
    resample: Resample = "cube",
) -> None:
    # At most `workers` series in flight, so a dead worker taints few series; they are retried
    # on the next run.
    writer = csv.DictWriter(handle, _columns(resample))

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
                            process, uid, root / "raw" / "series" / uid, cache, size, resample
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
    parser.add_argument("--resample", choices=["cube", "mm"], default="cube")
    parser.add_argument("--shard", type=_shard, help="K/N: every N-th series from the K-th")
    args = parser.parse_args(argv)
    size, resample = args.size, args.resample
    root = store()
    require_unzipped(root)
    cache = root / "cache" / cache_name(size, resample)
    cache.mkdir(parents=True, exist_ok=True)
    index, columns = cache / "index.csv", _columns(resample)
    k, n = args.shard or (0, 1)
    if args.shard is None:
        log, rows = index, _load_index(index, columns)
    else:  # index.csv is only read: other shards may be running
        log = cache / f"index-{k}of{n}.csv"
        rows = _final_rows(log)
        _write_index(log, rows, columns)
        rows += _final_rows(index)
    done = {row["SeriesInstanceUID"] for row in rows}
    uids = pd.read_csv(root / "raw" / "train.csv", usecols=["SeriesInstanceUID"], dtype=str)
    todo = [u for i, u in enumerate(uids["SeriesInstanceUID"]) if i % n == k and u not in done]
    workers = int(os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count() or 1))
    with log.open("a", newline="") as handle:
        _run(todo, root, cache, workers, handle, size, resample)
    table = pd.read_csv(log, dtype={"reason": str}).fillna({"reason": ""})
    reasons = Counter(table.loc[table["status"] == "excluded", "reason"])
    print(f"ok {(table['status'] == 'ok').sum()}, excluded {sum(reasons.values())}")
    for reason, count in sorted(reasons.items()):
        print(f"  excluded {reason}: {count}")
    print(f"seconds total {table['seconds'].sum():.0f}, mean {table['seconds'].mean():.2f}")


if __name__ == "__main__":
    main()
