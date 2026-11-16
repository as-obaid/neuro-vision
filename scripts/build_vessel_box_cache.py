"""Caches cut from the DICOM around each series' predicted vessel tree (`data.vbox.vessel_box`).

Usage (CPU node): PYTHONPATH=src python scripts/build_vessel_box_cache.py --fold 0
  default: the full version's input, a 192³ box of 0.5 mm voxels (cache vbox05-192);
  --spacing 1.0 --size 160: the 1 mm 160 mm head crop re-centred on the vessels (cache vc1mm-160).
Reads labels/vessel-boxes-fold<k>.csv (scripts/predict_vessel_boxes.py), the box start of the
1 mm cache (cache/b2mm-160/index.csv), the DICOM (raw/series/<uid>) and the Shi masks. Writes
cache/<out>/<uid>.npy (float16, normalised like every cache), its index.csv (status, reason,
box_fold, centre in 1 mm cache voxels, fallback, seg, seconds), cache/<out>-vessels/<uid>.npz
(key "seg": the 27 classes of `data.vessels`) and labels/points-<out>.csv (the expert points of
labels/points-b2mm-160.csv moved into each box, for the location-map head). A series whose
prediction held no vessel tree (fallback 1) is centred on the median centre of the others; at
1 mm the head crop instead moves by whole voxels (b2mm-160's own grid), and a series without a
tree keeps the head crop itself (no move).
Resumable: series with a final index row are skipped; rows whose reason starts with "error:"
are retried. --seg-only --uid-file F rewrites only the masks of the listed "ok" series (after a
change to the mask mapping), leaving the cut volumes and the index as they are. Then checks the
alignment: each expert point to the nearest voxel of its own aneurysm class in its box, in mm.
Prints counts only.
"""

import argparse
import csv
import json
import os
import time
from collections import Counter
from collections.abc import Sequence
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
import torch

from neurovision.data.dataset import load_index
from neurovision.data.dicom import read_header
from neurovision.data.labels import LOCATIONS
from neurovision.data.points import POINT_COLUMNS, points_file
from neurovision.data.preprocess import (
    ERROR_PREFIX,
    SPACING_MM,
    CacheKind,
    Geometry,
    GeometryError,
    cache_name,
    series_geometry,
)
from neurovision.data.vbox import BOX_SIZE, BOX_SPACING_MM, cache_from_box, vessel_box
from neurovision.data.vessels import SHI_DIR, SHI_ORIG, shi_to_seg, to_cache
from neurovision.paths import atomic_write, store

SOURCE_SIZE = 160  # the 1 mm cache the boxes are given in
OFFSETS = ("offset_mm_x", "offset_mm_y", "offset_mm_z")  # cache index: box start, mm
CENTRE = ("centre_x", "centre_y", "centre_z")
COLUMNS = (
    "SeriesInstanceUID",
    "status",
    "reason",
    "box_fold",
    *CENTRE,
    "fallback",
    "seg",
    "seconds",
)
Job = tuple[
    str, Path, tuple[float, ...], tuple[float, ...], int, int, float, Path, Path, Path, np.ndarray
]


def _one_thread() -> None:
    torch.set_num_threads(1)  # one process per CPU


def _save_npy(path: Path, volume: np.ndarray) -> None:
    with path.open("wb") as handle:  # a handle keeps the name: np.save(path) would add ".npy"
        np.save(handle, volume)


def centres(boxes: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """(centre per series, fallback flag): series without a tree take the median centre."""
    found = boxes[list(CENTRE)].notna().all(axis=1)
    if not found.any():
        raise SystemExit("no series has a predicted vessel tree")
    filled = boxes[list(CENTRE)].copy()
    filled.loc[~found] = boxes.loc[found, list(CENTRE)].median().to_numpy()
    return filled, (~found).astype(int)


def crop_centres(boxes: pd.DataFrame, size: int) -> tuple[pd.DataFrame, pd.Series]:
    """1 mm (vc1mm): (centre per series, fallback flag). The box starts a whole number of 1 mm
    voxels from the head crop's start, the one nearest the vessel centre, so it keeps b2mm-160's
    grid; a series without a tree keeps the head crop's own centre: no move."""
    found = boxes[list(CENTRE)].notna().all(axis=1)
    middle = (size - 1) / 2
    head = (SOURCE_SIZE - 1) / 2
    wanted = np.where(found.to_numpy()[:, None], boxes[list(CENTRE)].to_numpy(float), head)
    start = np.floor(wanted - middle + 0.5)  # round half up: the same on every platform
    filled = pd.DataFrame(start + middle, index=boxes.index, columns=list(CENTRE))
    return filled, (~found).astype(int)


def process(job: Job) -> dict[str, object]:
    uid, series, offset, centre, fallback, size, spacing, cache, seg_dir, mask, lut = job
    row: dict[str, object] = {"SeriesInstanceUID": uid, "fallback": fallback}
    row.update(dict(zip(CENTRE, centre, strict=True)))
    start = time.perf_counter()
    try:
        volume, geometry, box_from_cache = vessel_box(series, offset, centre, size, spacing)
    except GeometryError as error:
        row.update(status="excluded", reason=str(error), seg="")
    except Exception as error:  # one bad series must not stop the build
        row.update(status="excluded", reason=f"{ERROR_PREFIX} {type(error).__name__}", seg="")
    else:
        atomic_write(cache / f"{uid}.npy", lambda tmp: _save_npy(tmp, volume))
        row.update(
            status="ok",
            reason="",
            seg=_seg(mask, lut, geometry, offset, size, box_from_cache, seg_dir / f"{uid}.npz"),
        )
    row["seconds"] = round(time.perf_counter() - start, 2)
    return row


def _seg(
    mask_path: Path,
    lut: np.ndarray,
    geometry: Geometry,
    offset: tuple[float, ...],
    size: int,
    box_from_cache: np.ndarray,
    out: Path,
) -> str:
    """The Shi classes in the box: "written", "none" (no mask) or why not."""
    if not mask_path.exists():
        return "none"
    try:
        img = nib.load(mask_path)
        assert isinstance(img, nib.Nifti1Image)
        mask = np.asarray(img.dataobj).astype(np.uint8)
        if sorted(mask.shape) != sorted(geometry.size):
            return "shape"
        seg = to_cache(lut[mask], img.affine, geometry, offset, size, box_from_cache)
    except Exception as error:
        return f"error {type(error).__name__}"
    tmp = out.with_suffix(".tmp.npz")
    np.savez_compressed(tmp, seg=seg)
    tmp.replace(out)
    return "written"


SegJob = tuple[Path, np.ndarray, Path, tuple[float, ...], int, np.ndarray, Path]


def _seg_job(job: SegJob) -> str:
    mask, lut, series, offset, size, box_from_cache, out = job
    try:
        geometry = series_geometry([read_header(p) for p in sorted(series.glob("*.dcm"))])
    except Exception as error:
        return f"error {type(error).__name__}"
    return _seg(mask, lut, geometry, offset, size, box_from_cache, out)


def rebuild_segs(
    root: Path, size: int, out: str, spacing: float, uids: set[str], workers: int
) -> dict[str, int]:
    """Rewrite `<out>-vessels/<uid>.npz` of the listed "ok" series in place, at the box centre
    the index stores; the cut volumes and the index stay as they are. Counts by `_seg` result."""
    index = load_index(root / "cache" / out)
    ok = index[(index["status"] == "ok") & index.index.isin(list(uids))]
    source = load_index(root / "cache" / cache_name(SOURCE_SIZE, "mm"))
    lut = shi_to_seg(json.loads((root / SHI_DIR / "labels.json").read_text()))
    seg_dir = root / "cache" / f"{out}-vessels"
    jobs: list[SegJob] = [
        (
            root / SHI_DIR / SHI_ORIG / f"{uid}.nii.gz",
            lut,
            root / "raw" / "series" / uid,
            tuple(float(v) for v in source.loc[uid, list(OFFSETS)]),
            size,
            np.linalg.inv(cache_from_box(ok.loc[uid, list(CENTRE)].to_numpy(float), size, spacing)),
            seg_dir / f"{uid}.npz",
        )
        for uid in ok.index
    ]
    with ProcessPoolExecutor(workers, None, _one_thread) as pool:
        results = list(pool.map(_seg_job, jobs, chunksize=4))
    return dict(Counter(results))


def final_rows(index: Path) -> list[dict[str, str]]:
    """Complete, final rows (not an "error:" retry); a row cut short by a kill is dropped."""
    if not index.exists():
        return []
    lines = index.read_text().splitlines(keepends=True)
    if lines and not lines[-1].endswith("\n"):
        lines.pop()
    return [
        row
        for row in csv.DictReader(lines)
        if row["status"] in ("ok", "excluded") and not row["reason"].startswith(ERROR_PREFIX)
    ]


def build(
    root: Path,
    fold: int,
    size: int,
    workers: int,
    uids: set[str] | None,
    out: str,
    spacing: float = BOX_SPACING_MM,
) -> None:
    boxes = pd.read_csv(
        root / "labels" / f"vessel-boxes-fold{fold}.csv", dtype={"SeriesInstanceUID": str}
    )
    boxes = boxes.drop_duplicates("SeriesInstanceUID", keep="last").set_index("SeriesInstanceUID")
    if (boxes["box_fold"] != fold).any():
        raise SystemExit("the boxes file mixes folds")
    filled, fallback = crop_centres(boxes, size) if spacing == SPACING_MM else centres(boxes)
    source = load_index(root / "cache" / cache_name(SOURCE_SIZE, "mm"))
    cache = root / "cache" / out
    seg_dir = root / "cache" / f"{out}-vessels"
    cache.mkdir(parents=True, exist_ok=True)
    seg_dir.mkdir(parents=True, exist_ok=True)
    index = cache / "index.csv"
    kept = final_rows(index)
    done = {row["SeriesInstanceUID"] for row in kept}
    lut = shi_to_seg(json.loads((root / SHI_DIR / "labels.json").read_text()))
    todo = [u for u in boxes.index if u not in done and (uids is None or u in uids)]
    # every series gets a row (training refuses a cache index that misses one): those the 1 mm
    # cache excluded are excluded here too
    for uid in source.index[(source["status"] != "ok") & ~source.index.isin(done)]:
        if uids is None or uid in uids:
            reason = f"not in {cache_name(SOURCE_SIZE, 'mm')}: {source.loc[uid, 'reason']}"
            kept.append(
                {
                    "SeriesInstanceUID": uid,
                    "status": "excluded",
                    "reason": reason,
                    "box_fold": str(fold),
                    "seconds": "0",
                }
            )
    jobs: list[Job] = [
        (
            uid,
            root / "raw" / "series" / uid,
            tuple(float(v) for v in source.loc[uid, list(OFFSETS)]),
            tuple(round(float(v), 2) for v in filled.loc[uid]),  # cut where the index says
            int(fallback[uid]),
            size,
            spacing,
            cache,
            seg_dir,
            root / SHI_DIR / SHI_ORIG / f"{uid}.nii.gz",
            lut,
        )
        for uid in todo
    ]
    print(f"vessel-box cache {out}: {len(boxes)} boxes, {len(done)} done, {len(jobs)} to do")

    def write(tmp: Path) -> None:
        with tmp.open("w", newline="") as handle:
            w = csv.DictWriter(handle, COLUMNS)
            w.writeheader()
            w.writerows(kept)

    atomic_write(index, write)  # header plus the final rows: retries and torn rows go
    with (
        index.open("a", newline="") as handle,
        ProcessPoolExecutor(workers, None, _one_thread) as pool,
    ):
        writer = csv.DictWriter(handle, COLUMNS)
        futures = [pool.submit(process, job) for job in jobs]
        for future in as_completed(futures):
            writer.writerow({**future.result(), "box_fold": fold})
            handle.flush()
    table = pd.read_csv(index, dtype={"reason": str}).fillna({"reason": "", "seg": ""})
    print(table["status"].value_counts().to_string())
    print(table["seg"].value_counts().to_string())
    for reason, n in table.loc[table["status"] != "ok", "reason"].value_counts().head(10).items():
        print(f"  {n} x {reason}")
    print(f"seconds total {table['seconds'].sum():.0f}, mean {table['seconds'].mean():.2f}")


def write_points(root: Path, out: str, size: int, spacing: float) -> pd.DataFrame:
    """labels/points-<out>.csv: each expert point of an "ok" series in its box's voxels (the
    columns of labels/points-b2mm-160.csv; `points.load_points` drops those outside)."""
    index = load_index(root / "cache" / out)
    ok = index[index["status"] == "ok"]
    points = pd.read_csv(points_file(root, cache_name(SOURCE_SIZE, "mm")))
    points = points[points["SeriesInstanceUID"].isin(ok.index)].copy()
    for uid, rows in points.groupby("SeriesInstanceUID"):
        centre = ok.loc[uid, list(CENTRE)].to_numpy(float)
        box_from_cache = np.linalg.inv(cache_from_box(centre, size, spacing))
        q = rows[list(POINT_COLUMNS)].to_numpy(float)
        points.loc[rows.index, list(POINT_COLUMNS)] = (
            q @ box_from_cache[:3, :3].T + box_from_cache[:3, 3]
        )
    atomic_write(points_file(root, out), lambda tmp: points.to_csv(tmp, index=False))
    return points


def check(root: Path, size: int, out: str, spacing: float = BOX_SPACING_MM) -> None:
    """Expert point to the nearest voxel of its own aneurysm class in the box, in mm."""
    index = load_index(root / "cache" / out)
    ok = index[index["status"] == "ok"]
    seg_dir = root / "cache" / f"{out}-vessels"
    points = pd.read_csv(points_file(root, out))
    dist: list[float] = []
    outside = missing = 0
    for uid, rows in points[points["SeriesInstanceUID"].isin(ok.index)].groupby(
        "SeriesInstanceUID"
    ):
        path = seg_dir / f"{uid}.npz"
        if not path.exists():
            missing += 1
            continue
        seg = np.load(path)["seg"]
        for _, row in rows.iterrows():
            q = row[list(POINT_COLUMNS)].to_numpy(float)
            if not ((q >= -0.5) & (q <= size - 0.5)).all():
                outside += 1
                continue
            voxels = np.argwhere(seg == 1 + len(LOCATIONS) + LOCATIONS.index(row["location"]))
            if not len(voxels):
                dist.append(np.inf)
                continue
            # from the point's voxel to the nearest class voxel: the distance transform's value
            c = np.clip(np.rint(q), 0, size - 1)
            dist.append(float(np.sqrt(((voxels - c) ** 2).sum(1).min())) * spacing)
    v = np.array(dist)
    finite = v[np.isfinite(v)]
    print(
        f"point -> nearest own-class aneurysm voxel: n {len(v)}, none in mask "
        f"{int((~np.isfinite(v)).sum())}, outside the box {outside}, no mask file {missing}"
    )
    if finite.size:
        print(
            f"  median {np.median(finite):.1f} mm, p90 {np.percentile(finite, 90):.1f} mm, "
            f"within 3 mm {np.mean(v <= 3):.3f}, within 6 mm {np.mean(v <= 6):.3f}"
        )


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fold", type=int, required=True, help="fold of the box predictions")
    parser.add_argument("--size", type=int, default=BOX_SIZE, help="box edge in voxels")
    parser.add_argument("--spacing", type=float, default=BOX_SPACING_MM, help="voxel side, mm")
    parser.add_argument("--uid-file", type=Path, help="only these series, one per line")
    parser.add_argument("--out", help="cache folder (default vbox05-<size>, vc1mm-<size> at 1 mm)")
    parser.add_argument("--check-only", action="store_true", help="skip the build")
    parser.add_argument(
        "--seg-only", action="store_true", help="rewrite the listed series' masks only"
    )
    args = parser.parse_args(argv)
    root = store()
    kinds: dict[float, CacheKind] = {BOX_SPACING_MM: "vbox", SPACING_MM: "vcrop"}
    if args.out is None and args.spacing not in kinds:
        parser.error(f"--spacing {args.spacing} has no default cache name; pass --out")
    out = args.out or cache_name(args.size, kinds[args.spacing])
    uids = set(args.uid_file.read_text().split()) if args.uid_file else None
    workers = int(os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count() or 1))
    if args.seg_only:
        if uids is None:
            parser.error("--seg-only needs --uid-file")
        print(f"masks rewritten: {rebuild_segs(root, args.size, out, args.spacing, uids, workers)}")
    elif not args.check_only:
        build(root, args.fold, args.size, workers, uids, out, args.spacing)
        print(f"points: {len(write_points(root, out, args.size, args.spacing))} rows")
    check(root, args.size, out, args.spacing)


if __name__ == "__main__":
    main()
