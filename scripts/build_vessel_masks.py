"""Shi et al. vessel and aneurysm masks in the voxels of the 1 mm head-crop cache.

Usage (on a compute node): PYTHONPATH=src python scripts/build_vessel_masks.py [--size 160]
Reads `labels/shi2026-masks/` (labels.json, labelsTr_26classes_in_orig_space/<uid>.nii.gz), the
DICOM headers of each cached ("ok") series, and the box start in `cache/b2mm-{size}/index.csv`.
Writes `cache/b2mm-{size}-vessels/<uid>.npz` (key "seg": uint8 size³, classes of
`neurovision.data.vessels`); skips series already written, so a rerun resumes. Then checks the
alignment against the expert points (`labels/points-b2mm-{size}.csv`) and prints counts only.
"""

import argparse
import json
import os
from collections.abc import Sequence
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
from scipy import ndimage

from neurovision.data.dataset import load_index
from neurovision.data.dicom import read_header
from neurovision.data.labels import LOCATIONS
from neurovision.data.points import POINT_COLUMNS, inside, points_file
from neurovision.data.preprocess import cache_name, series_geometry
from neurovision.data.vessels import shi_to_seg, to_cache
from neurovision.paths import store

OFFSETS = ("offset_mm_x", "offset_mm_y", "offset_mm_z")
SHI = "labels/shi2026-masks"
ORIG = "labelsTr_26classes_in_orig_space"
Job = tuple[str, Path, Path, tuple[float, ...], np.ndarray, int, Path]


def _one(job: Job) -> str:
    uid, series_dir, mask_path, offset, lut, size, out = job
    if out.exists():
        return "kept"
    try:
        geometry = series_geometry([read_header(p) for p in sorted(series_dir.glob("*.dcm"))])
        img = nib.load(mask_path)
        assert isinstance(img, nib.Nifti1Image)
        mask = np.asarray(img.dataobj).astype(np.uint8)
        if sorted(mask.shape) != sorted(geometry.size):
            return f"shape {mask.shape} vs {geometry.size}"
        seg = to_cache(lut[mask], img.affine, geometry, offset, size)
    except Exception as err:  # one bad series must not stop the build
        return f"error {type(err).__name__}: {err}"
    tmp = out.with_suffix(".tmp.npz")
    np.savez_compressed(tmp, seg=seg)
    tmp.replace(out)
    return "written"


def build(root: Path, size: int, workers: int) -> pd.Series:
    labels = json.loads((root / SHI / "labels.json").read_text())
    lut = shi_to_seg(labels)
    index = load_index(root / "cache" / cache_name(size, "mm"))
    ok = index[index["status"] == "ok"]
    out_dir = root / "cache" / f"{cache_name(size, 'mm')}-vessels"
    out_dir.mkdir(parents=True, exist_ok=True)
    jobs: list[Job] = []
    for uid in ok.index:
        mask = root / SHI / ORIG / f"{uid}.nii.gz"
        if mask.exists():
            offset = tuple(float(v) for v in ok.loc[uid, list(OFFSETS)])
            series = root / "raw" / "series" / uid
            jobs.append((uid, series, mask, offset, lut, size, out_dir / f"{uid}.npz"))
    with ProcessPoolExecutor(workers) as pool:
        results = list(pool.map(_one, jobs, chunksize=4))
    print(f"cached ok series: {len(ok)}; with a Shi mask (orig space): {len(jobs)}")
    return pd.Series(results, index=[j[0] for j in jobs])


def check(root: Path, size: int) -> None:
    """Distance from each expert point inside the cube to the nearest voxel of its own aneurysm
    class, and to any aneurysm voxel; mm (1 mm voxels)."""
    out_dir = root / "cache" / f"{cache_name(size, 'mm')}-vessels"
    points = pd.read_csv(points_file(root, cache_name(size, "mm")))
    own: list[float] = []
    any_: list[float] = []
    missing = 0
    for uid, rows in points.groupby("SeriesInstanceUID"):
        path = out_dir / f"{uid}.npz"
        if not path.exists():
            missing += 1
            continue
        seg = np.load(path)["seg"]
        for _, row in rows.iterrows():
            q = row[list(POINT_COLUMNS)].to_numpy(float)
            if not inside(q, size, 0.0):
                continue
            k = LOCATIONS.index(row["location"])
            for target, sink in ((seg == 14 + k, own), (seg >= 14, any_)):
                if not target.any():
                    sink.append(np.inf)
                    continue
                dist = ndimage.distance_transform_edt(~target)
                c = np.clip(np.rint(q).astype(int), 0, size - 1)
                sink.append(float(dist[tuple(c)]))
    for name, values in (("own class", own), ("any aneurysm", any_)):
        v = np.array(values)
        finite = v[np.isfinite(v)]
        print(
            f"point -> nearest {name} voxel: n {len(v)}, none in mask "
            f"{int((~np.isfinite(v)).sum())}, median {np.median(finite):.1f} mm, "
            f"p90 {np.percentile(finite, 90):.1f} mm, within 3 mm {np.mean(v <= 3):.3f}, "
            f"within 6 mm {np.mean(v <= 6):.3f}"
        )
    print(f"series with points but no mask file: {missing}")


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--size", type=int, default=160, help="edge of the b2mm cache cube")
    parser.add_argument("--check-only", action="store_true", help="skip the build")
    args = parser.parse_args(argv)
    root = store()
    if not args.check_only:
        results = build(root, args.size, int(os.environ.get("SLURM_CPUS_PER_TASK", "4")))
        kinds = results.str.split(" ").str[0]
        print(kinds.value_counts().to_string())
        for reason, n in results[~kinds.isin(["written", "kept"])].value_counts().head(10).items():
            print(f"  {n} x {reason}")
    check(root, args.size)


if __name__ == "__main__":
    main()
