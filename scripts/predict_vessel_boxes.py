"""Vessel boxes for the full version: where the step-1 U-Net finds the vessel tree, per series.

Usage (GPU node): PYTHONPATH=src python scripts/predict_vessel_boxes.py --fold 0
Loads runs/vessel_r3d18_160/fold<k>/last.pt (the whole U-Net; refuses a run without meta.json
unless --allow-unfinished), predicts the 27 classes on cache/b2mm-160 for every "ok" series, and
appends one row per series to labels/vessel-boxes-fold<k>.csv: the predicted vessel voxels
(class > 0), their 99.5% box (lo, hi, centre) in 1 mm cache voxels, the centre in patient LPS mm
(DICOM headers; series without PixelSpacing use the cache's assumed field of view) and the extent
in mm. lo, hi and the centre are empty when fewer than `vbox.MIN_VESSEL_VOXELS` were found.
Series already in the file are skipped, so a rerun resumes. The fold's own series were never
seen by this U-Net; the other folds' series were its training series.
"""

import argparse
import csv
import time
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from neurovision.data.dataset import load_index
from neurovision.data.dicom import read_header
from neurovision.data.points import box_affine
from neurovision.data.preprocess import SPACING_MM, cache_name, series_geometry
from neurovision.data.vbox import trimmed_box
from neurovision.models.unet import R3DUNet
from neurovision.paths import store

RUN = "vessel_r3d18_160"
SIZE = 160
BATCH, WORKERS = 4, 8
OFFSETS = ("offset_mm_x", "offset_mm_y", "offset_mm_z")  # cache index: box start, mm
AXES = "xyz"
COLUMNS = (
    "SeriesInstanceUID",
    "box_fold",
    "model_epoch",
    "n_vessel",
    *(f"{name}_{a}" for name in ("lo", "hi", "centre", "centre_lps", "extent_mm") for a in AXES),
)


def boxes_file(root: Path, fold: int) -> Path:
    return root / "labels" / f"vessel-boxes-fold{fold}.csv"


class BoxInputs(Dataset[tuple[torch.Tensor, torch.Tensor, int]]):
    """(1 mm cache volume (1, S, S, S), LPS-from-cache-voxel 4x4 (NaN if unreadable), item)."""

    def __init__(self, cache: Path, raw: Path, offsets: pd.DataFrame) -> None:
        self.cache, self.raw, self.offsets = cache, raw, offsets
        self.uids = list(offsets.index)

    def __len__(self) -> int:
        return len(self.uids)

    def __getitem__(self, i: int) -> tuple[torch.Tensor, torch.Tensor, int]:
        uid = self.uids[i]
        volume = np.load(self.cache / f"{uid}.npy").astype(np.float32)
        return torch.from_numpy(volume)[None], torch.from_numpy(self._lps(uid)), i

    def _lps(self, uid: str) -> np.ndarray:
        try:
            files = sorted((self.raw / uid).glob("*.dcm"))
            geometry = series_geometry([read_header(p) for p in files])
            offset = tuple(float(v) for v in self.offsets.loc[uid, list(OFFSETS)])
            return np.asarray(geometry.affine @ np.linalg.inv(box_affine(geometry, offset)))
        except Exception:  # the box in cache voxels still serves; only the LPS columns are lost
            return np.full((4, 4), np.nan)


def box_row(
    uid: str, fold: int, epoch: int, seg: np.ndarray, lps_from_cache: np.ndarray
) -> dict[str, object]:
    """One output row from the predicted classes (S³, cache voxels) of one series."""
    lo, hi, n = trimmed_box(seg)
    centre = (lo + hi) / 2
    lps = (lps_from_cache @ np.append(centre, 1.0))[:3]
    row: dict[str, object] = {
        "SeriesInstanceUID": uid,
        "box_fold": fold,
        "model_epoch": epoch,
        "n_vessel": n,
    }
    values = {"lo": lo, "hi": hi, "centre": centre, "centre_lps": lps}
    values["extent_mm"] = (hi - lo + 1) * SPACING_MM
    for name, v in values.items():
        for a, value in zip(AXES, v, strict=True):
            row[f"{name}_{a}"] = "" if np.isnan(value) else round(float(value), 2)
    return row


def done_uids(path: Path) -> set[str]:
    """Series with a complete row in `path`; a row cut short by a kill is dropped."""
    if not path.exists():
        return set()
    lines = path.read_text().splitlines(keepends=True)
    if lines and not lines[-1].endswith("\n"):
        lines.pop()
        path.write_text("".join(lines))
    return {row["SeriesInstanceUID"] for row in csv.DictReader(lines)}


def load_unet(run_dir: Path, allow_unfinished: bool) -> tuple[R3DUNet, int]:
    if not allow_unfinished and not (run_dir / "meta.json").exists():
        raise SystemExit(f"{run_dir} has no meta.json: pretraining is not finished")
    state = torch.load(run_dir / "last.pt", map_location="cpu", weights_only=True)
    model = R3DUNet()
    model.load_state_dict(state["model"])
    return model, int(state["epoch"])


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fold", type=int, required=True)
    parser.add_argument("--allow-unfinished", action="store_true", help="smoke tests only")
    parser.add_argument("--limit", type=int, help="first N pending series (smoke tests)")
    parser.add_argument("--uid-file", type=Path, help="only these series, one per line")
    parser.add_argument("--out", type=Path, help="output CSV (default labels/vessel-boxes-...)")
    args = parser.parse_args(argv)
    root = store()
    out = args.out or boxes_file(root, args.fold)
    cache = root / "cache" / cache_name(SIZE, "mm")
    index = load_index(cache)
    ok = index[index["status"] == "ok"]
    if args.uid_file is not None:
        ok = ok[ok.index.isin(args.uid_file.read_text().split())]
    done = done_uids(out)
    todo = ok[~ok.index.isin(done)].iloc[: args.limit]
    print(
        f"vessel boxes fold {args.fold}: {len(ok)} ok series, {len(done)} done, {len(todo)} to do"
    )
    if todo.empty:
        return
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, epoch = load_unet(root / "runs" / RUN / f"fold{args.fold}", args.allow_unfinished)
    model = model.to(device).eval()
    inputs = BoxInputs(cache, root / "raw" / "series", todo)
    loader = DataLoader(inputs, batch_size=BATCH, num_workers=WORKERS, pin_memory=True)
    out.parent.mkdir(parents=True, exist_ok=True)
    new = not out.exists()
    t0 = time.perf_counter()
    with out.open("a", newline="") as handle, torch.no_grad():
        writer = csv.DictWriter(handle, COLUMNS)
        if new:
            writer.writeheader()
        for x, lps, items in loader:
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                seg = model(x.to(device, non_blocking=True)).argmax(1).to(torch.uint8).cpu()
            for s, m, i in zip(seg.numpy(), lps.numpy(), items.tolist(), strict=True):
                writer.writerow(box_row(inputs.uids[i], args.fold, epoch, s, m))
            handle.flush()
    table = pd.read_csv(out)
    tree = table["centre_x"].notna()
    print(
        f"rows {len(table)}, without a tree {int((~tree).sum())}, "
        f"tree but no LPS {int((tree & table['centre_lps_x'].isna()).sum())}; "
        f"{time.perf_counter() - t0:.0f} s"
    )
    for a in AXES:
        e = table[f"extent_mm_{a}"].dropna()
        print(f"extent {a} (mm): median {e.median():.1f}, p95 {e.quantile(0.95):.1f}")


if __name__ == "__main__":
    main()
