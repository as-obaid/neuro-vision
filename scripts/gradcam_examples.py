"""Grad-CAM of a CNN config on held-out positives: one figure and one overview example, no IDs.

Usage (on a compute node): PYTHONPATH=src python scripts/gradcam_examples.py \
    configs/final/b2_final.yaml <out_dir>

For each scan type the example is the held-out positive with the highest out-of-fold Aneurysm
Present probability whose localizer point lies inside the model input, at least `MARGIN_MM` from
its faces, and whose cached volume is most mirror-symmetric about axis 0 (left-right in RAS+), so
the axial view shown is axial. Each series is read by the fold model that never trained on it.
Writes `final-gradcam.png`; for the MRA example `overview-input.png` (axial MIP of the model
input; a CTA MIP shows mostly bone) and `overview-input.json` (14 probabilities, true artery).
`overview-located.*` is the same for the best MRA positive whose highest location probability is
a true artery. Needs resample "mm".

With `--pointing` it instead scores, over every held-out positive with a localizer point inside
the input, how often the CAM peak lies within `POINTING_MM` of the nearest point, against a fixed
point (the mean point) and a uniform random head voxel; aggregates only, in `pointing.json`.
"""

import argparse
import ast
import json
import os
from collections.abc import Callable, Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import nibabel.orientations as orientations
import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageDraw, ImageFont
from pydicom.dataset import Dataset
from torch import nn

from neurovision.config import Config, load_config
from neurovision.data.dataset import load_index
from neurovision.data.dicom import read_header
from neurovision.data.folds import load_folds
from neurovision.data.labels import ANEURYSM_PRESENT, LOCATIONS, TARGETS, load_labels
from neurovision.data.preprocess import (
    ASSUMED_FOV_MM,
    SPACING_MM,
    Geometry,
    _head_mask,
    cache_name,
    preprocess_with_crop,
    series_geometry,
)
from neurovision.evaluate.gradcam import grad_cam
from neurovision.infer.predict import predict_volumes
from neurovision.models.resnet3d import build_model
from neurovision.paths import store

SCAN_TYPES = ("CTA", "MRA", "MRI T2", "MRI T1post")
OVERVIEW = "MRA"  # its MIP shows the vessel tree; a CTA MIP shows the skull
ANGIOGRAPHY = {"CTA", "MRA"}  # vessels are bright: show a maximum-intensity slab
MARGIN_MM = 5.0
POINTING_MM = 20.0
OFFSETS = ("offset_mm_x", "offset_mm_y", "offset_mm_z")  # cache index: box start, mm
SLAB_MM = 5  # display slab through the expert's point; 1 mm voxels
SCALE = 2  # display pixels per voxel
FONT_DIR = Path("/usr/share/fonts/dejavu-sans-fonts")
BACKGROUND = (252, 252, 251)
INK = (31, 41, 55)
MUTED = (90, 99, 112)
RING = (34, 211, 238)
CAM_COLOURS = ((255, 236, 150), (247, 144, 30), (205, 30, 45))  # low, mid, high
SHORT = {
    "Internal Carotid Artery": "ICA",
    "Middle Cerebral Artery": "MCA",
    "Anterior Cerebral Artery": "ACA",
    "Communicating Artery": "communicating",
}


@dataclass(frozen=True)
class Example:
    scan_type: str
    artery: str
    point: tuple[float, float, float]  # model-input voxel (RAS+ axes)
    volume: np.ndarray  # the cached model input
    probabilities: np.ndarray  # (14,), from the fold model


def short_name(artery: str) -> str:
    for long, short in SHORT.items():
        artery = artery.replace(long, short)
    return artery


def pooled_oof(root: Path, name: str, folds: pd.Series) -> pd.DataFrame:
    """Every fold's oof.csv with its fold number; each series appears once."""
    parts = []
    for k in sorted(folds.unique()):
        oof = pd.read_csv(root / "runs" / name / f"fold{k}" / "oof.csv", index_col=0)
        parts.append(oof.assign(fold=int(k)))
    pooled = pd.concat(parts)
    if pooled.index.has_duplicates:
        raise SystemExit("a series appears in two folds' oof.csv")
    return pooled


def localizer_points(
    series_dir: Path,
    geometry: Geometry,
    affine: np.ndarray,
    rows: pd.DataFrame,
    headers: Sequence[Dataset] | None = None,
) -> list[tuple[str, np.ndarray]]:
    """(artery, model-input voxel) per localizer row; rows whose instance is missing are dropped.

    A point is (column, row) on one instance, plus a frame for multi-frame files; the box
    position is `affine` (`Crop.index_affine`) applied to (column, row, sorted slice rank).
    """
    if headers is None:
        headers = [read_header(p) for p in sorted(series_dir.glob("*.dcm"))]
    frames = [max(len(h.get("PerFrameFunctionalGroupsSequence") or []), 1) for h in headers]
    starts = np.cumsum([0, *frames])
    sops = [str(h.SOPInstanceUID) for h in headers]
    order = list(geometry.order)
    points = []
    columns = (rows["SOPInstanceUID"], rows["coordinates"], rows["location"])
    for sop, coords, artery in zip(*columns, strict=True):
        if sop not in sops:
            continue
        xy = ast.literal_eval(coords)
        k = sops.index(sop)
        frame = int(xy.get("f", 0))
        if starts[k] + frame >= starts[k + 1]:
            frame -= 1  # a 1-based frame number
        rank = order.index(int(starts[k] + frame))
        q = affine @ np.array([xy["x"], xy["y"], rank, 1.0])
        points.append((str(artery), q[:3]))
    return points


def box_affine(
    geometry: Geometry, offset_mm: Sequence[float], spacing_mm: float = SPACING_MM
) -> np.ndarray:
    """`Crop.index_affine` from the headers and the cache index's box start, without pixels.

    Mirrors `preprocess._mm_crop`; a test checks the two agree.
    """
    native = geometry.affine.copy()
    if geometry.spacing_assumed:
        native[:3, :2] *= ASSUMED_FOV_MM / max(geometry.size[:2])
    ornt = orientations.io_orientation(np.diag([-1.0, -1.0, 1.0, 1.0]) @ geometry.affine)
    to_native = orientations.inv_ornt_aff(ornt, geometry.size)
    spacing = np.linalg.norm((native @ to_native)[:3, :3], axis=0)
    shape = np.empty(3)
    for axis, (out, _) in enumerate(ornt):
        shape[int(out)] = geometry.size[axis]
    resampled = np.array(
        [max(1, round(n * s / spacing_mm)) for n, s in zip(shape, spacing, strict=True)]
    )
    zoom = resampled / shape
    start = np.round(np.asarray(offset_mm, dtype=float) / spacing_mm)
    to_box = np.eye(4)
    to_box[:3, :3] = np.diag(zoom)
    to_box[:3, 3] = 0.5 * zoom - 0.5 - start
    return to_box @ np.linalg.inv(to_native)


def mirror_axis(volume: np.ndarray) -> int:
    """The axis whose flip best matches the volume; a head in RAS+ order mirrors about axis 0."""
    keep = volume > volume.min() + 1e-3  # not padding
    scores = []
    for axis in range(volume.ndim):
        both = keep | np.flip(keep, axis)
        a = volume[both] - volume[both].mean()
        b = np.flip(volume, axis)[both] - np.flip(volume, axis)[both].mean()
        scores.append(float((a * b).sum() / max(np.sqrt((a * a).sum() * (b * b).sum()), 1e-12)))
    return int(np.argmax(scores))


def inside(q: np.ndarray, size: int, margin: float) -> bool:
    return bool(((q >= margin - 0.5) & (q <= size - 0.5 - margin)).all())


def load_fold_model(config: Config, root: Path, fold: int, device: torch.device) -> nn.Module:
    assert config.train is not None
    model = build_model(config.train.backbone, head=config.train.head)
    path = root / "runs" / config.name / f"fold{fold}" / "best.pt"
    model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
    return model.to(device).eval()


def find_example(
    candidates: pd.DataFrame,
    root: Path,
    cache: Path,
    size: int,
    localizers: pd.DataFrame,
    accept: Callable[[str, np.ndarray], bool] | None = None,
) -> tuple[str, str, np.ndarray] | None:
    """The first candidate (in the given order) with a usable point, as (series, artery, point).

    The cached volume must equal a fresh preprocessing run, so the crop that maps the point is
    the crop the model saw, and must mirror about axis 0 (`mirror_axis`).
    """
    for uid in candidates.index:
        rows = localizers[localizers["SeriesInstanceUID"] == uid]
        if rows.empty:
            continue
        series_dir = root / "raw" / "series" / str(uid)
        volume, geometry, crop = preprocess_with_crop(series_dir, size, "mm")
        assert crop is not None
        cached = np.load(cache / f"{uid}.npy").astype(np.float32)
        if not np.allclose(volume.astype(np.float32), cached, atol=1e-2):
            continue
        if mirror_axis(cached) != 0:
            continue
        for artery, q in localizer_points(series_dir, geometry, crop.index_affine, rows):
            if inside(q, size, MARGIN_MM) and (accept is None or accept(str(uid), q)):
                return str(uid), artery, q
    return None


def _box_points(job: tuple[Path, tuple[float, ...], pd.DataFrame]) -> list[np.ndarray]:
    """Model-input voxels of one series' localizer points, from headers only (process pool)."""
    series_dir, offset, rows = job
    torch.set_num_threads(1)
    headers = [read_header(p) for p in sorted(series_dir.glob("*.dcm"))]
    geometry = series_geometry(headers)
    affine = box_affine(geometry, offset)
    return [q for _, q in localizer_points(series_dir, geometry, affine, rows, headers)]


def head_chance(volume: np.ndarray, points: np.ndarray, radius: float) -> float:
    """Share of head-mask voxels within `radius` of the nearest point: a random guess's hit rate."""
    mask = _head_mask(volume)
    centres = np.argwhere(mask) * 2 + 0.5  # the mask is on a 2-voxel grid
    nearest = np.min(np.linalg.norm(centres[:, None, :] - points[None], axis=2), axis=1)
    return float((nearest <= radius).mean())


def pointing_summary(records: pd.DataFrame, radius: float) -> dict[str, object]:
    """Hit rates and median distances, overall and per scan type; counts only, no series."""

    def one(frame: pd.DataFrame) -> dict[str, float | int]:
        return {
            "series": len(frame),
            "cam_hit_rate": float((frame["cam_mm"] <= radius).mean()),
            "fixed_point_hit_rate": float((frame["fixed_mm"] <= radius).mean()),
            "random_head_voxel_hit_rate": float(frame["chance"].mean()),
            "cam_median_mm": float(frame["cam_mm"].median()),
            "fixed_point_median_mm": float(frame["fixed_mm"].median()),
        }

    return {
        "radius_mm": radius,
        "all": one(records),
        "by_scan_type": {k: one(g) for k, g in records.groupby("scan_type")},
    }


def pointing_game(
    config: Config,
    root: Path,
    oof: pd.DataFrame,
    positives: pd.DataFrame,
    index: pd.DataFrame,
    cache: Path,
    localizers: pd.DataFrame,
    device: torch.device,
) -> dict[str, object]:
    """CAM peak against the expert's points on every held-out positive with a point in the box."""
    assert config.train is not None
    size = config.train.input_size
    uids = sorted(set(positives.index) & set(localizers["SeriesInstanceUID"]))
    jobs = [
        (
            root / "raw" / "series" / uid,
            tuple(float(v) for v in index.loc[uid, list(OFFSETS)]),
            localizers[localizers["SeriesInstanceUID"] == uid],
        )
        for uid in uids
    ]
    workers = int(os.environ.get("SLURM_CPUS_PER_TASK", "4"))
    with ProcessPoolExecutor(workers) as pool:  # before CUDA starts: the pool forks
        mapped = list(pool.map(_box_points, jobs, chunksize=4))
    points = {
        uid: np.array([q for q in qs if inside(q, size, 0.0)])
        for uid, qs in zip(uids, mapped, strict=True)
    }
    points = {uid: q for uid, q in points.items() if len(q)}
    fixed = np.mean([q[0] for q in points.values()], axis=0)
    records = []
    for fold in sorted(oof.loc[list(points), "fold"].unique()):
        model = load_fold_model(config, root, int(fold), device)
        members = [u for u in points if oof.loc[u, "fold"] == fold]
        print(f"fold {fold}: {len(members)} positives with a point inside", flush=True)
        for uid in members:
            volume = np.load(cache / f"{uid}.npy").astype(np.float32)
            cam = grad_cam(model, torch.from_numpy(volume)[None, None])
            peak = np.array(np.unravel_index(int(np.argmax(cam)), cam.shape), dtype=float)
            q = points[uid]
            records.append(
                {
                    "scan_type": str(positives.loc[uid, "Modality"]),
                    "cam_mm": float(np.min(np.linalg.norm(q - peak, axis=1))),
                    "fixed_mm": float(np.min(np.linalg.norm(q - fixed, axis=1))),
                    "chance": head_chance(volume, q, POINTING_MM),
                }
            )
    summary = pointing_summary(pd.DataFrame(records), POINTING_MM)
    summary["positives_with_a_localizer"] = len(uids)
    summary["series_with_a_point_inside"] = len(points)
    return summary


def _font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    path = FONT_DIR / ("DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf")
    return ImageFont.truetype(str(path), size) if path.exists() else ImageFont.load_default(size)


def display(plane: np.ndarray) -> np.ndarray:
    """An axial plane (x left to right, y posterior to anterior) in radiological view.

    Rows run anterior (top) to posterior; columns run patient right (left) to patient left.
    """
    return plane.T[::-1, ::-1]


def display_point(q: np.ndarray, size: int) -> tuple[float, float]:
    """(column, row) of a model-input voxel in `display` coordinates."""
    return size - 1 - float(q[0]), size - 1 - float(q[1])


def grey(plane: np.ndarray) -> np.ndarray:
    """uint8 grey image, clipped at the 1st and 99.5th percentile of the non-padding voxels."""
    values = plane[plane > plane.min()]
    lo, hi = np.percentile(values if values.size else plane, [1, 99.5])
    return (np.clip((plane - lo) / max(hi - lo, 1e-6), 0, 1) * 255).astype(np.uint8)


def colourise(cam: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """RGB colour and alpha in [0, 1] per pixel; weak evidence (below 0.15) stays clear."""
    low, mid, high = (np.array(c, dtype=np.float32) for c in CAM_COLOURS)
    t = cam[..., None]
    rgb = np.where(t < 0.5, low + (mid - low) * (t / 0.5), mid + (high - mid) * ((t - 0.5) / 0.5))
    alpha = 0.7 * np.clip((cam - 0.15) / 0.85, 0, 1)
    return rgb, alpha


def overlay(image: np.ndarray, cam: np.ndarray) -> np.ndarray:
    rgb, alpha = colourise(cam)
    base = np.repeat(image[..., None].astype(np.float32), 3, axis=2)
    return (base * (1 - alpha[..., None]) + rgb * alpha[..., None]).astype(np.uint8)


def panels(example: Example, cam: np.ndarray) -> tuple[Image.Image, Image.Image, tuple[float, ...]]:
    """(input, overlay, point) for one example: an axial slab through the expert's point."""
    size = example.volume.shape[0]
    z = int(round(example.point[2]))
    lo, hi = max(z - SLAB_MM // 2, 0), min(z + SLAB_MM // 2 + 1, size)
    slab = example.volume[:, :, lo:hi].astype(np.float32)
    plane = slab.max(axis=2) if example.scan_type in ANGIOGRAPHY else slab.mean(axis=2)
    image = grey(display(plane))
    heat = display(cam[:, :, lo:hi].mean(axis=2))  # keeps the volume's own [0, 1] scale
    px = size * SCALE
    plain = Image.fromarray(image).convert("RGB").resize((px, px), Image.Resampling.BICUBIC)
    mixed = Image.fromarray(overlay(image, heat)).resize((px, px), Image.Resampling.BICUBIC)
    col, row = display_point(np.asarray(example.point), size)
    return plain, mixed, ((col + 0.5) * SCALE, (row + 0.5) * SCALE)


def _ring(image: Image.Image, at: tuple[float, ...]) -> None:
    draw, r = ImageDraw.Draw(image), 6 * SCALE
    x, y = at
    draw.ellipse((x - r, y - r, x + r, y + r), outline=RING, width=SCALE + 1)


def _text(draw: ImageDraw.ImageDraw, xy: tuple[float, float], text: str, **kw: object) -> None:
    draw.text(xy, text, anchor="ma", **kw)  # type: ignore[arg-type]


def render(examples: Sequence[Example], cams: Sequence[np.ndarray], path: Path) -> None:
    """Two rows (input, Grad-CAM overlay) by one column per example, with a legend."""
    px = examples[0].volume.shape[0] * SCALE
    gap, left, top, bottom = 18, 40, 92, 58
    width = left + len(examples) * (px + gap)
    height = top + 2 * px + gap + bottom
    canvas = Image.new("RGB", (width, height), BACKGROUND)
    draw = ImageDraw.Draw(canvas)
    title, body, small = _font(19, bold=True), _font(16), _font(14)
    for i, (example, cam) in enumerate(zip(examples, cams, strict=True)):
        x0 = left + i * (px + gap)
        centre = x0 + px / 2
        p = example.probabilities[TARGETS.index(ANEURYSM_PRESENT)]
        _text(draw, (centre, 10), example.scan_type, font=title, fill=INK)
        _text(draw, (centre, 38), short_name(example.artery), font=body, fill=INK)
        _text(draw, (centre, 62), f"P(aneurysm present) = {p:.2f}", font=small, fill=MUTED)
        plain, mixed, at = panels(example, cam)
        for image in (plain, mixed):
            _ring(image, at)
        marks = ImageDraw.Draw(plain)
        marks.text((8, 6), "R", font=small, fill=(235, 235, 235))
        marks.text((px - 8, 6), "L", font=small, fill=(235, 235, 235), anchor="ra")
        canvas.paste(plain, (x0, top))
        canvas.paste(mixed, (x0, top + px + gap))
    for label, y in (("Input", top + px / 2), ("Grad-CAM", top + px + gap + px / 2)):
        tile = Image.new("RGB", (px, 28), BACKGROUND)
        ImageDraw.Draw(tile).text((px / 2, 4), label, font=body, fill=INK, anchor="ma")
        canvas.paste(tile.rotate(90, expand=True), (6, int(y - px / 2)))
    # legend: the colour ramp and the ring
    y = top + 2 * px + gap + 22
    ramp = np.linspace(0, 1, 220, dtype=np.float32)[None, :].repeat(14, axis=0)
    bar = Image.fromarray(overlay(np.full(ramp.shape, 235, np.uint8), ramp))
    x = left + 34
    canvas.paste(bar, (x, y))
    draw.text((x - 6, y + 7), "low", font=small, fill=MUTED, anchor="rm")
    draw.text(
        (x + 226, y + 7),
        "high Grad-CAM evidence for Aneurysm Present",
        font=small,
        fill=MUTED,
        anchor="lm",
    )
    rx = x + 700
    draw.ellipse((rx - 9, y - 2, rx + 9, y + 16), outline=RING, width=3)
    draw.text((rx + 18, y + 7), "expert's aneurysm point", font=small, fill=MUTED, anchor="lm")
    draw.text(
        (width - gap, y + 7),
        "5 mm axial slab, radiological view",
        font=small,
        fill=MUTED,
        anchor="rm",
    )
    canvas.save(path)


def write_overview(example: Example, out: Path, stem: str, arteries: Sequence[str]) -> None:
    """The MIP PNG and a JSON; `arteries` are every location labelled positive for the series."""
    mip = grey(display(example.volume.astype(np.float32).max(axis=2)))
    px = example.volume.shape[0] * 3
    Image.fromarray(mip).resize((px, px), Image.Resampling.BICUBIC).save(out / f"{stem}.png")
    p = example.probabilities
    record = {
        "scan_type": example.scan_type,
        "true_arteries": list(arteries),
        "probabilities": {name: round(float(v), 4) for name, v in zip(TARGETS, p, strict=True)},
        "top_location": LOCATIONS[int(np.argmax(p[: len(LOCATIONS)]))],
        "view": "axial maximum-intensity projection of the model input, radiological view",
    }
    (out / f"{stem}.json").write_text(json.dumps(record, indent=2))


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("out", type=Path)
    parser.add_argument("--pointing", action="store_true", help="score CAM peaks; no figure")
    args = parser.parse_args(argv)
    config = load_config(args.config)
    assert config.train is not None and config.train.resample == "mm", "needs resample mm"
    size = config.train.input_size
    torch.set_num_threads(int(os.environ.get("SLURM_CPUS_PER_TASK", "4")))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    root = store()
    folds = load_folds(root)
    labels = load_labels(root, config.labels)
    cache = root / "cache" / cache_name(size, "mm")
    index = load_index(cache)
    ok = index.index[index["status"] == "ok"]
    localizers = pd.read_csv(root / "raw" / "train_localizers.csv")
    oof = pooled_oof(root, config.name, folds)
    positives = labels[(labels[ANEURYSM_PRESENT] == 1) & labels.index.isin(ok)]
    ranked = oof.loc[positives.index].join(positives["Modality"])
    ranked = ranked.sort_values(ANEURYSM_PRESENT, ascending=False, kind="stable")
    args.out.mkdir(parents=True, exist_ok=True)
    if args.pointing:
        summary = pointing_game(config, root, oof, positives, index, cache, localizers, device)
        (args.out / "pointing.json").write_text(json.dumps(summary, indent=2))
        print(json.dumps(summary, indent=2))
        return

    models: dict[int, nn.Module] = {}

    def example(uid: str, artery: str, q: np.ndarray) -> Example:
        fold = int(oof.loc[uid, "fold"])
        if fold not in models:
            models[fold] = load_fold_model(config, root, fold, device)
        volume = np.load(cache / f"{uid}.npy").astype(np.float32)
        x = torch.from_numpy(volume)[None, None]
        probs = predict_volumes(models[fold], x)[0]
        return Example(str(labels.loc[uid, "Modality"]), artery, tuple(q), volume, probs)

    def arteries(uid: str) -> list[str]:
        return [name for name in LOCATIONS if labels.loc[uid, name] == 1]

    examples, cams = [], []
    for scan_type in SCAN_TYPES:
        found = find_example(ranked[ranked["Modality"] == scan_type], root, cache, size, localizers)
        if found is None:
            raise SystemExit(f"{scan_type}: no positive with a point inside the input")
        ex = example(*found)
        model = models[int(oof.loc[found[0], "fold"])]
        cam = grad_cam(model, torch.from_numpy(ex.volume)[None, None])
        examples.append(ex)
        cams.append(cam)
        peak = np.unravel_index(int(np.argmax(cam)), cam.shape)
        distance = float(np.linalg.norm(np.array(peak) - np.array(ex.point)))
        p = ex.probabilities[TARGETS.index(ANEURYSM_PRESENT)]
        print(
            f"{scan_type}: {short_name(ex.artery)}; P(AP) {p:.3f}; CAM peak {distance:.0f} mm "
            f"from the point; CAM at the point {cam[tuple(np.round(ex.point).astype(int))]:.2f}"
        )
        if scan_type == OVERVIEW:
            write_overview(ex, args.out, "overview-input", arteries(found[0]))

    render(examples, cams, args.out / "final-gradcam.png")

    def located(uid: str, q: np.ndarray) -> bool:
        p = oof.loc[uid, list(LOCATIONS)].to_numpy(dtype=float)
        rows = localizers[localizers["SeriesInstanceUID"] == uid]
        return LOCATIONS[int(np.argmax(p))] in set(rows["location"])

    pool = ranked[ranked["Modality"] == OVERVIEW]
    found = find_example(pool, root, cache, size, localizers, located)
    if found is not None:
        write_overview(example(*found), args.out, "overview-located", arteries(found[0]))
    print(f"wrote {args.out / 'final-gradcam.png'}")


if __name__ == "__main__":
    main()
