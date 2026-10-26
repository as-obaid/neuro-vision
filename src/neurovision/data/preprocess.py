"""DICOM series to a float16 128-cubed RAS+ volume: the only DICOM-to-volume path."""

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import nibabel.orientations as orientations
import numpy as np
import torch
import torch.nn.functional as functional
from pydicom.dataset import Dataset

from neurovision.data.dicom import read_series

SIZE = 128
MIN_SLICES = 16
MAX_GAP_RATIO = 1.5
ATOL = 1e-3  # mm or direction cosine; spacing and position are stored to about 1e-4


class GeometryError(ValueError):
    pass


@dataclass(frozen=True)
class Geometry:
    order: tuple[int, ...]  # slice indices (files in name order, frames within a file), sorted
    affine: np.ndarray  # voxel (column, row, slice) to patient LPS, mm
    spacing: tuple[float, float, float]  # column, row, slice
    size: tuple[int, int, int]  # columns, rows, slices
    source: str  # "top-level", or "per-frame" if any geometry tag came from the frame groups


def _frame_value(ds: Dataset, frame: int, sequence: str, tag: str) -> Any:
    groups = ds.get("PerFrameFunctionalGroupsSequence") or []
    if frame >= len(groups):
        return None
    inner = groups[frame].get(sequence)
    return inner[0].get(tag) if inner else None


def _pick(top: Any, per_frame: Any) -> tuple[Any, bool]:
    return (top, False) if top is not None else (per_frame, per_frame is not None)


def _unique(values: Sequence[np.ndarray], reason: str) -> np.ndarray:
    first = values[0]
    if not all(np.allclose(v, first, atol=ATOL) for v in values):
        raise GeometryError(reason)
    return first


def series_geometry(datasets: Sequence[Dataset]) -> Geometry:
    if not datasets:
        raise GeometryError("no files")
    for ds in datasets:
        if ds.get("SamplesPerPixel") != 1 or ds.get("PhotometricInterpretation") not in (
            "MONOCHROME1",
            "MONOCHROME2",
        ):
            raise GeometryError("not single-sample grayscale")
    orients, spacings, positions, sizes = [], [], [], set()
    per_frame = False
    for ds in datasets:
        frames = len(ds.get("PerFrameFunctionalGroupsSequence") or [])
        sizes.add((ds.get("Rows"), ds.get("Columns")))
        for frame in range(max(frames, 1)):
            orient, used = _pick(
                ds.get("ImageOrientationPatient"),
                _frame_value(ds, frame, "PlaneOrientationSequence", "ImageOrientationPatient"),
            )
            per_frame |= used
            space, used = _pick(
                ds.get("PixelSpacing"),
                _frame_value(ds, frame, "PixelMeasuresSequence", "PixelSpacing"),
            )
            per_frame |= used
            if frames:
                position = _frame_value(ds, frame, "PlanePositionSequence", "ImagePositionPatient")
                per_frame |= position is not None
                missing = "multi-frame without positions"
            else:
                position, missing = ds.get("ImagePositionPatient"), "missing position"
            if orient is None:
                raise GeometryError("missing orientation")
            if space is None:
                raise GeometryError("missing pixel spacing")
            if position is None:
                raise GeometryError(missing)
            orients.append(np.array(orient, dtype=float))
            spacings.append(np.array(space, dtype=float))
            positions.append(np.array(position, dtype=float))
    if len(sizes) != 1:
        raise GeometryError("mixed size")
    orient = _unique(orients, "mixed orientation")
    row_spacing, col_spacing = _unique(spacings, "mixed pixel spacing")
    if len(positions) < MIN_SLICES:
        raise GeometryError("too few slices")
    col_dir, row_dir = orient[:3], orient[3:]  # direction of increasing column, of increasing row
    normal = np.cross(col_dir, row_dir)
    heights = np.array(positions) @ normal
    order = np.argsort(heights, kind="stable")
    gaps = np.diff(heights[order])
    if (gaps <= ATOL).any():
        raise GeometryError("duplicate slice positions")
    if (gaps > MAX_GAP_RATIO * np.median(gaps)).any():
        raise GeometryError("slice gap")
    step = (heights[order[-1]] - heights[order[0]]) / (len(order) - 1)
    affine = np.eye(4)
    affine[:3, 0], affine[:3, 1] = col_dir * col_spacing, row_dir * row_spacing
    affine[:3, 2], affine[:3, 3] = normal * step, positions[order[0]]
    rows, cols = next(iter(sizes))
    return Geometry(
        order=tuple(int(i) for i in order),
        affine=affine,
        spacing=(float(col_spacing), float(row_spacing), float(step)),
        size=(int(cols), int(rows), len(order)),
        source="per-frame" if per_frame else "top-level",
    )


def _slices(ds: Dataset) -> np.ndarray:
    frames = len(ds.get("PerFrameFunctionalGroupsSequence") or [])
    if frames:
        ds.NumberOfFrames = frames  # not a kept tag, so the decoder must be told
    pixels = ds.pixel_array.reshape(-1, ds.Rows, ds.Columns).astype(np.float32)
    with np.errstate(over="ignore"):
        pixels = pixels * np.float32(ds.get("RescaleSlope", 1)) + np.float32(
            ds.get("RescaleIntercept", 0)
        )
    return -pixels if ds.PhotometricInterpretation == "MONOCHROME1" else pixels


def preprocess_with_geometry(series_dir: Path) -> tuple[np.ndarray, Geometry]:
    datasets = read_series(series_dir)
    geometry = series_geometry(datasets)
    stack = np.concatenate([_slices(ds) for ds in datasets])[list(geometry.order)]
    if not np.isfinite(stack).all():
        raise GeometryError("non-finite pixels")
    # DICOM is LPS; flip x and y to RAS, then permute and flip the axes to RAS+ order.
    affine = np.diag([-1.0, -1.0, 1.0, 1.0]) @ geometry.affine
    ras = orientations.apply_orientation(
        stack.transpose(2, 1, 0), orientations.io_orientation(affine)
    )
    resampled = functional.interpolate(
        torch.from_numpy(np.ascontiguousarray(ras))[None, None],
        size=(SIZE, SIZE, SIZE),
        mode="trilinear",
        align_corners=False,
    )[0, 0].numpy()
    low, high = np.percentile(resampled, [0.5, 99.5])
    clipped = np.clip(resampled, low, high)
    z = (clipped - clipped.mean()) / max(float(clipped.std()), 1e-6)
    return np.ascontiguousarray(z, dtype=np.float16), geometry


def preprocess_series(series_dir: Path) -> np.ndarray:
    return preprocess_with_geometry(series_dir)[0]
