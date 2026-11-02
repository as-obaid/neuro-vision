"""DICOM series to a float16 cubic RAS+ volume (128 by default): the only DICOM-to-volume path.

resample "cube" (default) resizes the whole field of view to size³. resample "mm" resamples to
1 mm voxels and crops a size³ box around the head (see `head_box`).
"""

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import nibabel.orientations as orientations
import numpy as np
import torch
import torch.nn.functional as functional
from pydicom.dataset import Dataset
from scipy import ndimage

from neurovision.data.dicom import read_series

SIZE = 128
MIN_SLICES = 16
MAX_GAP_RATIO = 1.5
MAX_OBLIQUE_DEG = 10
# Standard orientations assumed when the series carries none, by the patient axis (LPS) of the
# slice direction: z axial, y coronal, x sagittal.
STANDARD_ORIENTATION = {
    0: np.array([0.0, 1, 0, 0, 0, -1]),
    1: np.array([1.0, 0, 0, 0, 0, -1]),
    2: np.array([1.0, 0, 0, 0, 1, 0]),
}
ERROR_PREFIX = "error:"  # index.csv reason of a transient failure, retried on the next run
ATOL = 1e-3  # mm or direction cosine; spacing and position are stored to about 1e-4
Resample = Literal["cube", "mm"]
SPACING_MM = 1.0  # voxel side of resample "mm"
SUPERIOR_MARGIN_MM = 10.0  # air kept above the head when the head is taller than the box
# In-plane field of view assumed when PixelSpacing is missing (multi-frame series whose shared
# groups are stripped): the median of the MR series that carry it is 197 mm (MRA) to 222 mm (T2).
ASSUMED_FOV_MM = 210.0


class GeometryError(ValueError):
    pass


@dataclass(frozen=True)
class Geometry:
    order: tuple[int, ...]  # slice indices (files in name order, frames within a file), sorted
    affine: np.ndarray  # voxel (column, row, slice) to patient LPS, mm
    spacing: tuple[float, float, float]  # column, row, slice
    size: tuple[int, int, int]  # columns, rows, slices
    # "top-level"; "per-frame" if a tag came from the frame groups; "assumed-orientation" or
    # "assumed-spacing" (orientation present, spacing defaulted) when a tag was missing everywhere
    source: str
    spacing_assumed: bool = False  # PixelSpacing missing everywhere: column and row spacing are 1


@dataclass(frozen=True)
class Crop:
    """Where the resample "mm" box sits; RAS+ axes, mm relative to the scan's first voxel."""

    index_affine: np.ndarray  # native voxel (column, row, sorted slice) to box voxel index
    kept_fraction: float  # share of the head mask inside the box
    offset_mm: tuple[float, float, float]  # box start
    extent_mm: tuple[float, float, float]  # scan extent


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


def _assume_orientation(positions: np.ndarray) -> np.ndarray:
    # Extent per axis rather than a step: it does not depend on the order the frames come in.
    extent = np.ptp(positions, axis=0)
    axis = int(np.argmax(extent))
    if extent[axis] < np.cos(np.radians(MAX_OBLIQUE_DEG)) * np.linalg.norm(extent):
        raise GeometryError("oblique slice direction")
    return STANDARD_ORIENTATION[axis]


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
            if position is None:
                raise GeometryError(missing)
            orients.append(orient)
            spacings.append(space)
            positions.append(np.array(position, dtype=float))
    if len(sizes) != 1:
        raise GeometryError("mixed size")
    if len(positions) < MIN_SLICES:
        raise GeometryError("too few slices")
    assumed = ""
    if all(o is None for o in orients):
        orient, assumed = _assume_orientation(np.array(positions)), "assumed-orientation"
    elif any(o is None for o in orients):
        raise GeometryError("missing orientation")
    else:
        orient = _unique([np.array(o, dtype=float) for o in orients], "mixed orientation")
    if all(s is None for s in spacings):
        row_spacing = col_spacing = 1.0  # irrelevant: the whole field of view is resampled
        assumed = assumed or "assumed-spacing"
    elif any(s is None for s in spacings):
        raise GeometryError("missing pixel spacing")
    else:
        row_spacing, col_spacing = _unique(
            [np.array(s, dtype=float) for s in spacings], "mixed pixel spacing"
        )
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
        source=assumed or ("per-frame" if per_frame else "top-level"),
        spacing_assumed=all(s is None for s in spacings),
    )


def antialias(volume: torch.Tensor, size: int | Sequence[int] = SIZE) -> torch.Tensor:
    """Gaussian-blur each axis that the resize to `size` downsamples (sigma as in scikit-image).

    `size` is one edge length for every axis, or one per axis.
    """
    for axis, n in enumerate(volume.shape):
        zoom = (size if isinstance(size, int) else size[axis]) / n
        if zoom >= 1:
            continue
        sigma = (1 / zoom - 1) / 2
        radius = int(np.ceil(4 * sigma))
        taps = torch.arange(-radius, radius + 1, dtype=volume.dtype)
        kernel = torch.exp(-(taps**2) / (2 * sigma**2))
        kernel /= kernel.sum()
        lines = volume.movedim(axis, -1).reshape(-1, 1, n)
        padded = functional.pad(lines, (radius, radius), mode="replicate")
        blurred = functional.conv1d(padded, kernel.view(1, 1, -1))
        volume = blurred.reshape(*volume.movedim(axis, -1).shape).movedim(-1, axis)
    return volume


def resample(volume: np.ndarray, size: int = SIZE) -> np.ndarray:
    blurred = antialias(torch.from_numpy(volume), size)
    return functional.interpolate(
        blurred[None, None], size=(size, size, size), mode="trilinear", align_corners=False
    )[0, 0].numpy()


def resample_mm(
    volume: np.ndarray, spacing: Sequence[float], spacing_mm: float = SPACING_MM
) -> np.ndarray:
    """Trilinear resample to `spacing_mm` voxels, blurring downsampled axes as `resample` does."""
    shape = tuple(
        max(1, round(n * s / spacing_mm)) for n, s in zip(volume.shape, spacing, strict=True)
    )
    blurred = antialias(torch.from_numpy(volume), shape)
    return functional.interpolate(
        blurred[None, None], size=shape, mode="trilinear", align_corners=False
    )[0, 0].numpy()


def _otsu(values: np.ndarray, bins: int = 256) -> float:
    hist, edges = np.histogram(values, bins=bins)
    below = np.cumsum(hist)
    mass = np.cumsum(hist * (edges[:-1] + edges[1:]) / 2)
    above = below[-1] - below
    with np.errstate(divide="ignore", invalid="ignore"):
        between = below * above * (mass / below - (mass[-1] - mass) / above) ** 2
    return float(edges[int(np.nanargmax(between[:-1])) + 1])


def _head_mask(volume: np.ndarray) -> np.ndarray:
    """Largest connected component above an Otsu threshold; on a 2-voxel grid for speed.

    Voxels at the minimum (padding outside a CT's circular field of view) are left out of the
    threshold, and an opening cuts thin bridges (a table, a head holder) before the components.
    """
    coarse = volume[::2, ::2, ::2]
    values = coarse[coarse > coarse.min()]
    if values.size == 0:
        return np.ones_like(coarse, dtype=bool)
    threshold = _otsu(values) if values.max() > values.min() else coarse.min()
    mask = ndimage.binary_opening(coarse > threshold)
    labels, count = ndimage.label(mask)
    if count == 0:
        return np.ones_like(coarse, dtype=bool)
    sizes = np.bincount(labels.ravel())
    sizes[0] = 0
    return np.asarray(labels == np.argmax(sizes))


def head_box(volume: np.ndarray, size: int, margin: int) -> tuple[tuple[int, int, int], float]:
    """Start index per axis of a size³ box on the head, and the share of the head mask inside it.

    RAS+ axes; axis 2 runs inferior to superior. Superior-inferior first: centred on the head mask
    when it fits, otherwise the box top sits `margin` voxels above the mask's top (brain kept, neck
    dropped). Left-right and anterior-posterior: centred on the part of the mask inside that slab.
    """
    mask = _head_mask(volume)
    voxels = np.argwhere(mask) * 2  # coarse to full index; each coarse voxel covers [2i, 2i + 2)
    lo_z, hi_z = int(voxels[:, 2].min()), int(voxels[:, 2].max()) + 2
    top = min(hi_z, volume.shape[2])
    start_z = (lo_z + top - size) // 2 if top - lo_z <= size else top + margin - size
    slab = voxels[(voxels[:, 2] + 2 > start_z) & (voxels[:, 2] < start_z + size)]
    slab = slab if len(slab) else voxels
    start = [
        (int(slab[:, a].min()) + min(int(slab[:, a].max()) + 2, volume.shape[a]) - size) // 2
        for a in (0, 1)
    ]
    box = (start[0], start[1], start_z)
    centres = voxels + 1
    inside = ((centres >= box) & (centres < np.add(box, size))).all(axis=1)
    return box, float(inside.mean())


def crop_box(volume: np.ndarray, start: Sequence[int], size: int) -> np.ndarray:
    """The size³ box at `start`, padded with the volume's minimum (air) outside the scan."""
    out = np.full((size,) * 3, volume.min(), dtype=volume.dtype)
    src, dst = [], []
    for a, n in enumerate(volume.shape):
        lo, hi = max(start[a], 0), min(start[a] + size, n)
        src.append(slice(lo, max(hi, lo)))
        dst.append(slice(lo - start[a], lo - start[a] + max(hi - lo, 0)))
    out[tuple(dst)] = volume[tuple(src)]
    return out


def cache_name(size: int, resample: Resample = "cube") -> str:
    """Cache folder under $NV_STORE/cache: b2-{size} for "cube", b2mm-{size} for "mm"."""
    return f"b2-{size}" if resample == "cube" else f"b2mm-{size}"


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


def _normalise(volume: np.ndarray) -> np.ndarray:
    low, high = np.percentile(volume, [0.5, 99.5])
    clipped = np.clip(volume, low, high)
    z = (clipped - clipped.mean()) / max(float(clipped.std()), 1e-6)
    return np.ascontiguousarray(z, dtype=np.float16)


def _cube(volume: np.ndarray, size: int) -> np.ndarray:
    return resample(volume, size)  # module-level: `resample` is a parameter name below


def _mm_crop(
    ras: np.ndarray, geometry: Geometry, ornt: np.ndarray, size: int, spacing_mm: float
) -> tuple[np.ndarray, Crop]:
    native = geometry.affine.copy()
    if geometry.spacing_assumed:
        native[:3, :2] *= ASSUMED_FOV_MM / max(geometry.size[:2])
    # oriented (RAS+ array) voxel to native voxel, and the oriented voxel sides in mm
    to_native = orientations.inv_ornt_aff(ornt, geometry.size)
    spacing = np.linalg.norm((native @ to_native)[:3, :3], axis=0)
    resampled = resample_mm(ras, tuple(float(s) for s in spacing), spacing_mm)
    start, kept = head_box(resampled, size, round(SUPERIOR_MARGIN_MM / spacing_mm))
    # native voxel -> oriented -> resampled (align_corners=False) -> box
    zoom = np.array(resampled.shape) / np.array(ras.shape)
    to_box = np.eye(4)
    to_box[:3, :3] = np.diag(zoom)
    to_box[:3, 3] = 0.5 * zoom - 0.5 - np.array(start)
    crop = Crop(
        index_affine=to_box @ np.linalg.inv(to_native),
        kept_fraction=kept,
        offset_mm=tuple(float(s * spacing_mm) for s in start),  # type: ignore[arg-type]
        extent_mm=tuple(float(n * s) for n, s in zip(ras.shape, spacing, strict=True)),  # type: ignore[arg-type]
    )
    return crop_box(resampled, start, size), crop


def preprocess_with_crop(
    series_dir: Path,
    size: int = SIZE,
    resample: Resample = "cube",
    spacing_mm: float = SPACING_MM,
) -> tuple[np.ndarray, Geometry, Crop | None]:
    """The normalised volume, the geometry, and for resample "mm" where the box sits."""
    datasets = read_series(series_dir)
    geometry = series_geometry(datasets)
    stack = np.concatenate([_slices(ds) for ds in datasets])
    if len(stack) != len(geometry.order):
        raise GeometryError("frame count mismatch")
    stack = stack[list(geometry.order)]
    if not np.isfinite(stack).all():
        raise GeometryError("non-finite pixels")
    # DICOM is LPS; flip x and y to RAS, then permute and flip the axes to RAS+ order.
    affine = np.diag([-1.0, -1.0, 1.0, 1.0]) @ geometry.affine
    ornt = orientations.io_orientation(affine)
    ras = np.ascontiguousarray(orientations.apply_orientation(stack.transpose(2, 1, 0), ornt))
    if resample == "cube":
        return _normalise(_cube(ras, size)), geometry, None
    volume, crop = _mm_crop(ras, geometry, ornt, size, spacing_mm)
    return _normalise(volume), geometry, crop


def preprocess_with_geometry(
    series_dir: Path, size: int = SIZE, resample: Resample = "cube"
) -> tuple[np.ndarray, Geometry]:
    volume, geometry, _ = preprocess_with_crop(series_dir, size, resample)
    return volume, geometry


def preprocess_series(
    series_dir: Path, size: int = SIZE, resample: Resample = "cube"
) -> np.ndarray:
    return preprocess_with_geometry(series_dir, size, resample)[0]
