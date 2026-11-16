"""The full version's input: a 96 mm cube of 0.5 mm voxels around the predicted vessel tree.

Coordinates: c is a voxel of the 1 mm head-crop cache (cache/b2mm-160, RAS+ axes, axis 0 left to
right); `points.box_affine` maps native DICOM voxels to it. The vessel box has the same axes and
BOX_SPACING_MM voxels; `cache_from_box` puts its centre on a point given in c. The box is cut from
the raw DICOM (`vessel_box`), not resampled from the 1 mm cache, and its centre is the middle of
the 99.5% vessel box (`trimmed_box`) of the step-1 U-Net's prediction on the 1 mm cache.
"""

from collections.abc import Sequence
from pathlib import Path

import numpy as np
import torch
from scipy import ndimage

from neurovision.data.points import box_affine
from neurovision.data.preprocess import SPACING_MM, Geometry, antialias, normalise, read_native

BOX_SIZE = 192
BOX_SPACING_MM = 0.5
BOX_MM = BOX_SIZE * BOX_SPACING_MM  # 96: holds 99.5% of the Shi vessel voxels in 94% of series
TRIM = 0.005  # each end of every axis: the 99.5% box ignores stray voxels
# fewer predicted vessel voxels at 1 mm: no box (the cache falls back to the median centre);
# a Shi tree on an MR T2 scan can hold under 1,000 voxels
MIN_VESSEL_VOXELS = 200


def cache_from_box(
    centre: Sequence[float], size: int = BOX_SIZE, spacing_mm: float = BOX_SPACING_MM
) -> np.ndarray:
    """Vessel-box voxel -> 1 mm cache voxel; the box's middle ((size - 1) / 2) lands on `centre`."""
    step = spacing_mm / SPACING_MM
    out = np.eye(4)
    out[:3, :3] *= step
    out[:3, 3] = np.asarray(centre, dtype=float) - (size - 1) / 2 * step
    return out


def trimmed_box(
    seg: np.ndarray, trim: float = TRIM, min_voxels: int = MIN_VESSEL_VOXELS
) -> tuple[np.ndarray, np.ndarray, int]:
    """(lo, hi, n): per-axis `trim` and 1 - `trim` quantiles of the voxels with a class > 0,
    in voxels of `seg`, and their count; lo and hi are NaN when there are fewer than
    `min_voxels`."""
    idx = np.nonzero(seg)
    n = int(idx[0].size)
    if n < min_voxels:
        return np.full(3, np.nan), np.full(3, np.nan), n
    lo = np.array([np.quantile(a, trim, method="lower") for a in idx], dtype=float)
    hi = np.array([np.quantile(a, 1 - trim, method="higher") for a in idx], dtype=float)
    return lo, hi, n


def cut_box(native: np.ndarray, native_from_box: np.ndarray, size: int) -> np.ndarray:
    """The size³ box sampled trilinearly from `native` (float32); outside the scan, its minimum.

    `native_from_box` maps a box voxel to a native voxel. Native axes finer than a box voxel are
    blurred first, as `preprocess.antialias` does for the 1 mm cache, on just the part of the
    scan the box covers (plus the blur's reach), which gives the same values as blurring it all.
    """
    box_from_native = np.linalg.inv(native_from_box)
    zoom = np.linalg.norm(box_from_native[:3, :3], axis=0)  # box voxels per native voxel
    sigma = np.where(zoom < 1, (1 / zoom - 1) / 2, 0.0)
    reach = np.ceil(4 * sigma).astype(int) + 2
    corners = np.array(np.meshgrid(*[[-0.5, size - 0.5]] * 3, indexing="ij")).reshape(3, -1)
    ends = native_from_box[:3, :3] @ corners + native_from_box[:3, 3:4]
    shape = np.array(native.shape)
    lo = np.clip(np.floor(ends.min(axis=1)).astype(int) - reach, 0, shape)
    hi = np.clip(np.ceil(ends.max(axis=1)).astype(int) + reach + 1, 0, shape)
    cval = float(native.min())
    if (hi <= lo).any():
        return np.full((size,) * 3, cval, dtype=np.float32)
    part = np.ascontiguousarray(native[lo[0] : hi[0], lo[1] : hi[1], lo[2] : hi[2]])
    # the exact zoom per axis (not a rounded size), so the part blurs like the whole scan
    target = [float(n * z) for n, z in zip(part.shape, zoom, strict=True)]
    blurred = antialias(torch.from_numpy(part.astype(np.float32)), target).numpy()
    return ndimage.affine_transform(
        blurred,
        native_from_box[:3, :3],
        native_from_box[:3, 3] - lo,
        output_shape=(size,) * 3,
        order=1,
        mode="constant",
        cval=cval,
    ).astype(np.float32)


def vessel_box(
    series_dir: Path,
    offset_mm: Sequence[float],
    centre: Sequence[float],
    size: int = BOX_SIZE,
    spacing_mm: float = BOX_SPACING_MM,
) -> tuple[np.ndarray, Geometry, np.ndarray]:
    """(normalised float16 size³ box, geometry, box_from_cache) for one DICOM series.

    `offset_mm` is the series' box start in the 1 mm cache index; `centre` the box centre in
    1 mm cache voxels. Normalised like every cache (`preprocess.normalise`) over the box itself.
    """
    native, geometry = read_native(series_dir)
    box_from_cache = np.linalg.inv(cache_from_box(centre, size, spacing_mm))
    native_from_box = np.linalg.inv(box_from_cache @ box_affine(geometry, offset_mm))
    return normalise(cut_box(native, native_from_box, size)), geometry, box_from_cache
