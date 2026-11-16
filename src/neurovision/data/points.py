"""Expert aneurysm points in model-input voxels, and the location map built from them.

`localizer_points` and `box_affine` map a `train_localizers.csv` row (column, row on one DICOM
instance) into the 1 mm head-crop cache (`resample: mm`). `ball_map` and `pool_map` turn the
points into the training target of the location-map head: a ball around each point, max-pooled
to the r3d_18 layer3 grid.
"""

import ast
from collections.abc import Sequence
from pathlib import Path

import nibabel.orientations as orientations
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from pydicom.dataset import Dataset

from neurovision.data.dicom import read_header
from neurovision.data.preprocess import ASSUMED_FOV_MM, SPACING_MM, Geometry, series_geometry

MAP_RADIUS_MM = 5.0  # ball around each expert point; the cache has 1 mm voxels
# r3d_18 layer3 stride on (axis 0, axis 1, axis 2): the stem keeps axis 0, then two stride-2 stages
MAP_STRIDE = (4, 8, 8)
POINT_COLUMNS = ("x", "y", "z")  # model-input voxel, RAS+ axes


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


def series_points(
    series_dir: Path, offset_mm: Sequence[float], rows: pd.DataFrame
) -> list[tuple[str, np.ndarray]]:
    """(artery, model-input voxel) of one series' localizer rows, from its headers only."""
    headers = [read_header(p) for p in sorted(series_dir.glob("*.dcm"))]
    geometry = series_geometry(headers)
    return localizer_points(series_dir, geometry, box_affine(geometry, offset_mm), rows, headers)


def points_file(root: Path, cache: str) -> Path:
    """The store's table of expert points in the voxels of cache `cache` (e.g. "b2mm-160")."""
    return root / "labels" / f"points-{cache}.csv"


def inside(q: np.ndarray, size: int, margin: float) -> bool:
    return bool(((q >= margin - 0.5) & (q <= size - 0.5 - margin)).all())


def load_points(path: Path, size: int) -> dict[str, np.ndarray]:
    """Series -> (k, 3) model-input voxels of its points inside a `size` cube; others dropped."""
    table = pd.read_csv(path)
    out: dict[str, np.ndarray] = {}
    for uid, rows in table.groupby("SeriesInstanceUID"):
        q = rows[list(POINT_COLUMNS)].to_numpy(dtype=np.float64)
        q = q[[inside(p, size, 0.0) for p in q]]
        if len(q):
            out[str(uid)] = q
    return out


def ball_map(points: np.ndarray, size: int, radius: float = MAP_RADIUS_MM) -> np.ndarray:
    """uint8 `size` cube: 1 within `radius` voxels of any point (k, 3), else 0."""
    out = np.zeros((size,) * 3, dtype=np.uint8)
    r = int(np.ceil(radius))
    for q in np.asarray(points, dtype=np.float64).reshape(-1, 3):
        centre = np.rint(q).astype(int)
        lo = np.maximum(centre - r, 0)
        hi = np.minimum(centre + r + 1, size)
        if (lo >= hi).any():
            continue
        grid = np.ogrid[lo[0] : hi[0], lo[1] : hi[1], lo[2] : hi[2]]
        d2 = sum((g - c) ** 2 for g, c in zip(grid, q, strict=True))
        out[lo[0] : hi[0], lo[1] : hi[1], lo[2] : hi[2]] |= (d2 <= radius**2).astype(np.uint8)
    return out


def pool_map(volume_map: torch.Tensor, stride: Sequence[int] = MAP_STRIDE) -> torch.Tensor:
    """(1, D, H, W) map -> (1, D/s0, H/s1, W/s2) float 0/1: a cell is 1 if any voxel is."""
    pooled = F.max_pool3d(volume_map.float()[None], kernel_size=tuple(stride))[0]
    return (pooled > 0.5).float()
