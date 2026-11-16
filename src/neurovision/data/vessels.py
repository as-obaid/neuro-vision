"""Shi et al. (2026) vessel and aneurysm masks as 27-class targets in the 1 mm head-crop cache.

Our classes: 0 background; 1 + k artery k and 14 + k an aneurysm on artery k, with k the index
of `LOCATIONS` (the Kaggle label order). The masks are NIfTI in the scan's own voxels (dicom2nifti,
affine to RAS+); `to_cache` resamples one with nearest neighbour into the cache box: mask voxel to
native voxel (`native_from_mask`), then the same affine as the localizer points
(`points.box_affine`).
"""

from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
from monai.utils import convert_to_tensor
from scipy import ndimage
from torch.utils.data import Dataset

from neurovision.data.augment import IMAGE, MAP
from neurovision.data.dataset import LR_AXIS, CacheDataset
from neurovision.data.labels import LOCATIONS, LR_SWAP
from neurovision.data.points import box_affine
from neurovision.data.preprocess import Geometry

N_SEG_CLASSES = 1 + 2 * len(LOCATIONS)  # 27
LPS_FROM_RAS = np.diag([-1.0, -1.0, 1.0, 1.0])
SHI_DIR = "labels/shi2026-masks"  # under the store: labels.json and the NIfTI folders
SHI_ORIG = "labelsTr_26classes_in_orig_space"  # <uid>.nii.gz in the scan's own voxels
_SHI_NAMES = {"BA-Tip": "Basilar Tip"}  # Shi's names that differ from the Kaggle labels
SEG_LR_SWAP: tuple[int, ...] = (
    0,
    *(1 + LR_SWAP[k] for k in range(len(LOCATIONS))),
    *(1 + len(LOCATIONS) + LR_SWAP[k] for k in range(len(LOCATIONS))),
)


def shi_to_seg(labels: Mapping[str, int]) -> np.ndarray:
    """Lookup table from a Shi class id (labels.json) to our class id."""
    lut = np.zeros(max(labels.values()) + 1, np.uint8)
    for name, shi in labels.items():
        if name == "background":
            continue
        aneurysm = name.endswith(" Aneurysm")
        artery = _SHI_NAMES.get(name.removesuffix(" Aneurysm"), name.removesuffix(" Aneurysm"))
        k = LOCATIONS.index(artery)
        lut[shi] = 1 + k + (len(LOCATIONS) if aneurysm else 0)
    return lut


def native_from_mask(
    mask_affine: np.ndarray, geometry: Geometry, mask_shape: Sequence[int]
) -> np.ndarray:
    """Mask voxel -> native voxel (column, row, sorted slice).

    Full geometry: through patient mm (RAS+ -> LPS -> native). A series whose orientation or
    pixel spacing our `Geometry` assumed (Kaggle's multi-frame MR files keep both only in the
    shared groups, which the test-time tags drop) has no true mm, while Shi's NIfTI came from
    the full headers: there the map is by voxel index. The NIfTI holds the scan's own voxels,
    reoriented (dicom2nifti, then nibabel to LPS; no resampling), so the map is the signed axis
    permutation between the two direction matrices, a flipped axis counted from its far end.
    """
    if not geometry.source.startswith("assumed"):
        return np.linalg.inv(geometry.affine) @ LPS_FROM_RAS @ mask_affine
    ours = geometry.affine[:3, :3] / np.linalg.norm(geometry.affine[:3, :3], axis=0)
    theirs = (LPS_FROM_RAS @ mask_affine)[:3, :3]
    theirs = theirs / np.linalg.norm(theirs, axis=0)
    turn = np.linalg.inv(ours) @ theirs  # unit-free: native axis a <- mask axis
    signed = np.round(turn)
    if np.abs(turn - signed).max() > 0.3 or not (np.abs(signed).sum(axis=0) == 1).all():
        raise ValueError("the mask's axes do not match the scan's axes")
    if (np.abs(signed).sum(axis=1) != 1).any():
        raise ValueError("the mask's axes do not match the scan's axes")
    out = np.eye(4)
    out[:3, :3] = signed
    for a in range(3):
        m = int(np.flatnonzero(signed[a])[0])
        if int(mask_shape[m]) != geometry.size[a]:
            raise ValueError(f"mask shape {tuple(mask_shape)} does not fit scan {geometry.size}")
        if signed[a, m] < 0:
            out[a, 3] = geometry.size[a] - 1
    return out


def cache_from_mask(
    mask_affine: np.ndarray,
    geometry: Geometry,
    offset_mm: Sequence[float],
    mask_shape: Sequence[int],
    box_from_cache: np.ndarray | None = None,
) -> np.ndarray:
    """Mask voxel -> native voxel (`native_from_mask`) -> cache box voxel; with
    `box_from_cache` (1 mm cache voxel -> another box, e.g. `data.vbox`), on into that box."""
    out = box_affine(geometry, offset_mm) @ native_from_mask(mask_affine, geometry, mask_shape)
    return out if box_from_cache is None else box_from_cache @ out


def to_cache(
    mask: np.ndarray,
    mask_affine: np.ndarray,
    geometry: Geometry,
    offset_mm: Sequence[float],
    size: int,
    box_from_cache: np.ndarray | None = None,
) -> np.ndarray:
    """The mask in the cache's size³ box (uint8, nearest neighbour; 0 outside the scan), or in
    the box `box_from_cache` leads to."""
    mask_from_cache = np.linalg.inv(
        cache_from_mask(mask_affine, geometry, offset_mm, mask.shape, box_from_cache)
    )
    out = ndimage.affine_transform(
        mask,
        mask_from_cache[:3, :3],
        mask_from_cache[:3, 3],
        output_shape=(size, size, size),
        order=0,
        mode="constant",
        cval=0,
    )
    return out.astype(np.uint8)


class VesselDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    """(volume (1, D, H, W) float32, class map (D, H, W) int64) from the cache and `seg_dir`.

    A left-right flip mirrors axis 0 of both and swaps each left/right class (`SEG_LR_SWAP`);
    draws come from `(flip_seed, i)` like `CacheDataset`.
    """

    def __init__(
        self, cache_dir: Path, seg_dir: Path, uids: Sequence[str], flip_lr: float = 0.0
    ) -> None:
        self.cache_dir, self.seg_dir, self.uids = cache_dir, seg_dir, list(uids)
        self.flip_lr = flip_lr
        self.flip_seed = 0
        self._swap = np.asarray(SEG_LR_SWAP, np.uint8)

    def __len__(self) -> int:
        return len(self.uids)

    def set_flip_seed(self, seed: int) -> None:
        self.flip_seed = seed

    def __getitem__(self, i: int) -> tuple[torch.Tensor, torch.Tensor]:
        uid = self.uids[i]
        volume = np.load(self.cache_dir / f"{uid}.npy").astype(np.float32)
        seg = np.load(self.seg_dir / f"{uid}.npz")["seg"]
        rng = np.random.default_rng([self.flip_seed, i])
        if self.flip_lr > 0 and rng.random() < self.flip_lr:
            volume = np.ascontiguousarray(np.flip(volume, axis=LR_AXIS))
            seg = self._swap[np.flip(seg, axis=LR_AXIS)]
        x = torch.from_numpy(volume).unsqueeze(0)
        return x, torch.from_numpy(np.ascontiguousarray(seg)).long()


class SegCacheDataset(CacheDataset):
    """(volume, 14 targets, class map (1, D, H, W) int64, keep) for the full version.

    The class map comes from `seg_dir/<uid>.npz` (key "seg"); keep = 0 and an empty map when the
    series has none, so it adds no segmentation loss. A left-right flip mirrors axis 0 of the
    volume and the map, swaps the paired labels (`LR_SWAP`) and the paired classes
    (`SEG_LR_SWAP`); draws come from `(flip_seed, i)` like `CacheDataset`. `transform` takes and
    returns a dict with "image" and "map" (`build_augment(..., with_map=True)`: the class map
    follows the image through the affine with nearest-neighbour sampling).
    """

    def __init__(
        self,
        cache_dir: Path,
        uids: list[str],
        targets: np.ndarray,
        transform: Callable[..., Any] | None,
        flip_lr: float,
        seg_dir: Path,
    ) -> None:
        super().__init__(cache_dir, uids, targets, transform, flip_lr)
        self.seg_dir = seg_dir
        self._swap = np.asarray(SEG_LR_SWAP, np.uint8)

    def __getitem__(self, i: int) -> tuple[torch.Tensor, ...]:
        uid = self.uids[i]
        volume = np.load(self.cache_dir / f"{uid}.npy").astype(np.float32)
        target = self.targets[i]
        path = self.seg_dir / f"{uid}.npz"
        keep = path.exists()
        seg = np.load(path)["seg"] if keep else np.zeros(volume.shape, np.uint8)
        if self._flips(i):
            volume = np.ascontiguousarray(np.flip(volume, axis=LR_AXIS))
            seg = self._swap[np.flip(seg, axis=LR_AXIS)]
            target = target[list(LR_SWAP)]
        x = torch.from_numpy(volume).unsqueeze(0)
        m = torch.from_numpy(np.ascontiguousarray(seg)).unsqueeze(0)
        if self.transform is not None:
            out = self.transform({IMAGE: x, MAP: m.float()})
            x = convert_to_tensor(out[IMAGE], track_meta=False)
            m = convert_to_tensor(out[MAP], track_meta=False).round()
        y = torch.from_numpy(np.ascontiguousarray(target))
        return x, y, m.long(), torch.tensor(float(keep))
