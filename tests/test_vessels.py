"""Shi et al. vessel and aneurysm masks mapped into the 1 mm head-crop cache."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from neurovision.data.dicom import read_header
from neurovision.data.labels import LOCATIONS, LR_SWAP
from neurovision.data.points import box_affine, localizer_points
from neurovision.data.preprocess import Geometry, series_geometry
from neurovision.data.vessels import (
    LPS_FROM_RAS,
    N_SEG_CLASSES,
    SEG_LR_SWAP,
    cache_from_mask,
    native_from_mask,
    shi_to_seg,
    to_cache,
)
from tests.dicom_fakes import write_series

SHI = {
    "background": 0,
    "Other Posterior Circulation": 1,
    "BA-Tip": 2,
    "Right Posterior Communicating Artery": 3,
    "Left Posterior Communicating Artery": 4,
    "Right Infraclinoid Internal Carotid Artery": 5,
    "Left Infraclinoid Internal Carotid Artery": 6,
    "Right Supraclinoid Internal Carotid Artery": 7,
    "Left Supraclinoid Internal Carotid Artery": 8,
    "Right Middle Cerebral Artery": 9,
    "Left Middle Cerebral Artery": 10,
    "Right Anterior Cerebral Artery": 11,
    "Left Anterior Cerebral Artery": 12,
    "Anterior Communicating Artery": 13,
    **{f"{name} Aneurysm": 14 + k for k, name in enumerate(LOCATIONS)},
}


def _geometry(size: tuple[int, int, int] = (40, 36, 30)) -> Geometry:
    """Axis-aligned 1 mm scan."""
    affine = np.eye(4)
    affine[:3, 3] = (-20.0, -18.0, -15.0)
    return Geometry(tuple(range(size[2])), affine, (1.0, 1.0, 1.0), size, "top-level")


def test_seg_classes_put_artery_k_at_1_plus_k_and_its_aneurysm_at_14_plus_k() -> None:
    lut = shi_to_seg(SHI)
    assert N_SEG_CLASSES == 27 and lut[0] == 0
    for k, name in enumerate(LOCATIONS):
        assert lut[SHI[name if name != "Basilar Tip" else "BA-Tip"]] == 1 + k
        assert lut[SHI[f"{name} Aneurysm"]] == 14 + k


def test_seg_lr_swap_mirrors_arteries_and_aneurysms_like_the_labels() -> None:
    for k in range(13):
        assert SEG_LR_SWAP[1 + k] == 1 + LR_SWAP[k]
        assert SEG_LR_SWAP[14 + k] == 14 + LR_SWAP[k]
    assert SEG_LR_SWAP[0] == 0
    assert sorted(SEG_LR_SWAP) == list(range(27))


def test_a_mask_voxel_lands_where_the_box_affine_puts_its_native_voxel() -> None:
    geometry = _geometry()
    offset = (2.0, 3.0, 1.0)
    mask = np.zeros(geometry.size, np.uint8)
    native = (25, 10, 12)
    mask[native] = 7
    # dicom2nifti keeps the voxel order here; its affine maps voxels to RAS+
    nifti_affine = np.linalg.inv(LPS_FROM_RAS) @ geometry.affine
    out = to_cache(mask, nifti_affine, geometry, offset, 32)
    assert out.dtype == np.uint8 and out.shape == (32, 32, 32)
    want = np.rint(box_affine(geometry, offset) @ [*native, 1.0])[:3].astype(int)
    assert out[tuple(want)] == 7
    assert np.count_nonzero(out) == 1


def test_a_flipped_nifti_voxel_order_maps_to_the_same_place() -> None:
    geometry = _geometry()
    offset = (0.0, 0.0, 0.0)
    mask = np.zeros(geometry.size, np.uint8)
    mask[5, 6, 7] = 3
    base = np.linalg.inv(LPS_FROM_RAS) @ geometry.affine
    flip = np.eye(4)
    flip[0, 0], flip[0, 3] = -1, geometry.size[0] - 1  # nifti column i is native column n-1-i
    a = to_cache(mask, base, geometry, offset, 32)
    b = to_cache(mask[::-1].copy(), base @ flip, geometry, offset, 32)
    np.testing.assert_array_equal(a, b)


# --- Series without their orientation or pixel spacing ----------------------------------------
# Kaggle's multi-frame MR files keep orientation and pixel spacing only in the shared groups, which
# the test-time tags drop: our Geometry assumes a standard orientation and 1 mm pixels, so its mm
# are not the mask's (Shi's NIfTI came from the full headers, 0.5 mm pixels). The mask holds the
# same voxels, so it maps by voxel index.


def _assumed_series(tmp_path: Path, shape: tuple[int, int, int] = (18, 200, 210)) -> Geometry:
    """A multi-frame series like Kaggle's: positions per frame, nothing else (slices, rows, cols).
    210 columns: the assumed 210 mm field of view gives 1 mm pixels in the cache."""
    files = write_series(
        tmp_path / "s",
        shape=shape,
        pixel_spacing=(0.5, 0.5),
        slice_spacing=2.0,
        multiframe=True,
        top_level_geometry=False,
        frame_orientation=False,
    )
    geometry = series_geometry([read_header(p) for p in files])
    assert geometry.source == "assumed-orientation" and geometry.spacing_assumed
    return geometry


def _shi_affine(geometry: Geometry, pixel_mm: float = 0.5) -> np.ndarray:
    """dicom2nifti from the full headers, reoriented to LPS: same voxels, true pixel size, RAS+."""
    lps = geometry.affine.copy()
    lps[:3, :2] *= pixel_mm / np.linalg.norm(lps[:3, :2], axis=0)
    return np.linalg.inv(LPS_FROM_RAS) @ lps


def _centred(geometry: Geometry, native: np.ndarray, size: int) -> tuple[float, ...]:
    """Box start (mm, 1 mm voxels) that puts native voxel `native` near the box centre."""
    q = box_affine(geometry, (0.0, 0.0, 0.0)) @ [*native, 1.0]
    return tuple(float(v) for v in np.floor(q[:3] - size / 2))


def _blob(geometry: Geometry, label: int) -> tuple[np.ndarray, np.ndarray]:
    mask = np.zeros(geometry.size, np.uint8)
    mask[120:126, 60:66, 4:7] = label
    return mask, np.argwhere(mask == label).mean(axis=0)


def test_an_assumed_spacing_series_maps_its_mask_by_voxel_index(tmp_path: Path) -> None:
    geometry = _assumed_series(tmp_path)
    mask, centre = _blob(geometry, 7)
    offset = _centred(geometry, centre, 32)
    out = to_cache(mask, _shi_affine(geometry), geometry, offset, 32)
    want = (box_affine(geometry, offset) @ [*centre, 1.0])[:3]
    found = np.argwhere(out == 7)
    assert len(found)
    np.testing.assert_allclose(found.mean(axis=0), want, atol=1.0)


def test_an_assumed_spacing_mask_lands_on_the_expert_point_of_its_voxel(tmp_path: Path) -> None:
    geometry = _assumed_series(tmp_path)
    header = read_header(next((tmp_path / "s").glob("*.dcm")))
    column, row, frame = 150, 80, 5
    rows = pd.DataFrame(
        {
            "SOPInstanceUID": [str(header.SOPInstanceUID)],
            "coordinates": [str({"x": column, "y": row, "f": frame})],
            "location": [LOCATIONS[4]],
        }
    )
    offset = _centred(geometry, np.array([column, row, frame]), 32)
    affine = box_affine(geometry, offset)
    [(_, point)] = localizer_points(tmp_path / "s", geometry, affine, rows, [header])
    mask = np.zeros(geometry.size, np.uint8)
    mask[column - 1 : column + 2, row - 1 : row + 2, frame] = 14 + 4
    out = to_cache(mask, _shi_affine(geometry), geometry, offset, 32)
    found = np.argwhere(out == 14 + 4)
    assert len(found) and np.linalg.norm(found - point, axis=1).min() <= 1.0


def test_an_assumed_spacing_series_follows_a_flipped_and_permuted_mask(tmp_path: Path) -> None:
    geometry = _assumed_series(tmp_path)
    mask, centre = _blob(geometry, 2)
    offset = _centred(geometry, centre, 32)
    base = _shi_affine(geometry)
    # the NIfTI stores (row reversed, column, slice): its voxel (i, j, k) is native (j, n-1-i, k)
    swap = np.zeros((4, 4))
    swap[0, 1] = 1.0
    swap[1, 0], swap[1, 3] = -1.0, geometry.size[1] - 1
    swap[2, 2] = swap[3, 3] = 1.0
    stored = np.ascontiguousarray(mask.transpose(1, 0, 2)[::-1])
    want = to_cache(mask, base, geometry, offset, 32)
    assert np.count_nonzero(want)
    np.testing.assert_array_equal(to_cache(stored, base @ swap, geometry, offset, 32), want)


def test_full_geometry_keeps_the_mm_chain() -> None:
    geometry = _geometry()
    nifti = np.linalg.inv(LPS_FROM_RAS) @ geometry.affine
    nifti[:3, 3] += 0.3  # a sub-voxel shift in mm moves the mask, as it always did
    chain = np.linalg.inv(geometry.affine) @ LPS_FROM_RAS @ nifti
    np.testing.assert_array_equal(native_from_mask(nifti, geometry, geometry.size), chain)
    np.testing.assert_array_equal(
        cache_from_mask(nifti, geometry, (0, 0, 0), geometry.size),
        box_affine(geometry, (0, 0, 0)) @ chain,
    )


def test_an_assumed_series_refuses_a_mask_on_other_axes_or_of_another_size(
    tmp_path: Path,
) -> None:
    geometry = _assumed_series(tmp_path)
    base = _shi_affine(geometry)
    turn = np.eye(4)
    c, s = np.cos(np.radians(45)), np.sin(np.radians(45))
    turn[:2, :2] = [[c, -s], [s, c]]
    with pytest.raises(ValueError, match="axes"):
        native_from_mask(turn @ base, geometry, geometry.size)
    with pytest.raises(ValueError, match="shape"):
        native_from_mask(base, geometry, (geometry.size[0] + 1, *geometry.size[1:]))
