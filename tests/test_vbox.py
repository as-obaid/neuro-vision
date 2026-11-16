"""The full version's input: a 0.5 mm cube around the predicted vessel tree, cut from the DICOM."""

from pathlib import Path

import numpy as np
import pytest
import torch
from scipy import ndimage

from neurovision.data.points import box_affine
from neurovision.data.preprocess import Geometry, antialias, preprocess_with_crop
from neurovision.data.vbox import (
    BOX_SIZE,
    BOX_SPACING_MM,
    cache_from_box,
    cut_box,
    trimmed_box,
    vessel_box,
)
from neurovision.data.vessels import LPS_FROM_RAS, to_cache
from tests.dicom_fakes import write_series


def test_the_default_box_is_a_96_mm_cube_of_half_mm_voxels() -> None:
    assert (BOX_SIZE, BOX_SPACING_MM) == (192, 0.5)


def test_cache_from_box_puts_the_box_centre_on_the_given_point() -> None:
    centre = np.array([80.0, 70.5, 60.25])
    m = cache_from_box(centre, size=16, spacing_mm=0.5)
    np.testing.assert_allclose((m @ [7.5, 7.5, 7.5, 1.0])[:3], centre)
    # one box voxel is half a cache voxel along every axis; the faces sit 4 cache voxels out
    np.testing.assert_allclose(m[:3, :3], 0.5 * np.eye(3))
    np.testing.assert_allclose((m @ [-0.5, -0.5, -0.5, 1.0])[:3], centre - 4.0)
    np.testing.assert_allclose((m @ [15.5, 15.5, 15.5, 1.0])[:3], centre + 4.0)


def test_trimmed_box_ignores_stray_voxels_and_reports_the_count() -> None:
    seg = np.zeros((40, 40, 40), np.uint8)
    seg[10:20, 12:30, 5:25] = 3  # 3,600 voxels
    seg[38, 38, 38] = 1  # one stray voxel: outside the 99.5% box
    lo, hi, n = trimmed_box(seg, min_voxels=100)
    assert n == 3601
    np.testing.assert_array_equal(lo, [10, 12, 5])
    np.testing.assert_array_equal(hi, [19, 29, 24])


def test_trimmed_box_is_nan_below_the_minimum_count() -> None:
    seg = np.zeros((20, 20, 20), np.uint8)
    seg[5, 5, 5] = 2
    lo, hi, n = trimmed_box(seg, min_voxels=10)
    assert n == 1 and np.isnan(lo).all() and np.isnan(hi).all()


def _smooth(shape: tuple[int, int, int]) -> np.ndarray:
    rng = np.random.default_rng(0)
    return ndimage.gaussian_filter(rng.normal(size=shape), 2).astype(np.float32)


def test_cut_box_crops_the_native_volume_without_changing_the_result() -> None:
    native = _smooth((60, 50, 40))
    spacing = np.array([0.3, 0.4, 0.8])  # mm per native voxel: the first two axes are blurred
    size, step = 12, 0.5
    box_from_native = np.eye(4)
    box_from_native[:3, :3] = np.diag(spacing / step)
    box_from_native[:3, 3] = [-30.0, -22.0, -20.0]
    native_from_box = np.linalg.inv(box_from_native)
    got = cut_box(native, native_from_box, size)
    shape = [float(n * s / step) for n, s in zip(native.shape, spacing, strict=True)]
    blurred = antialias(torch.from_numpy(native), shape).numpy()
    want = ndimage.affine_transform(
        blurred,
        native_from_box[:3, :3],
        native_from_box[:3, 3],
        output_shape=(size,) * 3,
        order=1,
        mode="constant",
        cval=float(native.min()),
    )
    np.testing.assert_allclose(got, want, atol=1e-5)


def test_cut_box_outside_the_scan_is_the_volume_minimum() -> None:
    native = _smooth((20, 20, 20))
    native_from_box = np.eye(4)
    native_from_box[:3, 3] = 500.0
    out = cut_box(native, native_from_box, 8)
    assert out.shape == (8, 8, 8) and np.all(out == native.min())


def _blob_series(tmp_path: Path) -> tuple[Path, np.ndarray]:
    n, rows, cols = 40, 48, 44
    pixels = np.zeros((n, rows, cols), dtype=np.uint16)
    pixels[18:22, 20:26, 15:21] = 1000  # centred on slice 19.5, row 22.5, column 17.5
    write_series(
        tmp_path / "s",
        shape=(n, rows, cols),
        pixel_spacing=(0.8, 0.7),
        slice_spacing=1.2,
        orientation=(0, 1, 0, 0, 0, -1),  # sagittal, so the axes permute and flip
        pixels=pixels,
    )
    return tmp_path / "s", np.array([17.5, 22.5, 19.5, 1.0])


@pytest.mark.parametrize("shift", [(0.0, 0.0, 0.0), (2.0, -1.0, 1.5)])
def test_vessel_box_centres_the_cut_on_a_point_given_in_cache_voxels(
    tmp_path: Path, shift: tuple[float, float, float]
) -> None:
    series, native_blob = _blob_series(tmp_path)
    _, _, crop = preprocess_with_crop(series, 48, "mm")
    assert crop is not None
    blob = (crop.index_affine @ native_blob)[:3]  # the blob in 1 mm cache voxels
    size = 24
    volume, geometry, box_from_cache = vessel_box(
        series, crop.offset_mm, blob + shift, size=size, spacing_mm=0.5
    )
    assert volume.shape == (size,) * 3 and volume.dtype == np.float16
    values = volume.astype(np.float32)
    found = np.argwhere(values > (values.max() + values.min()) / 2).mean(axis=0)
    want = (size - 1) / 2 - np.array(shift) / 0.5
    assert np.abs(found - want).max() < 0.6, (found, want)
    np.testing.assert_allclose((box_from_cache @ [*blob, 1.0])[:3], want)
    assert geometry.size == (44, 48, 40)


def test_a_mask_maps_into_the_vessel_box_where_the_image_goes() -> None:
    affine = np.eye(4)
    affine[:3, :3] = np.diag([0.8, 0.8, 1.5])
    geometry = Geometry(tuple(range(30)), affine, (0.8, 0.8, 1.5), (40, 36, 30), "top-level")
    offset = (2.0, 1.0, 3.0)
    native = np.array([25.0, 10.0, 12.0, 1.0])
    mask = np.zeros(geometry.size, np.uint8)
    mask[25, 10, 12] = 9
    centre = (box_affine(geometry, offset) @ native)[:3] + [1.0, 0.0, -0.5]
    box_from_cache = np.linalg.inv(cache_from_box(centre, size=20, spacing_mm=0.5))
    nifti_affine = np.linalg.inv(LPS_FROM_RAS) @ geometry.affine
    out = to_cache(mask, nifti_affine, geometry, offset, 20, box_from_cache)
    assert out.shape == (20, 20, 20) and set(np.unique(out)) == {0, 9}
    # a 0.8 x 0.8 x 1.5 mm voxel covers about 1.6 x 1.6 x 3 box voxels around its centre
    want = (box_from_cache @ box_affine(geometry, offset) @ native)[:3]
    np.testing.assert_allclose(np.argwhere(out == 9).mean(axis=0), want, atol=0.6)
    np.testing.assert_allclose(want, [9.5 - 2.0, 9.5, 9.5 + 1.0])
