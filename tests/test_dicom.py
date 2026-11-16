from pathlib import Path

import numpy as np
import pytest

from neurovision.data.dicom import TEST_TAGS, read_header, read_series
from tests.dicom_fakes import fake_pixels, write_series

SHAPE = (4, 8, 6)
SPACING = {"pixel_spacing": (0.5, 0.75), "slice_spacing": 1.5}


def test_test_tags_are_26_unique_keywords() -> None:
    assert len(TEST_TAGS) == 26
    assert len(set(TEST_TAGS)) == 26


def test_read_header_keeps_only_test_tags(tmp_path: Path) -> None:
    path = write_series(
        tmp_path, shape=SHAPE, rescale_slope=2.0, rescale_intercept=-1024.0, **SPACING
    )[0]
    ds = read_header(path)
    assert {elem.keyword for elem in ds} <= set(TEST_TAGS)
    assert "PatientName" not in ds and "StudyDate" not in ds and "PixelData" not in ds
    assert (ds.Rows, ds.Columns) == (8, 6)
    assert list(ds.PixelSpacing) == [0.5, 0.75]
    assert ds.RescaleSlope == 2.0 and ds.RescaleIntercept == -1024.0


def test_read_series_decodes_plain_pixels(tmp_path: Path) -> None:
    write_series(tmp_path, shape=SHAPE, **SPACING)
    slices = read_series(tmp_path)
    assert len(slices) == 4
    assert {elem.keyword for elem in slices[0]} <= set(TEST_TAGS) | {"PixelData"}
    stacked = np.stack(
        [ds.pixel_array[0] if ds.pixel_array.ndim == 3 else ds.pixel_array for ds in slices]
    )
    np.testing.assert_array_equal(stacked, fake_pixels(SHAPE))


def test_read_series_decodes_rle_pixels(tmp_path: Path) -> None:
    write_series(tmp_path, shape=SHAPE, rle=True, **SPACING)
    slices = read_series(tmp_path)
    stacked = np.stack([ds.pixel_array.reshape(8, 6) for ds in slices])
    np.testing.assert_array_equal(stacked, fake_pixels(SHAPE))


@pytest.mark.filterwarnings("ignore:.*frames have been found")  # NumberOfFrames is not kept
def test_read_series_multiframe_is_one_dataset(tmp_path: Path) -> None:
    write_series(tmp_path, shape=SHAPE, multiframe=True, rle=True, **SPACING)
    slices = read_series(tmp_path)
    assert len(slices) == 1
    assert len(slices[0].PerFrameFunctionalGroupsSequence) == 4
    np.testing.assert_array_equal(slices[0].pixel_array, fake_pixels(SHAPE))


def test_read_series_ignores_other_files(tmp_path: Path) -> None:
    write_series(tmp_path, shape=SHAPE, **SPACING)
    (tmp_path / "notes.txt").write_text("x")
    assert len(read_series(tmp_path)) == 4
