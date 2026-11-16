"""resample: mm, physical-spacing resample and head crop."""

import inspect
from pathlib import Path

import numpy as np
import pydicom
import pytest

from neurovision.data import preprocess
from neurovision.data.preprocess import (
    ASSUMED_FOV_MM,
    SUPERIOR_MARGIN_MM,
    cache_name,
    crop_box,
    head_box,
    preprocess_series,
    preprocess_with_crop,
    resample_mm,
)
from neurovision.infer.predict import predict_series
from tests.dicom_fakes import write_series
from tests.script_loader import import_script
from tests.test_predict_series import _model

KW = {"shape": (20, 16, 16), "pixel_spacing": (1.0, 1.0), "slice_spacing": 2.0}


def ellipsoid(shape: tuple[int, int, int], centre: tuple[float, ...], radii: tuple[float, ...]):
    grid = np.indices(shape, dtype=np.float32)
    r2 = sum(((g - c) / r) ** 2 for g, c, r in zip(grid, centre, radii, strict=True))
    return r2 <= 1


def test_anisotropic_spacing_becomes_isotropic() -> None:
    volume = np.random.default_rng(0).normal(size=(40, 30, 20)).astype(np.float32)
    assert resample_mm(volume, (0.5, 1.0, 2.0), 1.0).shape == (20, 30, 40)
    assert resample_mm(volume, (0.5, 1.0, 2.0), 2.0).shape == (10, 15, 20)


def test_series_volume_is_a_size_cube_with_mm_voxels(tmp_path: Path) -> None:
    write_series(tmp_path / "s", shape=(20, 40, 30), pixel_spacing=(0.5, 0.5), slice_spacing=3.0)
    volume, geometry, crop = preprocess_with_crop(tmp_path / "s", 24, "mm")
    assert volume.shape == (24, 24, 24) and volume.dtype == np.float16
    assert crop is not None
    # the scan is 15 x 20 x 60 mm along RAS+ axes (columns, rows, slices for an axial series)
    assert crop.extent_mm == pytest.approx((15.0, 20.0, 60.0))
    # one crop voxel is 1 mm: the native step along each axis maps to spacing/1 mm crop voxels
    np.testing.assert_allclose(np.abs(crop.index_affine[:3, :3]).sum(axis=0), (0.5, 0.5, 3.0))
    assert geometry.spacing == (0.5, 0.5, 3.0)


def test_head_box_keeps_the_head_and_drops_a_detached_table() -> None:
    shape = (200, 220, 200)
    volume = np.zeros(shape, dtype=np.float32)
    head = ellipsoid(shape, (100, 120, 110), (65, 75, 60))
    volume[head] = 1000.0
    volume[:, 10:22, :] = 1000.0  # table slab as bright as the head, posterior, 23 mm away
    start, kept = head_box(volume, 160, 10)
    lo, hi = np.argwhere(head).min(axis=0), np.argwhere(head).max(axis=0)
    for axis in range(3):
        assert start[axis] <= lo[axis] and hi[axis] < start[axis] + 160
    assert start[1] >= 22  # the table lies wholly outside the box
    assert kept == pytest.approx(1.0)
    cropped = crop_box(volume, start, 160)
    assert (cropped == 1000.0).sum() == head.sum()


def test_tall_foreground_is_anchored_at_the_superior_end() -> None:
    shape = (120, 120, 300)
    volume = np.zeros(shape, dtype=np.float32)
    volume[30:90, 30:90, 20:280] = 1.0  # RAS+ axis 2 runs inferior to superior: 279 is the top
    start, kept = head_box(volume, 160, 10)
    assert start[2] == 280 + 10 - 160
    assert start[0] == (30 + 90 - 160) // 2 and start[1] == (30 + 90 - 160) // 2
    assert 0.5 < kept < 0.6  # 150 of 260 slices


def test_short_foreground_is_centred_on_the_superior_axis() -> None:
    volume = np.zeros((100, 100, 100), dtype=np.float32)
    volume[20:80, 20:80, 40:70] = 1.0
    start, _ = head_box(volume, 64, 10)
    assert start == ((20 + 80 - 64) // 2, (20 + 80 - 64) // 2, (40 + 70 - 64) // 2)


def test_in_plane_centre_ignores_foreground_below_the_box() -> None:
    volume = np.zeros((300, 120, 300), dtype=np.float32)
    volume[100:160, 30:90, 120:280] = 1.0  # head, left of the scan centre
    volume[20:280, 30:90, 0:120] = 1.0  # shoulders, touching the head, much wider
    start, _ = head_box(volume, 160, 10)
    assert start[2] == 280 + 10 - 160  # the box spans slices 130-289: no shoulder in it
    assert start[0] == (100 + 160 - 160) // 2  # centred on the head, not the shoulders


def test_padding_outside_the_scan_uses_the_minimum() -> None:
    volume = np.full((30, 30, 30), 5.0, dtype=np.float32)
    volume[0, 0, 0] = -1000.0
    cropped = crop_box(volume, (-10, -10, -10), 50)
    assert cropped.shape == (50, 50, 50)
    assert (cropped[:10] == -1000.0).all() and (cropped[40:] == -1000.0).all()
    assert cropped[10, 10, 10] == -1000.0 and cropped[20, 20, 20] == 5.0


def test_index_affine_maps_a_native_voxel_into_the_crop(tmp_path: Path) -> None:
    n, rows, cols = 30, 40, 36
    pixels = np.zeros((n, rows, cols), dtype=np.uint16)
    pixels[6:10, 12:22, 16:26] = 1000  # blob centred on slice 7.5, row 16.5, column 20.5
    write_series(
        tmp_path / "s",
        shape=(n, rows, cols),
        pixel_spacing=(0.8, 0.8),
        slice_spacing=1.5,
        orientation=(0, 1, 0, 0, 0, -1),  # sagittal, so the axes permute and flip
        pixels=pixels,
    )
    volume, _, crop = preprocess_with_crop(tmp_path / "s", 32, "mm")
    assert crop is not None
    point = crop.index_affine @ np.array([20.5, 16.5, 7.5, 1.0])
    values = volume.astype(np.float32)
    centre = np.argwhere(values > values.max() / 2).mean(axis=0)
    assert np.abs(point[:3] - centre).max() < 1.0, (point, centre)


def test_missing_pixel_spacing_assumes_a_typical_field_of_view(tmp_path: Path) -> None:
    for path in write_series(tmp_path / "s", shape=(20, 64, 48), pixel_spacing=(1.0, 1.0),
                             slice_spacing=2.0):  # fmt: skip
        ds = pydicom.dcmread(path)
        del ds.PixelSpacing
        ds.save_as(path)
    _, geometry, crop = preprocess_with_crop(tmp_path / "s", 24, "mm")
    assert geometry.spacing_assumed and geometry.source == "assumed-spacing"
    assert crop is not None
    step = ASSUMED_FOV_MM / 64
    assert crop.extent_mm == pytest.approx((48 * step, 64 * step, 40.0))


def test_defaults_stay_cube(tmp_path: Path) -> None:
    for fn in (preprocess.preprocess_series, preprocess.preprocess_with_geometry, predict_series):
        assert inspect.signature(fn).parameters["resample"].default == "cube"
    assert preprocess.SIZE == 128 and SUPERIOR_MARGIN_MM == 10.0
    write_series(tmp_path / "s", **KW)
    cube = preprocess_series(tmp_path / "s")
    assert np.array_equal(cube, preprocess_series(tmp_path / "s", 128, "cube"))
    volume, _, crop = preprocess_with_crop(tmp_path / "s")
    assert crop is None and np.array_equal(volume, cube)


def test_cache_name_per_resample() -> None:
    assert cache_name(128, "cube") == "b2-128"
    assert cache_name(160, "mm") == "b2mm-160"


def test_script_mm_writes_its_own_cache_with_crop_columns(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path
    write_series(root / "raw" / "series" / "1.2.1", **KW)
    (root / "raw" / "train.csv").write_text("SeriesInstanceUID\n1.2.1\n")
    (root / "raw" / ".complete").touch()
    monkeypatch.setenv("NV_STORE", str(root))
    monkeypatch.setenv("SLURM_CPUS_PER_TASK", "1")
    script = import_script("preprocess", monkeypatch)
    script.main(["--size", "16", "--resample", "mm"])
    cache = root / "cache" / "b2mm-16"
    assert np.load(cache / "1.2.1.npy").shape == (16, 16, 16)
    header = (cache / "index.csv").read_text().splitlines()[0].split(",")
    assert header == [*script.COLUMNS, *script.CROP_COLUMNS]
    row = dict(
        zip(header, (cache / "index.csv").read_text().splitlines()[1].split(","), strict=True)
    )
    assert float(row["extent_mm_z"]) == pytest.approx(40.0)
    assert 0 < float(row["kept_fraction"]) <= 1
    assert not (root / "cache" / "b2-16").exists()
    script.main(["--size", "16"])
    header = (root / "cache" / "b2-16" / "index.csv").read_text().splitlines()[0].split(",")
    assert header == list(script.COLUMNS)


def test_predict_series_mm_matches_the_cache_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from neurovision.infer import predict

    write_series(tmp_path / "s", **KW)
    seen: list[str] = []
    real = predict.preprocess_series
    monkeypatch.setattr(
        predict, "preprocess_series", lambda d, s, r: seen.append(r) or real(d, s, r)
    )
    probs = predict.predict_series(tmp_path / "s", _model(), np.full(14, 0.5), 16, "mm")
    assert probs.shape == (14,) and seen == ["mm"]


def test_shards_split_the_work_and_the_next_full_run_merges_them(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    uids = ["1.2.1", "1.2.2", "1.2.3"]
    for uid in uids:
        write_series(tmp_path / "raw" / "series" / uid, **KW)
    (tmp_path / "raw" / "train.csv").write_text("SeriesInstanceUID\n" + "\n".join(uids) + "\n")
    (tmp_path / "raw" / ".complete").touch()
    monkeypatch.setenv("NV_STORE", str(tmp_path))
    monkeypatch.setenv("SLURM_CPUS_PER_TASK", "1")
    script = import_script("preprocess", monkeypatch)
    cache = tmp_path / "cache" / "b2mm-16"
    script.main(["--size", "16", "--resample", "mm", "--shard", "1/2"])
    assert sorted(p.name for p in cache.glob("*.npy")) == ["1.2.2.npy"]
    assert not (cache / "index.csv").exists()
    script.main(["--size", "16", "--resample", "mm", "--shard", "0/2"])
    assert len((cache / "index-0of2.csv").read_text().splitlines()) == 3  # header, 1.2.1, 1.2.3
    first = (cache / "1.2.2.npy").stat().st_mtime_ns
    script.main(["--size", "16", "--resample", "mm"])
    lines = (cache / "index.csv").read_text().splitlines()
    assert lines[0].split(",") == [*script.COLUMNS, *script.CROP_COLUMNS]
    assert sorted(line.split(",")[0] for line in lines[1:]) == uids
    assert not list(cache.glob("index-*"))
    assert (cache / "1.2.2.npy").stat().st_mtime_ns == first
    with pytest.raises(SystemExit):
        script.main(["--size", "16", "--shard", "2/2"])
