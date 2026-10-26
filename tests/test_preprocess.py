from collections.abc import Sequence
from concurrent.futures import Future
from concurrent.futures.process import BrokenProcessPool
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
import pandas as pd
import pydicom
import pytest
import torch

from neurovision.data.dicom import read_header
from neurovision.data.preprocess import (
    GeometryError,
    antialias,
    preprocess_series,
    preprocess_with_geometry,
    resample,
    series_geometry,
)
from tests.dicom_fakes import fake_pixels, write_series
from tests.script_loader import import_script

SHAPE = (20, 16, 16)
KW = {"shape": SHAPE, "pixel_spacing": (1.0, 1.0), "slice_spacing": 2.0}


def edit(path: Path, **tags: object) -> None:
    ds = pydicom.dcmread(path)
    for name, value in tags.items():
        setattr(ds, name, value)
    ds.save_as(path)


def test_shape_dtype_contiguity_and_normalisation(tmp_path: Path) -> None:
    write_series(tmp_path / "s", **KW)
    vol = preprocess_series(tmp_path / "s")
    assert vol.shape == (128, 128, 128)
    assert vol.dtype == np.float16
    assert vol.flags["C_CONTIGUOUS"]
    assert abs(float(vol.astype(np.float32).mean())) < 0.02
    assert abs(float(vol.astype(np.float32).std()) - 1.0) < 0.1


def cube_series(folder: Path, orientation: Sequence[float], *, stripped: bool = False) -> None:
    """Fill the Left, Anterior, Superior octant (LPS: x > 0, y < 0, z > 0) of a 60 mm box."""
    n, rows, cols = 20, 20, 20
    step = 3.0
    x_dir, y_dir = np.array(orientation[:3]), np.array(orientation[3:])
    normal = np.cross(x_dir, y_dir)
    start = -(x_dir * (cols - 1) + y_dir * (rows - 1) + normal * (n - 1)) * step / 2
    k, j, i = np.meshgrid(range(n), range(rows), range(cols), indexing="ij")
    points = (
        start
        + k[..., None] * step * normal
        + j[..., None] * step * y_dir
        + i[..., None] * step * x_dir
    )
    bright = (points[..., 0] > 6) & (points[..., 1] < -6) & (points[..., 2] > 6)
    positions = [tuple(start + normal * step * s) for s in range(n)]
    write_series(
        folder,
        shape=(n, rows, cols),
        pixel_spacing=(step, step),
        slice_spacing=step,
        orientation=orientation,
        positions=positions,
        pixels=(bright * 1000).astype(np.uint16),
        multiframe=stripped,
        top_level_geometry=not stripped,
        frame_orientation=not stripped,
    )


@pytest.mark.parametrize(
    "orientation",
    [(1, 0, 0, 0, 1, 0), (0, 1, 0, 0, 0, -1), (1, 0, 0, 0, 0, -1)],
    ids=["axial", "sagittal", "coronal"],
)
def test_cube_lands_in_left_anterior_superior_octant(
    tmp_path: Path, orientation: Sequence[float]
) -> None:
    cube_series(tmp_path / "s", orientation)
    vol = preprocess_series(tmp_path / "s")
    centre = np.argwhere(vol > 0).mean(axis=0)
    # RAS+: axis 0 Left to Right, axis 1 Posterior to Anterior, axis 2 Inferior to Superior
    assert centre[0] < 64 - 10
    assert centre[1] > 64 + 10
    assert centre[2] > 64 + 10


@pytest.mark.parametrize(
    "orientation",
    [(1, 0, 0, 0, 1, 0), (0, 1, 0, 0, 0, -1), (1, 0, 0, 0, 0, -1)],
    ids=["axial", "sagittal", "coronal"],
)
def test_stripped_multiframe_assumes_standard_orientation(
    tmp_path: Path, orientation: Sequence[float]
) -> None:
    cube_series(tmp_path / "s", orientation, stripped=True)
    vol, geometry = preprocess_with_geometry(tmp_path / "s")
    assert geometry.source == "assumed-orientation"
    centre = np.argwhere(vol > 0).mean(axis=0)
    assert centre[0] < 64 - 10
    assert centre[1] > 64 + 10
    assert centre[2] > 64 + 10


def test_stripped_series_with_reversed_frames_keeps_slice_direction(tmp_path: Path) -> None:
    cube_series(tmp_path / "a", (1, 0, 0, 0, 1, 0), stripped=True)
    (path,) = (tmp_path / "a").glob("*.dcm")
    ds = pydicom.dcmread(path)
    pixels = ds.pixel_array[::-1].copy()
    ds.PerFrameFunctionalGroupsSequence = list(ds.PerFrameFunctionalGroupsSequence)[::-1]
    ds.PixelData = pixels.tobytes()
    ds.save_as(path)
    vol = preprocess_series(tmp_path / "a")
    assert np.argwhere(vol > 0).mean(axis=0)[2] > 64 + 10


def test_rejects_oblique_step_without_orientation(tmp_path: Path) -> None:
    positions = [(0.0, 3.0 * i, 3.0 * i) for i in range(20)]
    write_series(
        tmp_path / "s",
        **KW,
        positions=positions,
        multiframe=True,
        top_level_geometry=False,
        frame_orientation=False,
    )
    with pytest.raises(GeometryError, match="oblique slice direction"):
        preprocess_series(tmp_path / "s")


def test_file_order_does_not_matter(tmp_path: Path) -> None:
    write_series(tmp_path / "a", **KW)
    write_series(tmp_path / "b", **KW)
    files = sorted((tmp_path / "b").glob("*.dcm"))
    for n, path in enumerate(files):
        path.rename(path.with_name(f"z{(n * 7) % len(files):04d}.dcm"))
    assert np.array_equal(preprocess_series(tmp_path / "a"), preprocess_series(tmp_path / "b"))


def test_slope_and_intercept_are_applied(tmp_path: Path) -> None:
    pixels = fake_pixels(SHAPE)
    write_series(tmp_path / "a", **KW, pixels=pixels)
    for path in sorted((tmp_path / "a").glob("*.dcm"))[10:]:
        edit(path, RescaleSlope=2.0, RescaleIntercept=500.0)
    expected = pixels.copy()
    expected[10:] = pixels[10:] * 2 + 500
    write_series(tmp_path / "b", **KW, pixels=expected)
    assert np.array_equal(preprocess_series(tmp_path / "a"), preprocess_series(tmp_path / "b"))


def test_rle_decodes(tmp_path: Path) -> None:
    write_series(tmp_path / "a", **KW)
    write_series(tmp_path / "b", **KW, rle=True)
    assert np.array_equal(preprocess_series(tmp_path / "a"), preprocess_series(tmp_path / "b"))


def test_monochrome1_is_inverted(tmp_path: Path) -> None:
    write_series(tmp_path / "a", **KW)
    write_series(tmp_path / "b", **KW, photometric="MONOCHROME1")
    a = preprocess_series(tmp_path / "a").astype(np.float32)
    b = preprocess_series(tmp_path / "b").astype(np.float32)
    assert np.allclose(a, -b, atol=2e-2)


def test_multiframe_matches_single_frame_and_reads_per_frame_geometry(tmp_path: Path) -> None:
    write_series(tmp_path / "single", **KW)
    write_series(tmp_path / "multi", **KW, multiframe=True, top_level_geometry=False)
    vol, geometry = preprocess_with_geometry(tmp_path / "multi")
    assert np.array_equal(preprocess_series(tmp_path / "single"), vol)
    assert geometry.source == "per-frame"
    assert preprocess_with_geometry(tmp_path / "single")[1].source == "top-level"


def test_header_only_geometry(tmp_path: Path) -> None:
    files = write_series(tmp_path / "s", **KW)
    geometry = series_geometry([read_header(p) for p in files])
    assert geometry.size == (16, 16, 20)
    assert geometry.spacing == (1.0, 1.0, 2.0)


def test_rejects_single_slice_gap(tmp_path: Path) -> None:
    positions = [(0.0, 0.0, 2.0 * i) for i in range(21) if i != 8]
    write_series(tmp_path / "s", **{**KW, "shape": (20, 16, 16)}, positions=positions)
    with pytest.raises(GeometryError, match="slice gap"):
        preprocess_series(tmp_path / "s")


def test_rejects_mixed_orientation(tmp_path: Path) -> None:
    files = write_series(tmp_path / "s", **KW)
    edit(files[3], ImageOrientationPatient=[0, 1, 0, 1, 0, 0])
    with pytest.raises(GeometryError, match="mixed orientation"):
        preprocess_series(tmp_path / "s")


def test_rejects_too_few_slices(tmp_path: Path) -> None:
    write_series(tmp_path / "s", **{**KW, "shape": (15, 16, 16)})
    with pytest.raises(GeometryError, match="too few slices"):
        preprocess_series(tmp_path / "s")


def test_rejects_missing_position_in_both_places(tmp_path: Path) -> None:
    write_series(tmp_path / "s", **KW, top_level_geometry=False)
    with pytest.raises(GeometryError, match="missing position"):
        preprocess_series(tmp_path / "s")


def test_rejects_multiframe_without_positions(tmp_path: Path) -> None:
    (path,) = write_series(tmp_path / "s", **KW, multiframe=True)
    ds = pydicom.dcmread(path)
    for group in ds.PerFrameFunctionalGroupsSequence:
        del group.PlanePositionSequence
    ds.save_as(path)
    with pytest.raises(GeometryError, match="multi-frame without positions"):
        preprocess_series(tmp_path / "s")


def test_rejects_colour(tmp_path: Path) -> None:
    write_series(tmp_path / "s", **KW, samples_per_pixel=3)
    with pytest.raises(GeometryError, match="not single-sample grayscale"):
        preprocess_series(tmp_path / "s")


def test_rejects_non_finite_pixels(tmp_path: Path) -> None:
    write_series(tmp_path / "s", **KW, rescale_slope=1e38)
    with pytest.raises(GeometryError, match="non-finite pixels"):
        preprocess_series(tmp_path / "s")


def test_script_records_unexpected_errors_and_continues(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "store"
    series = root / "raw" / "series"
    write_series(series / "1.2.1", **KW)
    files = write_series(series / "1.2.2", **KW)
    files[4].write_bytes(b"not dicom")
    pd.DataFrame({"SeriesInstanceUID": ["1.2.2", "1.2.1"]}).to_csv(
        root / "raw" / "train.csv", index=False
    )
    (root / "raw" / ".complete").touch()
    monkeypatch.setenv("NV_STORE", str(root))
    monkeypatch.setenv("SLURM_CPUS_PER_TASK", "2")
    import_script("preprocess", monkeypatch).main([])
    index = pd.read_csv(root / "cache" / "b2-128" / "index.csv").set_index("SeriesInstanceUID")
    assert index.loc["1.2.1", "status"] == "ok"
    assert index.loc["1.2.2", "status"] == "excluded"
    assert str(index.loc["1.2.2", "reason"]).startswith("error: ")


def test_script_writes_cache_and_index_and_resumes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    root = tmp_path / "store"
    series = root / "raw" / "series"
    write_series(series / "1.2.1", **KW)
    write_series(series / "1.2.2", **KW)
    write_series(series / "1.2.3", **{**KW, "shape": (15, 16, 16)})
    pd.DataFrame({"SeriesInstanceUID": ["1.2.1", "1.2.2", "1.2.3"], "x": [0, 1, 0]}).to_csv(
        root / "raw" / "train.csv", index=False
    )
    (root / "raw" / ".complete").touch()
    monkeypatch.setenv("NV_STORE", str(root))
    monkeypatch.setenv("SLURM_CPUS_PER_TASK", "2")
    script = import_script("preprocess", monkeypatch)

    script.main([])
    cache = root / "cache" / "b2-128"
    index = pd.read_csv(cache / "index.csv", dtype=str).set_index("SeriesInstanceUID")
    assert index.loc["1.2.1", "status"] == "ok"
    assert index.loc["1.2.1", "geometry_source"] == "top-level"
    assert index.loc["1.2.3", "status"] == "excluded"
    assert index.loc["1.2.3", "reason"] == "too few slices"
    assert sorted(p.name for p in cache.glob("*.npy")) == ["1.2.1.npy", "1.2.2.npy"]
    assert np.load(cache / "1.2.1.npy").shape == (128, 128, 128)
    assert "ok 2" in capsys.readouterr().out

    first = (cache / "1.2.1.npy").stat().st_mtime_ns
    write_series(series / "1.2.4", **KW)
    pd.DataFrame({"SeriesInstanceUID": ["1.2.1", "1.2.2", "1.2.3", "1.2.4"]}).to_csv(
        root / "raw" / "train.csv", index=False
    )
    script.main([])
    assert (cache / "1.2.1.npy").stat().st_mtime_ns == first
    assert len(pd.read_csv(cache / "index.csv")) == 4
    assert not list(cache.glob("*.tmp"))


def test_spacing_default_is_recorded_separately_from_orientation(tmp_path: Path) -> None:
    for path in write_series(tmp_path / "s", **KW):
        ds = pydicom.dcmread(path)
        del ds.PixelSpacing
        ds.save_as(path)
    assert preprocess_with_geometry(tmp_path / "s")[1].source == "assumed-spacing"


def test_rejects_frame_count_mismatch(tmp_path: Path) -> None:
    files = write_series(tmp_path / "s", **KW)
    ds = pydicom.dcmread(files[5])
    ds.NumberOfFrames = 2
    ds.PixelData = ds.PixelData * 2
    ds.save_as(files[5])
    with (
        pytest.warns(UserWarning, match="sufficient to contain 2 frames"),
        pytest.raises(GeometryError, match="frame count mismatch"),
    ):
        preprocess_series(tmp_path / "s")


def test_one_voxel_sheet_survives_a_fourfold_downsample() -> None:
    volume = np.zeros((512, 16, 16), dtype=np.float32)
    volume[255] = 1.0  # fourfold along axis 0: plain trilinear samples between voxels and misses it
    peak = resample(volume).max()
    assert peak > 0.1


def test_antialias_leaves_upsampled_axes_unchanged() -> None:
    volume = torch.rand(16, 16, 16)
    assert torch.equal(antialias(volume), volume)


def test_antialias_blurs_only_the_downsampled_axis() -> None:
    volume = torch.rand(512, 16, 16)
    volume[:] = volume[0]  # constant along axis 0, so blurring that axis changes nothing
    assert torch.allclose(antialias(volume), volume, atol=1e-6)
    volume = torch.rand(16, 512, 16)
    assert not torch.allclose(antialias(volume), volume, atol=1e-3)


def make_raw(root: Path, uids: list[str], **kw: object) -> Path:
    series = root / "raw" / "series"
    for uid in uids:
        write_series(series / uid, **{**KW, **kw})
    pd.DataFrame({"SeriesInstanceUID": uids}).to_csv(root / "raw" / "train.csv", index=False)
    (root / "raw" / ".complete").touch()
    return root / "cache" / "b2-128" / "index.csv"


def run_script(root: Path, monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    monkeypatch.setenv("NV_STORE", str(root))
    monkeypatch.setenv("SLURM_CPUS_PER_TASK", "2")
    script = import_script("preprocess", monkeypatch)
    script.main([])
    return script


def test_script_refuses_without_complete_marker(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    make_raw(tmp_path, ["1.2.1"])
    (tmp_path / "raw" / ".complete").unlink()
    monkeypatch.setenv("NV_STORE", str(tmp_path))
    with pytest.raises(SystemExit, match=r"\.complete"):
        import_script("preprocess", monkeypatch).main([])
    assert not (tmp_path / "cache").exists()


def test_resume_retries_error_rows_and_keeps_geometry_exclusions(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    index = make_raw(tmp_path, ["1.2.1", "1.2.2"])
    index.parent.mkdir(parents=True)
    script = import_script("preprocess", monkeypatch)
    rows = [
        {"SeriesInstanceUID": "1.2.1", "status": "excluded", "reason": "error: OSError"},
        {"SeriesInstanceUID": "1.2.2", "status": "excluded", "reason": "slice gap"},
    ]
    pd.DataFrame(rows, columns=list(script.COLUMNS)).to_csv(index, index=False)
    run_script(tmp_path, monkeypatch)
    table = pd.read_csv(index, dtype=str, keep_default_na=False).set_index("SeriesInstanceUID")
    assert len(table) == 2
    assert table.loc["1.2.1", "status"] == "ok"
    assert table.loc["1.2.2", "reason"] == "slice gap"
    assert (index.parent / "1.2.1.npy").exists()
    assert not (index.parent / "1.2.2.npy").exists()


@pytest.mark.parametrize("content", ["", None], ids=["empty", "header-only"])
def test_index_gets_exactly_one_header(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, content: str | None
) -> None:
    index = make_raw(tmp_path, ["1.2.1"])
    index.parent.mkdir(parents=True)
    script = import_script("preprocess", monkeypatch)
    index.write_text("" if content == "" else ",".join(script.COLUMNS) + "\n")
    run_script(tmp_path, monkeypatch)
    lines = index.read_text().splitlines()
    assert sum(line.startswith("SeriesInstanceUID,") for line in lines) == 1
    assert len(lines) == 2


def test_resume_tolerates_a_trailing_partial_line(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    index = make_raw(tmp_path, ["1.2.1", "1.2.2"])
    index.parent.mkdir(parents=True)
    script = import_script("preprocess", monkeypatch)
    done = {"SeriesInstanceUID": "1.2.1", "status": "excluded", "reason": "slice gap"}
    pd.DataFrame([done], columns=list(script.COLUMNS)).to_csv(index, index=False)
    with index.open("a") as handle:
        handle.write("1.2.2,ok,,top-lev")
    run_script(tmp_path, monkeypatch)
    table = pd.read_csv(index, dtype=str, keep_default_na=False).set_index("SeriesInstanceUID")
    assert table["status"].to_dict() == {"1.2.1": "excluded", "1.2.2": "ok"}
    assert table.loc["1.2.2", "size_z"] == "20"
    assert (index.parent / "1.2.2.npy").exists()


class KillingPool:
    """In-process stand-in for the pool: the first run of the 'killer' series breaks the pool."""

    created = 0
    killer = "1.2.2"
    killed = False

    def __init__(self, *_: object) -> None:
        type(self).created += 1

    def __enter__(self) -> "KillingPool":
        return self

    def __exit__(self, *_: object) -> None:
        return None

    def submit(self, fn: Any, uid: str, *rest: object) -> Future[Any]:
        future: Future[Any] = Future()
        if uid == self.killer and not type(self).killed:
            type(self).killed = True
            future.set_exception(BrokenProcessPool("worker died"))
        else:
            future.set_result(fn(uid, *rest))
        return future


def test_dead_worker_marks_series_and_the_run_continues(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    index = make_raw(tmp_path, ["1.2.1", "1.2.2", "1.2.3"])
    monkeypatch.setenv("NV_STORE", str(tmp_path))
    monkeypatch.setenv("SLURM_CPUS_PER_TASK", "1")
    script = import_script("preprocess", monkeypatch)
    monkeypatch.setattr(script, "ProcessPoolExecutor", KillingPool)
    script.main([])
    table = pd.read_csv(index, dtype=str, keep_default_na=False).set_index("SeriesInstanceUID")
    assert table.loc["1.2.2", "reason"] == "error: worker died"
    assert table.loc["1.2.1", "status"] == "ok"
    assert table.loc["1.2.3", "status"] == "ok"
    assert KillingPool.created >= 2
    assert "ok 2, excluded 1" in capsys.readouterr().out
