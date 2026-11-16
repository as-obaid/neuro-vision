"""Full version stages 1 and 2: vessel boxes from the U-Net, and the 0.5 mm vessel-box cache."""

import json
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
import pytest
import torch

from neurovision.data.labels import LOCATIONS
from neurovision.data.points import box_affine
from neurovision.data.preprocess import preprocess_with_crop
from neurovision.data.vessels import LPS_FROM_RAS, SHI_DIR, SHI_ORIG
from tests.dicom_fakes import write_series
from tests.script_loader import import_script, load_script
from tests.test_vessels import SHI

RMCA = LOCATIONS.index("Right Middle Cerebral Artery")


# --- stage 1: boxes ------------------------------------------------------------------------


def test_box_row_gives_the_trimmed_box_in_cache_voxels_lps_and_mm() -> None:
    boxes = load_script("predict_vessel_boxes")
    seg = np.zeros((40, 40, 40), np.uint8)
    seg[10:20, 12:30, 5:25] = 4
    lps = np.eye(4)
    lps[:3, 3] = [100.0, -50.0, 7.0]
    row = boxes.box_row("u", 0, 11, seg, lps)
    assert row["n_vessel"] == 3600 and row["box_fold"] == 0 and row["model_epoch"] == 11
    assert (row["lo_x"], row["hi_x"], row["centre_x"]) == (10.0, 19.0, 14.5)
    assert (row["centre_lps_x"], row["centre_lps_y"], row["centre_lps_z"]) == (114.5, -29.5, 21.5)
    assert (row["extent_mm_x"], row["extent_mm_y"], row["extent_mm_z"]) == (10.0, 18.0, 20.0)
    assert set(row) == set(boxes.COLUMNS)


def test_box_row_without_a_tree_leaves_the_box_empty() -> None:
    boxes = load_script("predict_vessel_boxes")
    row = boxes.box_row("u", 0, 3, np.zeros((20, 20, 20), np.uint8), np.eye(4))
    assert row["n_vessel"] == 0 and row["centre_x"] == "" and row["centre_lps_z"] == ""


def test_done_uids_drops_a_row_cut_short(tmp_path: Path) -> None:
    boxes = load_script("predict_vessel_boxes")
    path = tmp_path / "b.csv"
    path.write_text("SeriesInstanceUID,n_vessel\na,1\nb,2\nc,")
    assert boxes.done_uids(path) == {"a", "b"}
    assert path.read_text().endswith("b,2\n")
    assert boxes.done_uids(tmp_path / "none.csv") == set()


def _series(root: Path, uid: str) -> tuple[np.ndarray, tuple[float, ...], object, np.ndarray]:
    """A DICOM series with a bright blob; (native blob centre, 1 mm offset, geometry, blob in c)."""
    n, rows, cols = 40, 48, 44
    pixels = np.zeros((n, rows, cols), dtype=np.uint16)
    pixels[18:22, 20:26, 15:21] = 1000  # slices 18-21, rows 20-25, columns 15-20
    write_series(
        root / "raw" / "series" / uid,
        shape=(n, rows, cols),
        pixel_spacing=(0.8, 0.7),
        slice_spacing=1.2,
        orientation=(0, 1, 0, 0, 0, -1),
        pixels=pixels,
    )
    _, geometry, crop = preprocess_with_crop(root / "raw" / "series" / uid, 160, "mm")
    assert crop is not None
    native = np.array([17.5, 22.5, 19.5, 1.0])
    offset = tuple(float(np.round(o, 1)) for o in crop.offset_mm)
    return native, offset, geometry, (box_affine(geometry, offset) @ native)[:3]


def test_box_inputs_maps_cache_voxels_to_patient_lps(tmp_path: Path) -> None:
    boxes = load_script("predict_vessel_boxes")
    native, offset, geometry, blob = _series(tmp_path, "u")
    cache = tmp_path / "cache"
    cache.mkdir()
    np.save(cache / "u.npy", np.zeros((8, 8, 8), np.float16))
    offsets = pd.DataFrame([offset], index=pd.Index(["u"]), columns=list(boxes.OFFSETS))
    x, lps, i = boxes.BoxInputs(cache, tmp_path / "raw" / "series", offsets)[0]
    assert x.shape == (1, 8, 8, 8) and i == 0
    want = geometry.affine @ native  # type: ignore[attr-defined]
    np.testing.assert_allclose((lps.numpy() @ np.append(blob, 1.0))[:3], want[:3], atol=1e-6)


# --- stage 2: the vessel-box cache ---------------------------------------------------------


def _store(root: Path) -> np.ndarray:
    """Series "a" (an R MCA aneurysm on the blob, with a box) and "b" (no tree predicted)."""
    rows, centres = [], []
    for uid in ("a", "b"):
        native, offset, geometry, blob = _series(root, uid)
        rows.append(
            {
                "SeriesInstanceUID": uid,
                "status": "ok",
                "reason": "",
                **dict(zip(("offset_mm_x", "offset_mm_y", "offset_mm_z"), offset, strict=True)),
            }
        )
        centres.append(blob if uid == "a" else [np.nan] * 3)
    rows.append({"SeriesInstanceUID": "c", "status": "excluded", "reason": "slice gap"})
    cache = root / "cache" / "b2mm-160"
    cache.mkdir(parents=True)
    pd.DataFrame(rows).to_csv(cache / "index.csv", index=False)
    labels = root / "labels"
    labels.mkdir(exist_ok=True)
    boxes = pd.DataFrame(centres, columns=["centre_x", "centre_y", "centre_z"])
    boxes.insert(0, "SeriesInstanceUID", ["a", "b"])
    boxes.insert(1, "box_fold", 0)
    boxes.to_csv(labels / "vessel-boxes-fold0.csv", index=False)
    shi = root / SHI_DIR
    (shi / SHI_ORIG).mkdir(parents=True)
    (shi / "labels.json").write_text(json.dumps(SHI))
    mask = np.zeros(geometry.size, np.uint8)  # type: ignore[attr-defined]
    mask[15:21, 20:26, 18:22] = SHI["Right Middle Cerebral Artery Aneurysm"]
    affine = np.linalg.inv(LPS_FROM_RAS) @ geometry.affine  # type: ignore[attr-defined]
    nib.save(nib.Nifti1Image(mask, affine), shi / SHI_ORIG / "a.nii.gz")
    blob_a = np.asarray(centres[0])
    pd.DataFrame(
        [
            {
                "SeriesInstanceUID": "a",
                "location": LOCATIONS[RMCA],
                "x": blob_a[0],
                "y": blob_a[1],
                "z": blob_a[2],
            }
        ]
    ).to_csv(labels / "points-b2mm-160.csv", index=False)
    return blob_a


def test_build_cuts_each_box_maps_the_mask_and_resumes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("NV_STORE", str(tmp_path))
    blob = _store(tmp_path)
    script = import_script("build_vessel_box_cache", monkeypatch)
    script.build(tmp_path, 0, 24, 1, None, "vbox05-24")
    cache = tmp_path / "cache" / "vbox05-24"
    index = pd.read_csv(cache / "index.csv").set_index("SeriesInstanceUID")
    assert set(index.index) == {"a", "b", "c"}  # every series has a row
    assert index.loc["c", "status"] == "excluded" and "slice gap" in index.loc["c", "reason"]
    index = index.drop("c")
    assert (index["status"] == "ok").all()
    assert index.loc["a", "fallback"] == 0 and index.loc["b", "fallback"] == 1
    assert (index["box_fold"] == 0).all()
    np.testing.assert_allclose(
        index.loc["b", ["centre_x", "centre_y", "centre_z"]].to_numpy(float), np.round(blob, 2)
    )  # the median of the series with a tree
    assert index.loc["a", "seg"] == "written" and index.loc["b", "seg"] == "none"
    volume = np.load(cache / "a.npy")
    assert volume.shape == (24, 24, 24) and volume.dtype == np.float16
    seg = np.load(tmp_path / "cache" / "vbox05-24-vessels" / "a.npz")["seg"]
    assert set(np.unique(seg)) == {0, 1 + len(LOCATIONS) + RMCA}
    np.testing.assert_allclose(np.argwhere(seg > 0).mean(axis=0), [11.5] * 3, atol=0.6)
    bright = volume.astype(np.float32)
    found = np.argwhere(bright > (bright.max() + bright.min()) / 2).mean(axis=0)
    np.testing.assert_allclose(found, [11.5] * 3, atol=0.6)  # the image sits on its mask
    points = script.write_points(tmp_path, "vbox05-24", 24, 0.5)
    # the box is cut at the centre the index stores (0.01 voxel), and the points use the same
    np.testing.assert_allclose(points[["x", "y", "z"]].to_numpy(float), [[11.5] * 3], atol=0.011)
    assert (tmp_path / "labels" / "points-vbox05-24.csv").exists()
    script.check(tmp_path, 24, "vbox05-24")
    assert "n 1, none in mask 0, outside the box 0" in capsys.readouterr().out
    before = (cache / "a.npy").stat().st_mtime_ns
    script.build(tmp_path, 0, 24, 1, None, "vbox05-24")  # nothing left to do
    assert (cache / "a.npy").stat().st_mtime_ns == before
    assert len(pd.read_csv(cache / "index.csv")) == 3


def test_the_1mm_recut_matches_the_head_crop_shifted_and_moves_the_points(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """--spacing 1.0: the same 1 mm grid as b2mm-160, shifted by a whole number of voxels."""
    monkeypatch.setenv("NV_STORE", str(tmp_path))
    blob = _store(tmp_path)
    size, shift = 40, np.array([3.0, -2.0, 1.0])
    centre = (size - 1) / 2 + np.round(blob - (size - 1) / 2) + shift  # whole-voxel offset
    boxes = pd.read_csv(tmp_path / "labels" / "vessel-boxes-fold0.csv")
    boxes.loc[boxes["SeriesInstanceUID"] == "a", ["centre_x", "centre_y", "centre_z"]] = centre
    boxes.to_csv(tmp_path / "labels" / "vessel-boxes-fold0.csv", index=False)
    script = import_script("build_vessel_box_cache", monkeypatch)
    script.build(tmp_path, 0, size, 1, {"a"}, "vc1mm-40", 1.0)
    assert np.load(tmp_path / "cache" / "vc1mm-40" / "a.npy").shape == (size,) * 3
    # the pixels, before each box's own clip and z-score: same grid, shifted by `start`
    from neurovision.data import preprocess, vbox

    raw = lambda v: v.astype(np.float32)  # noqa: E731
    monkeypatch.setattr(preprocess, "normalise", raw)
    monkeypatch.setattr(vbox, "normalise", raw)
    series = tmp_path / "raw" / "series" / "a"
    _, _, crop = preprocess.preprocess_with_crop(series, 160, "mm")
    head = preprocess.preprocess_with_crop(series, 160, "mm")[0]
    assert crop is not None
    cube = vbox.vessel_box(series, crop.offset_mm, centre, size, 1.0)[0]
    start = (centre - (size - 1) / 2).astype(int)  # cube voxel v = head-crop voxel v + start
    old = head[tuple(slice(s, s + size) for s in start)]
    # the head crop blurs with a rounded zoom (31 of 44 columns), the re-cut with the exact one
    assert np.abs(cube - old).max() < 0.03 * 1000 and np.abs(cube - old).mean() < 1.0
    points = script.write_points(tmp_path, "vc1mm-40", size, 1.0)
    np.testing.assert_allclose(points[["x", "y", "z"]].to_numpy(float)[0], blob - start, atol=1e-6)
    assert (tmp_path / "labels" / "points-vc1mm-40.csv").exists()


def test_the_1mm_recut_moves_whole_voxels_and_keeps_the_head_crop_without_a_tree(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """--spacing 1.0: a vessel centre between voxels snaps to a whole-voxel move; series "b"
    (no predicted tree) is not moved to the median centre but keeps its head crop: zero shift."""
    monkeypatch.setenv("NV_STORE", str(tmp_path))
    blob = _store(tmp_path)
    size, head = 160, 79.5
    cols = ["centre_x", "centre_y", "centre_z"]
    boxes = pd.read_csv(tmp_path / "labels" / "vessel-boxes-fold0.csv")
    boxes.loc[boxes["SeriesInstanceUID"] == "a", cols] = head + np.array([10.3, -7.6, 0.5])
    boxes.to_csv(tmp_path / "labels" / "vessel-boxes-fold0.csv", index=False)
    script = import_script("build_vessel_box_cache", monkeypatch)
    script.build(tmp_path, 0, size, 1, None, "vc1mm-160", 1.0)
    index = pd.read_csv(tmp_path / "cache" / "vc1mm-160" / "index.csv").set_index(
        "SeriesInstanceUID"
    )
    assert (index.loc[["a", "b"], "status"] == "ok").all()
    assert index.loc["a", "fallback"] == 0 and index.loc["b", "fallback"] == 1
    move = np.array([10.0, -8.0, 1.0])  # nearest whole voxels, halves rounded up
    np.testing.assert_array_equal(index.loc["a", cols].to_numpy(float), head + move)
    np.testing.assert_array_equal(index.loc["b", cols].to_numpy(float), [head] * 3)
    points = script.write_points(tmp_path, "vc1mm-160", size, 1.0)
    np.testing.assert_allclose(points[["x", "y", "z"]].to_numpy(float)[0], blob - move, atol=1e-9)
    # "b" is the head crop itself: the same pixels before each box's own clip and z-score
    from neurovision.data import preprocess, vbox

    raw = lambda v: v.astype(np.float32)  # noqa: E731
    monkeypatch.setattr(preprocess, "normalise", raw)
    monkeypatch.setattr(vbox, "normalise", raw)
    series = tmp_path / "raw" / "series" / "b"
    head_crop, _, crop = preprocess.preprocess_with_crop(series, size, "mm")
    assert crop is not None
    cube = vbox.vessel_box(series, crop.offset_mm, index.loc["b", cols].to_numpy(float), size, 1.0)
    diff = np.abs(cube[0] - head_crop)
    assert diff.max() < 0.03 * 1000 and diff.mean() < 1.0


def test_crop_centres_round_halves_up_and_keep_the_head_crop_without_a_tree() -> None:
    script = load_script("build_vessel_box_cache")
    boxes = pd.DataFrame(
        {
            "centre_x": [79.5, 80.0, np.nan],
            "centre_y": [60.2, 79.0, np.nan],
            "centre_z": [100.7] * 3,
        }
    )
    boxes.loc[2, "centre_z"] = np.nan
    filled, fallback = script.crop_centres(boxes, 160)
    np.testing.assert_array_equal(
        filled.to_numpy(), [[79.5, 60.5, 100.5], [80.5, 79.5, 100.5], [79.5, 79.5, 79.5]]
    )
    assert fallback.tolist() == [0, 0, 1]
    # a smaller box: the start (centre - 19.5) stays a whole number of voxels from the head crop's
    np.testing.assert_array_equal(script.crop_centres(boxes, 40)[0].iloc[2].to_numpy(), [79.5] * 3)


def test_seg_only_rewrites_the_masks_of_the_listed_series_and_nothing_else(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("NV_STORE", str(tmp_path))
    _store(tmp_path)
    script = import_script("build_vessel_box_cache", monkeypatch)
    script.build(tmp_path, 0, 24, 1, None, "vbox05-24")
    cache = tmp_path / "cache" / "vbox05-24"
    seg_path = tmp_path / "cache" / "vbox05-24-vessels" / "a.npz"
    good = np.load(seg_path)["seg"]
    np.savez_compressed(seg_path, seg=np.zeros_like(good))  # a mask mapped the old, wrong way
    volume = (cache / "a.npy").stat().st_mtime_ns
    index = (cache / "index.csv").read_text()
    counts = script.rebuild_segs(tmp_path, 24, "vbox05-24", 0.5, {"a", "b", "c"}, 1)
    np.testing.assert_array_equal(np.load(seg_path)["seg"], good)
    assert counts == {"written": 1, "none": 1}  # b has no mask; c is not an ok row
    assert (cache / "a.npy").stat().st_mtime_ns == volume
    assert (cache / "index.csv").read_text() == index


def test_build_retries_error_rows_and_drops_a_torn_row(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("NV_STORE", str(tmp_path))
    _store(tmp_path)
    script = import_script("build_vessel_box_cache", monkeypatch)
    cache = tmp_path / "cache" / "vbox05-24"
    cache.mkdir(parents=True)
    header = ",".join(script.COLUMNS)
    (cache / "index.csv").write_text(
        f"{header}\na,excluded,error: worker died,0,1,2,3,0,,1.0\nb,ok,,0,1,2"
    )
    script.build(tmp_path, 0, 24, 1, None, "vbox05-24")
    index = pd.read_csv(cache / "index.csv").set_index("SeriesInstanceUID").drop("c")
    assert sorted(index.index) == ["a", "b"] and (index["status"] == "ok").all()


def test_centres_need_one_series_with_a_tree() -> None:
    script = load_script("build_vessel_box_cache")
    boxes = pd.DataFrame({"centre_x": [np.nan], "centre_y": [np.nan], "centre_z": [np.nan]})
    with pytest.raises(SystemExit, match="no series"):
        script.centres(boxes)


def test_unet_checkpoint_needs_a_finished_run(tmp_path: Path) -> None:
    boxes = load_script("predict_vessel_boxes")
    from neurovision.models.unet import R3DUNet

    torch.save({"model": R3DUNet().state_dict(), "epoch": 4}, tmp_path / "last.pt")
    with pytest.raises(SystemExit, match="not finished"):
        boxes.load_unet(tmp_path, allow_unfinished=False)
    model, epoch = boxes.load_unet(tmp_path, allow_unfinished=True)
    assert isinstance(model, R3DUNet) and epoch == 4


# --- Slurm ---------------------------------------------------------------------------------

SLURM = Path(__file__).resolve().parents[1] / "slurm"


@pytest.mark.parametrize("name", ["vessel_boxes.sbatch", "vbox_cache.sbatch"])
@pytest.mark.parametrize("args", [("x",), ("-1",), ()])
def test_vessel_box_jobs_refuse_a_bad_fold(name: str, args: tuple[str, ...]) -> None:
    import subprocess

    done = subprocess.run(["bash", str(SLURM / name), *args], capture_output=True, text=True)
    assert done.returncode == 1 and done.stderr.startswith("refused:")


def test_vessel_box_jobs_run_their_script_for_an_hour() -> None:
    boxes = (SLURM / "vessel_boxes.sbatch").read_text()
    cache = (SLURM / "vbox_cache.sbatch").read_text()
    assert "--partition=sharing" in boxes and "--gres=gpu:h200:1" in boxes
    assert "scripts/predict_vessel_boxes.py" in boxes
    assert "--partition=short" in cache and "--gres" not in cache
    assert "scripts/build_vessel_box_cache.py" in cache
    assert "--time=01:00:00" in boxes and "--time=01:00:00" in cache
