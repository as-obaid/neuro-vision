from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
import pytest

from neurovision.data.labels import TARGETS
from tests.script_loader import import_script, load_script

MASKS = "labels/shi2026-masks"
FOLDERS = ("labelsTr_26classes_in_orig_space", "labelsTr_multiframe_cor_sag")


def write_mask(store: Path, folder: str, uid: str, classes: tuple[int, ...]) -> None:
    out = store / MASKS / folder
    out.mkdir(parents=True, exist_ok=True)
    data = np.zeros((4, 4, 4), dtype=np.uint8)
    for i, c in enumerate(classes):
        data[i, 0, 0] = c
    nib.save(nib.Nifti1Image(data, np.eye(4)), out / f"{uid}.nii.gz")


def derive(store: Path, monkeypatch: pytest.MonkeyPatch, expected: dict[str, int]) -> int:
    module = import_script("derive_labels", monkeypatch)
    monkeypatch.setattr(module, "EXPECTED_COUNTS", expected)
    monkeypatch.setenv("NV_STORE", str(store))
    monkeypatch.delenv("SLURM_CPUS_PER_TASK", raising=False)
    try:
        module.main([])
    except SystemExit as exc:
        return 1 if isinstance(exc.code, str) else int(exc.code or 0)
    return 0


def counts(a: int, b: int) -> dict[str, int]:
    return {FOLDERS[0]: a, FOLDERS[1]: b}


def test_classes_map_to_targets_and_aneurysm_present_is_or(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    write_mask(tmp_path, FOLDERS[0], "b", (3, 14, 26))  # vessel class 3 is ignored
    write_mask(tmp_path, FOLDERS[1], "a", (20,))
    write_mask(tmp_path, FOLDERS[0], "c", (1, 13))  # vessels only
    assert derive(tmp_path, monkeypatch, counts(2, 1)) == 0
    out = pd.read_csv(tmp_path / "labels" / "corrected.csv")
    assert list(out.columns) == ["SeriesInstanceUID", *TARGETS]
    assert list(out["SeriesInstanceUID"]) == ["a", "b", "c"]
    expected = {"a": {6, 13}, "b": {0, 12, 13}, "c": set()}
    for uid, positives in expected.items():
        row = out.set_index("SeriesInstanceUID").loc[uid]
        assert {i for i, t in enumerate(TARGETS) if row[t] == 1} == positives
        assert set(row.unique()) <= {0, 1}


def test_prints_counts_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    write_mask(tmp_path, FOLDERS[0], "secret-uid", (14,))
    write_mask(tmp_path, FOLDERS[1], "other-uid", (15,))
    assert derive(tmp_path, monkeypatch, counts(1, 1)) == 0
    out = capsys.readouterr().out
    assert "secret-uid" not in out
    assert out.splitlines() == [
        f"{FOLDERS[0]}: 1 masks",
        f"{FOLDERS[1]}: 1 masks",
        "wrote 2 series, 2 with an aneurysm",
    ]


def test_count_differing_from_manifest_exits_1(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    write_mask(tmp_path, FOLDERS[0], "a", (14,))
    write_mask(tmp_path, FOLDERS[1], "b", (15,))
    assert derive(tmp_path, monkeypatch, counts(2, 1)) == 1
    assert not (tmp_path / "labels" / "corrected.csv").exists()


def test_same_series_in_both_folders_fails(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    write_mask(tmp_path, FOLDERS[0], "a", (14,))
    write_mask(tmp_path, FOLDERS[1], "a", (15,))
    assert derive(tmp_path, monkeypatch, counts(1, 1)) == 1


@pytest.mark.parametrize("folder", FOLDERS)
def test_missing_mask_folder_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, folder: str
) -> None:
    other = FOLDERS[1] if folder == FOLDERS[0] else FOLDERS[0]
    write_mask(tmp_path, other, "a", (14,))
    assert derive(tmp_path, monkeypatch, {other: 1}) == 1


def test_non_integer_mask_is_refused(tmp_path: Path) -> None:
    module = load_script("derive_labels")
    path = tmp_path / "m.nii.gz"
    nib.save(nib.Nifti1Image(np.full((2, 2, 2), 14.0, dtype=np.float32), np.eye(4)), path)
    with pytest.raises(ValueError, match="m.nii.gz"):
        module.locations_present(path)
