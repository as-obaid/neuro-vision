import os
import subprocess
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
import pytest

from neurovision.data.labels import TARGETS

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "derive_labels.py"
MASKS = "labels/shi2026-masks"
FOLDERS = ("labelsTr_26classes_in_orig_space", "labelsTr_multiframe_cor_sag")


def write_mask(store: Path, folder: str, uid: str, classes: tuple[int, ...]) -> None:
    out = store / MASKS / folder
    out.mkdir(parents=True, exist_ok=True)
    data = np.zeros((4, 4, 4), dtype=np.uint8)
    for i, c in enumerate(classes):
        data[i, 0, 0] = c
    nib.save(nib.Nifti1Image(data, np.eye(4)), out / f"{uid}.nii.gz")


def derive(store: Path) -> subprocess.CompletedProcess[str]:
    env = {**os.environ, "NV_STORE": str(store)}
    env.pop("SLURM_CPUS_PER_TASK", None)
    return subprocess.run([sys.executable, str(SCRIPT)], env=env, capture_output=True, text=True)


def test_classes_map_to_targets_and_aneurysm_present_is_or(tmp_path: Path) -> None:
    write_mask(tmp_path, FOLDERS[0], "b", (3, 14, 26))  # vessel class 3 is ignored
    write_mask(tmp_path, FOLDERS[1], "a", (20,))
    write_mask(tmp_path, FOLDERS[0], "c", (1, 13))  # vessels only
    done = derive(tmp_path)
    assert done.returncode == 0, done.stderr
    out = pd.read_csv(tmp_path / "labels" / "corrected.csv")
    assert list(out.columns) == ["SeriesInstanceUID", *TARGETS]
    assert list(out["SeriesInstanceUID"]) == ["a", "b", "c"]
    expected = {"a": {6, 13}, "b": {0, 12, 13}, "c": set()}
    for uid, positives in expected.items():
        row = out.set_index("SeriesInstanceUID").loc[uid]
        assert {i for i, t in enumerate(TARGETS) if row[t] == 1} == positives
        assert set(row.unique()) <= {0, 1}


def test_prints_counts_only(tmp_path: Path) -> None:
    write_mask(tmp_path, FOLDERS[0], "secret-uid", (14,))
    write_mask(tmp_path, FOLDERS[1], "other-uid", (15,))
    done = derive(tmp_path)
    assert "secret-uid" not in done.stdout + done.stderr
    assert "2" in done.stdout


def test_same_series_in_both_folders_fails(tmp_path: Path) -> None:
    write_mask(tmp_path, FOLDERS[0], "a", (14,))
    write_mask(tmp_path, FOLDERS[1], "a", (15,))
    assert derive(tmp_path).returncode != 0


@pytest.mark.parametrize("folder", FOLDERS)
def test_missing_mask_folder_fails(tmp_path: Path, folder: str) -> None:
    other = FOLDERS[1] if folder == FOLDERS[0] else FOLDERS[0]
    write_mask(tmp_path, other, "a", (14,))
    assert derive(tmp_path).returncode != 0
