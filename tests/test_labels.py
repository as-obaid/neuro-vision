import hashlib
from pathlib import Path

import pandas as pd
import pytest
import yaml

from neurovision.data import labels
from neurovision.data.labels import ANEURYSM_PRESENT, LOCATIONS, TARGETS, load_labels

ROOT = Path(__file__).resolve().parents[1]

# Class names from the source's labels.json; classes 14..26 are aneurysms.
SOURCE_ANEURYSM_CLASSES = [
    "Left Infraclinoid Internal Carotid Artery Aneurysm",
    "Right Infraclinoid Internal Carotid Artery Aneurysm",
    "Left Supraclinoid Internal Carotid Artery Aneurysm",
    "Right Supraclinoid Internal Carotid Artery Aneurysm",
    "Left Middle Cerebral Artery Aneurysm",
    "Right Middle Cerebral Artery Aneurysm",
    "Anterior Communicating Artery Aneurysm",
    "Left Anterior Cerebral Artery Aneurysm",
    "Right Anterior Cerebral Artery Aneurysm",
    "Left Posterior Communicating Artery Aneurysm",
    "Right Posterior Communicating Artery Aneurysm",
    "Basilar Tip Aneurysm",
    "Other Posterior Circulation Aneurysm",
]


def test_targets_are_14_unique_names_with_aneurysm_present_last():
    assert len(TARGETS) == 14
    assert len(set(TARGETS)) == 14
    assert TARGETS[-1] == "Aneurysm Present"


def test_locations_and_aneurysm_present_split_targets():
    assert TARGETS[:13] == LOCATIONS
    assert TARGETS[13] == ANEURYSM_PRESENT


def test_target_order_matches_kaggle_label_cols():
    assert TARGETS[:4] == (
        "Left Infraclinoid Internal Carotid Artery",
        "Right Infraclinoid Internal Carotid Artery",
        "Left Supraclinoid Internal Carotid Artery",
        "Right Supraclinoid Internal Carotid Artery",
    )
    assert TARGETS[11:] == ("Basilar Tip", "Other Posterior Circulation", "Aneurysm Present")


def test_source_aneurysm_classes_follow_target_order():
    assert [f"{t} Aneurysm" for t in LOCATIONS] == SOURCE_ANEURYSM_CLASSES


def targets_row(positives: set[int]) -> dict[str, int]:
    row = {t: int(i in positives) for i, t in enumerate(TARGETS[:13])}
    row[ANEURYSM_PRESENT] = int(bool(positives))
    return row


def make_store(tmp_path: Path) -> Path:
    train = pd.DataFrame(
        [
            {"SeriesInstanceUID": "s1", "Modality": "CTA", "PatientAge": 50, "PatientSex": "Male"}
            | targets_row({0}),
            {"SeriesInstanceUID": "s2", "Modality": "MRA", "PatientAge": 61, "PatientSex": "Female"}
            | targets_row(set()),
            {"SeriesInstanceUID": "s3", "Modality": "MRI T2", "PatientAge": 7, "PatientSex": "Male"}
            | targets_row({4, 5}),
        ]
    )
    (tmp_path / "raw").mkdir()
    train.to_csv(tmp_path / "raw" / "train.csv", index=False)
    corrected = pd.DataFrame(
        [
            {"SeriesInstanceUID": "s2"} | targets_row({12}),
            {"SeriesInstanceUID": "s3"} | targets_row({5, 4}),
        ]
    )
    (tmp_path / "labels").mkdir()
    corrected.to_csv(tmp_path / "labels" / "corrected.csv", index=False)
    return tmp_path


@pytest.fixture
def store(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    made = make_store(tmp_path)
    digest = hashlib.sha256((made / "labels" / "corrected.csv").read_bytes()).hexdigest()
    monkeypatch.setattr(labels, "CORRECTED_SHA256", digest)
    return made


def test_original_reads_train_csv(store: Path):
    out = load_labels(store, "original")
    assert out.index.name == "SeriesInstanceUID"
    assert list(out.index) == ["s1", "s2", "s3"]
    assert list(out.columns) == [*TARGETS, "PatientAge", "PatientSex", "Modality"]
    assert all(out[t].dtype == "int8" for t in TARGETS)
    assert out.loc["s2", ANEURYSM_PRESENT] == 0


def test_corrected_replaces_listed_series_only(store: Path):
    out = load_labels(store, "corrected")
    assert list(out.columns) == [*TARGETS, "PatientAge", "PatientSex", "Modality"]
    assert all(out[t].dtype == "int8" for t in TARGETS)
    original = load_labels(store, "original")
    assert out.loc["s1"].equals(original.loc["s1"])
    assert out.loc["s2", TARGETS[12]] == 1
    assert out.loc["s2", ANEURYSM_PRESENT] == 1
    assert out.loc["s2", "Modality"] == "MRA"
    assert out.loc["s2", "PatientAge"] == 61


def test_sha_mismatch_raises(store: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(labels, "CORRECTED_SHA256", "0" * 64)
    with pytest.raises(ValueError, match="sha256"):
        load_labels(store, "corrected")


def test_missing_corrections_file_raises(store: Path):
    (store / "labels" / "corrected.csv").unlink()
    with pytest.raises(ValueError, match="corrected.csv"):
        load_labels(store, "corrected")


def test_unknown_series_in_corrections_raises(store: Path, monkeypatch: pytest.MonkeyPatch):
    path = store / "labels" / "corrected.csv"
    extra = pd.DataFrame([{"SeriesInstanceUID": "nope"} | targets_row({1})])
    pd.concat([pd.read_csv(path), extra]).to_csv(path, index=False)
    monkeypatch.setattr(labels, "CORRECTED_SHA256", hashlib.sha256(path.read_bytes()).hexdigest())
    with pytest.raises(ValueError, match="not in train.csv"):
        load_labels(store, "corrected")


def test_non_binary_target_raises(store: Path):
    path = store / "raw" / "train.csv"
    frame = pd.read_csv(path)
    frame.loc[0, TARGETS[0]] = 2
    frame.to_csv(path, index=False)
    with pytest.raises(ValueError, match="0 or 1"):
        load_labels(store, "original")


def test_duplicate_series_raises(store: Path):
    path = store / "raw" / "train.csv"
    frame = pd.read_csv(path)
    pd.concat([frame, frame.iloc[[0]]]).to_csv(path, index=False)
    with pytest.raises(ValueError, match="unique"):
        load_labels(store, "original")


def test_missing_target_column_raises(store: Path):
    path = store / "raw" / "train.csv"
    pd.read_csv(path).drop(columns=[TARGETS[3]]).to_csv(path, index=False)
    with pytest.raises(ValueError, match="missing"):
        load_labels(store, "original")


def test_manifest_sha_equals_module_constant():
    manifest = yaml.safe_load((ROOT / "data" / "manifests" / "labels.yaml").read_text())
    assert manifest["corrected_csv_sha256"] == labels.CORRECTED_SHA256
