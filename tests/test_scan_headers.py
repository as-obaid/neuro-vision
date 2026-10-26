import importlib
import os
from pathlib import Path
from types import ModuleType

import pandas as pd
import pytest

from neurovision.data.dicom import TEST_TAGS
from tests.dicom_fakes import write_series

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"


def load_script(monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    # Importable by name so pool workers can import it too.
    monkeypatch.syspath_prepend(str(SCRIPTS))
    monkeypatch.setenv("PYTHONPATH", str(SCRIPTS) + os.pathsep + os.environ.get("PYTHONPATH", ""))
    return importlib.import_module("scan_headers")


def make_store(root: Path) -> Path:
    shape = (3, 4, 4)
    kw = {"shape": shape, "pixel_spacing": (0.5, 0.5), "slice_spacing": 1.0}
    # 1.2.3.1 and 1.2.3.2 share a patient (same sex, ages 2 years apart); 1.2.3.3 and 1.2.3.4
    # share a study date only; 1.2.3.5 is not in train.csv and must be ignored.
    specs = {
        "1.2.3.1": {"patient_id": "A", "study_date": "20200101", "n": 3},
        "1.2.3.2": {"patient_id": "A", "study_date": "20200202", "n": 2},
        "1.2.3.3": {"patient_id": "B", "study_date": "20200303", "n": 3, "multi": True},
        "1.2.3.4": {"patient_id": "C", "study_date": "20200303", "n": 1},
        "1.2.3.5": {"patient_id": "D", "study_date": "20200404", "n": 1},
    }
    for uid, spec in specs.items():
        write_series(
            root / "raw" / "series" / uid,
            **{**kw, "shape": (spec["n"], 4, 4)},
            series_uid=uid,
            patient_id=spec["patient_id"],
            study_date=spec["study_date"],
            multiframe=bool(spec.get("multi")),
        )
    train = pd.DataFrame(
        {
            "SeriesInstanceUID": ["1.2.3.1", "1.2.3.2", "1.2.3.3", "1.2.3.4"],
            "PatientAge": [50, 52, 40, 70],
            "PatientSex": ["Male", "Male", "Female", "Male"],
        }
    )
    train.to_csv(root / "raw" / "train.csv", index=False)
    (root / "raw" / ".complete").touch()
    return root


def test_scan_writes_headers_and_prints_counts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    store = make_store(tmp_path)
    monkeypatch.setenv("NV_STORE", str(store))
    monkeypatch.setenv("SLURM_CPUS_PER_TASK", "2")
    script = load_script(monkeypatch)

    script.main([])

    headers = pd.read_csv(store / "labels" / "headers.csv", dtype=str, keep_default_na=False)
    assert list(headers.columns) == [
        "SeriesInstanceUID",
        "n_files",
        *TEST_TAGS,
        "StudyDate",
        "error",
    ]
    assert headers["SeriesInstanceUID"].tolist() == ["1.2.3.1", "1.2.3.2", "1.2.3.3", "1.2.3.4"]
    assert headers["n_files"].tolist() == ["3", "2", "1", "1"]
    assert headers["PerFrameFunctionalGroupsSequence"].tolist() == ["", "", "3", ""]
    assert headers["TransferSyntaxUID"].eq("1.2.840.10008.1.2.1").all()
    assert headers["PatientID"].tolist() == ["A", "A", "B", "C"]
    assert headers["StudyDate"].tolist() == ["20200101", "20200202", "20200303", "20200303"]
    assert headers["error"].eq("").all()
    assert headers["ImageOrientationPatient"][0] == "1.0\\0.0\\0.0\\0.0\\1.0\\0.0"

    out = capsys.readouterr().out
    assert "PatientID: non-missing 4, distinct 3, series per value max 2 mean 1.33" in out
    assert "shared by 2+ series 1, consistent sex and age 1" in out
    assert "StudyDate: non-missing 4, distinct 3, series per value max 2 mean 1.33" in out
    assert "shared by 2+ series 1, consistent sex and age 0" in out
    assert "1.2.3.1" not in out


def test_scan_refuses_without_complete_marker(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = make_store(tmp_path)
    (store / "raw" / ".complete").unlink()
    monkeypatch.setenv("NV_STORE", str(store))
    with pytest.raises(SystemExit, match=r"\.complete"):
        load_script(monkeypatch).main([])
    assert not (store / "labels").exists()


def test_scan_records_a_bad_series_and_continues(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    store = make_store(tmp_path)
    (store / "raw" / "series" / "1.2.3.2" / "0000.dcm").write_bytes(b"not dicom")
    for path in (store / "raw" / "series" / "1.2.3.4").glob("*.dcm"):
        path.unlink()
    monkeypatch.setenv("NV_STORE", str(store))
    monkeypatch.setenv("SLURM_CPUS_PER_TASK", "2")
    load_script(monkeypatch).main([])
    headers = pd.read_csv(store / "labels" / "headers.csv", dtype=str, keep_default_na=False)
    by_uid = headers.set_index("SeriesInstanceUID")
    assert by_uid.loc["1.2.3.1", "error"] == ""
    assert by_uid.loc["1.2.3.2", "error"] != ""
    assert by_uid.loc["1.2.3.4", "error"] != ""
    assert by_uid.loc["1.2.3.4", "n_files"] == "0"
    assert by_uid.loc["1.2.3.1", "PatientID"] == "A"
    assert "series with errors 2" in capsys.readouterr().out
