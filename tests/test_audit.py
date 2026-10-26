import importlib
import json
from pathlib import Path
from types import ModuleType

import pandas as pd
import pytest

from neurovision.data.labels import TARGETS

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"

# uid, Modality, PatientAge, PatientSex, positive targets
ROWS = [
    ("s1", "CTA", 35, "Male", [TARGETS[0], TARGETS[13]]),
    ("s2", "CTA", 45, "Female", []),
    ("s3", "MRA", 65, "Female", [TARGETS[0], TARGETS[4], TARGETS[13]]),
    ("s4", "MRA", 85, None, [TARGETS[13]]),
    ("s5", "MRI T2", None, "Male", []),
]


def load_script(monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    monkeypatch.syspath_prepend(str(SCRIPTS))
    return importlib.import_module("audit")


def make_store(root: Path, complete: bool = True, skip: str | None = None) -> Path:
    raw = root / "raw"
    (raw / "series").mkdir(parents=True)
    (raw / "segmentations").mkdir()
    records = []
    for uid, modality, age, sex, positives in ROWS:
        row: dict[str, object] = {"SeriesInstanceUID": uid}
        row.update({t: int(t in positives) for t in TARGETS})
        row.update({"PatientAge": age, "PatientSex": sex, "Modality": modality})
        records.append(row)
        if uid != skip:
            (raw / "series" / uid).mkdir()
            (raw / "series" / uid / "a.dcm").touch()
    pd.DataFrame(records).to_csv(raw / "train.csv", index=False)
    (raw / "series" / "s1" / "b.dcm").touch()
    (raw / "series" / "extra").mkdir()
    (raw / "series" / "extra" / "a.dcm").touch()
    (raw / "segmentations" / "x.nii").touch()
    if complete:
        (raw / ".complete").touch()
    return root


def run_audit(module: ModuleType, store: Path, monkeypatch: pytest.MonkeyPatch, *args: str) -> int:
    monkeypatch.setenv("NV_STORE", str(store))
    return module.main(list(args))


def test_audit_counts_and_json(tmp_path, monkeypatch, capsys):
    store = make_store(tmp_path)
    assert run_audit(load_script(monkeypatch), store, monkeypatch) == 0
    out = json.loads((store / "logs" / "audit.json").read_text())
    assert out["n_series"] == 5
    assert out["modality"] == {"CTA": 2, "MRA": 2, "MRI T2": 1}
    assert out["positives"][TARGETS[0]] == {"CTA": 1, "MRA": 1, "MRI T2": 0, "total": 2}
    assert out["positives"][TARGETS[13]]["total"] == 3
    assert out["age"]["count"] == 4
    assert out["age"]["mean"] == 57.5
    assert out["age"]["median"] == 55.0
    assert (out["age"]["min"], out["age"]["max"]) == (35, 85)
    assert out["age"]["bands"] == {"<40": 1, "40-59": 1, "60-79": 1, ">=80": 1}
    assert out["sex"] == {"Male": 2, "Female": 2}
    assert out["missing"]["PatientAge"] == 1
    assert out["missing"]["PatientSex"] == 1
    assert out["missing"]["Modality"] == 0
    assert "| Modality |" in capsys.readouterr().out


def test_verify_counts(tmp_path, monkeypatch, capsys):
    store = make_store(tmp_path)
    assert run_audit(load_script(monkeypatch), store, monkeypatch, "--verify") == 0
    out = capsys.readouterr().out
    for line in (
        "present: 5",
        "missing: 0",
        "empty: 0",
        "dcm files: 6",
        "single-file series: 4",
        "not in train.csv: 1",
        "segmentation files: 1",
    ):
        assert line in out


def test_verify_refuses_without_complete(tmp_path, monkeypatch):
    store = make_store(tmp_path, complete=False)
    with pytest.raises(SystemExit, match=r"\.complete"):
        run_audit(load_script(monkeypatch), store, monkeypatch, "--verify")


def test_verify_fails_on_missing_series(tmp_path, monkeypatch, capsys):
    store = make_store(tmp_path, skip="s3")
    assert run_audit(load_script(monkeypatch), store, monkeypatch, "--verify") == 1
    assert "missing: 1" in capsys.readouterr().out


def test_verify_fails_on_empty_series(tmp_path, monkeypatch, capsys):
    store = make_store(tmp_path)
    (store / "raw" / "series" / "s2" / "a.dcm").unlink()
    assert run_audit(load_script(monkeypatch), store, monkeypatch, "--verify") == 1
    assert "empty: 1" in capsys.readouterr().out


def test_verify_reports_missing_segmentations_folder(tmp_path, monkeypatch, capsys):
    store = make_store(tmp_path)
    (store / "raw" / "segmentations" / "x.nii").unlink()
    (store / "raw" / "segmentations").rmdir()
    assert run_audit(load_script(monkeypatch), store, monkeypatch, "--verify") == 0
    out = capsys.readouterr().out
    assert "segmentation files: 0" in out
    assert "warning: raw/segmentations is missing" in out
