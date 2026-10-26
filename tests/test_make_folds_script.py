import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from tests.label_fakes import fake_labels
from tests.script_loader import load_script


@pytest.fixture
def setup(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    module = load_script("make_folds")
    labels = fake_labels()
    monkeypatch.setattr(module, "load_labels", lambda store, source: labels)
    monkeypatch.setenv("NV_STORE", str(tmp_path))
    (tmp_path / "labels").mkdir()
    rng = np.random.default_rng(1)
    headers = pd.DataFrame(
        {"PatientID": [f"p{i // 2}" for i in range(len(labels))]}, index=labels.index
    )
    headers.loc[headers.index[:5], "PatientID"] = np.nan
    headers.loc[rng.choice(headers.index[10:], 3), "PatientID"] = "p0"
    headers.to_csv(tmp_path / "labels" / "headers.csv")
    return module, labels, tmp_path


def test_writes_folds_and_manifest(setup, capsys) -> None:
    module, labels, store = setup
    module.main(["--group-key", "PatientID"])
    folds = pd.read_csv(store / "folds" / "folds.csv", index_col="SeriesInstanceUID")["fold"]
    assert folds.index.equals(labels.index)
    assert sorted(folds.unique()) == [0, 1, 2, 3, 4]
    headers = pd.read_csv(store / "labels" / "headers.csv", index_col=0)["PatientID"].dropna()
    assert (folds.loc[headers.index].groupby(headers).nunique() == 1).all()
    manifest = yaml.safe_load((store / "folds" / "folds.yaml").read_text())
    assert manifest["n_folds"] == 5
    assert manifest["seed"] == 0
    assert manifest["group_key"] == "PatientID"
    assert manifest["labels"] == "corrected"
    assert (
        manifest["folds_sha256"]
        == hashlib.sha256((store / "folds" / "folds.csv").read_bytes()).hexdigest()
    )
    out = capsys.readouterr().out
    assert "groups" in out
    assert "s0000" not in out


def test_key_none_makes_ungrouped_folds(setup) -> None:
    module, _, store = setup
    module.main(["--group-key", "none"])
    manifest = yaml.safe_load((store / "folds" / "folds.yaml").read_text())
    assert manifest["group_key"] == "none"


def test_refuses_to_overwrite(setup) -> None:
    module, _, store = setup
    module.main(["--group-key", "none"])
    before = (store / "folds" / "folds.csv").read_bytes()
    with pytest.raises(SystemExit, match="refusing to overwrite"):
        module.main(["--group-key", "PatientID"])
    assert (store / "folds" / "folds.csv").read_bytes() == before


def test_text_ids_and_blanks_are_read_as_text(setup, capsys) -> None:
    module, labels, store = setup
    rest = [f"g{i // 2}" for i in range(len(labels) - 6)]
    ids = ["00123", "123", "NA", "NA", "", ""] + rest
    pd.DataFrame({"PatientID": ids}, index=labels.index).to_csv(store / "labels" / "headers.csv")
    module.main(["--group-key", "PatientID"])
    folds = pd.read_csv(store / "folds" / "folds.csv", index_col="SeriesInstanceUID")["fold"]
    # "00123" and "123" differ, "NA" is a real shared value, each blank is its own group.
    expected = 2 + 1 + 2 + len(set(rest))
    assert f"groups: {expected}," in capsys.readouterr().out
    assert folds.iloc[2] == folds.iloc[3]
