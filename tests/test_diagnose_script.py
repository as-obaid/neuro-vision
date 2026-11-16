import hashlib
import json
import re
from pathlib import Path

import pytest

from neurovision.data.folds import make_folds
from neurovision.evaluate.bootstrap import METRICS
from tests.label_fakes import fake_labels
from tests.script_loader import load_script


@pytest.fixture
def setup(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    train, diagnose = load_script("train"), load_script("diagnose")
    labels = fake_labels()
    for module in (train, diagnose):
        monkeypatch.setattr(module, "load_labels", lambda store, source: labels)
    monkeypatch.setenv("NV_STORE", str(tmp_path))
    (tmp_path / "folds").mkdir()
    make_folds(labels, 5, 0, None).to_csv(tmp_path / "folds" / "folds.csv")
    sha = hashlib.sha256((tmp_path / "folds" / "folds.csv").read_bytes()).hexdigest()
    (tmp_path / "folds" / "folds.yaml").write_text(f"folds_sha256: {sha}\n")
    configs = tmp_path / "configs"
    configs.mkdir()
    (configs / "b0.yaml").write_text("kind: prior\nseed: 0\nlabels: corrected\n")
    (configs / "b1.yaml").write_text(
        "kind: gbm\nseed: 0\nlabels: corrected\n"
        "gbm: {max_iter: 10, learning_rate: 0.1, max_depth: 2}\n"
    )
    b0, b1 = str(configs / "b0.yaml"), str(configs / "b1.yaml")
    train.main([b0])
    train.main([b1])
    return diagnose, tmp_path, b0, b1


def test_writes_aggregates_for_both_configs(setup) -> None:
    diagnose, store, b0, b1 = setup
    diagnose.main([b1, "--against", b0])
    path = store / "runs" / "b1" / "diagnostics.json"
    result = json.loads(path.read_text())
    assert (result["config"], result["against"], result["n"]) == ("b1", "b0", 200)
    for name in ("b1", "b0"):
        assert set(result["per_fold"][name]) == {"0", "1", "2", "3", "4"}
        assert set(result["by_modality"][name]) == {"CTA", "MRA", "MRI T1post", "MRI T2"}
        assert {"youden", "sensitivity_0.90"} <= result["aneurysm_present"][name].keys()
        assert result["aneurysm_present"][name]["calibration"]["bins"]
    pooled = result["pooled"]
    assert result["bootstrap"]["score"]["difference"] == pytest.approx(
        pooled["b1"]["score"] - pooled["b0"]["score"]
    )
    assert pooled["b0"]["score"] == 0.5
    assert not re.search(r"s\d{4}", path.read_text()), "a series id leaked into the output"


def test_a_config_against_itself_differs_by_zero(setup) -> None:
    diagnose, store, _, b1 = setup
    diagnose.main([b1, "--against", b1])
    result = json.loads((store / "runs" / "b1" / "diagnostics.json").read_text())
    for metric in METRICS:
        assert result["bootstrap"][metric]["lo"] == result["bootstrap"][metric]["hi"] == 0.0


def test_refuses_configs_with_different_series(setup) -> None:
    diagnose, store, b0, b1 = setup
    (store / "runs" / "b0" / "fold4" / "oof.csv").unlink()
    with pytest.raises(SystemExit, match="different series"):
        diagnose.main([b1, "--against", b0])
