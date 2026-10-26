import hashlib
import json
import re
from pathlib import Path

import pytest

from neurovision.config import load_config
from neurovision.track import log
from neurovision.track.meta import run_meta


def write_config(tmp_path: Path):
    path = tmp_path / "b0.yaml"
    path.write_text("kind: prior\nseed: 0\nlabels: corrected\n")
    return load_config(path)


def test_run_meta_fields(tmp_path: Path) -> None:
    config = write_config(tmp_path)
    meta = run_meta(config)
    assert {
        "commit",
        "dirty",
        "config_sha256",
        "host",
        "gpu",
        "python",
        "torch",
        "monai",
        "started_at",
    } <= meta.keys()
    assert (
        meta["config_sha256"]
        == hashlib.sha256(json.dumps(config.model_dump(), sort_keys=True).encode()).hexdigest()
    )
    assert re.fullmatch(r"\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d(\.\d+)?\+00:00", meta["started_at"])
    assert isinstance(meta["dirty"], bool) or meta["commit"] == "unknown"
    json.dumps(meta)


def test_run_meta_outside_git_is_unknown(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("neurovision.track.meta.REPO", tmp_path)
    meta = run_meta(write_config(tmp_path))
    assert meta["commit"] == "unknown"
    assert meta["dirty"] == "unknown"


def test_log_start_writes_offline_run(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("WANDB_PROJECT", "test-project")
    monkeypatch.setenv("WANDB_ENTITY", "test-entity")
    monkeypatch.setenv("WANDB_SILENT", "true")
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    run = log.start(write_config(tmp_path), 2, run_dir)
    run.log({"loss": 1.0}, step=1)
    run.finish({"best": 0.7})
    offline = list((run_dir / "wandb").glob("offline-run-*"))
    assert len(offline) == 1
