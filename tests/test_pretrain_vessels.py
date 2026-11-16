"""Vessel pretraining script: the run folder it writes, and its Slurm job."""

import json
import subprocess
from pathlib import Path

import pytest

from tests.script_loader import load_script

SLURM = Path(__file__).resolve().parents[1] / "slurm"


def test_run_defaults_to_the_step1_run_and_takes_another_name() -> None:
    script = load_script("pretrain_vessels")
    assert script.parse_args(["--fold", "1"]).run == "vessel_r3d18_160"
    args = script.parse_args(["--fold", "1", "--run", "vessel_r3d18_160_v2"])
    assert (args.fold, args.run) == (1, "vessel_r3d18_160_v2")


@pytest.mark.parametrize("run", ["", "../x", "a/b", ".hidden"])
def test_run_must_be_a_plain_folder_name(run: str) -> None:
    script = load_script("pretrain_vessels")
    with pytest.raises(SystemExit):
        script.parse_args(["--fold", "0", "--run", run])


def test_a_finished_run_is_left_alone_and_the_default_run_untouched(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("NV_STORE", str(tmp_path))
    done = tmp_path / "runs" / "vessel_r3d18_160_v2" / "fold2"
    done.mkdir(parents=True)
    (done / "meta.json").write_text(json.dumps({"run": "vessel_r3d18_160_v2"}))
    load_script("pretrain_vessels").main(["--fold", "2", "--run", "vessel_r3d18_160_v2"])
    assert sorted(p.name for p in done.iterdir()) == ["meta.json"]
    assert not (tmp_path / "runs" / "vessel_r3d18_160").exists()


@pytest.mark.parametrize("args", [("x",), ("-1",), (), ("0", "../x"), ("0", "a/b")])
def test_pretrain_job_refuses_a_bad_fold_or_run(args: tuple[str, ...]) -> None:
    done = subprocess.run(
        ["bash", str(SLURM / "pretrain.sbatch"), *args], capture_output=True, text=True
    )
    assert done.returncode == 1 and done.stderr.startswith("refused:")


def test_pretrain_job_passes_the_run_to_the_script() -> None:
    text = (SLURM / "pretrain.sbatch").read_text()
    assert 'scripts/pretrain_vessels.py --fold "$fold" --run "$run"' in text
    assert "run=${2:-vessel_r3d18_160}" in text
