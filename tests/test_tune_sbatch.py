import subprocess
from pathlib import Path

SBATCH = Path(__file__).resolve().parents[1] / "slurm" / "tune.sbatch"


def test_requests_one_l40s_on_sharing_for_an_hour() -> None:
    text = SBATCH.read_text()
    for line in (
        "#SBATCH --partition=sharing",
        "#SBATCH --gres=gpu:l40s:1",
        "#SBATCH --time=01:00:00",
    ):
        assert line in text.splitlines()


def test_delegates_to_train_sbatch() -> None:
    assert 'exec bash slurm/train.sbatch "$@"' in SBATCH.read_text()


def test_train_argument_checks_apply() -> None:
    done = subprocess.run(
        ["bash", str(SBATCH), "scripts/train.py", "0"], capture_output=True, text=True
    )
    assert done.returncode == 1
    assert done.stderr.startswith("refused:")
