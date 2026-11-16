import subprocess
from pathlib import Path

import pytest

SBATCH = Path(__file__).resolve().parents[1] / "slurm" / "train.sbatch"


@pytest.mark.parametrize(
    "args",
    [
        ("scripts/train.py", "0"),
        ("configs/../x.yaml", "0"),
        ("configs/base/b2.yaml", "x"),
        ("configs/base/b2.yaml", "-1"),
        ("configs/base/b2.yaml",),
    ],
)
def test_refuses_bad_arguments(args: tuple[str, ...]) -> None:
    done = subprocess.run(["bash", str(SBATCH), *args], capture_output=True, text=True)
    assert done.returncode == 1
    assert done.stderr.startswith("refused:")
