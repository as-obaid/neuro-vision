import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "slurm" / "purge_check.py"


def iso(epoch: int) -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S+00:00", time.gmtime(epoch))


def purge(tmp_path: Path, now: int, *args: str) -> subprocess.CompletedProcess[str]:
    dates = tmp_path / "dates"
    if not dates.exists():
        dates.write_text(f"{iso(10_000)} {iso(20_000)}\n")
    return subprocess.run(
        [sys.executable, str(SCRIPT), "--now", str(now), *args, "--file", str(dates)],
        capture_output=True,
        text=True,
    )


def test_windows(tmp_path: Path) -> None:
    assert purge(tmp_path, 0, "--limit", "1:00:00").returncode == 0  # ends at 3,600: before
    assert purge(tmp_path, 7_000, "--limit", "1:00:00").returncode == 1  # runs into it
    assert purge(tmp_path, 15_000, "--limit", "00:10").returncode == 1  # inside it
    assert purge(tmp_path, 20_000, "--limit", "1-00:00:00").returncode == 0  # after it
    ahead = purge(tmp_path, 0, "--limit", "1:00:00", "--ahead", "1:00:00", "--ahead", "1:00:00")
    assert ahead.returncode == 1  # 3 h of queued work pushes it past 10,000 s
    assert "a /scratch purge window starts" in ahead.stderr


def test_errors(tmp_path: Path) -> None:
    assert purge(tmp_path, 0, "--limit", "soon").returncode == 2
    (tmp_path / "dates").write_text("2026-11-02T18:00:00 2026-11-04T00:00:00\n")
    done = purge(tmp_path, 0, "--limit", "1:00:00")
    assert done.returncode == 2
    assert "UTC offset" in done.stderr


def test_committed_purge_dates_parse() -> None:
    sys.path.insert(0, str(ROOT / "slurm"))
    try:
        import purge_check
    finally:
        sys.path.pop(0)
    windows = purge_check.read_windows(str(ROOT / "slurm" / "purge_dates"))
    assert [start.isoformat() for start, _ in windows] == [
        "2026-11-02T18:00:00-05:00",
        "2026-11-30T18:00:00-05:00",
    ]
    assert purge_check.limit_seconds("1-02:03:04") == 93784
    assert purge_check.limit_seconds("05:06") == 306
    assert purge_check.limit_seconds("1-12:00") == 129600


def test_clear_job_prints_the_next_window_start_in_local_time(tmp_path: Path) -> None:
    dates = tmp_path / "dates"
    dates.write_text(f"{iso(10_000)} {iso(20_000)}\n{iso(50_000)} {iso(60_000)}\n")
    env = {"TZ": "UTC"}
    out = subprocess.run(
        [sys.executable, str(SCRIPT), "--now", "21000", "--limit", "1:00:00", "--file", str(dates)],
        capture_output=True,
        text=True,
        env=env,
    )
    assert out.returncode == 0
    assert out.stdout.strip() == "1970-01-01T13:53:20"
    past = purge(tmp_path, 70_000, "--limit", "1:00:00")
    assert past.returncode == 0 and past.stdout == ""
