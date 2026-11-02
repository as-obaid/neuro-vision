import os
import shutil
import subprocess
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
GIT = ["git", "-c", "user.name=t", "-c", "user.email=t@x.invalid", "-c", "commit.gpgsign=false"]


def run(args: list[str], cwd: Path, env: dict[str, str] | None = None):
    full = {k: v for k, v in os.environ.items() if not k.startswith("GIT_")}
    full.update(env or {})
    return subprocess.run(args, cwd=cwd, capture_output=True, text=True, env=full)


def iso(epoch: float) -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S+00:00", time.gmtime(epoch))


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    root = tmp_path / "repo"
    for rel in ("src", "scripts", "configs", "data/manifests", "slurm"):
        (root / rel).mkdir(parents=True)
    for name in ("submit.sh", "purge_check.py"):
        shutil.copy(ROOT / "slurm" / name, root / "slurm" / name)
    (root / "slurm" / "purge_dates").write_text("")
    (root / "slurm" / "job.sbatch").write_text("#!/bin/bash\n#SBATCH --time=02:00:00\nexit 0\n")
    (root / "pyproject.toml").write_text("x\n")
    (root / "src" / "a.py").write_text("A = 1\n")
    (root / "notes.md").write_text("n\n")
    assert run(["git", "init", "-q"], root).returncode == 0
    assert run(["git", "add", "."], root).returncode == 0
    assert run([*GIT, "commit", "-qm", "init"], root).returncode == 0
    return root


@pytest.fixture
def fake_bin(tmp_path: Path) -> Path:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    sbatch = bin_dir / "sbatch"
    sbatch.write_text(f'#!/bin/bash\necho "$@" > "{tmp_path}/sbatch.args"\necho 4242\n')
    sbatch.chmod(0o755)
    return bin_dir


def submit(repo: Path, fake_bin: Path, *args: str, cwd: Path | None = None):
    env = {"PATH": f"{fake_bin}:{os.environ['PATH']}", "TZ": "UTC"}
    return run(["bash", str(repo / "slurm" / "submit.sh"), *args], cwd or repo, env)


def sbatch_args(fake_bin: Path) -> str | None:
    record = fake_bin.parent / "sbatch.args"
    return record.read_text().strip() if record.exists() else None


def test_clean_tree_submits_with_the_next_window_as_deadline(repo: Path, fake_bin: Path):
    start = time.time() + 30 * 86400
    (repo / "slurm" / "purge_dates").write_text(f"{iso(start)} {iso(start + 86400)}\n")
    assert run(["git", "add", "slurm"], repo).returncode == 0
    assert run([*GIT, "commit", "-qm", "dates"], repo).returncode == 0
    done = submit(repo, fake_bin, "slurm/job.sbatch", "scripts/x.py", "--flag")
    assert done.returncode == 0, done.stderr
    assert done.stdout.strip() == "4242"
    deadline = time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(int(start)))
    expected = f"--parsable --deadline={deadline} slurm/job.sbatch scripts/x.py --flag"
    assert sbatch_args(fake_bin) == expected
    assert (repo / "slurm" / "logs").is_dir()


def test_no_window_left_means_no_deadline(repo: Path, fake_bin: Path):
    done = submit(repo, fake_bin, "slurm/job.sbatch")
    assert done.returncode == 0, done.stderr
    assert sbatch_args(fake_bin) == "--parsable slurm/job.sbatch"


@pytest.mark.parametrize("path", ["src/a.py", "src/new.py", "slurm/job.sbatch"])
def test_dirty_tree_is_refused(repo: Path, fake_bin: Path, path: str):
    (repo / path).write_text("changed\n")
    done = submit(repo, fake_bin, "slurm/job.sbatch")
    assert done.returncode == 1
    assert "uncommitted" in done.stderr
    assert sbatch_args(fake_bin) is None


def test_changes_outside_the_code_paths_are_ignored(repo: Path, fake_bin: Path):
    (repo / "notes.md").write_text("changed\n")
    assert submit(repo, fake_bin, "slurm/job.sbatch").returncode == 0


@pytest.mark.parametrize("batch", ["slurm/missing.sbatch", "job.sbatch", "slurm/../x.sbatch"])
def test_bad_sbatch_path_is_refused(repo: Path, fake_bin: Path, batch: str):
    (repo / "job.sbatch").write_text("#SBATCH --time=01:00:00\n")
    done = submit(repo, fake_bin, batch)
    assert done.returncode == 1
    assert "slurm/*.sbatch" in done.stderr
    assert sbatch_args(fake_bin) is None


def test_missing_sbatch_argument_is_refused(repo: Path, fake_bin: Path):
    assert submit(repo, fake_bin).returncode == 1


def test_not_the_repo_root_is_refused(repo: Path, fake_bin: Path):
    done = submit(repo, fake_bin, "slurm/job.sbatch", cwd=repo / "slurm")
    assert done.returncode == 1
    assert "repo root" in done.stderr
    assert sbatch_args(fake_bin) is None


def test_sbatch_without_a_time_limit_is_refused(repo: Path, fake_bin: Path):
    (repo / "slurm" / "job.sbatch").write_text("#!/bin/bash\n")
    assert run(["git", "add", "slurm"], repo).returncode == 0
    assert run([*GIT, "commit", "-qm", "no time"], repo).returncode == 0
    done = submit(repo, fake_bin, "slurm/job.sbatch")
    assert done.returncode == 1
    assert "--time" in done.stderr


def test_purge_window_overlap_is_refused(repo: Path, fake_bin: Path):
    start = time.time() + 600  # the job's 2 h limit reaches it
    (repo / "slurm" / "purge_dates").write_text(f"{iso(start)} {iso(start + 86400)}\n")
    assert run(["git", "add", "slurm"], repo).returncode == 0
    assert run([*GIT, "commit", "-qm", "dates"], repo).returncode == 0
    done = submit(repo, fake_bin, "slurm/job.sbatch")
    assert done.returncode == 1
    assert "purge window" in done.stderr
    assert sbatch_args(fake_bin) is None


def test_begin_after_a_window_is_checked_from_the_begin_time(repo: Path, fake_bin: Path):
    start = time.time() + 600  # from now, the job's 2 h limit would reach it
    end = start + 86400
    (repo / "slurm" / "purge_dates").write_text(f"{iso(start)} {iso(end)}\n")
    assert run(["git", "add", "slurm"], repo).returncode == 0
    assert run([*GIT, "commit", "-qm", "dates"], repo).returncode == 0
    begin = time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(int(end) + 60))
    env = {"PATH": f"{fake_bin}:{os.environ['PATH']}", "TZ": "UTC", "SUBMIT_BEGIN": begin}
    done = run(["bash", str(repo / "slurm" / "submit.sh"), "slurm/job.sbatch", "a"], repo, env)
    assert done.returncode == 0, done.stderr
    assert sbatch_args(fake_bin) == f"--parsable --begin={begin} slurm/job.sbatch a"


def test_begin_inside_a_window_is_refused(repo: Path, fake_bin: Path):
    start = time.time() + 600
    (repo / "slurm" / "purge_dates").write_text(f"{iso(start)} {iso(start + 86400)}\n")
    assert run(["git", "add", "slurm"], repo).returncode == 0
    assert run([*GIT, "commit", "-qm", "dates"], repo).returncode == 0
    begin = time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(int(start) + 3600))
    env = {"PATH": f"{fake_bin}:{os.environ['PATH']}", "TZ": "UTC", "SUBMIT_BEGIN": begin}
    done = run(["bash", str(repo / "slurm" / "submit.sh"), "slurm/job.sbatch"], repo, env)
    assert done.returncode == 1
    assert "purge window" in done.stderr
    assert sbatch_args(fake_bin) is None


@pytest.mark.parametrize("begin", ["tomorrow", "2026-11-04 00:05", "2026-11-04T00:05:00;x"])
def test_bad_begin_is_refused(repo: Path, fake_bin: Path, begin: str):
    env = {"PATH": f"{fake_bin}:{os.environ['PATH']}", "TZ": "UTC", "SUBMIT_BEGIN": begin}
    done = run(["bash", str(repo / "slurm" / "submit.sh"), "slurm/job.sbatch"], repo, env)
    assert done.returncode == 1
    assert "SUBMIT_BEGIN" in done.stderr
    assert sbatch_args(fake_bin) is None


@pytest.fixture
def cpu(tmp_path: Path) -> Path:
    work = tmp_path / "work"
    (work / "slurm").mkdir(parents=True)
    shutil.copy(ROOT / "slurm" / "cpu.sbatch", work / "slurm" / "cpu.sbatch")
    return work


def run_cpu(work: Path, *args: str, env: dict[str, str] | None = None):
    base = {"SLURM_SUBMIT_DIR": str(work)}
    return run(["bash", str(work / "slurm" / "cpu.sbatch"), *args], work, {**base, **(env or {})})


def fake_store(tmp_path: Path) -> Path:
    py = tmp_path / "store" / "envs" / "neuro-vision" / "bin" / "python"
    py.parent.mkdir(parents=True)
    py.write_text(
        "#!/bin/bash\n"
        'if [ -n "${SECRET:-}" ]; then s=set; else s=unset; fi\n'
        'echo "py $* PYTHONPATH=$PYTHONPATH SECRET=$s"\n'
    )
    py.chmod(0o755)
    return tmp_path / "store"


@pytest.mark.parametrize("script", ["other/x.py", "scripts/x.sh", "scripts/../x.py", "x.py"])
def test_cpu_refuses_a_script_outside_scripts(cpu: Path, tmp_path: Path, script: str):
    done = run_cpu(cpu, script, env={"NV_STORE": str(fake_store(tmp_path))})
    assert done.returncode == 1
    assert "scripts/*.py" in done.stderr


def test_cpu_without_arguments_is_refused(cpu: Path, tmp_path: Path):
    assert run_cpu(cpu, env={"NV_STORE": str(fake_store(tmp_path))}).returncode == 1


def test_cpu_requires_nv_store(cpu: Path):
    done = run_cpu(cpu, "scripts/x.py", env={"NV_STORE": ""})
    assert done.returncode == 1
    assert "NV_STORE" in done.stderr


def test_cpu_runs_the_store_python_with_src_on_the_path(cpu: Path, tmp_path: Path):
    done = run_cpu(cpu, "scripts/x.py", "a", env={"NV_STORE": str(fake_store(tmp_path))})
    assert done.returncode == 0, done.stderr
    assert done.stdout.strip() == f"py scripts/x.py a PYTHONPATH={cpu}/src SECRET=unset"


def test_cpu_sources_env_without_printing_it(cpu: Path, tmp_path: Path):
    (cpu / ".env").write_text("SECRET=hunter2\n")
    store = fake_store(tmp_path)
    done = run_cpu(cpu, "scripts/x.py", env={"NV_STORE": str(store)})
    assert done.returncode == 0, done.stderr
    assert "SECRET=set" in done.stdout  # the child received it
    assert "hunter2" not in done.stdout
    assert "hunter2" not in done.stderr


def test_cpu_env_file_can_supply_nv_store(cpu: Path, tmp_path: Path):
    (cpu / ".env").write_text(f"NV_STORE={fake_store(tmp_path)}\n")
    done = run_cpu(cpu, "scripts/x.py", env={"NV_STORE": ""})
    assert done.returncode == 0, done.stderr


def test_cpu_header_has_the_contract_values():
    text = (ROOT / "slurm" / "cpu.sbatch").read_text()
    for line in (
        "#SBATCH --job-name=nv-cpu",
        "#SBATCH --account=ostadabbas",
        "#SBATCH --partition=short",
        "#SBATCH --cpus-per-task=4",
        "#SBATCH --mem=16G",
        "#SBATCH --time=01:00:00",
        "#SBATCH --output=slurm/logs/%x_%j.out",
        "#SBATCH --error=slurm/logs/%x_%j.err",
    ):
        assert line in text.splitlines()
