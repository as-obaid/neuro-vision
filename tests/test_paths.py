from pathlib import Path

import pytest

from neurovision.paths import atomic_write, fold_dir, require_unzipped, store


def test_store_reads_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("NV_STORE", str(tmp_path))
    assert store() == tmp_path


@pytest.mark.parametrize("value", [None, ""])
def test_store_unset_or_empty_raises(monkeypatch: pytest.MonkeyPatch, value: str | None) -> None:
    if value is None:
        monkeypatch.delenv("NV_STORE", raising=False)
    else:
        monkeypatch.setenv("NV_STORE", value)
    with pytest.raises(RuntimeError, match="NV_STORE"):
        store()


def test_atomic_write_target_has_full_content(tmp_path: Path) -> None:
    target = tmp_path / "out.txt"
    atomic_write(target, lambda tmp: tmp.write_text("full"))
    assert target.read_text() == "full"
    assert list(tmp_path.iterdir()) == [target]


def test_atomic_write_failure_leaves_no_target(tmp_path: Path) -> None:
    target = tmp_path / "out.txt"

    def fail(tmp: Path) -> None:
        tmp.write_text("partial")
        raise OSError("disk full")

    with pytest.raises(OSError, match="disk full"):
        atomic_write(target, fail)
    assert not target.exists()


def test_atomic_write_failure_keeps_previous_target(tmp_path: Path) -> None:
    target = tmp_path / "out.txt"
    target.write_text("old")
    with pytest.raises(OSError, match="boom"):
        atomic_write(target, lambda tmp: (_ for _ in ()).throw(OSError("boom")))
    assert target.read_text() == "old"


def test_require_unzipped(tmp_path: Path) -> None:
    with pytest.raises(SystemExit, match=r"raw/\.complete is missing"):
        require_unzipped(tmp_path)
    (tmp_path / "raw").mkdir()
    (tmp_path / "raw" / ".complete").touch()
    require_unzipped(tmp_path)


def test_fold_dir(tmp_path: Path) -> None:
    assert fold_dir(tmp_path, "b2", 3) == tmp_path / "runs" / "b2" / "fold3"
