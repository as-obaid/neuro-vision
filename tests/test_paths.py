from pathlib import Path

import pytest

from neurovision.paths import store


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
