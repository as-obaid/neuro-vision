"""Import a file from scripts/ as a module (scripts/ is not a package)."""

import importlib
import importlib.util
import os
from pathlib import Path
from types import ModuleType

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"


def load_script(name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(f"script_{name}", SCRIPTS / f"{name}.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def import_script(name: str, monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    """Import by name, so that process-pool workers can import it too."""
    monkeypatch.syspath_prepend(str(SCRIPTS))
    monkeypatch.setenv("PYTHONPATH", str(SCRIPTS) + os.pathsep + os.environ.get("PYTHONPATH", ""))
    return importlib.import_module(name)
