"""Location of the data store."""

import os
from collections.abc import Callable
from pathlib import Path


def store() -> Path:
    value = os.environ.get("NV_STORE")
    if not value:
        raise RuntimeError("NV_STORE is not set; point it at the data store directory")
    return Path(value)


def atomic_write(path: Path, write: Callable[[Path], object]) -> None:
    """Let `write` fill a temporary sibling, then rename it over `path`."""
    tmp = path.with_name(path.name + ".tmp")
    write(tmp)
    os.replace(tmp, path)


def require_unzipped(root: Path) -> None:
    if not (root / "raw" / ".complete").exists():
        raise SystemExit("raw/.complete is missing: the unzip has not finished")


def fold_dir(root: Path, config_name: str, fold: int) -> Path:
    return root / "runs" / config_name / f"fold{fold}"
