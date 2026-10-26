"""Location of the data store."""

import os
from pathlib import Path


def store() -> Path:
    value = os.environ.get("NV_STORE")
    if not value:
        raise RuntimeError("NV_STORE is not set; point it at the data store directory")
    return Path(value)
