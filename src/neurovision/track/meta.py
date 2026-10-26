"""Where a run came from: commit, config, host, hardware, versions."""

import hashlib
import json
import platform
import socket
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import monai
import torch

from neurovision.config import Config

REPO = Path(__file__).resolve().parents[3]


def _git(*args: str) -> str | None:
    result = subprocess.run(["git", "-C", str(REPO), *args], capture_output=True, text=True)
    return result.stdout.strip() if result.returncode == 0 else None


def run_meta(config: Config) -> dict[str, Any]:
    commit = _git("rev-parse", "HEAD")
    status = _git("status", "--porcelain")
    resolved = json.dumps(config.model_dump(), sort_keys=True)
    return {
        "commit": commit or "unknown",
        "dirty": bool(status) if status is not None else "unknown",
        "config_sha256": hashlib.sha256(resolved.encode()).hexdigest(),
        "host": socket.gethostname(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "python": platform.python_version(),
        "torch": torch.__version__,
        "monai": monai.__version__,
        "started_at": datetime.now(UTC).isoformat(),
    }
