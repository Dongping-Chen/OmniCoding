"""Runtime path configuration that works in source and wheel installs."""

from __future__ import annotations

import os
from pathlib import Path


def runtime_root() -> Path:
    """Return the writable root for workspaces and optional local config."""
    configured = os.environ.get("OMNICODING_RUNTIME_ROOT", "").strip()
    return Path(configured).expanduser().resolve() if configured else Path.cwd().resolve()
