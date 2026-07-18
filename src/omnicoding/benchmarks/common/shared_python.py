"""Shared Python environment defaults for coding-agent benchmark runners."""

from __future__ import annotations

import os
from pathlib import Path


DEFAULT_SHARED_PYTHON_ENV_ENVVAR = "OMNICODING_SHARED_PYTHON_ENV"


def default_shared_python_env() -> str | None:
    configured = os.environ.get(DEFAULT_SHARED_PYTHON_ENV_ENVVAR)
    if configured is not None:
        return configured.strip() or None
    return None


def normalize_shared_python_env(raw_value: str | None) -> str | None:
    if not raw_value:
        return None
    env_root = Path(raw_value).expanduser().resolve()
    env_python = env_root / "bin" / "python"
    if not env_python.exists():
        raise FileNotFoundError(f"Shared Python environment is missing bin/python: {env_python}")
    return str(env_root)
