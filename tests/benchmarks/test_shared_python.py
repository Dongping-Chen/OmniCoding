from __future__ import annotations

from pathlib import Path

import pytest

from omnicoding.benchmarks.common.shared_python import (
    DEFAULT_SHARED_PYTHON_ENV_ENVVAR,
    default_shared_python_env,
    normalize_shared_python_env,
)


def test_shared_python_env_is_opt_in(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(DEFAULT_SHARED_PYTHON_ENV_ENVVAR, raising=False)
    assert default_shared_python_env() is None


def test_shared_python_env_uses_explicit_environment(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    env_root = tmp_path / "venv"
    env_python = env_root / "bin" / "python"
    env_python.parent.mkdir(parents=True)
    env_python.touch()
    monkeypatch.setenv(DEFAULT_SHARED_PYTHON_ENV_ENVVAR, str(env_root))
    assert normalize_shared_python_env(default_shared_python_env()) == str(env_root.resolve())


def test_shared_python_env_rejects_missing_python(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        normalize_shared_python_env(str(tmp_path / "missing"))
