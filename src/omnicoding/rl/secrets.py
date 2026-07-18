"""Load coordinator credentials without placing them in process arguments."""

from __future__ import annotations

import os
import stat
from pathlib import Path

_MAX_SECRET_BYTES = 4096


def _read_restricted_file(raw_path: str) -> str:
    path = Path(raw_path).expanduser()
    if not path.is_absolute():
        raise RuntimeError("ROLLOUT_COORDINATOR_TOKEN_FILE must be an absolute path")

    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise RuntimeError(f"cannot open coordinator token file: {exc}") from exc

    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode):
            raise RuntimeError("coordinator token path must be a regular file")
        if metadata.st_uid != os.getuid():
            raise RuntimeError("coordinator token file must be owned by the current user")
        if stat.S_IMODE(metadata.st_mode) & 0o077:
            raise RuntimeError("coordinator token file must not be accessible by group or other")
        if metadata.st_size > _MAX_SECRET_BYTES:
            raise RuntimeError("coordinator token file is unexpectedly large")
        value = os.read(descriptor, _MAX_SECRET_BYTES + 1).decode("utf-8").strip()
    finally:
        os.close(descriptor)

    if not value or any(character.isspace() for character in value):
        raise RuntimeError("coordinator token file must contain one non-empty token")
    return value


def load_coordinator_token() -> str:
    """Read the bearer token from a protected file or legacy environment value."""
    token_file = os.environ.get("ROLLOUT_COORDINATOR_TOKEN_FILE", "").strip()
    if token_file:
        return _read_restricted_file(token_file)

    token = os.environ.get("ROLLOUT_COORDINATOR_TOKEN", "").strip()
    if not token:
        raise RuntimeError(
            "set ROLLOUT_COORDINATOR_TOKEN_FILE (recommended) or "
            "ROLLOUT_COORDINATOR_TOKEN"
        )
    return token
