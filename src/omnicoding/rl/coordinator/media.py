"""Stage a record's media files into a kira workspace via copies.

The agent sees the same relative subpath it appears as in the record (e.g.
``media/audios/000052.wav``) so the agent can run shell tools on it
without us rewriting paths in the question text.
"""

from __future__ import annotations

import logging
import shutil
from pathlib import Path

from .dataset import Record

LOGGER = logging.getLogger("coordinator.media")


def stage_media(record: Record, workspace: Path, dataset_root: Path) -> list[str]:
    """Copy media into ``workspace/media/<kind>/<file>``.

    Returns the list of relative paths that ended up staged (for inclusion in
    the kira instruction prompt). Copies intentionally hide the source dataset
    root from the agent process; symlink targets would disclose it.
    """
    staged: list[str] = []
    root = dataset_root.resolve()
    workspace_root = workspace.resolve()
    for kind in ("videos", "audios", "images"):
        for rel in record.media.get(kind, []):
            relative = Path(rel)
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError(f"unsafe media path for record {record.id}: {rel!r}")
            src = (root / relative).resolve()
            try:
                src.relative_to(root)
            except ValueError as exc:
                raise ValueError(
                    f"media path escapes dataset root for record {record.id}: {rel!r}"
                ) from exc
            if not src.is_file():
                LOGGER.warning("media missing: %s (record %s)", src, record.id)
                continue
            dst = (workspace_root / relative).resolve()
            try:
                dst.relative_to(workspace_root)
            except ValueError as exc:
                raise ValueError(
                    f"media destination escapes workspace for record {record.id}: {rel!r}"
                ) from exc
            dst.parent.mkdir(parents=True, exist_ok=True)
            if dst.exists() or dst.is_symlink():
                dst.unlink()
            shutil.copy2(src, dst)
            staged.append(rel)
    LOGGER.info("staged %d media files for %s in %s", len(staged), record.id, workspace)
    return staged
