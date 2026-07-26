"""Stage record media using the same paths as the Kira SFT harness."""

from __future__ import annotations

import logging
import shutil
from pathlib import Path

from .dataset import Record

LOGGER = logging.getLogger("coordinator.media")


def stage_media(record: Record, workspace: Path, dataset_root: Path) -> list[str]:
    """Copy media into the SFT harness's ``workspace/inputs/...`` layout.

    Returns the list of relative paths that ended up staged (for inclusion in
    the kira instruction prompt). Copies intentionally hide the source dataset
    root from the agent process; symlink targets would disclose it.  A single
    audio-video file can appear under both ``media.videos`` and
    ``media.audios`` in the RL schema.  It is staged and listed once, matching
    the video BenchSpec used to collect Kira SFT trajectories.
    """
    staged: list[str] = []
    seen_sources: set[Path] = set()
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
            if src in seen_sources:
                continue
            seen_sources.add(src)

            if record.source_dataset == "Omnimodal-Agent-SFT-2K":
                # OmniGAIA's SFT BenchSpec preserves the dataset-relative path
                # beneath inputs/ (for example inputs/media/images/x.jpg).
                staged_relative = Path("inputs") / relative
            else:
                # LVOmniBench-style SFT specs stage their one video as
                # inputs/videos/<name>, independent of its source location.
                staged_relative = Path("inputs") / kind / src.name

            dst = (workspace_root / staged_relative).resolve()
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
            staged.append(staged_relative.as_posix())
    LOGGER.info("staged %d media files for %s in %s", len(staged), record.id, workspace)
    return staged
