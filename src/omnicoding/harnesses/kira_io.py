"""I/O + workspace helpers split out of ``run_bench_kira.py``.

Pure utilities — no global state, no harness/spec coupling. Tested by
the data-dispatch and harness regression tests. Keep this file
under 400 lines so the main driver can stay under the 800-line cap.
"""
from __future__ import annotations

import base64
import json
import logging
import subprocess
import time
from pathlib import Path
from typing import Any

LOGGER = logging.getLogger("run_bench_kira")


def decode_subcall_images_to_dir(jsonl_path: Path, images_dir: Path) -> int:
    """Decode every ``status==ok`` record's ``image_b64`` into a real
    image file under ``images_dir``. Returns count written.

    File-name convention: ``<stem-of-original-arg>.<ext>``. Multiple
    image_reads on the same path land on the same file (idempotent).
    Multimodal SFT prep references these paths so the train rows do not
    depend on the JSONL still being on disk."""
    images_dir.mkdir(parents=True, exist_ok=True)
    n = 0
    for line in jsonl_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        if rec.get("status") != "ok" or not rec.get("image_b64"):
            continue
        path_arg = rec.get("file_path_arg") or ""
        mime = rec.get("mime") or "image/png"
        ext = mime.split("/")[-1]
        if ext == "jpeg":
            ext = "jpg"
        stem = Path(path_arg).stem or f"img_{n}"
        target = images_dir / f"{stem}.{ext}"
        if not target.exists():
            target.write_bytes(base64.b64decode(rec["image_b64"]))
        n += 1
    LOGGER.info("kira.driver decoded %d images → %s", n, images_dir)
    return n


def setup_per_job_venv(workspace: Path, base_venv: Path) -> Path:
    """Create a thin per-job venv that LAYERS over the readonly base.

    1. ``<base_venv>/bin/python -m venv --copies <ws>/.venv`` (~5 s) —
       fresh empty venv with its own bin/python, bin/pip, pip+setuptools
       in its site-packages. ~73 MB.
    2. Drop ``_kira_base_layer.pth`` into the new venv's site-packages
       pointing at the base's site-packages. Python's site.py appends
       that dir to ``sys.path`` AFTER the overlay's own site-packages, so:
         - ``import numpy`` → falls through to base (read shared)
         - ``pip install foo`` → writes into overlay (no base mutation)
         - ``pip install -U numpy`` → writes overlay's numpy, which
           shadows base on import (overlay comes first in sys.path)

    Trade-off vs full ``cp -a`` (round-17.6 originally tried): cp -a was
    60 s + 6.5 GB per job; .pth is 5 s + 73 MB. Same isolation, same
    writability, no base inode mutation. Verified via
    ``test_per_job_venv_pth_overlay_isolates_pip_install``.

    Returns the path to the per-job venv root (use it as
    ``shared_python_env`` and prepend ``<root>/bin`` to PATH).
    """
    started = time.monotonic()
    venv_root = workspace / ".venv"
    base_lib_dirs = sorted((base_venv / "lib").glob("python*"))
    if not base_lib_dirs:
        raise FileNotFoundError(f"base venv has no lib/python*: {base_venv}")
    py_dirname = base_lib_dirs[0].name
    base_sp = base_venv / "lib" / py_dirname / "site-packages"
    if not base_sp.exists():
        raise FileNotFoundError(f"base venv site-packages not found: {base_sp}")

    subprocess.run(
        [str(base_venv / "bin" / "python"), "-m", "venv", "--copies", str(venv_root)],
        check=True, capture_output=True,
    )
    overlay_sp = venv_root / "lib" / py_dirname / "site-packages"
    pth_file = overlay_sp / "_kira_base_layer.pth"
    pth_file.write_text(f"{base_sp}\n", encoding="utf-8")

    elapsed = time.monotonic() - started
    LOGGER.info(
        "kira.driver per-job venv ready %s (took %.1fs, layered over %s)",
        venv_root, elapsed, base_sp,
    )
    return venv_root


def filter_by_source_indices(
    items: list[dict[str, Any]], spec_str: str | None,
) -> list[dict[str, Any]]:
    """Keep only items whose ``__source_index__`` is in ``spec_str``
    (comma-separated ints). No-op when ``spec_str`` is empty."""
    if not spec_str:
        return items
    wanted = {int(x.strip()) for x in spec_str.split(",") if x.strip()}
    out = [it for it in items if int(it.get("__source_index__", -1)) in wanted]
    LOGGER.info(
        "kira.driver source_indices filter %s: %d → %d items",
        sorted(wanted), len(items), len(out),
    )
    return out


def load_existing_results(
    out_dir: Path, filename: str = "results.json", min_attempt: int = 1,
) -> dict[Any, dict[str, Any]]:
    """Read ``out_dir/<filename>`` (if any) → ``{source_index: row}``.

    Used by the auto-resume path: any row keyed by source_index whose
    ``error`` is falsy AND whose ``attempt >= min_attempt`` is reused
    as-is. ``--attempt=2`` (escalation) sets ``min_attempt=2`` to ignore
    pass-1 rows so the run actually re-fires the item. Missing file or
    bad JSON → empty (logged at WARNING).
    """
    p = out_dir / filename
    if not p.exists():
        return {}
    try:
        rows = json.loads(p.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        LOGGER.warning("kira.driver could not parse existing %s: %s", p, exc)
        return {}
    out: dict[Any, dict[str, Any]] = {}
    for r in rows or []:
        if not isinstance(r, dict):
            continue
        si = r.get("source_index")
        if si is None:
            continue
        if int(r.get("attempt", 1)) < min_attempt:
            continue
        out[si] = r
    return out


def load_prior_rows(
    out_dir: Path, filename: str = "results.json",
) -> dict[Any, dict[str, Any]]:
    """All rows from the prior file, regardless of attempt. Used to
    archive attempt-1 data into the new attempt-2 row's ``prior_attempts``
    list so escalation never silently destroys earlier-attempt evidence."""
    p = out_dir / filename
    if not p.exists():
        return {}
    try:
        rows = json.loads(p.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    out: dict[Any, dict[str, Any]] = {}
    for r in rows or []:
        if not isinstance(r, dict):
            continue
        si = r.get("source_index")
        if si is None:
            continue
        out[si] = r
    return out


_PRIOR_ATTEMPT_FIELDS = (
    "attempt", "reasoning_effort", "exit_reason", "predicted_answer",
    "tool_call_num", "elapsed_s", "error", "step_limit_used",
    "prompt_tokens", "completion_tokens", "reasoning_tokens",
)


def archive_prior_row(new_row: dict[str, Any], prior: dict[str, Any] | None) -> None:
    """If a prior-attempt row exists for the same si and the new row is
    a higher attempt, attach a slim summary of the prior row to the new
    row's ``prior_attempts`` list. Slim, not full — full messages.json /
    trajectory_steps would balloon results.json on every escalation."""
    if not prior:
        return
    new_attempt = int(new_row.get("attempt", 1))
    prior_attempt = int(prior.get("attempt", 1))
    if prior_attempt >= new_attempt:
        return
    archive = list(prior.get("prior_attempts") or [])
    archive.append({k: prior.get(k) for k in _PRIOR_ATTEMPT_FIELDS})
    new_row["prior_attempts"] = archive


def atomic_write_results(
    out_dir: Path, rows: list[dict[str, Any]], filename: str = "results.json",
) -> None:
    """Write ``<filename>`` via tmp+rename so a kill mid-write never
    leaves a partially-flushed file the next resume would mis-parse."""
    p = out_dir / filename
    tmp = p.with_name(p.name + ".tmp")
    tmp.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(p)


def sorted_rows(rows_by_si: dict[Any, dict[str, Any]]) -> list[dict[str, Any]]:
    """Stable order: numeric source_index first, then string fallback."""
    def _key(item):
        si = item[0]
        try:
            return (0, int(si))
        except (TypeError, ValueError):
            return (1, str(si))
    return [r for _, r in sorted(rows_by_si.items(), key=_key)]
