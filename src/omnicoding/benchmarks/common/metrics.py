"""Atomic JSON write + accuracy/by-dimension metrics computation.

Centralises what every per-bench runner used to copy: the
`_atomic_write_json` write-tmp-then-rename, the `_calc_metrics`
overall+by-key payload, and the same final-results sort key.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import tempfile
from pathlib import Path
from typing import Any, Iterable

LOGGER = logging.getLogger(__name__)


def atomic_write_json(path: Path, payload: Any) -> None:
    """Write `payload` as JSON to `path` atomically (tmp + rename).

    Important: the streaming runners write the same path many times as
    items complete. Without atomic rename, a concurrent reader (the user
    tailing the file) sees half-written JSON.
    """

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_fd, tmp_name = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(tmp_fd, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, ensure_ascii=False, indent=2)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp_name, path)
    except Exception:
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise


def sort_results(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(results, key=lambda row: row.get("source_index", row.get("__source_index__", 0)))


def calc_metrics(results: list[dict[str, Any]]) -> dict[str, Any]:
    """Overall accuracy + tool-call + non-empty stats.

    `is_correct=None` (free-form benchmarks) is not counted toward
    accuracy, only toward `non_empty_ratio`.
    """

    total = len(results)
    if total == 0:
        return {"count": 0, "correct_count": 0, "accuracy": 0.0, "non_empty_ratio": 0.0, "avg_tool_calls": 0.0}

    correct = sum(1 for row in results if row.get("is_correct") is True)
    non_empty = sum(1 for row in results if row.get("predicted_option") or row.get("prediction"))
    tool_calls = [int(row.get("tool_call_num", 0) or 0) for row in results]
    grounded_total = sum(1 for row in results if row.get("is_correct") is not None)
    accuracy = correct / grounded_total if grounded_total > 0 else 0.0
    return {
        "count": total,
        "correct_count": correct,
        "accuracy": accuracy,
        "non_empty_ratio": non_empty / total if total else 0.0,
        "avg_tool_calls": sum(tool_calls) / total if total else 0.0,
    }


def metrics_by_key(results: list[dict[str, Any]], key: str) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in results:
        k = row.get(key)
        if k is None:
            continue
        grouped.setdefault(str(k), []).append(row)
    return {k: calc_metrics(rows) for k, rows in grouped.items()}


def build_metrics_payload(
    results: list[dict[str, Any]],
    args: argparse.Namespace,
    *,
    by_keys: Iterable[str] = ("difficulty", "question_type", "video_category"),
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "overall": calc_metrics(results),
        "args": {
            "bench": getattr(args, "bench", None),
            "model_name": getattr(args, "model_name", None),
            "model_reasoning_effort": getattr(args, "model_reasoning_effort", None),
            "concurrent_limit": getattr(args, "concurrent_limit", None),
            "item_timeout": getattr(args, "item_timeout", None),
            "sandbox": getattr(args, "sandbox", None),
            "allow_shell_network": getattr(args, "allow_shell_network", None),
            "allow_shell_gpu": getattr(args, "allow_shell_gpu", None),
            "outer_sandbox": getattr(args, "outer_sandbox", None),
            "agent_md_path": getattr(args, "agent_md_path", None),
        },
    }
    for key in by_keys:
        breakdown = metrics_by_key(results, key)
        if breakdown:
            payload[f"by_{key}"] = breakdown
    return payload


def write_results_and_metrics(
    results: list[dict[str, Any]],
    *,
    results_path: Path,
    metrics_path: Path,
    args: argparse.Namespace,
) -> None:
    sorted_results = sort_results(results)
    atomic_write_json(results_path, sorted_results)
    atomic_write_json(metrics_path, build_metrics_payload(sorted_results, args))


def build_run_paths(
    args: argparse.Namespace, *, dirname: str, timestamp: str,
) -> tuple[Path, Path, Path]:
    """Return (output_root, results_path, metrics_path).

    `output_root` is `args.output_dir / dirname / `.
    """

    output_root = Path(args.output_dir).resolve() / dirname
    output_root.mkdir(parents=True, exist_ok=True)
    return (
        output_root,
        output_root / f"run_{timestamp}_results.json",
        output_root / f"run_{timestamp}_metrics.json",
    )


def build_run_dir_name(model_name: str, effort: str | None, *, suffix: str = "") -> str:
    safe_model = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in str(model_name).strip()) or "model"
    parts = [safe_model]
    if effort:
        parts.append(effort)
    if suffix:
        parts.append(suffix)
    return "_".join(parts)
