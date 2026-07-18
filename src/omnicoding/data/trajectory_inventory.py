#!/usr/bin/env python3
"""Inventory Kira trajectories and select one correct complete attempt per item."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


BENCHMARKS = (
    "omnigaia",
    "lvomnibench",
    "socialomni_l1",
    "socialomni_l2",
    "videozerobench",
)
PREDICTION_FIELDS = {
    "omnigaia": "predicted_answer",
    "lvomnibench": "predicted_option",
    "socialomni_l1": "predicted_option",
    "socialomni_l2": "predicted_option",
    "videozerobench": "level3_answer",
}


def infer_benchmark(path: Path) -> str | None:
    text = "/".join(path.parts).lower()
    return next((benchmark for benchmark in BENCHMARKS if benchmark in text), None)


def stable_item_id(row: dict[str, Any], benchmark: str) -> str | None:
    if benchmark == "omnigaia":
        value = row.get("id")
    else:
        value = row.get("question_id")
        if value is None:
            value = row.get("source_question_id")
    if value is not None:
        return str(value)
    if benchmark == "lvomnibench" and row.get("video_id") and row.get("question"):
        digest = hashlib.sha256(
            f"{row['video_id']}\n{row['question']}".encode("utf-8")
        ).hexdigest()[:16]
        return f"lv-{digest}"
    return None


def _normalise_prediction(value: Any) -> str:
    text = "" if value is None else str(value)
    return re.sub(r"\s+", " ", text).strip().casefold()


def _bool_verdict(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if value in (0, 1):
        return bool(value)
    if isinstance(value, str):
        normalised = value.strip().lower()
        if normalised in {"true", "yes", "correct", "1"}:
            return True
        if normalised in {"false", "no", "incorrect", "0"}:
            return False
    return None


def load_judgements(
    specs: Iterable[str],
) -> dict[tuple[str, str, str, str], tuple[bool, str]]:
    """Load RUN:BENCH:PATH:ID_FIELD:PRED_FIELD:VERDICT_FIELD specs."""
    judgements: dict[tuple[str, str, str, str], tuple[bool, str]] = {}
    for spec in specs:
        parts = spec.split(":", 5)
        if len(parts) != 6:
            raise ValueError(
                "judge spec must be RUN:BENCH:PATH:ID_FIELD:PRED_FIELD:VERDICT_FIELD"
            )
        run, benchmark, raw_path, id_field, prediction_field, verdict_field = parts
        if benchmark not in BENCHMARKS:
            raise ValueError(f"unknown benchmark in judge spec: {benchmark}")
        path = Path(raw_path)
        payload = json.loads(path.read_text(encoding="utf-8"))
        rows = payload.get("results") if isinstance(payload, dict) else payload
        if not isinstance(rows, list):
            raise ValueError(f"judge file {path} must contain a list")
        for row in rows:
            verdict = _bool_verdict(row.get(verdict_field))
            item_id = row.get(id_field)
            if verdict is None or item_id is None:
                continue
            key = (
                run,
                benchmark,
                str(item_id),
                _normalise_prediction(row.get(prediction_field)),
            )
            judgements[key] = (verdict, path.name)
    return judgements


def _iter_result_files(root: Path) -> Iterable[Path]:
    for directory, dirnames, filenames in os.walk(root):
        dirnames[:] = [name for name in dirnames if not name.startswith(".")]
        if "results.json" in filenames:
            yield Path(directory) / "results.json"


def _result_rows(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload.get("results") if isinstance(payload, dict) else payload
    if not isinstance(rows, list):
        raise ValueError(f"{path} must be a list or contain results[]")
    return [row for row in rows if isinstance(row, dict)]


def _has_nonempty_file(path: Path) -> bool:
    return path.is_file() and path.stat().st_size > 2


def _has_valid_json(path: Path) -> bool:
    if not _has_nonempty_file(path):
        return False
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return False
    return isinstance(payload, (list, dict)) and bool(payload)


def artifact_status(
    result_path: Path,
    row: dict[str, Any],
    *,
    validate_messages: bool = False,
) -> dict[str, bool]:
    source_index = row.get("source_index")
    try:
        item_dir = result_path.parent / f"item_{int(source_index):04d}"
    except (TypeError, ValueError):
        item_dir = result_path.parent / "item_unknown"
    images_dir = item_dir / "images"
    processed_images = images_dir.is_dir() and any(
        child.is_file() for child in images_dir.iterdir()
    )
    messages_path = item_dir / "messages.json"
    messages_present = _has_nonempty_file(messages_path)
    return {
        "messages_json": messages_present,
        "messages_json_valid": (
            _has_valid_json(messages_path) if validate_messages else messages_present
        ),
        "trajectory_json": _has_nonempty_file(item_dir / "trajectory.json"),
        "final_text": _has_nonempty_file(item_dir / "final_text.txt"),
        "image_subcalls": _has_nonempty_file(item_dir / "image_subcalls.jsonl"),
        "processed_images": processed_images,
    }


def _candidate(
    label: str,
    root: Path,
    result_path: Path,
    row: dict[str, Any],
    benchmark: str,
    judgements: dict[tuple[str, str, str, str], tuple[bool, str]],
) -> dict[str, Any] | None:
    item_id = stable_item_id(row, benchmark)
    if item_id is None:
        return None
    prediction = row.get(PREDICTION_FIELDS[benchmark])
    correctness = row.get("is_correct")
    correctness_source = "results.json:is_correct"
    if not isinstance(correctness, bool):
        match = judgements.get(
            (label, benchmark, item_id, _normalise_prediction(prediction))
        )
        if match:
            correctness, judge_name = match
            correctness_source = f"external_judge:{judge_name}"
        else:
            correctness = None
            correctness_source = "unscored"

    artifacts = artifact_status(
        result_path, row, validate_messages=correctness is True
    )
    complete = bool(
        not row.get("error")
        and artifacts["messages_json_valid"]
        and artifacts["final_text"]
    )
    source_index = row.get("source_index")
    try:
        item_dir = result_path.parent / f"item_{int(source_index):04d}"
        item_relative = str(item_dir.relative_to(root))
    except (TypeError, ValueError):
        item_relative = ""
    return {
        "run": label,
        "benchmark": benchmark,
        "item_id": item_id,
        "harness": row.get("harness", "kira"),
        "model": row.get("model"),
        "attempt": int(row.get("attempt") or 0),
        "correct": correctness,
        "correctness_source": correctness_source,
        "complete": complete,
        "protocol_completed": bool(row.get("completed")),
        "exit_reason": row.get("exit_reason"),
        "artifacts": artifacts,
        "result_file": str(result_path.relative_to(root)),
        "item_dir": item_relative,
    }


def _selection_score(candidate: dict[str, Any]) -> tuple[Any, ...]:
    artifacts = candidate["artifacts"]
    return (
        candidate["complete"],
        artifacts["trajectory_json"],
        artifacts["image_subcalls"],
        artifacts["processed_images"],
        candidate["attempt"],
        candidate["result_file"],
    )


def inventory_run(
    label: str,
    root: Path,
    judgements: dict[tuple[str, str, str, str], tuple[bool, str]] | None = None,
) -> dict[str, Any]:
    if not root.is_dir():
        raise FileNotFoundError(root)
    judgements = judgements or {}
    candidates: list[dict[str, Any]] = []
    unreadable: list[str] = []
    for result_path in _iter_result_files(root):
        benchmark = infer_benchmark(result_path.relative_to(root))
        if benchmark is None:
            continue
        try:
            rows = _result_rows(result_path)
        except (OSError, ValueError, json.JSONDecodeError):
            unreadable.append(str(result_path.relative_to(root)))
            continue
        for row in rows:
            candidate = _candidate(
                label, root, result_path, row, benchmark, judgements
            )
            if candidate:
                candidates.append(candidate)

    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for candidate in candidates:
        grouped[(candidate["benchmark"], candidate["item_id"])].append(candidate)

    selected: list[dict[str, Any]] = []
    for group in grouped.values():
        eligible = [row for row in group if row["correct"] is True and row["complete"]]
        if eligible:
            selected.append(max(eligible, key=_selection_score))
    selected.sort(key=lambda row: (row["benchmark"], row["item_id"]))

    summary: dict[str, dict[str, Any]] = {}
    for benchmark in BENCHMARKS:
        bench_rows = [row for row in candidates if row["benchmark"] == benchmark]
        if not bench_rows:
            continue
        unique_ids = {row["item_id"] for row in bench_rows}
        correct_rows = [row for row in bench_rows if row["correct"] is True]
        correct_ids = {row["item_id"] for row in correct_rows}
        selected_rows = [row for row in selected if row["benchmark"] == benchmark]
        selected_ids = {row["item_id"] for row in selected_rows}
        availability = Counter()
        for row in selected_rows:
            for key, present in row["artifacts"].items():
                availability[key] += int(present)
        summary[benchmark] = {
            "candidate_rows": len(bench_rows),
            "unique_items": len(unique_ids),
            "correct_candidate_rows": len(correct_rows),
            "correct_unique_items": len(correct_ids),
            "correct_complete_selected": len(selected_rows),
            "correct_items_missing_complete_artifacts": len(correct_ids - selected_ids),
            "correct_item_ids_missing_complete_artifacts": sorted(
                correct_ids - selected_ids
            ),
            "unscored_candidate_rows": sum(row["correct"] is None for row in bench_rows),
            "unselected_items": len(unique_ids) - len(selected_rows),
            "selected_artifact_availability": dict(availability),
        }

    for row in selected:
        blockers = ["content_sanitization_required"]
        if row["artifacts"]["processed_images"]:
            blockers.append("derived_benchmark_media_license_review")
        row["release_blockers"] = blockers
    return {
        "schema_version": 1,
        "run": label,
        "selection_policy": (
            "one externally or internally judged-correct, complete attempt per stable "
            "benchmark item; artifact completeness is independent of task_complete "
            "protocol status; prefer richer artifacts, then attempt number and stable path"
        ),
        "summary": summary,
        "unreadable_results": unreadable,
        "selected": selected,
    }


def _parse_run(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise ValueError("run must be LABEL=PATH")
    label, raw_path = value.split("=", 1)
    if not label or not raw_path:
        raise ValueError("run must be LABEL=PATH")
    return label, Path(raw_path).resolve()


def _parse_runs(values: Iterable[str]) -> list[tuple[str, Path]]:
    runs = [_parse_run(value) for value in values]
    labels = [label for label, _ in runs]
    duplicates = sorted(label for label, count in Counter(labels).items() if count > 1)
    if duplicates:
        raise ValueError(f"run labels must be unique; duplicates: {duplicates}")
    return runs


def _validate_output_location(output: Path, runs: list[tuple[str, Path]]) -> None:
    resolved = output.resolve()
    for label, root in runs:
        if resolved.is_relative_to(root):
            raise ValueError(
                f"output must be outside read-only run root {label!r}: {root}"
            )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="append", required=True, help="LABEL=PATH; repeatable")
    parser.add_argument(
        "--judge",
        action="append",
        default=[],
        help="RUN:BENCH:PATH:ID_FIELD:PRED_FIELD:VERDICT_FIELD; repeatable",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    try:
        runs = _parse_runs(args.run)
        _validate_output_location(args.output, runs)
        judgements = load_judgements(args.judge)
        payload = {
            "schema_version": 1,
            "runs": [inventory_run(label, root, judgements) for label, root in runs],
        }
    except (FileNotFoundError, ValueError) as exc:
        parser.error(str(exc))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
