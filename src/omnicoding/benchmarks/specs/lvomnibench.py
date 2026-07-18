"""LVOmniBench BenchSpec.

LVOmniBench items are MCQ questions over a single video file. The spec
loads the data.json, filters by question_id/video_id/difficulty,
stages the per-item video into `inputs/videos/`, builds an MCQ prompt,
and extracts a single A/B/C/D letter from `<answer>X</answer>`.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path
from typing import Any, Optional

from omnicoding.benchmarks.common.spec import (
    BenchSpec,
    BuildPromptCtx,
    ResultRowCtx,
    filter_by_field_set,
    load_json_items,
)
from omnicoding.benchmarks.prompts.lvomnibench_prompting import (
    build_claude_prompt as _build_lvo_claude_prompt,
    build_codex_prompt as _build_lvo_codex_prompt,
    build_system_prefix as _build_lvo_system_prefix,
    build_user_question as _build_lvo_user_question,
    extract_choice as _lvo_extract_choice,
)


def _filter(items: list[dict[str, Any]], args: argparse.Namespace) -> list[dict[str, Any]]:
    items = filter_by_field_set(items, "question_id", getattr(args, "question_ids", None))
    items = filter_by_field_set(items, "video_id", getattr(args, "video_ids", None))
    if getattr(args, "difficulty", None):
        wanted = {value.strip().capitalize() for value in args.difficulty}
        items = [item for item in items if str(item.get("difficulty", "")).capitalize() in wanted]
    if getattr(args, "max_items", None):
        items = items[: args.max_items]
    return items


def _stage_inputs(item: dict[str, Any], dataset_root: Path, workspace: Path) -> list[Path]:
    video_id = str(item.get("video_id", "")).strip()
    if not video_id:
        return []
    candidates = [
        dataset_root / "videos" / f"{video_id}.mp4",
        dataset_root / "videos" / f"{video_id}.mkv",
        dataset_root / "videos" / video_id,
    ]
    source = next((p for p in candidates if p.exists()), None)
    if source is None:
        raise FileNotFoundError(
            f"LVOmniBench video not found for question_id={item.get('question_id')} "
            f"under {dataset_root}/videos (tried {[c.name for c in candidates]})."
        )
    target = workspace / "inputs" / "videos" / source.name
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)
    return [target.relative_to(workspace)]


def _build_codex(ctx: BuildPromptCtx) -> str:
    return _build_lvo_codex_prompt(ctx)


def _build_system(ctx: BuildPromptCtx) -> str:
    return _build_lvo_system_prefix(ctx)


def _build_user(ctx: BuildPromptCtx) -> str:
    return _build_lvo_user_question(ctx)


def _build_claude(ctx: BuildPromptCtx) -> str:
    return _build_lvo_claude_prompt(
        item=ctx.item,
        staged_paths=ctx.staged_paths,
        provider="claude",
        effort=None,
        gpu_enabled=ctx.allow_shell_gpu,
        shared_python_env=ctx.shared_python_env,
        extra_system_prompt=ctx.extra_system_prompt,
    )


def _extract(text: str, item: dict[str, Any]) -> str:
    return _lvo_extract_choice(text, list(item.get("options") or []))


def _is_correct(item: dict[str, Any], prediction: str) -> Optional[bool]:
    correct = str(item.get("correct_option", "")).strip().upper()
    if not correct:
        return None
    return bool(prediction) and prediction.upper() == correct


def _result_row(ctx: ResultRowCtx) -> dict[str, Any]:
    item = ctx.item
    row: dict[str, Any] = {
        "source_index": item.get("__source_index__", ctx.item_index),
        "question_id": item.get("question_id"),
        "video_id": item.get("video_id"),
        "video_category": item.get("video_category"),
        "sub_category": item.get("sub_category"),
        "duration": item.get("duration"),
        "question_type": item.get("question_type"),
        "difficulty": item.get("difficulty"),
        "question": item.get("question"),
        "options": item.get("options"),
        "predicted_option": ctx.prediction,
        "is_correct": ctx.is_correct,
        "raw_model_output": ctx.raw_model_output,
        "tool_call_num": ctx.tool_call_num,
        "return_code": ctx.return_code,
        "timed_out": ctx.timed_out,
        "stdout_text": ctx.stdout_text or "",
        "stderr_text": ctx.stderr_text or "",
    }
    row.update(ctx.extra)
    if ctx.include_gold_fields:
        row["answer"] = item.get("answer")
        row["correct_option"] = str(item.get("correct_option", "")).strip().upper()
    if ctx.keep_workdirs:
        row["workspace_dir"] = str(ctx.workspace_dir)
    return row


SPEC = BenchSpec(
    name="lvomnibench",
    iterate_items=load_json_items,
    filter_items=_filter,
    stage_inputs=_stage_inputs,
    build_codex_prompt=_build_codex,
    build_claude_prompt=_build_claude,
    extract_prediction=_extract,
    is_correct=_is_correct,
    result_row=_result_row,
    item_id=lambda item: str(item.get("question_id", item.get("__source_index__", "?"))),
    answer_format_hint="a single XML tag like <answer>A</answer> containing one option letter from A, B, C, or D",
    build_system_prefix=_build_system,
    build_user_question=_build_user,
)
