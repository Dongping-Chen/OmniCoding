"""SocialOmni Level-2 question-1 BenchSpec (yes/no MCQ over A/B).

Level-2 items have a `question_1` (yes/no MCQ) and a `question_2`
(free-form utterance). This spec covers question_1 only — the
extracted prediction is the canonical "A" or "B" letter. Question_2
would be a separate spec; it's not exercised by the smoke tests.
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
from omnicoding.benchmarks.prompts.socialomni_prompting import (
    build_level2_q1_prompt,
    build_level2_q1_user_question as _build_level2_q1_user_question,
    build_system_prefix as _build_socialomni_system_prefix,
    extract_answer_text,
    normalize_level2_q1_prediction,
)


def _filter(items: list[dict[str, Any]], args: argparse.Namespace) -> list[dict[str, Any]]:
    items = filter_by_field_set(items, "video_id", getattr(args, "video_ids", None))
    if getattr(args, "max_items", None):
        items = items[: args.max_items]
    return items


def _stage_inputs(item: dict[str, Any], dataset_root: Path, workspace: Path) -> list[Path]:
    rel = str(item.get("video_file") or "").strip()
    if not rel:
        return []
    source = (dataset_root / rel).resolve() if not Path(rel).is_absolute() else Path(rel)
    if not source.exists():
        return []
    target = workspace / "inputs" / Path(rel).name
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)
    return [target.relative_to(workspace)]


def _build_codex(ctx: BuildPromptCtx) -> str:
    """Concat of unified split builders so codex-cli / mini-swe-agent /
    kira all see the same prompt structure."""
    return _build_socialomni_system_prefix(ctx) + "\n\n" + _build_level2_q1_user_question(ctx)


def _build_claude(ctx: BuildPromptCtx) -> str:
    rel = ctx.staged_paths[0].as_posix() if ctx.staged_paths else ""
    return build_level2_q1_prompt(ctx.item, include_audio=True, local_video_path=rel)


def _extract(text: str, item: dict[str, Any]) -> str:
    return normalize_level2_q1_prediction(extract_answer_text(text))


def _is_correct(item: dict[str, Any], prediction: str) -> Optional[bool]:
    q1 = item.get("question_1") or {}
    correct = str(q1.get("correct_answer") or "").strip().upper()
    if not correct:
        return None
    return bool(prediction) and prediction.upper() == correct


def _result_row(ctx: ResultRowCtx) -> dict[str, Any]:
    item = ctx.item
    q1 = item.get("question_1") or {}
    row: dict[str, Any] = {
        "source_index": item.get("__source_index__", ctx.item_index),
        "question_id": item.get("video_id"),
        "video_file": item.get("video_file"),
        "level": "level2_q1",
        "question": q1.get("question"),
        "options": [f"A. {q1.get('option_A') or 'YES'}", f"B. {q1.get('option_B') or 'NO'}"],
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
        row["correct_answer"] = str(q1.get("correct_answer") or "").strip().upper()
    if ctx.keep_workdirs:
        row["workspace_dir"] = str(ctx.workspace_dir)
    return row


SPEC = BenchSpec(
    name="socialomni_l2",
    iterate_items=load_json_items,
    filter_items=_filter,
    stage_inputs=_stage_inputs,
    build_codex_prompt=_build_codex,
    build_claude_prompt=_build_claude,
    extract_prediction=_extract,
    is_correct=_is_correct,
    result_row=_result_row,
    item_id=lambda item: str(item.get("video_id", item.get("__source_index__", "?"))),
    answer_format_hint="a single XML tag like <answer>A</answer> containing exactly A or B",
    build_system_prefix=_build_socialomni_system_prefix,
    build_user_question=_build_level2_q1_user_question,
)
