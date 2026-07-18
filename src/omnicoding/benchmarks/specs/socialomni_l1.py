"""SocialOmni Level-1 BenchSpec (one MCQ per sample, A/B/C/D)."""

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
    build_level1_prompt,
    build_level1_user_question as _build_level1_user_question,
    build_system_prefix as _build_socialomni_system_prefix,
    extract_answer_text,
    normalize_mcq_prediction,
)


def _filter(items: list[dict[str, Any]], args: argparse.Namespace) -> list[dict[str, Any]]:
    items = filter_by_field_set(items, "id", getattr(args, "question_ids", None))
    if getattr(args, "max_items", None):
        items = items[: args.max_items]
    return items


def _stage_inputs(item: dict[str, Any], dataset_root: Path, workspace: Path) -> list[Path]:
    rel = str(item.get("video_path") or "").strip()
    if not rel:
        return []
    source = (dataset_root / rel).resolve() if not Path(rel).is_absolute() else Path(rel)
    if not source.exists():
        return []
    target = workspace / "inputs" / Path(rel).name
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)
    return [target.relative_to(workspace)]


def _allowed_letters(item: dict[str, Any]) -> str:
    options = item.get("options") or []
    letters = "".join(str(opt).split(".")[0].strip().upper() for opt in options if "." in str(opt))
    return letters or "ABCD"


def _build_codex(ctx: BuildPromptCtx) -> str:
    """Concat of unified split builders so codex-cli / mini-swe-agent /
    kira all see the same prompt structure."""
    return _build_socialomni_system_prefix(ctx) + "\n\n" + _build_level1_user_question(ctx)


def _build_claude(ctx: BuildPromptCtx) -> str:
    rel = ctx.staged_paths[0].as_posix() if ctx.staged_paths else ""
    return build_level1_prompt(ctx.item, include_audio=True, local_video_path=rel)


def _extract(text: str, item: dict[str, Any]) -> str:
    answer = extract_answer_text(text)
    return normalize_mcq_prediction(answer, _allowed_letters(item))


def _is_correct(item: dict[str, Any], prediction: str) -> Optional[bool]:
    correct = str(item.get("correct_answer") or "").strip().upper()
    if not correct:
        return None
    return bool(prediction) and prediction.upper() == correct


def _result_row(ctx: ResultRowCtx) -> dict[str, Any]:
    item = ctx.item
    row: dict[str, Any] = {
        "source_index": item.get("__source_index__", ctx.item_index),
        "question_id": item.get("id"),
        "video_path": item.get("video_path"),
        "level": "level1",
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
        row["correct_answer"] = str(item.get("correct_answer") or "").strip().upper()
    if ctx.keep_workdirs:
        row["workspace_dir"] = str(ctx.workspace_dir)
    return row


SPEC = BenchSpec(
    name="socialomni_l1",
    iterate_items=load_json_items,
    filter_items=_filter,
    stage_inputs=_stage_inputs,
    build_codex_prompt=_build_codex,
    build_claude_prompt=_build_claude,
    extract_prediction=_extract,
    is_correct=_is_correct,
    result_row=_result_row,
    item_id=lambda item: str(item.get("id", item.get("__source_index__", "?"))),
    answer_format_hint="a single XML tag like <answer>A</answer> containing one option letter",
    build_system_prefix=_build_socialomni_system_prefix,
    build_user_question=_build_level1_user_question,
)
