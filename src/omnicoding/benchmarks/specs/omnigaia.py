"""OmniGAIA BenchSpec.

Each item carries an `omni_modal_input` array of (type, id, path) media
references plus a free-form question. Staging copies every referenced
media file into the workspace under its original relative path.
Correctness is left to the downstream evaluator (free-text answers
need LLM-as-judge); the spec emits `is_correct=None` and a result row
that the evaluator can post-process.
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
from omnicoding.benchmarks.prompts.omnigaia_prompting import (
    build_claude_prompt as _build_omnigaia_claude_prompt,
    build_codex_prompt as _build_omnigaia_codex_prompt,
    build_system_prefix as _build_omnigaia_system_prefix,
    build_user_question as _build_omnigaia_user_question,
    extract_answer as _omnigaia_extract_answer,
)


def _filter(items: list[dict[str, Any]], args: argparse.Namespace) -> list[dict[str, Any]]:
    items = filter_by_field_set(items, "id", getattr(args, "question_ids", None))
    if getattr(args, "categories", None):
        wanted = {str(value).strip() for value in args.categories}
        items = [item for item in items if str(item.get("category", "")).strip() in wanted]
    if getattr(args, "max_items", None):
        items = items[: args.max_items]
    return items


def _resolve_media_source(dataset_root: Path, rel_path: str) -> Path:
    """Resolve a media reference against the dataset root.

    Items typically reference paths like `videos/<id>.mp4`,
    `images/<id>.png`, or `audios/<id>.wav`. We honor `..`-free joins
    to keep the staging confined to dataset_root."""
    rel = Path(rel_path)
    if rel.is_absolute():
        # Absolute paths in the dataset are unusual; fall back to a
        # treat-as-relative interpretation by stripping the leading slash.
        rel = Path(*rel.parts[1:]) if len(rel.parts) > 1 else rel
    return (dataset_root / rel).resolve()


def _stage_inputs(item: dict[str, Any], dataset_root: Path, workspace: Path) -> list[Path]:
    staged: list[Path] = []
    for media in item.get("omni_modal_input") or []:
        if not isinstance(media, dict):
            continue
        rel = media.get("path")
        if not isinstance(rel, str) or not rel.strip():
            continue
        source = _resolve_media_source(dataset_root, rel)
        if not source.exists():
            # Mirror the legacy behaviour: warn-but-skip so a single
            # missing file doesn't block the whole question. The prompt
            # still references whatever was staged successfully.
            continue
        target = (workspace / "inputs" / rel).resolve()
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        staged.append(target.relative_to(workspace))
    return staged


def _build_codex(ctx: BuildPromptCtx) -> str:
    return _build_omnigaia_codex_prompt(ctx)


def _build_system(ctx: BuildPromptCtx) -> str:
    return _build_omnigaia_system_prefix(ctx)


def _build_user(ctx: BuildPromptCtx) -> str:
    return _build_omnigaia_user_question(ctx)


def _build_claude(ctx: BuildPromptCtx) -> str:
    return _build_omnigaia_claude_prompt(ctx)


def _extract(text: str, item: dict[str, Any]) -> str:
    return _omnigaia_extract_answer(text)


def _is_correct(item: dict[str, Any], prediction: str) -> Optional[bool]:
    """OmniGAIA answers are free-text; correctness needs an LLM judge.
    Mark `is_correct=None` here and rely on the post-hoc evaluator."""
    return None


def _result_row(ctx: ResultRowCtx) -> dict[str, Any]:
    item = ctx.item
    row: dict[str, Any] = {
        "source_index": item.get("__source_index__", ctx.item_index),
        "id": item.get("id"),
        "category": item.get("category"),
        "Level": item.get("Level"),
        "task_type": item.get("task_type"),
        "total_steps": item.get("total_steps"),
        "required_external_tools": item.get("required_external_tools"),
        "question": item.get("question"),
        "omni_modal_input": item.get("omni_modal_input"),
        "predicted_answer": ctx.prediction,
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
        row["annotated_solution"] = item.get("annotated_solution")
        row["sources"] = item.get("sources")
    if ctx.keep_workdirs:
        row["workspace_dir"] = str(ctx.workspace_dir)
    return row


SPEC = BenchSpec(
    name="omnigaia",
    iterate_items=load_json_items,
    filter_items=_filter,
    stage_inputs=_stage_inputs,
    build_codex_prompt=_build_codex,
    build_claude_prompt=_build_claude,
    extract_prediction=_extract,
    is_correct=_is_correct,
    result_row=_result_row,
    item_id=lambda item: str(item.get("id", item.get("__source_index__", "?"))),
    answer_format_hint="a single XML tag like <answer>FINAL_ANSWER</answer> containing only the computed answer text",
    build_system_prefix=_build_system,
    build_user_question=_build_user,
)
