"""VideoZeroBench BenchSpec — Level-3 only, one inference per question.

Each item is one VZB question. The model sees one staged video plus the
question text and emits a single ``<answer>...</answer>`` tag. We score
Level-3 with normalize-exact-match against ``answer`` in the annotation
file (the same normalization the official evaluator uses for L3) and
write ``level3_answer`` at the result row's top level so
``evaluate_videozerobench.evaluate`` can read it directly without a
shape adapter.
"""

from __future__ import annotations

import argparse
import logging
import math
import re
import shutil
import unicodedata
from pathlib import Path
from typing import Any, Optional

from omnicoding.benchmarks.common.spec import (
    BenchSpec,
    BuildPromptCtx,
    ResultRowCtx,
    filter_by_field_set,
    load_json_items,
)
from omnicoding.benchmarks.prompts.videozerobench_prompting import (
    build_claude_prompt as _build_claude_prompt,
    build_codex_prompt as _build_codex_prompt,
    build_system_prefix as _build_vzb_system_prefix,
    build_user_question as _build_vzb_user_question,
    extract_answer_text,
)


LOGGER = logging.getLogger(__name__)


def _filter(items: list[dict[str, Any]], args: argparse.Namespace) -> list[dict[str, Any]]:
    items = filter_by_field_set(items, "question_id", getattr(args, "question_ids", None))
    items = filter_by_field_set(items, "video_id", getattr(args, "video_ids", None))
    if getattr(args, "languages", None):
        wanted = {str(value).strip() for value in args.languages}
        items = [item for item in items if str(item.get("language", "")).strip() in wanted]
    if getattr(args, "categories", None):
        wanted = {str(value).strip() for value in args.categories}
        items = [item for item in items if str(item.get("category", "")).strip() in wanted]
    if getattr(args, "max_items", None):
        items = items[: args.max_items]
    return items


def _stage_inputs(item: dict[str, Any], dataset_root: Path, workspace: Path) -> list[Path]:
    video = str(item.get("video") or item.get("video_id") or "").strip()
    if not video:
        return []
    source = (dataset_root / "compressed" / video).resolve()
    if not source.exists():
        LOGGER.warning("vzb.stage_inputs missing video=%s expected=%s", video, source)
        return []
    target = workspace / "inputs" / "videos" / source.name
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)
    return [target.relative_to(workspace)]


def _build_codex(ctx: BuildPromptCtx) -> str:
    return _build_codex_prompt(ctx)


def _build_claude(ctx: BuildPromptCtx) -> str:
    return _build_claude_prompt(ctx)


def _extract(text: str, item: dict[str, Any]) -> str:
    """Return the ``<answer>`` payload as a free-text string. Empty
    string ⇒ no tag found, which triggers the harness's continue-retry."""
    answer = extract_answer_text(text)
    LOGGER.info(
        "vzb.extract qid=%s len_in=%d len_out=%d",
        item.get("question_id"), len(text or ""), len(answer),
    )
    return answer


# Mirrors evaluate_videozerobench.normalize_answer: numeric values
# canonicalize via float (so "8", "8.0", "08" all match), otherwise
# NFKC + lowercase + replace punctuation/symbols with spaces + collapse
# whitespace. Replicated here (not imported) because:
#   - evaluate_videozerobench lives in the videozerobench/ data dir,
#     not on the spec's import path under harnesses/run_*;
#   - the function is small and stable (the official paper pinned this
#     normalization rule).
_NUMERIC_RE = re.compile(r"[-+]?[0-9]+(?:\.[0-9]+)?")


def _normalize(value: Any) -> str:
    if value is None:
        return ""
    text = unicodedata.normalize("NFKC", str(value)).strip()
    if not text:
        return ""
    cleaned = text.replace(",", "")
    if _NUMERIC_RE.fullmatch(cleaned):
        numeric = float(cleaned)
        if math.isfinite(numeric) and numeric.is_integer():
            return str(int(numeric))
        return f"{numeric:.6f}".rstrip("0").rstrip(".")
    chars: list[str] = []
    for ch in text.lower():
        category = unicodedata.category(ch)
        if category.startswith("P") or category.startswith("S"):
            chars.append(" ")
        else:
            chars.append(ch)
    return re.sub(r"\s+", " ", "".join(chars)).strip()


def _is_correct(item: dict[str, Any], prediction: str) -> Optional[bool]:
    gold = item.get("answer")
    if gold is None or str(gold).strip() == "":
        return None
    if not prediction:
        return False
    correct = _normalize(prediction) == _normalize(gold)
    LOGGER.info(
        "vzb.is_correct qid=%s pred=%r gold=%r => %s",
        item.get("question_id"), prediction, gold, correct,
    )
    return correct


def _result_row(ctx: ResultRowCtx) -> dict[str, Any]:
    item = ctx.item
    row: dict[str, Any] = {
        "source_index": item.get("__source_index__", ctx.item_index),
        "question_id": item.get("question_id"),
        "video_id": item.get("video_id"),
        "video": item.get("video"),
        "language": item.get("language"),
        "category": item.get("category"),
        "level3_answer": ctx.prediction,
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
    if ctx.keep_workdirs:
        row["workspace_dir"] = str(ctx.workspace_dir)
    return row


SPEC = BenchSpec(
    name="videozerobench",
    iterate_items=load_json_items,
    filter_items=_filter,
    stage_inputs=_stage_inputs,
    build_codex_prompt=_build_codex,
    build_claude_prompt=_build_claude,
    extract_prediction=_extract,
    is_correct=_is_correct,
    result_row=_result_row,
    item_id=lambda item: str(item.get("question_id", item.get("__source_index__", "?"))),
    answer_format_hint=(
        "a single XML tag like <answer>YOUR_ANSWER</answer> containing "
        "the concise final answer text — no JSON, no markdown, nothing else"
    ),
    build_system_prefix=lambda ctx: _build_vzb_system_prefix(ctx),
    build_user_question=lambda ctx: _build_vzb_user_question(ctx),
)
