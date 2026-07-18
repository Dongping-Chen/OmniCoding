"""Per-item prompt builders + answer extractor for OmniGAIA.

OmniGAIA items combine open-ended factual reasoning with multi-modal
evidence: every question carries an `omni_modal_input` array of
video / image / audio paths plus a free-text question whose answer
typically requires both inspecting those media files AND searching the
web for grounding (the dataset's own annotated_solution explicitly
prescribes `web_search` as Step 2 in many cases).

Output contract: a single `<answer>...</answer>` tag holding the final
computed answer (string, number, or short phrase). `is_correct` is left
to the downstream evaluator — exact-match on free-text answers needs
LLM-as-judge to be meaningful.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from omnicoding.benchmarks.common.spec import BuildPromptCtx, render_system_prefix, render_user_question

# Spec-level constants — what makes OmniGAIA different from peers.
_BENCHMARK_NAME = "OmniGAIA benchmark item"
_STAGED_FILE_DESC = "multimodal files listed below"
_SCOPE = "item"
_MAX_COMMANDS = 12

WEB_SEARCH_HINT = (
    "A web_search tool is available (via shell command for mini-swe, MCP for opencode "
    "and claude). Many OmniGAIA items explicitly require web search to find a "
    "grounding fact (study size, regulatory threshold, person's affiliation, etc.). "
    "If the staged media alone cannot decide the answer, search the web — but never "
    "search for the dataset name, the question text, or the answer itself."
)

# OmniGAIA-specific extras appended after the standard tool_workflow
# block. Order matters: web_search hint LAST so it stays close to the
# task description that follows in role=user.
# Open-ended multimodal + web_search QA benefits from careful exploration.
# SFT-tuned variants short-circuit (~8 steps avg, vs base ~20+) and
# accuracy drops on hard items. See agent.md "lvomnibench SFT regression"
# notes (2026-05-05) for the same pattern.
_OMNIGAIA_EXPLORATION_RULE = (
    "Take time to triangulate before answering: many items require "
    "multiple web_search queries with different framings, cross-checking "
    "the returned URLs / dates / numbers, and ``image_read`` on any "
    "embedded media the question references. Do NOT call ``task_complete`` "
    "until at least two independent sources agree on the final value. "
    "Premature termination is the most common failure mode — when in "
    "doubt, run one more search or load one more image."
)
_OMNIGAIA_EXTRAS = [WEB_SEARCH_HINT, _OMNIGAIA_EXPLORATION_RULE]

def _build_question_block(item: dict[str, Any]) -> list[str]:
    # The general "wrap your answer in <answer></answer> + emit as plain
    # text + then task_complete" protocol is in role=system via
    # ``common/spec.py:FINAL_ANSWER_PROTOCOL`` — no need to re-state it
    # per item. Per-item content rules (e.g. MCQ "Answer with the option
    # letter only (e.g., A).") are already embedded in ``item['question']``
    # by the dataset, so the question text carries the per-item format
    # constraint.
    block = [
        "",
        "Question:",
        (item.get("question") or "").strip(),
    ]
    # MCQ items carry an `options` list (e.g. ["A. foo", "B. bar"]). Without
    # rendering it, the model gets a multiple-choice prompt with no choices —
    # accuracy collapses to ~25% (round 17.8 scale-up bug). Show the options
    # verbatim so the model can pick a letter and answer matches the
    # ground_truth list (which enumerates `A`, `A.`, `A. foo` variants).
    options = item.get("options")
    if isinstance(options, list) and options:
        block.append("")
        block.append("Options:")
        block.extend(str(opt) for opt in options)
    return block


def build_system_prefix(ctx: BuildPromptCtx) -> str:
    """Static OmniGAIA prefix — workspace boundary, network/sandbox
    rules, tool-workflow rules, web_search hint. Byte-identical across
    every item in a dispatch run, so the LLM provider's prompt cache
    can hold it once and only re-encode the per-item question."""
    return render_system_prefix(
        ctx=ctx,
        benchmark_name=_BENCHMARK_NAME,
        staged_file_description=_STAGED_FILE_DESC,
        scope=_SCOPE,
        max_commands=_MAX_COMMANDS,
        extras=_OMNIGAIA_EXTRAS,
    )


def build_user_question(ctx: BuildPromptCtx) -> str:
    """Per-item OmniGAIA user message — staged file list + Question +
    Options + answer-format hint. Pair with ``build_system_prefix``."""
    return render_user_question(
        staged_paths=ctx.staged_paths,
        question_block=_build_question_block(ctx.item),
    )


def build_codex_prompt(ctx: BuildPromptCtx) -> str:
    """Legacy concat shape (system_prefix + user_question) for callers
    without a real role=system slot — codex ``codex exec`` (single
    prompt arg) and mini-swe-agent's user-only flow. Kira's
    ``run_bench_kira.py`` calls the split builders directly so the
    static prefix can ride the LLM provider's prompt cache."""
    return build_system_prefix(ctx) + "\n\n" + build_user_question(ctx)


def build_claude_prompt(ctx: BuildPromptCtx) -> str:
    """Claude-style prompt — same content as the codex prompt; the
    claude-runner ingests the concat shape and the Anthropic API splits
    into role=system / role=user as needed."""
    return build_system_prefix(ctx) + "\n\n" + build_user_question(ctx)


_ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", re.DOTALL | re.IGNORECASE)
_PLACEHOLDER_ANSWERS = frozenset(
    {"...", "..", ".", "YOUR_ANSWER", "FINAL_ANSWER", "ANSWER", "TBD", "N/A"}
)


def _is_placeholder(s: str) -> bool:
    s = s.strip()
    if not s:
        return True
    upper = s.upper()
    if upper in _PLACEHOLDER_ANSWERS:
        return True
    # Strings of repeated punctuation only ("...", "----", etc.).
    return all(not c.isalnum() for c in s)


def extract_answer(text: str) -> str:
    """Return the LAST non-placeholder ``<answer>...</answer>`` payload.

    Why not just LAST: 9B-class SFT models sometimes write a real answer
    first and then keep thinking, emitting a stray placeholder ``<answer>...
    </answer>`` (e.g. literal "..." or "YOUR_ANSWER") at the very end. The
    placeholder shadows the real answer if we pick LAST blindly. Skipping
    obvious placeholders preserves the true intent without flipping cases
    where a clean self-correction overrides an earlier wrong answer.

    Falls back to the LAST raw block when every match is a placeholder.
    """
    if not text:
        return ""
    matches = [m.strip() for m in _ANSWER_RE.findall(text)]
    if not matches:
        return ""
    nontrivial = [m for m in matches if not _is_placeholder(m)]
    if nontrivial:
        return nontrivial[-1]
    return matches[-1]
