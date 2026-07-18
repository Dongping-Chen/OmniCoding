"""Prompt and answer helpers for LVOmniBench coding-agent runners."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from omnicoding.benchmarks.common.spec import BuildPromptCtx, render_system_prefix, render_user_question


# Spec-level constants — what makes LVOmniBench different from peers.
_BENCHMARK_NAME = "LVOmniBench long audio-video benchmark item"
_STAGED_FILE_DESC = "video file(s) listed below"
_SCOPE = "multiple-choice item"
_MAX_COMMANDS = 8


# Per-item content rule — LVOmniBench is always 4-option MCQ with
# letters A/B/C/D, so we tell the model the valid letter set here.
# The general "wrap in <answer>X</answer> + emit as plain text + then
# task_complete" protocol lives in role=system (FINAL_ANSWER_PROTOCOL
# in common/spec.py) and is not re-stated per item.
_LVO_LETTER_RULE = (
    "Answer with exactly one option letter from A, B, C, or D."
)


# Long-video MCQ benefits from sustained, careful exploration: extract
# multiple frame samples across the full timeline, transcribe relevant
# audio segments, ``image_read`` the candidate frames before deciding.
# Without this nudge the SFT-tuned 27B short-circuits to ~24 steps
# (vs ~58 for the base model on the same items) and accuracy drops
# ~12 percentage points. See agent.md "lvomnibench SFT regression"
# notes (2026-05-05) for the diagnosis.
_LVO_EXPLORATION_RULE = (
    "Take time to explore: this is a long-video item where details "
    "matter. Extract multiple frame samples from the full timeline (not "
    "just start/middle/end), transcribe relevant audio segments, and "
    "``image_read`` the candidate frames BEFORE committing to an answer. "
    "Do NOT call ``task_complete`` until you have triangulated the "
    "answer with at least three independent pieces of evidence "
    "(frames + audio + visible text/objects). Premature termination is "
    "the most common failure mode here — when in doubt, look at one "
    "more frame."
)


def build_options_str(options: list[str]) -> str:
    return "\n".join(str(option).strip() for option in options if str(option).strip())


def _build_question_block(item: dict[str, Any]) -> list[str]:
    question = str(item.get("question", "")).strip()
    options_str = build_options_str(item.get("options") or [])
    return [
        "",
        f"Question: {question}",
        "Options:",
        options_str,
        "",
        _LVO_LETTER_RULE,
    ]


def build_system_prefix(ctx: BuildPromptCtx) -> str:
    """Static LVOmniBench prefix using the unified renderer."""
    return render_system_prefix(
        ctx=ctx,
        benchmark_name=_BENCHMARK_NAME,
        staged_file_description=_STAGED_FILE_DESC,
        scope=_SCOPE,
        max_commands=_MAX_COMMANDS,
        extras=[_LVO_EXPLORATION_RULE],
    )


def build_user_question(ctx: BuildPromptCtx) -> str:
    """Per-item LVOmniBench user message — staged file list + Question
    + Options + final-answer-format instruction."""
    return render_user_question(
        staged_paths=ctx.staged_paths,
        question_block=_build_question_block(ctx.item),
    )


def build_codex_prompt(ctx: BuildPromptCtx) -> str:
    """Legacy concat shape for codex-cli + mini-swe-agent."""
    return build_system_prefix(ctx) + "\n\n" + build_user_question(ctx)


def build_claude_prompt(ctx: BuildPromptCtx) -> str:
    """Claude-style prompt — same content as the codex prompt; the
    claude-runner ingests the concat shape and the Anthropic API splits
    into role=system / role=user as needed."""
    return build_system_prefix(ctx) + "\n\n" + build_user_question(ctx)


_OPT_LIST_LINE = re.compile(r"^\s*[A-D][\.\)]\s+\S", re.IGNORECASE)
_LETTER_RE = re.compile(r"\b([A-D])\b")
# "the answer is A", "Answer: A", "I choose A", "final answer A",
# "correct option is A", "option A". Catches model prose answers
# that don't use the <answer>X</answer> tag.
_ANSWER_PHRASE_RE = re.compile(
    r"\b(?:answer\s+is\s*|answer\s*[:=]\s*|"
    r"(?:i\s+(?:will\s+)?(?:chose|choose|pick|select)\s+)|"
    r"final\s+answer\s*[:=]?\s*|"
    r"correct\s+(?:option|answer)\s+is\s*|"
    r"option\s+)\(?\*?([A-D])\*?\)?\b",
    re.IGNORECASE,
)


def extract_choice(text: str, options: list[str]) -> str:
    """Recover an A/B/C/D letter from agent output.

    Layered to keep false positives down — round 12 audit found three
    real-world traps: substring fallback hitting "Wife" in the prompt's
    option list, ``\\b[A-D]\\b`` matching ``Disp.A`` from ``nvidia-smi``,
    and the regex hitting the prompt's option list ``A. Wife`` letter
    when the kira loop replays the original instruction in a tool reply.
    """

    cleaned = (text or "").strip()
    if not cleaned:
        return ""

    # Layer 1: <answer>X</answer> — most reliable. Take the LAST tag so
    # a model that rehearses "<answer>A</answer> example" earlier in
    # thinking can still deliver "<answer>D</answer>" as the final.
    answer_matches = re.findall(r"<answer>(.*?)</answer>", cleaned, flags=re.IGNORECASE | re.DOTALL)
    for raw in reversed(answer_matches):
        tagged = raw.strip().upper()
        tagged_match = re.fullmatch(r"[\s\*\(\[]*([A-D])[\s\*\)\]\.\,\:\;\!\?]*", tagged)
        if tagged_match:
            return tagged_match.group(1)

    # Layer 2: explicit answer phrases ("the answer is A", "I choose D").
    # Take the LAST match so an early "first I considered A" doesn't
    # win over a later "the answer is D".
    phrase_matches = list(_ANSWER_PHRASE_RE.finditer(cleaned))
    if phrase_matches:
        return phrase_matches[-1].group(1).upper()

    # Layer 3: standalone-letter scan AFTER stripping option-list lines
    # (so the prompt's "A. Wife" doesn't leak through) AND with
    # adjacency guards against embedded tokens (so ``Disp.A | Volatile``
    # from nvidia-smi doesn't either). Match the original case so
    # "I need to use a tool" doesn't become a false ``A``. Take the LAST
    # surviving match.
    filtered_lines = [
        line for line in cleaned.splitlines() if not _OPT_LIST_LINE.match(line)
    ]
    filtered = "\n".join(filtered_lines)
    candidates: list[str] = []
    for m in _LETTER_RE.finditer(filtered):
        i = m.start()
        prev_ch = filtered[i - 1] if i > 0 else ""
        next_ch = filtered[m.end()] if m.end() < len(filtered) else ""
        # Reject letters preceded by '.' (token middle: ``Disp.A``)
        # or '/' / '_' / '-' (path / identifier middles like ``opt/A``,
        # ``var-A``). Letters followed by an alpha-numeric run via
        # punctuation (``A.Wife``) are caught by the ``_OPT_LIST_LINE``
        # filter above when they appear as their own line; mid-line
        # ``A.Wife`` is rare enough that we don't try to reject it
        # specifically.
        if prev_ch in {".", "/", "_", "-"}:
            continue
        # Reject when the next char is a hyphen-concatenated identifier
        # ("A-Class") — no real benchmark answer looks like that.
        if next_ch in {"-", "_"}:
            continue
        candidates.append(m.group(1))
    if candidates:
        return candidates[-1]

    # Layer 4: substring fallback — only when EXACTLY ONE option's text
    # appears. Multiple matches usually means the model enumerated
    # the choices ("Options are Wife, Friend, Mother, Colleague") not
    # answered, so we'd rather return "" than guess.
    normalized = " ".join(cleaned.split()).lower()
    option_map: dict[str, str] = {}
    for option in options:
        match = re.match(r"\s*([A-D])\.\s*(.+?)\s*$", str(option), flags=re.IGNORECASE)
        if not match:
            continue
        letter = match.group(1).upper()
        answer_text = match.group(2).strip().lower()
        option_map[answer_text] = letter

    if normalized in option_map:
        return option_map[normalized]

    matched_letters = {
        letter
        for answer_text, letter in option_map.items()
        if answer_text and answer_text in normalized
    }
    if len(matched_letters) == 1:
        return matched_letters.pop()
    return ""
