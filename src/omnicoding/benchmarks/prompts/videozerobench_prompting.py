"""Single-question VideoZeroBench prompt + answer extraction.

VZB now runs as one inference per question: model sees the staged video
plus the question text, then emits its final answer in a single
``<answer>...</answer>`` tag (free text). Level-1/2/4/5 are not
exercised; only Level-3 (answer-only, no benchmark-provided evidence)
is wired through this code path. The official evaluator scores by
normalize-exact-match against ``answer`` in the annotation file —
``specs.videozerobench`` writes ``level3_answer`` at the row's top
level so ``evaluate_videozerobench.evaluate`` reads it directly.

Mirrors the LVOmniBench / SocialOmni layout: ``build_codex_prompt``
adds the codex sandbox preamble, ``build_claude_prompt`` skips it; the
inner body and the ``<answer>`` output instruction are shared.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

from omnicoding.benchmarks.common.spec import BuildPromptCtx, render_system_prefix, render_user_question


LOGGER = logging.getLogger(__name__)


# Spec-level constants — what makes VideoZeroBench different from peers.
_BENCHMARK_NAME = "VideoZeroBench item"
_STAGED_FILE_DESC = "video file listed below"
_SCOPE = "single VideoZeroBench item"
_MAX_COMMANDS = 12


_ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", re.DOTALL | re.IGNORECASE)

# BUG-X1 lenient-fallback pattern. Only fires when the strict
# ``<answer>`` tag is absent. Anchored to natural-language "this is my
# final answer" phrasing so it cannot match early-trajectory ffprobe /
# find / ls output. The capture group must be a short (<=80 char)
# one-line payload — long captures are dropped (fail-closed) so a
# runaway prose block can't be mistaken for an answer. ``finditer``
# yields matches in order; the LAST one wins (matches the strict-parse
# behavior of preferring the model's last commitment over earlier
# rehearsal).
#
# All variants require an explicit "answer" / "final answer" cue before
# the capture; we deliberately do NOT match "the X is Y" (would pull in
# a sentence subject like "username", "result", etc. with no
# ground-truth correspondence).
#
# Order of alternatives matters: the longer "the answer is/=/:"
# alternative comes first so we capture the full cue before falling
# back to the bare "final answer" / "answer" prefix (otherwise
# "the answer is X" would split into a bare "answer" cue plus an
# "is X" payload).
_FINAL_ANSWER_RE = re.compile(
    r"(?im)"
    r"(?:^|[\.\n,;:]\s*)"              # boundary: BOL or punctuation
    r"(?:(?:my|our|so\s+the|therefore\s+the|hence\s+the)\s+)?"
    r"(?:\*\*)?"                       # optional bold open
    r"(?:"
    r"the\s*answer\s*(?:is|=|:)"      # "the answer is/=/:"
    r"|"
    r"final\s*answer"                  # "final answer"
    r"|"
    r"answer"                          # bare "answer"
    r")"
    r"(?:\*\*)?"                       # optional bold close
    r"\s*[:=]?\s*"                     # optional colon/equals
    r"(?:\*\*)?"                       # optional bold around payload
    r"`?([^\n`*]{1,80}?)`?"            # 1..80 chars, no newlines/backticks/asterisks
    r"(?:\*\*)?"                       # optional bold close after payload
    r"\s*[\.,;!\?]?\s*(?:$|\n)"        # end of line / sentence
)
# Trim trailing markdown emphasis / punctuation we never want in the
# answer payload (model often writes e.g. "**5**" or "the answer is 8.").
_TRAILING_TRIM_RE = re.compile(r"[\s\.\*`'\":,;!\?]+$")
_LEADING_TRIM_RE = re.compile(r"^[\s\*`'\":]+")
# Reject candidates that are pure filler (e.g. "the", "yes", "above")
# rather than a substantive answer. Empty string after stripping these
# is treated as no match. We also reject candidates that *start* with
# a non-committal hedge (e.g. "unclear from..."): a confident answer
# would not lead with these words.
_REJECT_FILLER = {
    "", "the", "above", "below", "yes", "no", "none", "n/a", "na",
    "unknown", "unclear", "inconclusive", "indeterminate", "ambiguous",
    "unable", "uncertain",
}
_REJECT_PREFIX = (
    "unclear", "unknown", "uncertain", "unable", "inconclusive",
    "indeterminate", "ambiguous", "not ", "i ", "we ", "based ",
    "somewhere ", "probably ", "possibly ", "likely ",
)


def _lenient_final_answer(text: str) -> str:
    """Last-resort recovery for runs where the model wrote its final
    answer in prose without the ``<answer>`` wrapper (typically because
    its shell echo hung and the model gave up but still claimed an
    answer in its closing paragraph).

    Conservative by design: only the LAST ~4 KB of ``text`` is searched
    so we never pick up early ``find . -name "*.mp4"`` output as the
    "answer"; only patterns anchored on an explicit "Final Answer" /
    "the answer is" / "Answer:" cue fire. Returns "" if nothing safe
    matches.
    """
    if not text:
        return ""
    tail = text[-4000:]
    last_match: str | None = None
    for m in _FINAL_ANSWER_RE.finditer(tail):
        cand = m.group(1).strip()
        cand = _TRAILING_TRIM_RE.sub("", cand)
        cand = _LEADING_TRIM_RE.sub("", cand)
        if not cand:
            continue
        # Reject obvious non-answers: whole sentences, filler tokens,
        # or anything with embedded newlines.
        if len(cand) > 80 or "\n" in cand:
            continue
        cand_lower = cand.lower()
        if cand_lower in _REJECT_FILLER:
            continue
        if any(cand_lower.startswith(p) for p in _REJECT_PREFIX):
            continue
        last_match = cand
    return last_match or ""


def extract_answer_text(text: str) -> str:
    """Return the LAST ``<answer>...</answer>`` match, stripped.

    Empty string when no tag is present and the lenient fallback also
    finds nothing. Last-match (not first) handles the Qwen3.6-thinking
    case where the model rehearses an example tag inside ``<think>``
    before committing the real one outside.

    BUG-X1: when no strict tag is present we run a conservative
    lenient fallback (see ``_lenient_final_answer``) so that runs where
    the model wrote its answer in prose only — typically because its
    shell echo hung — can still be scored against the gold normaliser
    rather than counted as silent zeros.
    """
    matches = _ANSWER_RE.findall(text or "")
    if matches:
        return matches[-1].strip()
    return _lenient_final_answer(text or "")


_VZB_WORKING_PROCESS_HINT = "\n".join(
    [
        "Working process:",
        "- Inspect the staged video with ffprobe / ffmpeg / python before answering.",
        "- Stop inspecting as soon as you have direct visual, audio, or textual evidence to decide.",
    ]
)

# VZB-specific content rule: the answer is free text (number, phrase,
# or short label) — there are no options. The general "wrap in
# <answer>X</answer> + emit as plain text + then task_complete"
# protocol lives in role=system (FINAL_ANSWER_PROTOCOL in
# common/spec.py); the per-item rule below only adds VZB-specific
# guidance about what KIND of payload goes in the wrapper.
_VZB_PAYLOAD_RULE = (
    "Inside the <answer> wrapper, put only a concise final answer — a number, "
    "phrase, or short label. No reasoning, labels, quotes, markdown, or extra "
    "commentary."
)

# Long-video QA benefits from sustained, careful exploration. SFT-tuned
# variants tend to short-circuit (~24 steps) vs the base model (~58 steps)
# and accuracy drops. See agent.md "lvomnibench SFT regression" notes
# (2026-05-05) — same root cause shows up on VZB long-tail items.
_VZB_EXPLORATION_RULE = (
    "Take time to explore: this is a video item where details matter. "
    "Sample multiple frames across the full timeline (not just start/middle/"
    "end), transcribe relevant audio segments, and ``image_read`` the "
    "candidate frames BEFORE committing to an answer. Do NOT call "
    "``task_complete`` until you have triangulated the answer with at least "
    "three independent pieces of evidence. Premature termination is the "
    "most common failure mode here — when in doubt, look at one more frame."
)

# Extras appended to the unified system prefix for VZB. Working-process
# advice + payload rule are both harness-static (don't vary per item).
_VZB_EXTRAS = [_VZB_WORKING_PROCESS_HINT, _VZB_EXPLORATION_RULE, _VZB_PAYLOAD_RULE]


def build_codex_prompt(ctx: BuildPromptCtx) -> str:
    """Codex-style single-question VZB prompt — concat of the unified
    split builders so codex-cli, mini-swe-agent and kira all see the
    same prompt structure (system_prefix + user_question)."""
    LOGGER.info(
        "vzb.build_codex_prompt qid=%s video=%s sandbox=%s gpu=%s",
        ctx.item.get("question_id"),
        ctx.item.get("video"),
        ctx.sandbox,
        ctx.allow_shell_gpu,
    )
    return build_system_prefix(ctx) + "\n\n" + build_user_question(ctx)


def _build_vzb_question_block(item: dict) -> list[str]:
    """Per-item Question block: video id + question + final-answer hint
    (the working-process / answer-format text is in the system prefix
    via ``_VZB_EXTRAS`` so it stays static across items)."""
    video_id = str(item.get("video") or item.get("video_id") or "")
    question = str(item.get("question") or "").strip()
    return [
        "",
        f"Video id / filename: {video_id}",
        "",
        f"Question:\n{question}",
    ]


def build_system_prefix(ctx: BuildPromptCtx) -> str:
    """Static VZB prefix using the unified renderer. NETWORK_FORBIDDEN_
    TARGET is part of the unified ``network_instructions`` call (we pass
    ``allow_shell_network`` through ctx — the helper's default forbidden
    target is broad enough to cover VZB's restrictions); the working-
    process / final-answer hints land in extras."""
    return render_system_prefix(
        ctx=ctx,
        benchmark_name=_BENCHMARK_NAME,
        staged_file_description=_STAGED_FILE_DESC,
        scope=_SCOPE,
        max_commands=_MAX_COMMANDS,
        extras=_VZB_EXTRAS,
    )


def build_user_question(ctx: BuildPromptCtx) -> str:
    """Per-item VZB user message — staged video + video id + question."""
    return render_user_question(
        staged_paths=ctx.staged_paths,
        question_block=_build_vzb_question_block(ctx.item),
    )


def build_claude_prompt(ctx: BuildPromptCtx) -> str:
    """Claude-style single-question VZB prompt — same content as the
    codex prompt; claude-runner ingests the concat shape and lets the
    Anthropic API split into role=system / role=user as needed."""
    LOGGER.info(
        "vzb.build_claude_prompt qid=%s video=%s gpu=%s",
        ctx.item.get("question_id"),
        ctx.item.get("video"),
        ctx.allow_shell_gpu,
    )
    return build_system_prefix(ctx) + "\n\n" + build_user_question(ctx)
