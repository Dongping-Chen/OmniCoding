"""Prompt and answer helpers for SocialOmni coding-agent runners."""

from __future__ import annotations

import re

from omnicoding.benchmarks.common.agent_environment import tool_workflow_instructions
from omnicoding.benchmarks.common.spec import BuildPromptCtx, render_system_prefix, render_user_question


# Spec-level constants — what makes SocialOmni different from peers.
# All three levels share the same preamble (workspace + network + gpu)
# but vary in their per-item question body.
_BENCHMARK_NAME = "SocialOmni benchmark item"
_STAGED_FILE_DESC = "files listed below"
_SCOPE = "multiple-choice item"
_MAX_COMMANDS = 5


def extract_answer_text(agent_output: str) -> str:
    matches = re.findall(r"<answer>(.*?)</answer>", agent_output or "", re.DOTALL | re.IGNORECASE)
    if matches:
        return matches[-1].strip()
    return (agent_output or "").strip()


def agentic_mcq_output_instruction(letters: str) -> str:
    """Per-item MCQ content rule: which letters are valid + an example.

    The general "wrap in <answer>X</answer> + emit as plain text + then
    task_complete" protocol lives in role=system (``FINAL_ANSWER_PROTOCOL``
    in ``common/spec.py``) for kira's split-prompt path. This function is
    still called by the legacy single-string Claude/codex paths
    (``build_level1_prompt`` etc.) which DON'T see the system prefix, so
    we keep the wrapper rule here for those callers — but minimal,
    de-duplicated against the system protocol.
    """
    normalized = ", ".join(letter for letter in letters.upper())
    example = letters.upper()[0]
    return (
        "Working process:\n"
        f"{tool_workflow_instructions(scope='multiple-choice item', max_commands=5)}\n"
        "- If a transcript, metadata result, or other tool output maps clearly to an option, answer immediately instead of collecting more evidence.\n"
        "\n"
        f"Answer with one option letter from: {normalized}. "
        f"Format: <answer>{example}</answer>. Do not put option text, "
        "explanation, punctuation, or extra words inside the tag."
    )


def agentic_utterance_output_instruction() -> str:
    """Per-item utterance rule (Level-2 Q2). See
    :func:`agentic_mcq_output_instruction` for why the wrapper rule
    is still here despite being in the system protocol — same single-
    string Claude/codex compatibility reason."""
    return (
        "Working process:\n"
        f"{tool_workflow_instructions(scope='single item', max_commands=5)}\n"
        "\n"
        "Answer with the concise utterance itself wrapped in "
        "<answer>utterance</answer>. No explanation, reasoning, labels, "
        "quotes, or markdown inside the tag."
    )


def normalize_mcq_prediction(answer_text: str, letters: str) -> str:
    text = extract_answer_text(answer_text)
    if not text:
        return ""

    allowed = set(letters.upper())
    allowed_pattern = "".join(sorted(allowed))
    upper = text.upper().strip()

    labeled_matches = re.findall(
        rf"\b(?:OPTION|ANSWER|CHOICE|CORRECT ANSWER)(?:\s+IS)?\s*[:：]?\s*[\*\(\[]*([{allowed_pattern}])(?:[\*\)\]\.\,\:\;\!\?\s]|$)",
        upper,
    )
    if labeled_matches:
        return labeled_matches[-1]

    bare = re.fullmatch(rf"[\s\*\(\[]*([{allowed_pattern}])[\s\*\)\]\.\,\:\;\!\?]*", upper)
    if bare:
        return bare.group(1)

    for raw_line in reversed(text.splitlines()):
        line = re.sub(r"^[>\-\*\s]+", "", raw_line).strip().upper()
        if not line:
            continue
        line_bare = re.fullmatch(rf"[\*\(\[]*([{allowed_pattern}])[\*\)\]\.\,\:\;\!\?]*", line)
        if line_bare:
            return line_bare.group(1)
        line_labeled = re.search(
            rf"\b(?:OPTION|ANSWER|CHOICE|CORRECT ANSWER)(?:\s+IS)?\s*[:：]?\s*[\*\(\[]*([{allowed_pattern}])(?:[\*\)\]\.\,\:\;\!\?\s]|$)",
            line,
        )
        if line_labeled:
            return line_labeled.group(1)

    tail = upper[-80:]
    tail_match = re.search(
        rf"[\s\*\(\[]([{allowed_pattern}])[\s\*\)\]\.\,\:\;\!\?]*$",
        f" {tail}",
    )
    if tail_match:
        return tail_match.group(1)

    return ""


def normalize_level2_q1_prediction(answer_text: str) -> str:
    choice = normalize_mcq_prediction(answer_text, "AB")
    if choice:
        return choice

    text = extract_answer_text(answer_text).lower()
    has_yes = bool(re.search(r"\byes\b", text))
    has_no = bool(re.search(r"\bno\b", text))
    if has_yes and not has_no:
        return "A"
    if has_no and not has_yes:
        return "B"
    return ""


def build_level1_prompt(
    sample: dict,
    *,
    include_audio: bool,
    local_video_path: str = "",
    user_prompt_base: str = "",
) -> str:
    asr_content = str(sample.get("asr_content") or "").strip() if include_audio else ""
    options = sample.get("options") or []

    prompt_parts: list[str] = [
        "[TASK]\n"
        "Solve this multiple-choice video/audio benchmark item by inspecting the staged local media with tools before choosing."
    ]
    if local_video_path:
        prompt_parts.append(f"[LOCAL_VIDEO]\nAnalyze the local video file at: {local_video_path}")
    if asr_content:
        prompt_parts.append(f"[ASR]\n{asr_content}")
    if options:
        prompt_parts.append("Options:\n" + "\n".join(str(option) for option in options))
    if user_prompt_base:
        prompt_parts.append(user_prompt_base)
    question = str(sample.get("question", "")).strip()
    if question:
        prompt_parts.append(f"Question:\n{question}")
    prompt_parts.append(agentic_mcq_output_instruction("ABCD"))
    return "\n\n".join(prompt_parts)


def build_level2_q1_prompt(
    sample: dict,
    *,
    include_audio: bool,
    local_video_path: str = "",
    user_prompt_base: str = "",
) -> str:
    q1 = sample.get("question_1", {}) or {}
    asr_content = str(sample.get("full_asr") or "").strip() if include_audio else ""
    option_a = str(q1.get("option_A") or "YES").strip()
    option_b = str(q1.get("option_B") or "NO").strip()

    prompt_parts: list[str] = [
        "[TASK]\n"
        "Solve this yes/no video/audio benchmark item by inspecting the staged local media with tools before choosing."
    ]
    if local_video_path:
        prompt_parts.append(f"[LOCAL_VIDEO]\nAnalyze the local prefix video file at: {local_video_path}")
    if asr_content:
        prompt_parts.append(f"[ASR]\n{asr_content}")
    prompt_parts.append(f"Options:\nA. {option_a}\nB. {option_b}")
    if user_prompt_base:
        prompt_parts.append(user_prompt_base)
    question = str(q1.get("question", "")).strip()
    if question:
        prompt_parts.append(f"Question:\n{question}")
    prompt_parts.append(agentic_mcq_output_instruction("AB"))
    return "\n\n".join(prompt_parts)


def build_level2_q2_prompt(
    sample: dict,
    *,
    include_audio: bool,
    local_video_path: str = "",
    user_prompt_base: str = "",
) -> str:
    q2 = sample.get("question_2", {}) or {}
    asr_content = str(sample.get("full_asr") or "").strip() if include_audio else ""

    prompt_parts: list[str] = [
        "[TASK]\n"
        "Answer this follow-up video/audio benchmark item by inspecting the staged local media with tools before responding."
    ]
    if local_video_path:
        prompt_parts.append(f"[LOCAL_VIDEO]\nAnalyze the local prefix video file at: {local_video_path}")
    if asr_content:
        prompt_parts.append(f"[ASR]\n{asr_content}")
    if user_prompt_base:
        prompt_parts.append(user_prompt_base)
    question = str(q2.get("question", "")).strip()
    if question:
        prompt_parts.append(f"Question:\n{question}")
    prompt_parts.append(agentic_utterance_output_instruction())
    return "\n\n".join(prompt_parts)


# ---------- unified split builders (kira / role-aware harnesses) ----

def build_system_prefix(ctx: BuildPromptCtx) -> str:
    """Static SocialOmni prefix using the unified renderer. Same across
    all three levels — only the per-item user_question varies."""
    return render_system_prefix(
        ctx=ctx,
        benchmark_name=_BENCHMARK_NAME,
        staged_file_description=_STAGED_FILE_DESC,
        scope=_SCOPE,
        max_commands=_MAX_COMMANDS,
        extras=None,
    )


def build_level1_user_question(ctx: BuildPromptCtx) -> str:
    """Per-item Level-1 user message — staged files + level-1 body
    (Question + Options + ABCD MCQ output instruction)."""
    rel = ctx.staged_paths[0].as_posix() if ctx.staged_paths else ""
    body = build_level1_prompt(ctx.item, include_audio=True, local_video_path=rel)
    return render_user_question(
        staged_paths=ctx.staged_paths,
        question_block=["", body],
    )


def build_level2_q1_user_question(ctx: BuildPromptCtx) -> str:
    """Per-item Level-2 Q1 user message (yes/no MCQ)."""
    rel = ctx.staged_paths[0].as_posix() if ctx.staged_paths else ""
    body = build_level2_q1_prompt(ctx.item, include_audio=True, local_video_path=rel)
    return render_user_question(
        staged_paths=ctx.staged_paths,
        question_block=["", body],
    )


def build_level2_q2_user_question(ctx: BuildPromptCtx) -> str:
    """Per-item Level-2 Q2 user message (free-text utterance)."""
    rel = ctx.staged_paths[0].as_posix() if ctx.staged_paths else ""
    body = build_level2_q2_prompt(ctx.item, include_audio=True, local_video_path=rel)
    return render_user_question(
        staged_paths=ctx.staged_paths,
        question_block=["", body],
    )
