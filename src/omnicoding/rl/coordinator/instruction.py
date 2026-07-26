"""Build Kira prompts for OmniCoding RL with the SFT harness contract.

The SFT trajectories are collected through :mod:`omnicoding.harnesses.kira`.
RL must therefore preserve the same split prompt shape:

``Kira SYSTEM_PROMPT + BenchSpec-style system prefix + user question``.

This module deliberately reuses the benchmark prompt renderers instead of
maintaining a second, shorter "RL prompt".  The latter is superficially
reasonable but is out-of-distribution for the 9B SFT checkpoint.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

from omnicoding.benchmarks.common.spec import (
    RELATIVE_PATH_HINT,
    BuildPromptCtx,
    build_continue_prompt,
)
from omnicoding.benchmarks.prompts.lvomnibench_prompting import (
    build_system_prefix as build_video_system_prefix,
    build_user_question as build_video_user_question,
)
from omnicoding.benchmarks.prompts.omnigaia_prompting import (
    build_system_prefix as build_omnigaia_system_prefix,
    build_user_question as build_omnigaia_user_question,
)
from omnicoding.harnesses.kira import WEB_SEARCH_PROMPT_HINT

from .dataset import Record


@dataclass(frozen=True, slots=True)
class KiraPrompt:
    """The three prompt pieces consumed by the production Kira loop."""

    system_prefix: str
    user_question: str
    continue_prompt: str


_OPEN_ANSWER_HINT = (
    "a single XML tag like <answer>FINAL_ANSWER</answer> containing only "
    "the computed answer text"
)
_MCQ_ANSWER_HINT = (
    "a single XML tag like <answer>A</answer> containing one option letter "
    "from the options shown in the task"
)


def _prompt_context(
    record: Record,
    staged_media: list[str],
    *,
    shared_python_env: str | None,
) -> BuildPromptCtx:
    return BuildPromptCtx(
        item={
            "id": record.id,
            "question": record.question,
            "options": list(record.options or []),
            "answer_type": record.answer_type,
            "source_dataset": record.source_dataset,
            "category": record.category,
        },
        staged_paths=[Path(value) for value in staged_media],
        sandbox="workspace-write",
        allow_shell_network=True,
        allow_shell_gpu=True,
        shared_python_env=shared_python_env,
        disable_native_vision=False,
        extra_system_prompt="",
    )


def build_kira_prompt(
    record: Record,
    staged_media: list[str],
    *,
    shared_python_env: str | None,
) -> KiraPrompt:
    """Return the same prompt layout used to collect Kira SFT data.

    ``Omnimodal-Agent-SFT-2K`` is an OmniGAIA source, so it uses the exact
    OmniGAIA renderer.  The three video MCQ sources use the established
    long-video MCQ renderer.  Both go through the common
    ``render_system_prefix`` / ``render_user_question`` implementation used
    by the SFT harness.
    """

    ctx = _prompt_context(
        record,
        staged_media,
        shared_python_env=shared_python_env,
    )
    if record.source_dataset == "Omnimodal-Agent-SFT-2K":
        system_prefix = build_omnigaia_system_prefix(ctx)
        user_question = build_omnigaia_user_question(ctx)
    else:
        system_prefix = build_video_system_prefix(ctx)
        user_question = build_video_user_question(ctx)

    # These are appended by run_bench_kira after the BenchSpec renderer.
    system_prefix += WEB_SEARCH_PROMPT_HINT + RELATIVE_PATH_HINT
    answer_hint = (
        _MCQ_ANSWER_HINT if record.answer_type == "mcq" else _OPEN_ANSWER_HINT
    )
    continue_prompt = build_continue_prompt(
        SimpleNamespace(answer_format_hint=answer_hint)
    )
    return KiraPrompt(
        system_prefix=system_prefix,
        user_question=user_question,
        continue_prompt=continue_prompt,
    )


def build_instruction(record: Record, staged_media: list[str]) -> str:
    """Compatibility wrapper returning only the per-item user message."""

    return build_kira_prompt(
        record,
        staged_media,
        shared_python_env="/workspace/.venv",
    ).user_question
