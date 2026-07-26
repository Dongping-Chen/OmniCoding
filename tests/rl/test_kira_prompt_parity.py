from __future__ import annotations

from pathlib import Path

from omnicoding.benchmarks.common.spec import RELATIVE_PATH_HINT, BuildPromptCtx
from omnicoding.benchmarks.prompts.lvomnibench_prompting import (
    build_system_prefix as build_video_system_prefix,
    build_user_question as build_video_user_question,
)
from omnicoding.benchmarks.prompts.omnigaia_prompting import (
    build_system_prefix as build_omnigaia_system_prefix,
    build_user_question as build_omnigaia_user_question,
)
from omnicoding.harnesses.kira import WEB_SEARCH_PROMPT_HINT
from omnicoding.rl.coordinator.dataset import Record
from omnicoding.rl.coordinator.instruction import build_kira_prompt


def _ctx(record: Record, media: list[str]) -> BuildPromptCtx:
    return BuildPromptCtx(
        item={
            "id": record.id,
            "question": record.question,
            "options": list(record.options or []),
            "answer_type": record.answer_type,
            "source_dataset": record.source_dataset,
            "category": record.category,
        },
        staged_paths=[Path(value) for value in media],
        sandbox="workspace-write",
        allow_shell_network=True,
        allow_shell_gpu=True,
        shared_python_env="/workspace/.venv",
        disable_native_vision=False,
        extra_system_prompt="",
    )


def _record(source: str, answer_type: str) -> Record:
    return Record(
        id="fixture:1",
        question="What is shown?",
        answer_type=answer_type,
        ground_truth=["secret-gold"],
        options=["A. cat", "B. dog"] if answer_type == "mcq" else None,
        media={"videos": ["ignored"], "audios": [], "images": []},
        source_dataset=source,
        category="fixture",
    )


def test_omnigaia_rl_prompt_is_exact_sft_harness_shape() -> None:
    record = _record("Omnimodal-Agent-SFT-2K", "open")
    media = ["inputs/media/images/sample.jpg"]
    prompt = build_kira_prompt(
        record, media, shared_python_env="/workspace/.venv"
    )
    ctx = _ctx(record, media)

    assert prompt.system_prefix == (
        build_omnigaia_system_prefix(ctx)
        + WEB_SEARCH_PROMPT_HINT
        + RELATIVE_PATH_HINT
    )
    assert prompt.user_question == build_omnigaia_user_question(ctx)
    assert "secret-gold" not in prompt.system_prefix + prompt.user_question
    assert "Please continue solving the task" in prompt.continue_prompt


def test_video_mcq_rl_prompt_is_exact_sft_harness_shape() -> None:
    record = _record("AVUTBenchmark", "mcq")
    media = ["inputs/videos/sample.mp4"]
    prompt = build_kira_prompt(
        record, media, shared_python_env="/workspace/.venv"
    )
    ctx = _ctx(record, media)

    assert prompt.system_prefix == (
        build_video_system_prefix(ctx)
        + WEB_SEARCH_PROMPT_HINT
        + RELATIVE_PATH_HINT
    )
    assert prompt.user_question == build_video_user_question(ctx)
    assert "<answer>A</answer>" in prompt.continue_prompt
