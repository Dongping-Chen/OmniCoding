"""Run one kira trajectory end-to-end.

One worker call = one full agent loop. Returns a ``Trajectory`` ready for the
HTTP response. The kira run itself is sync; we wrap with ``asyncio.to_thread``
so the FastAPI loop can interleave many trajectories concurrently.
"""

from __future__ import annotations

import asyncio
import logging
import time
import traceback
from pathlib import Path

from omnicoding.agents.kira import AgentResult, KiraAgent

from omnicoding.rl.reward import (
    extract_answer_from_messages,
    normalize,
    should_remove_trajectory,
)
from omnicoding.rl.schemas import RolloutRequest, Trajectory

from .dataset import Record
from .instruction import CONTINUE_PROMPT, build_instruction

LOGGER = logging.getLogger("omnicoding.rl.coordinator.worker")


def _map_exit_reason(kira_exit: str) -> str:
    # Kira exit_reason set: task_complete | step_limit | no_tool_calls | error.
    # Schema also allows `timeout` and `context_overflow` for our own outer wrap.
    known = {"task_complete", "step_limit", "no_tool_calls", "context_overflow", "error"}
    return kira_exit if kira_exit in known else "error"


def _build_agent(workspace: Path, req: RolloutRequest) -> KiraAgent:
    return KiraAgent(
        workspace=workspace,
        model_name=req.sglang_model_name,
        provider="qwen",
        api_base=req.sglang_base_url,
        api_key="EMPTY",
        continue_prompt=CONTINUE_PROMPT,
        step_limit=req.max_turns,
        request_timeout_s=req.request_timeout_s,
        block_timeout_s=req.block_timeout_s,
        enable_thinking=bool(req.sampling_params.enable_thinking),
        temperature=req.sampling_params.temperature,
        top_p=req.sampling_params.top_p,
        max_tokens=req.sampling_params.max_tokens,
        seed=req.sampling_params.seed,
        # Slurm uses these variables to launch the worker. They are not agent
        # inputs and must not expose coordinator scratch paths to its shell.
        extra_env={"REQUEST_FILES": "", "RESULT_FILES": "", "OMNICODING_PYTHON": ""},
        image_subcall_log=workspace / "image_subcalls.jsonl",
    )


def _failed_trajectory(sample_index: int, reason: str, error: str) -> Trajectory:
    reward_details = {
        "score": 0.0,
        "correctness": 0.0,
        "raw_acc": 0.0,
        "format": 0.0,
        "modality_penalty": 0.0,
        "bad_tool_penalty": 0.0,
        "modality_match": 0.0,
        "p_bad_tool": 0.0,
        "n_tool": 0.0,
        "n_unparseable": 0.0,
        "n_disallowed": 0.0,
        "n_escape": 0.0,
        "n_syntax_fail": 0.0,
        "removed": 1.0,
        "num_steps": 0.0,
    }
    return Trajectory(
        sample_index=sample_index,
        messages=[],
        final_text="",
        extracted_answer=None,
        prediction_normalized=None,
        reward=0.0,
        outcome_reward=0.0,
        raw_outcome_reward=0.0,
        format_reward=0.0,
        removed=True,
        reward_details=reward_details,
        exit_reason=reason,  # "error" or "timeout"
        n_steps=0,
        n_tool_calls=0,
        cumulative_prompt_tokens=0,
        cumulative_completion_tokens=0,
        cumulative_reasoning_tokens=0,
        error=error,
    )


def _trajectory_from_result(
    sample_index: int,
    result: AgentResult,
) -> Trajectory:
    """Convert an agent result without access to answer keys.

    The coordinator applies reward grading after the worker returns. Keeping
    this worker-side object deliberately ungraded prevents answer keys from
    crossing the Slurm trust boundary.
    """
    extracted = extract_answer_from_messages(result.messages)
    exit_reason = _map_exit_reason(result.exit_reason)
    removed = should_remove_trajectory(exit_reason, extracted)
    return Trajectory(
        sample_index=sample_index,
        messages=result.messages,
        final_text=result.final_text,
        extracted_answer=extracted,
        prediction_normalized=normalize(extracted) if extracted else None,
        reward=0.0,
        outcome_reward=0.0,
        raw_outcome_reward=0.0,
        format_reward=0.0,
        removed=removed,
        reward_details={"ungraded": 1.0, "num_steps": float(result.n_steps)},
        exit_reason=exit_reason,
        n_steps=result.n_steps,
        n_tool_calls=result.n_tool_calls,
        cumulative_prompt_tokens=result.cumulative_prompt_tokens,
        cumulative_completion_tokens=result.cumulative_completion_tokens,
        cumulative_reasoning_tokens=result.cumulative_reasoning_tokens,
        error=result.error,
    )


async def run_one_trajectory(
    record: Record,
    sample_index: int,
    req: RolloutRequest,
    workspace: Path,
    staged_media: list[str],
) -> Trajectory:
    """Run one kira agent loop. Always returns a ``Trajectory`` — never raises."""

    workspace = workspace.resolve()
    workspace.mkdir(parents=True, exist_ok=True)
    started = time.time()
    LOGGER.info("rollout start id=%s sample=%d ws=%s", record.id, sample_index, workspace)

    try:
        instruction = build_instruction(record, staged_media)
        agent = _build_agent(workspace, req)

        # Cap entire trajectory wall clock so a single run can't hang the batch.
        deadline = req.request_timeout_s * (req.max_turns + 4)
        try:
            result = await asyncio.wait_for(asyncio.to_thread(agent.run, instruction), timeout=deadline)
        except asyncio.TimeoutError:
            LOGGER.warning("rollout timeout id=%s sample=%d after %ds", record.id, sample_index, deadline)
            return _failed_trajectory(sample_index, "timeout", f"trajectory exceeded {deadline}s wall clock")

        traj = _trajectory_from_result(sample_index, result)
        LOGGER.info(
            "rollout done id=%s sample=%d exit=%s steps=%d reward=%.1f elapsed=%.1fs",
            record.id, sample_index, traj.exit_reason, traj.n_steps, traj.reward, time.time() - started,
        )
        return traj
    except Exception as exc:  # noqa: BLE001 — surface ANY worker crash as a failed trajectory
        LOGGER.error("rollout error id=%s sample=%d: %s\n%s", record.id, sample_index, exc, traceback.format_exc())
        return _failed_trajectory(sample_index, "error", f"{type(exc).__name__}: {exc}")
