"""Run one kira trajectory end-to-end.

One worker call = one full agent loop. Returns a ``Trajectory`` ready for the
HTTP response. The kira run itself is sync; we wrap with ``asyncio.to_thread``
so the FastAPI loop can interleave many trajectories concurrently.
"""

from __future__ import annotations

import asyncio
import logging
import os
import subprocess
import sys
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
from .instruction import KiraPrompt, build_kira_prompt

LOGGER = logging.getLogger("omnicoding.rl.coordinator.worker")


def _map_exit_reason(kira_exit: str) -> str:
    # Kira exit_reason set: task_complete | step_limit | no_tool_calls | error.
    # Schema also allows `timeout` and `context_overflow` for our own outer wrap.
    known = {"task_complete", "step_limit", "no_tool_calls", "context_overflow", "error"}
    return kira_exit if kira_exit in known else "error"


def _setup_writable_python(workspace: Path) -> tuple[Path, dict[str, str]]:
    """Create a private package overlay inside the immutable container.

    The worker image is intentionally read-only.  A lightweight
    ``--system-site-packages`` venv gives the model writable ``pip install``
    semantics without copying or modifying the shared image packages.
    """

    venv_root = workspace / ".venv"
    if not (venv_root / "bin" / "python").exists():
        subprocess.run(
            [
                sys.executable,
                "-m",
                "venv",
                "--copies",
                "--system-site-packages",
                str(venv_root),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
    inherited_path = os.environ.get("PATH", "/usr/local/bin:/usr/bin:/bin")
    harness_venv_raw = os.environ.get("OMNICODING_HARNESS_VENV", "").strip()
    harness_bin: Path | None = None
    harness_library_paths: list[str] = []
    if harness_venv_raw:
        harness_venv = Path(harness_venv_raw)
        harness_bin = harness_venv / "bin"
        harness_site_packages = sorted(
            (harness_venv / "lib").glob("python*/site-packages")
        )
        if not harness_site_packages:
            raise RuntimeError(
                f"harness venv has no site-packages directory: {harness_venv}"
            )
        local_site_packages = (
            venv_root
            / "lib"
            / f"python{sys.version_info.major}.{sys.version_info.minor}"
            / "site-packages"
        )
        local_site_packages.mkdir(parents=True, exist_ok=True)
        (local_site_packages / "omnicoding_harness.pth").write_text(
            "\n".join(str(path) for path in harness_site_packages) + "\n"
        )
        for site_packages in harness_site_packages:
            harness_library_paths.extend(
                str(path) for path in sorted((site_packages / "nvidia").glob("*/lib"))
            )

    path_parts = [str(venv_root / "bin")]
    if harness_bin is not None:
        path_parts.append(str(harness_bin))
    path_parts.extend(["/usr/local/bin", inherited_path])
    env = {
        "PATH": os.pathsep.join(path_parts),
        "VIRTUAL_ENV": str(venv_root),
        "PIP_CACHE_DIR": str(workspace / ".cache" / "pip"),
        "PYTHONUSERBASE": str(workspace / ".local"),
        "REQUEST_FILES": "",
        "RESULT_FILES": "",
        "OMNICODING_PYTHON": "",
    }
    if harness_library_paths:
        inherited_library_path = os.environ.get("LD_LIBRARY_PATH", "")
        env["LD_LIBRARY_PATH"] = os.pathsep.join(
            harness_library_paths
            + ([inherited_library_path] if inherited_library_path else [])
        )
    return venv_root, env


def _build_agent(
    workspace: Path,
    req: RolloutRequest,
    *,
    prompt: KiraPrompt,
    extra_env: dict[str, str],
) -> KiraAgent:
    return KiraAgent(
        workspace=workspace,
        model_name=req.sglang_model_name,
        provider="qwen",
        api_base=req.sglang_base_url,
        api_key="EMPTY",
        continue_prompt=prompt.continue_prompt,
        step_limit=req.max_turns,
        request_timeout_s=req.request_timeout_s,
        block_timeout_s=req.block_timeout_s,
        enable_thinking=bool(req.sampling_params.enable_thinking),
        temperature=req.sampling_params.temperature,
        top_p=req.sampling_params.top_p,
        max_tokens=req.sampling_params.max_tokens,
        seed=req.sampling_params.seed,
        # Slurm uses the cleared variables to launch the worker. They are not
        # agent inputs and must not expose coordinator scratch paths.
        extra_env=extra_env,
        image_subcall_log=workspace / "image_subcalls.jsonl",
        max_tool_reminders=10,
        enable_summarize=True,
        image_read_mode="native",
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
        job_venv, extra_env = await asyncio.to_thread(
            _setup_writable_python, workspace
        )
        prompt = build_kira_prompt(
            record,
            staged_media,
            shared_python_env=str(job_venv),
        )
        agent = _build_agent(
            workspace,
            req,
            prompt=prompt,
            extra_env=extra_env,
        )

        # Cap entire trajectory wall clock so a single run can't hang the batch.
        deadline = req.request_timeout_s * (req.max_turns + 4)
        try:
            result = await asyncio.wait_for(
                asyncio.to_thread(
                    agent.run,
                    prompt.user_question,
                    system_prefix=prompt.system_prefix,
                ),
                timeout=deadline,
            )
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
