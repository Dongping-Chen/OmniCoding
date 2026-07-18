"""Per-item codex CLI runner with continue-retry.

Replaces the duplicated `_run_item` + `_run_item_with_retries` blocks
that lived in each per-bench `run_codex_cli_<bench>.py`.

The harness entrypoint passes a `BenchSpec` and an `argparse.Namespace`;
this module handles workspace staging, codex invocation, output
recovery (assistant_message → reasoning → raw stdout), session-resume
retry when no answer was extracted, and result-row construction.

`codex exec resume` only accepts a subset of the parent flags. The
session's sandbox, cwd, and approval_policy are inherited from the
persisted record; passing them again yields
`error: unexpected argument '--sandbox' found`. We therefore build the
resume command with only `--model`, `-c model_reasoning_effort`, and
custom `-c key=value` overrides.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
from pathlib import Path
from typing import Any, Optional

from omnicoding.benchmarks.common.codex_cli import (
    count_codex_tool_calls,
    extract_codex_last_assistant_text,
    extract_codex_last_reasoning_text,
    parse_codex_jsonl,
    summarize_codex_event_line,
)
from omnicoding.benchmarks.common.processes import execute_prompt_command
from omnicoding.benchmarks.common.runtime import (
    INTERNAL_WORKSPACE_ROOT,
    ItemWorkspace,
    assign_gpu_device,
    cleanup_workspace,
    make_workspace,
    should_retry_without_outer_sandbox,
    workspace_paths_for_invocation,
    wrap_outer_sandbox,
)
from omnicoding.benchmarks.common.spec import (
    CONTINUE_RETRY_LIMIT,
    RELATIVE_PATH_HINT,
    BenchSpec,
    BuildPromptCtx,
    ResultRowCtx,
    build_continue_prompt,
)

LOGGER = logging.getLogger(__name__)

HARNESS_NAME = "codex-cli"
HARNESS_VERSION = "0.124.1"


def build_codex_command(
    *,
    args: argparse.Namespace,
    cwd: Path,
    output_message_path: Path,
    is_resume: bool = False,
    thread_id: Optional[str] = None,
) -> list[str]:
    """Build the codex CLI command. `is_resume=True` switches to
    `codex exec resume`, which inherits sandbox/cwd/approval_policy
    from the persisted session and rejects them on the command line.
    """

    if is_resume:
        if not thread_id:
            raise ValueError("thread_id is required when is_resume=True")
        cmd: list[str] = [
            args.codex_bin, "exec", "resume",
            "--skip-git-repo-check",
            "--json",
            "-o", str(output_message_path),
        ]
    else:
        cmd = [
            args.codex_bin, "exec",
            "--skip-git-repo-check",
            "--sandbox", args.sandbox,
            "--cd", str(cwd),
            "--json",
            "-o", str(output_message_path),
        ]
        cmd.extend(["-c", f'approval_policy="{args.approval_policy}"'])

    if args.model_name:
        cmd.extend(["--model", args.model_name])
    if getattr(args, "model_reasoning_effort", None):
        cmd.extend(["-c", f'model_reasoning_effort="{args.model_reasoning_effort}"'])
    for override in getattr(args, "codex_config_overrides", None) or []:
        cmd.extend(["-c", override])

    if is_resume:
        cmd.extend([thread_id, "-"])
    else:
        cmd.append("-")
    return cmd


def resolve_agent_output(
    *,
    stdout_text: str,
    stderr_text: str,
    output_message_path: Path,
    events: list[dict[str, Any]],
    timed_out: bool,
    item_timeout: int,
) -> str:
    """Recover the model's final visible answer.

    Recovery order: file from `--output-last-message` → last
    `assistant_message` event → last `reasoning` event → raw
    stderr+stdout. The reasoning fallback exists because Qwen3.6 +
    thinking ON often emits `<answer>X</answer>` inside `<think>` so it
    arrives as a Responses-API `reasoning` item rather than an
    `assistant_message`.
    """

    if timed_out:
        return f"Error: Codex item timeout after {item_timeout} seconds."
    if output_message_path.exists():
        text = output_message_path.read_text(encoding="utf-8").strip()
        if text:
            return text
    recovered = extract_codex_last_assistant_text(events)
    if recovered.strip():
        return recovered
    recovered_reasoning = extract_codex_last_reasoning_text(events)
    if recovered_reasoning.strip():
        return recovered_reasoning
    combined = "\n".join(part for part in [stderr_text.strip(), stdout_text.strip()] if part).strip()
    return combined or "Error: Codex produced no final message."


def extract_thread_id(events: list[dict[str, Any]]) -> str:
    for event in events:
        thread = event.get("thread_id") or (event.get("item") or {}).get("thread_id")
        if isinstance(thread, str) and thread.strip():
            return thread.strip()
    return ""


async def run_codex_item(
    *,
    item_index: int,
    item: dict[str, Any],
    spec: BenchSpec,
    args: argparse.Namespace,
    semaphore: asyncio.Semaphore,
    gpu_slot_queue: Optional[asyncio.Queue[str]] = None,
    extra_system_prompt: str = "",
) -> dict[str, Any]:
    """Run one item end-to-end through Codex CLI: stage → run → extract
    → retry-if-empty → build result row."""

    async with semaphore:
        return await _run_codex_item_inner(
            item_index=item_index,
            item=item,
            spec=spec,
            args=args,
            gpu_slot_queue=gpu_slot_queue,
            extra_system_prompt=extra_system_prompt,
        )


async def _run_codex_item_inner(
    *,
    item_index: int,
    item: dict[str, Any],
    spec: BenchSpec,
    args: argparse.Namespace,
    gpu_slot_queue: Optional[asyncio.Queue[str]],
    extra_system_prompt: str,
) -> dict[str, Any]:
    item_id = spec.item_id(item)
    live_prefix = f"[{spec.name} {item_id}] "

    workspace = make_workspace(
        args=args,
        item_id=item_id,
        cli_kind="codex",
        binary_path=Path(args.codex_bin),
    )
    assigned_gpu: Optional[str] = None

    try:
        if gpu_slot_queue is not None and getattr(args, "allow_shell_gpu", False):
            assigned_gpu = await gpu_slot_queue.get()
            assign_gpu_device(workspace.env, assigned_gpu)

        staged_paths = spec.stage_inputs(item, args.dataset_root, workspace.workspace_dir)
        workspace.staged_paths = list(staged_paths)

        prompt = spec.build_codex_prompt(_make_prompt_ctx(args, item, staged_paths, extra_system_prompt)) + RELATIVE_PATH_HINT
        (workspace.artifacts_dir / "prompt.txt").write_text(prompt, encoding="utf-8")

        cwd, output_message_path = workspace_paths_for_invocation(workspace, outer_sandbox=args.outer_sandbox)
        inner_command = build_codex_command(
            args=args, cwd=cwd, output_message_path=output_message_path,
        )
        exec_command = (
            wrap_outer_sandbox(
                inner_command=inner_command,
                workspace=workspace,
                args=args,
                binary_path=Path(args.codex_bin).resolve(),
            )
            if args.outer_sandbox
            else inner_command
        )

        stdout_text, stderr_text, return_code, timed_out = await execute_prompt_command(
            command=exec_command,
            prompt=prompt,
            timeout=args.item_timeout,
            env=workspace.env,
            cwd=None if args.outer_sandbox else str(workspace.workspace_dir),
            live_prefix=live_prefix,
            live_output=args.live_output,
            stdout_summarizer=summarize_codex_event_line,
        )

        outer_sandbox_retry_fallback = False
        if args.outer_sandbox and should_retry_without_outer_sandbox(return_code, timed_out, stderr_text):
            outer_sandbox_retry_fallback = True
            LOGGER.warning("%souter sandbox failed; retrying without it.", live_prefix)
            stdout_text, stderr_text, return_code, timed_out = await execute_prompt_command(
                command=inner_command,
                prompt=prompt,
                timeout=args.item_timeout,
                env=workspace.env,
                cwd=str(workspace.workspace_dir),
                live_prefix=live_prefix,
                live_output=args.live_output,
                stdout_summarizer=summarize_codex_event_line,
            )

        events = parse_codex_jsonl(stdout_text)
        agent_output = resolve_agent_output(
            stdout_text=stdout_text,
            stderr_text=stderr_text,
            output_message_path=workspace.output_message_path,
            events=events,
            timed_out=timed_out,
            item_timeout=args.item_timeout,
        )

        thread_id = extract_thread_id(events)
        agent_output, retry_meta = await _maybe_continue_retry(
            spec=spec,
            item=item,
            args=args,
            workspace=workspace,
            events=events,
            stdout_text=stdout_text,
            stderr_text=stderr_text,
            initial_output=agent_output,
            timed_out=timed_out,
            thread_id=thread_id,
            live_prefix=live_prefix,
        )

        prediction = spec.extract_prediction(agent_output, item)
        is_correct = spec.is_correct(item, prediction)

        ctx = ResultRowCtx(
            item=item,
            item_index=item_index,
            prediction=prediction,
            is_correct=is_correct,
            raw_model_output=agent_output,
            tool_call_num=count_codex_tool_calls(events),
            return_code=return_code,
            timed_out=timed_out,
            stdout_text=stdout_text,
            stderr_text=stderr_text,
            workspace_dir=workspace.workspace_dir,
            keep_workdirs=args.keep_workdirs,
            include_gold_fields=getattr(args, "include_gold_fields_in_results", False),
            extra={
                "harness": HARNESS_NAME,
                "harness_version": HARNESS_VERSION,
                "outer_sandbox": bool(args.outer_sandbox),
                "outer_sandbox_retry_fallback": outer_sandbox_retry_fallback,
                "gpu_device_assigned": assigned_gpu,
                "thread_id": thread_id,
                **retry_meta,
            },
        )
        return spec.result_row(ctx)
    finally:
        if gpu_slot_queue is not None and assigned_gpu is not None:
            await gpu_slot_queue.put(assigned_gpu)
        cleanup_workspace(workspace, keep=args.keep_workdirs)


def _make_prompt_ctx(
    args: argparse.Namespace,
    item: dict[str, Any],
    staged_paths: list[Path],
    extra_system_prompt: str,
) -> BuildPromptCtx:
    return BuildPromptCtx(
        item=item,
        staged_paths=staged_paths,
        sandbox=getattr(args, "sandbox", "workspace-write"),
        allow_shell_network=bool(getattr(args, "allow_shell_network", False)),
        allow_shell_gpu=bool(getattr(args, "allow_shell_gpu", False)),
        shared_python_env=getattr(args, "shared_python_env", None),
        disable_native_vision=bool(getattr(args, "disable_native_vision", False)),
        extra_system_prompt=extra_system_prompt or getattr(args, "agent_md_prompt", "") or "",
    )


async def _maybe_continue_retry(
    *,
    spec: BenchSpec,
    item: dict[str, Any],
    args: argparse.Namespace,
    workspace: ItemWorkspace,
    events: list[dict[str, Any]],
    stdout_text: str,
    stderr_text: str,
    initial_output: str,
    timed_out: bool,
    thread_id: str,
    live_prefix: str,
) -> tuple[str, dict[str, Any]]:
    """If the model engaged with tools but never committed an
    `<answer>X</answer>` (Qwen3.6 + thinking ON failure mode), use
    `codex exec resume <thread_id>` with a permissive continue prompt
    to nudge it. Capped at 2 retries.
    """

    agent_output = initial_output
    attempts: list[dict[str, Any]] = []
    if timed_out or not thread_id:
        return agent_output, {"retry_attempts": attempts}

    continue_prompt = build_continue_prompt(spec)
    cwd, output_message_path = workspace_paths_for_invocation(workspace, outer_sandbox=args.outer_sandbox)

    for attempt in range(1, CONTINUE_RETRY_LIMIT + 1):
        if spec.extract_prediction(agent_output, item):
            break
        resume_inner = build_codex_command(
            args=args, cwd=cwd, output_message_path=output_message_path,
            is_resume=True, thread_id=thread_id,
        )
        resume_exec = (
            wrap_outer_sandbox(
                inner_command=resume_inner,
                workspace=workspace,
                args=args,
                binary_path=Path(args.codex_bin).resolve(),
            )
            if args.outer_sandbox
            else resume_inner
        )
        resume_stdout, resume_stderr, _, resume_timed_out = await execute_prompt_command(
            command=resume_exec,
            prompt=continue_prompt,
            timeout=args.item_timeout,
            env=workspace.env,
            cwd=None if args.outer_sandbox else str(workspace.workspace_dir),
            live_prefix=f"{live_prefix}resume{attempt} ",
            live_output=args.live_output,
            stdout_summarizer=summarize_codex_event_line,
        )
        events.extend(parse_codex_jsonl(resume_stdout))
        attempts.append(
            {
                "attempt": attempt,
                "thread_id": thread_id,
                "prompt": continue_prompt,
                "stdout_chars": len(resume_stdout),
                "stderr_chars": len(resume_stderr),
                "timed_out": resume_timed_out,
            }
        )
        LOGGER.info(
            "%scodex resume attempt %s/%s thread=%s stdout_chars=%s",
            live_prefix, attempt, CONTINUE_RETRY_LIMIT, thread_id[:8], len(resume_stdout),
        )
        if resume_timed_out:
            break
        agent_output = resolve_agent_output(
            stdout_text=stdout_text + "\n" + resume_stdout,
            stderr_text=stderr_text + "\n" + resume_stderr,
            output_message_path=workspace.output_message_path,
            events=events,
            timed_out=False,
            item_timeout=args.item_timeout,
        )
    return agent_output, {"retry_attempts": attempts, "retry_count": len(attempts)}
