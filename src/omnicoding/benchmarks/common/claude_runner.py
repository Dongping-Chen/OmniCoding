"""Per-item Claude Code CLI runner with continue-retry.

Mirror of `common.codex_runner` for Claude. Differences vs. codex:

* Output stream is the Claude-Code stream-json shape (`type: assistant`,
  `type: result`); recovery falls through to `thinking` blocks because
  Qwen3.6 + thinking ON often emits `<answer>X</answer>` inside the
  `<think>` block, which the proxy translates to a Claude `thinking`
  content block that the CLI never re-emits as `text`.
* Session resume uses `claude --resume <UUID>`. We must NOT pass
  `--no-session-persistence` on the initial run; otherwise the SQLite
  session file isn't written and resume fails.
* Optional usage-limit gate (only meaningful for hosted Claude;
  local proxy auto-disables it via `is_local_claude_provider`).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Optional

from omnicoding.benchmarks.common.claude_cli import (
    build_claude_isolation_flags,
    count_claude_tool_calls,
    extract_claude_final_result,
    parse_claude_stream_json,
    sum_claude_tokens,
    summarize_claude_stream_line,
    walk_claude_full_transcript,
)
from omnicoding.benchmarks.common.claude_provider import is_local_claude_provider, should_use_usage_limit_gate
from omnicoding.benchmarks.common.processes import execute_prompt_command
from omnicoding.benchmarks.common.runtime import (
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

HARNESS_NAME = "claude-code"
HARNESS_VERSION = "3.4.1"

TAVILY_MCP_BIN = os.environ.get("OMNICODING_TAVILY_MCP_BIN", "").strip()
MCP_TAVILY_TOOL = "mcp__tavily__web_search"
DEFAULT_ALLOWED_TOOLS = ("Bash", "Read", "Glob", "Grep", "Edit", "Write", MCP_TAVILY_TOOL)


def _write_mcp_config(workspace_dir: Path) -> Path:
    """Drop a per-workspace MCP config that registers the Tavily server.

    Used together with `--mcp-config <path>` and `--strict-mcp-config`
    so the workspace is the only source of MCP servers (no leakage from
    a global ~/.claude/mcp.json)."""
    cfg_path = workspace_dir / ".mcp.json"
    cfg_path.write_text(
        json.dumps(
            {
                "mcpServers": {
                    "tavily": {
                        "command": TAVILY_MCP_BIN or sys.executable,
                        "args": [] if TAVILY_MCP_BIN else [
                            "-m", "omnicoding.tools.tavily_mcp",
                        ],
                    }
                }
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return cfg_path


def build_claude_command(
    *,
    args: argparse.Namespace,
    is_local_proxy: bool,
    mcp_config_path: Optional[Path] = None,
    resume_session_id: Optional[str] = None,
) -> list[str]:
    """Build the Claude Code CLI command. With `resume_session_id`
    supplied, becomes a `--resume <UUID>` invocation; otherwise the
    initial run.
    """

    cmd: list[str] = [
        args.claude_bin,
        "--print",
        "--verbose",
        "--output-format", "stream-json",
        "--dangerously-skip-permissions",
    ]
    if mcp_config_path is not None:
        cmd.extend(["--mcp-config", str(mcp_config_path)])
    cmd.extend(build_claude_isolation_flags(DEFAULT_ALLOWED_TOOLS, bare=is_local_proxy))
    if args.model_name:
        cmd.extend(["--model", args.model_name])
    if getattr(args, "effort", None):
        cmd.extend(["--effort", args.effort])
    if resume_session_id:
        cmd.extend(["--resume", resume_session_id])
    cmd.append("-")  # prompt arrives via stdin
    return cmd


async def run_claude_item(
    *,
    item_index: int,
    item: dict[str, Any],
    spec: BenchSpec,
    args: argparse.Namespace,
    semaphore: asyncio.Semaphore,
    gpu_slot_queue: Optional[asyncio.Queue[str]] = None,
    extra_system_prompt: str = "",
) -> dict[str, Any]:
    async with semaphore:
        return await _run_claude_item_inner(
            item_index=item_index,
            item=item,
            spec=spec,
            args=args,
            gpu_slot_queue=gpu_slot_queue,
            extra_system_prompt=extra_system_prompt,
        )


async def _run_claude_item_inner(
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
        cli_kind="claude",
        binary_path=Path(args.claude_bin),
    )
    workspace.env.setdefault("CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC", "1")
    assigned_gpu: Optional[str] = None

    is_local_proxy = is_local_claude_provider(
        provider_name=os.environ.get("ANTHROPIC_PROVIDER"),
        model_name=getattr(args, "model_name", None),
        env_overrides={"ANTHROPIC_BASE_URL": os.environ.get("ANTHROPIC_BASE_URL", "")},
    )

    try:
        if gpu_slot_queue is not None and getattr(args, "allow_shell_gpu", False):
            assigned_gpu = await gpu_slot_queue.get()
            assign_gpu_device(workspace.env, assigned_gpu)

        staged_paths = spec.stage_inputs(item, args.dataset_root, workspace.workspace_dir)
        workspace.staged_paths = list(staged_paths)

        prompt = spec.build_claude_prompt(_make_prompt_ctx(args, item, staged_paths, extra_system_prompt)) + RELATIVE_PATH_HINT
        (workspace.artifacts_dir / "prompt.txt").write_text(prompt, encoding="utf-8")

        mcp_cfg_path = _write_mcp_config(workspace.workspace_dir)
        inner_command = build_claude_command(
            args=args, is_local_proxy=is_local_proxy, mcp_config_path=mcp_cfg_path,
        )
        exec_command = (
            wrap_outer_sandbox(
                inner_command=inner_command,
                workspace=workspace,
                args=args,
                binary_path=Path(args.claude_bin).resolve(),
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
            stdout_summarizer=summarize_claude_stream_line,
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
                stdout_summarizer=summarize_claude_stream_line,
            )

        events, _result_event, tool_call_num = parse_claude_stream_json(stdout_text)
        agent_output = extract_claude_final_result(events, stdout_text, stderr_text)
        if timed_out:
            agent_output = f"Error: Claude Code item timeout after {args.item_timeout} seconds."

        session_id = _extract_session_id(events)
        agent_output, retry_meta = await _maybe_continue_retry(
            spec=spec,
            item=item,
            args=args,
            workspace=workspace,
            events=events,
            initial_output=agent_output,
            timed_out=timed_out,
            session_id=session_id,
            is_local_proxy=is_local_proxy,
            live_prefix=live_prefix,
        )

        prediction = spec.extract_prediction(agent_output, item)
        is_correct = spec.is_correct(item, prediction)

        # Round-12 BUG-C1: aggregate prompt/completion tokens from the
        # claude SDK stream-json events. Pre-fix the row left both
        # fields as None and wide-smoke cost analysis missed claude.
        prompt_tokens, completion_tokens = sum_claude_tokens(events)

        # Round-12 BUG-C3: ``raw_model_output`` should hold the full
        # rolling assistant transcript (text + thinking) tail-sliced to
        # 8000 chars, so the wide-smoke audit can scan claude rows the
        # same way it scans kira/mini_swe. ``agent_output`` (just the
        # last visible turn) is still the right input for
        # ``extract_prediction`` because the answer is at the end —
        # we keep the two paths separate.
        full_transcript = walk_claude_full_transcript(events)
        raw_for_row = full_transcript[-8000:] if full_transcript else agent_output

        ctx = ResultRowCtx(
            item=item,
            item_index=item_index,
            prediction=prediction,
            is_correct=is_correct,
            raw_model_output=raw_for_row,
            tool_call_num=tool_call_num + count_claude_tool_calls(events),
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
                "session_id": session_id,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
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


def _extract_session_id(events: list[dict[str, Any]]) -> str:
    for event in events:
        if event.get("type") not in {"system", "init", "system.init"}:
            continue
        for key in ("session_id", "id"):
            value = event.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
        if isinstance(event.get("data"), dict):
            value = event["data"].get("session_id")
            if isinstance(value, str) and value.strip():
                return value.strip()
    return ""


async def _maybe_continue_retry(
    *,
    spec: BenchSpec,
    item: dict[str, Any],
    args: argparse.Namespace,
    workspace: ItemWorkspace,
    events: list[dict[str, Any]],
    initial_output: str,
    timed_out: bool,
    session_id: str,
    is_local_proxy: bool,
    live_prefix: str,
) -> tuple[str, dict[str, Any]]:
    """When the initial run produced no extractable answer, resume the
    Claude session and ask it to commit to the answer-tag format. We
    log the gate result so it's clear in the slurm log whether retry
    actually fired (regression diagnostic from r14)."""

    agent_output = initial_output
    attempts: list[dict[str, Any]] = []
    initial_prediction = spec.extract_prediction(agent_output, item)
    LOGGER.info(
        "%sclaude retry-gate: prediction=%r timed_out=%s session=%s",
        live_prefix, initial_prediction, timed_out, session_id[:8] if session_id else "(none)",
    )
    if timed_out or not session_id or initial_prediction:
        return agent_output, {"retry_attempts": attempts}

    continue_prompt = build_continue_prompt(spec)

    for attempt in range(1, CONTINUE_RETRY_LIMIT + 1):
        if spec.extract_prediction(agent_output, item):
            break
        resume_inner = build_claude_command(
            args=args, is_local_proxy=is_local_proxy,
            mcp_config_path=workspace.workspace_dir / ".mcp.json",
            resume_session_id=session_id,
        )
        resume_exec = (
            wrap_outer_sandbox(
                inner_command=resume_inner,
                workspace=workspace,
                args=args,
                binary_path=Path(args.claude_bin).resolve(),
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
            stdout_summarizer=summarize_claude_stream_line,
        )
        new_events, _, _ = parse_claude_stream_json(resume_stdout)
        events.extend(new_events)
        attempts.append(
            {
                "attempt": attempt,
                "session_id": session_id,
                "prompt": continue_prompt,
                "stdout_chars": len(resume_stdout),
                "stderr_chars": len(resume_stderr),
                "timed_out": resume_timed_out,
            }
        )
        LOGGER.info(
            "%sclaude resume attempt %s/%s session=%s stdout_chars=%s timed_out=%s",
            live_prefix, attempt, CONTINUE_RETRY_LIMIT, session_id[:8],
            len(resume_stdout), resume_timed_out,
        )
        # Round-12 BUG-C2: always re-extract from the accumulated
        # events, even on timeout. Pre-fix the loop broke without
        # updating agent_output on timeout, so the row's
        # raw_model_output stayed as the previous (also-empty)
        # attempt's text. A SocialOmni L1 smoke case timed
        # out mid tool-call XML at attempt 2 and ended up with a
        # 99-char ``<function=Read>...`` snippet instead of the model's
        # last complete assistant turn from earlier in the trajectory.
        # extract_claude_final_result walks the events for the last
        # complete assistant text/thinking, which survives a mid-XML
        # truncation cleanly.
        agent_output = extract_claude_final_result(
            events, agent_output + "\n" + resume_stdout, resume_stderr,
        )
        if resume_timed_out:
            break
    return agent_output, {"retry_attempts": attempts, "retry_count": len(attempts)}


def use_usage_limit_gate(args: argparse.Namespace) -> bool:
    """Convenience predicate the harness can call to decide whether to
    spawn the usage-limit reset poller. Local providers always return
    False; hosted Claude defers to the user via --usage_limit_gate."""

    return should_use_usage_limit_gate(
        getattr(args, "usage_limit_gate", "auto"),
        provider_name=os.environ.get("ANTHROPIC_PROVIDER"),
        model_name=getattr(args, "model_name", None),
        env_overrides={"ANTHROPIC_BASE_URL": os.environ.get("ANTHROPIC_BASE_URL", "")},
    )
