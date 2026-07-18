"""Single-entry Claude Code CLI runner across all registered benchmarks.

Replaces the per-bench `run_claude_code_<bench>.py` files. Same shape
as `run_codex.py`; delegates to `common.claude_runner`.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from omnicoding.benchmarks import specs  # noqa: E402
from omnicoding.paths import runtime_root

REPO_ROOT = runtime_root()
from omnicoding.benchmarks.common.cli_args import (  # noqa: E402
    add_agent_md_args,
    add_claude_args,
    add_filter_args,
    add_io_args,
    add_run_args,
    add_sandbox_args,
    finalize_args,
)
from omnicoding.benchmarks.common.claude_runner import run_claude_item  # noqa: E402
from omnicoding.benchmarks.common.metrics import (  # noqa: E402
    atomic_write_json,
    build_metrics_payload,
    build_run_dir_name,
    build_run_paths,
    sort_results,
)
from omnicoding.benchmarks.common.spec import BenchSpec  # noqa: E402

try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover
    tqdm = None

LOGGER = logging.getLogger("benchmarks.harnesses.run_claude")
LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Claude Code CLI across registered benchmarks.")
    add_io_args(parser, default_workspace_subdir=".claude_workspaces")
    add_filter_args(parser)
    add_run_args(parser)
    add_sandbox_args(parser)
    add_agent_md_args(parser)
    add_claude_args(parser)
    return parser.parse_args()


def _resolve_claude_bin(raw: str) -> str:
    candidate = shutil.which(raw) if raw == "claude" else raw
    if candidate is None:
        raise FileNotFoundError(
            "Claude Code CLI binary not found on PATH. "
            "Pass --claude_bin <absolute path> or install claude.",
        )
    if not Path(candidate).exists():
        raise FileNotFoundError(f"--claude_bin not found: {candidate}")
    return str(candidate)


async def _run_per_item_loop(
    spec: BenchSpec,
    items: list[dict[str, Any]],
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    semaphore = asyncio.Semaphore(args.concurrent_limit)
    gpu_slot_queue: Optional[asyncio.Queue[str]] = None
    if args.allow_shell_gpu and args.gpu_device_pool:
        gpu_slot_queue = asyncio.Queue()
        for device in args.gpu_device_pool:
            gpu_slot_queue.put_nowait(device)

    tasks = [
        asyncio.create_task(
            run_claude_item(
                item_index=index,
                item=item,
                spec=spec,
                args=args,
                semaphore=semaphore,
                gpu_slot_queue=gpu_slot_queue,
                extra_system_prompt=getattr(args, "agent_md_prompt", "") or "",
            )
        )
        for index, item in enumerate(items)
    ]
    return await _drain_with_streaming_writes(tasks, args)


async def _run_batch_loop(
    spec: BenchSpec,
    items: list[dict[str, Any]],
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    semaphore = asyncio.Semaphore(args.concurrent_limit)
    assert spec.groups_by is not None
    groups = spec.groups_by(items)

    async def _run_group(index: int, members: list[dict[str, Any]]) -> list[dict[str, Any]]:
        head = dict(members[0])
        head["__group__"] = members
        head["__group_index__"] = index
        single_row = await run_claude_item(
            item_index=index,
            item=head,
            spec=spec,
            args=args,
            semaphore=semaphore,
            extra_system_prompt=getattr(args, "agent_md_prompt", "") or "",
        )
        from omnicoding.benchmarks.common.spec import ResultRowCtx
        rows: list[dict[str, Any]] = []
        for member in members:
            rows.append(
                spec.result_row(
                    ResultRowCtx(
                        item=member,
                        item_index=member.get("__source_index__", 0),
                        prediction=spec.extract_prediction(single_row.get("raw_model_output", ""), member),
                        is_correct=spec.is_correct(member, ""),
                        raw_model_output=single_row.get("raw_model_output", ""),
                        tool_call_num=int(single_row.get("tool_call_num", 0) or 0),
                        return_code=single_row.get("return_code"),
                        timed_out=bool(single_row.get("timed_out")),
                        stdout_text=single_row.get("stdout_text", ""),
                        stderr_text=single_row.get("stderr_text", ""),
                        workspace_dir=Path(single_row.get("workspace_dir", "/tmp")),
                        keep_workdirs=args.keep_workdirs,
                        include_gold_fields=getattr(args, "include_gold_fields_in_results", False),
                        extra={k: v for k, v in single_row.items() if k.startswith(("outer_", "retry_", "thread_", "session_", "gpu_"))},
                    )
                )
            )
        return rows

    tasks = [asyncio.create_task(_run_group(index, members)) for index, (_video, members) in enumerate(groups)]
    nested = await _drain_with_streaming_writes(tasks, args)
    flat: list[dict[str, Any]] = []
    for entry in nested:
        if isinstance(entry, list):
            flat.extend(entry)
        else:
            flat.append(entry)
    return flat


async def _drain_with_streaming_writes(
    tasks: list[asyncio.Task[Any]],
    args: argparse.Namespace,
) -> list[Any]:
    results: list[Any] = []
    progress = tqdm(total=len(tasks), desc=f"{args.bench}") if tqdm else None
    try:
        for task in asyncio.as_completed(tasks):
            result = await task
            if isinstance(result, list):
                results.extend(result)
            else:
                results.append(result)
            _flush_progress(results, args)
            if progress is not None:
                progress.update(1)
    finally:
        if progress is not None:
            progress.close()
    return results


def _flush_progress(results: list[Any], args: argparse.Namespace) -> None:
    if not getattr(args, "_results_path", None):
        return
    flat = [row for row in results if isinstance(row, dict)]
    sorted_results = sort_results(flat)
    atomic_write_json(args._results_path, sorted_results)
    atomic_write_json(args._metrics_path, build_metrics_payload(sorted_results, args))


async def main_async(args: argparse.Namespace) -> None:
    spec = specs.get(args.bench)
    args.claude_bin = _resolve_claude_bin(args.claude_bin)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = build_run_dir_name(args.model_name, getattr(args, "model_reasoning_effort", None))
    output_root, results_path, metrics_path = build_run_paths(
        args, dirname=f"{spec.name}_claude_{run_dir}", timestamp=timestamp,
    )
    args.output_root = output_root
    args._results_path = results_path
    args._metrics_path = metrics_path

    items_all = spec.iterate_items(args.input_path)
    items = spec.filter_items(items_all, args)
    if not items:
        raise ValueError(f"No items matched filters for bench={spec.name}")
    LOGGER.info("bench=%s loaded=%s after_filter=%s", spec.name, len(items_all), len(items))

    if spec.groups_by is None:
        results = await _run_per_item_loop(spec, items, args)
    else:
        results = await _run_batch_loop(spec, items, args)

    sorted_results = sort_results(results)
    atomic_write_json(results_path, sorted_results)
    atomic_write_json(metrics_path, build_metrics_payload(sorted_results, args))

    if not args.keep_workdirs:
        shutil.rmtree(args.workspace_root, ignore_errors=True)

    LOGGER.info("Saved results: %s", results_path)
    LOGGER.info("Saved metrics: %s", metrics_path)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format=LOG_FORMAT, datefmt="%Y-%m-%d %H:%M:%S")
    args = parse_args()
    finalize_args(
        args,
        repo_root=REPO_ROOT,
        runner_root=Path(__file__).resolve().parent,
        workspace_subdir=".claude_workspaces",
    )
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
