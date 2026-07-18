"""Shared argparse blocks + post-parse validation for harness runners.

The codex and claude harness entrypoints (`harnesses/run_codex.py`,
`harnesses/run_claude.py`) attach these arg groups, then call
`finalize_args()` once to apply the cross-flag adjustments that used to
live duplicated in every per-bench `main()`.

Keeping this in one place is what lets us delete the giant argparse
walls in `run_codex_cli_<bench>.py`. New harness flags (e.g. a future
`--budget` or `--fail-fast`) can be added once here.
"""

from __future__ import annotations

import argparse
import logging
import shutil
from pathlib import Path
from typing import Optional

from omnicoding.benchmarks.common.shared_python import default_shared_python_env, normalize_shared_python_env
from omnicoding.benchmarks.common.workspace import (
    iter_gpu_device_paths,
    normalize_gpu_visible_devices,
    parse_gpu_device_pool,
)


LOGGER = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Argparse groups
# ---------------------------------------------------------------------------


def add_io_args(parser: argparse.ArgumentParser, *, default_workspace_subdir: str) -> None:
    parser.add_argument("--bench", required=True, help="Benchmark name registered in benchmarks/specs.")
    parser.add_argument("--input_file", required=True, help="Path to the benchmark JSON.")
    parser.add_argument(
        "--dataset_root",
        default=None,
        help="Dataset root containing media. Defaults to the parent of --input_file.",
    )
    parser.add_argument("--output_dir", default="./outputs", help="Run output root.")
    parser.add_argument(
        "--workspace_root",
        default=None,
        help=f"Per-item workspace root. Defaults to <repo>/{default_workspace_subdir}.",
    )
    parser.add_argument("--keep_workdirs", action="store_true")


def add_filter_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--max_items", type=int, default=None)
    parser.add_argument("--question_ids", nargs="*", default=None)
    parser.add_argument("--video_ids", nargs="*", default=None)
    parser.add_argument("--difficulty", nargs="*", default=None)
    parser.add_argument("--languages", nargs="*", default=None)
    parser.add_argument("--categories", nargs="*", default=None)


def add_run_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--concurrent_limit", type=int, default=1)
    parser.add_argument("--item_timeout", type=int, default=5400)
    parser.add_argument("--live_output", action="store_true", default=True)
    parser.add_argument("--no_live_output", action="store_false", dest="live_output")
    parser.add_argument(
        "--include_gold_fields_in_results",
        action="store_true",
        help="Include gold answer fields in saved results.",
    )


def add_sandbox_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--sandbox",
        default="workspace-write",
        choices=["read-only", "workspace-write", "danger-full-access"],
    )
    parser.add_argument("--allow_shell_network", action="store_true")
    parser.add_argument("--allow_shell_gpu", action="store_true")
    parser.add_argument("--gpu_visible_devices", default="0")
    parser.add_argument("--gpu_max_parallel_items", type=int, default=1)
    parser.add_argument("--gpu_soft_memory_fraction", type=float, default=None)
    parser.add_argument("--gpu_device_pool", default=None)
    parser.add_argument("--shared_python_env", default=default_shared_python_env())
    parser.add_argument("--outer_sandbox", action="store_true", default=True)
    parser.add_argument("--no_outer_sandbox", action="store_false", dest="outer_sandbox")
    parser.add_argument("--workspace_only_outer_sandbox", action="store_true")


def add_agent_md_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--use_agent_md_system_prompt", action="store_true")
    parser.add_argument(
        "--agent_md_variant",
        default="auto",
        choices=["auto", "default", "gpu", "gpu_new"],
    )
    parser.add_argument(
        "--agent_md_file",
        default=None,
        help="Explicit path to an agent-guide markdown. Overrides --agent_md_variant.",
    )
    parser.add_argument(
        "--enable_agent_stack",
        action="store_true",
        help="Convenience: enable allow_shell_network + allow_shell_gpu + use_agent_md_system_prompt.",
    )
    parser.add_argument("--disable_native_vision", action="store_true")


def add_codex_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--codex_bin", default="codex")
    parser.add_argument("--model_name", required=True)
    parser.add_argument(
        "--model_reasoning_effort",
        default=None,
        choices=["minimal", "low", "medium", "high", "xhigh", "extra-high", "extra_high"],
    )
    parser.add_argument(
        "--codex_config_override",
        action="append",
        dest="codex_config_overrides",
        default=[],
    )
    parser.add_argument(
        "--approval_policy",
        default="never",
        choices=["untrusted", "on-failure", "on-request", "never"],
    )


def add_claude_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--claude_bin", default="claude")
    parser.add_argument("--model_name", required=True)
    parser.add_argument(
        "--model_reasoning_effort",
        default=None,
        choices=["minimal", "low", "medium", "high", "xhigh", "extra-high", "extra_high"],
    )
    parser.add_argument(
        "--effort",
        default=None,
        choices=["low", "medium", "high", "max"],
        help="Optional --effort override.",
    )
    parser.add_argument(
        "--usage_limit_gate",
        default="auto",
        choices=["auto", "on", "off"],
    )
    parser.add_argument("--usage_poll_interval", type=int, default=600)


# ---------------------------------------------------------------------------
# Post-parse normalisation
# ---------------------------------------------------------------------------


def _normalize_reasoning_effort(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    canonical = value.replace("-", "_")
    if canonical in {"xhigh", "extra_high"}:
        return "xhigh"
    return canonical


def finalize_args(
    args: argparse.Namespace,
    *,
    repo_root: Path,
    runner_root: Path,
    workspace_subdir: str,
) -> None:
    """Apply cross-flag adjustments that used to live duplicated in every
    per-bench `main()`. Mutates `args` in place.

    Validates GPU/sandbox/outer_sandbox interactions, resolves
    workspace/dataset/input paths, and stamps a few helper attributes
    (`repo_root`, `input_path`, `dataset_root`, `agent_md_*`,
    `gpu_visible_devices`, `gpu_device_pool`).
    """

    if hasattr(args, "model_reasoning_effort"):
        args.model_reasoning_effort = _normalize_reasoning_effort(args.model_reasoning_effort)
    args.gpu_visible_devices = normalize_gpu_visible_devices(getattr(args, "gpu_visible_devices", None))
    args.gpu_device_pool = parse_gpu_device_pool(getattr(args, "gpu_device_pool", None))

    if getattr(args, "enable_agent_stack", False):
        args.allow_shell_network = True
        args.allow_shell_gpu = True
        args.use_agent_md_system_prompt = True

    sandbox = getattr(args, "sandbox", None)
    if sandbox is not None:
        if args.allow_shell_network and sandbox != "danger-full-access":
            LOGGER.warning("allow_shell_network → forcing sandbox to danger-full-access (was %s).", sandbox)
            args.sandbox = "danger-full-access"
        if args.allow_shell_gpu and args.sandbox != "danger-full-access":
            LOGGER.warning("allow_shell_gpu → forcing sandbox to danger-full-access (was %s).", args.sandbox)
            args.sandbox = "danger-full-access"
        if args.gpu_device_pool and args.sandbox != "danger-full-access":
            LOGGER.warning("gpu_device_pool → forcing sandbox to danger-full-access (was %s).", args.sandbox)
            args.sandbox = "danger-full-access"

    if getattr(args, "workspace_only_outer_sandbox", False) and not args.outer_sandbox:
        LOGGER.warning("workspace_only_outer_sandbox requires outer_sandbox; enabling outer_sandbox.")
        args.outer_sandbox = True

    if args.gpu_soft_memory_fraction is not None and not (0.0 < args.gpu_soft_memory_fraction <= 1.0):
        raise ValueError("--gpu_soft_memory_fraction must be in (0, 1].")
    if args.allow_shell_gpu and args.gpu_max_parallel_items < 1:
        raise ValueError("--gpu_max_parallel_items must be >= 1 when allow_shell_gpu is enabled.")
    if args.gpu_device_pool and not args.allow_shell_gpu:
        LOGGER.warning("gpu_device_pool set → enabling allow_shell_gpu.")
        args.allow_shell_gpu = True

    if args.outer_sandbox and not shutil.which("bwrap"):
        LOGGER.warning("bubblewrap not available; disabling outer_sandbox.")
        args.outer_sandbox = False

    args.repo_root = repo_root
    input_path = Path(args.input_file).resolve()
    if not input_path.exists():
        raise FileNotFoundError(f"--input_file not found: {input_path}")
    args.input_path = input_path

    dataset_root = (
        Path(args.dataset_root).resolve() if args.dataset_root else input_path.parent
    )
    if not dataset_root.exists():
        raise FileNotFoundError(f"--dataset_root not found: {dataset_root}")
    args.dataset_root = dataset_root

    if args.workspace_root is None:
        args.workspace_root = str((repo_root / workspace_subdir).resolve())
    args.workspace_root = Path(args.workspace_root).resolve()
    args.workspace_root.mkdir(parents=True, exist_ok=True)

    if getattr(args, "shared_python_env", None):
        args.shared_python_env = normalize_shared_python_env(args.shared_python_env)

    args.runner_root = runner_root

    args.agent_md_path, args.agent_md_prompt = _resolve_agent_md(
        args, runner_root=runner_root, repo_root=repo_root,
    )

    if args.allow_shell_gpu:
        gpu_devices = iter_gpu_device_paths()
        if gpu_devices:
            LOGGER.info("GPU passthrough enabled; visible devices: %s", ", ".join(str(p) for p in gpu_devices))
        else:
            LOGGER.warning("allow_shell_gpu set but no /dev/nvidia* devices found.")
        if args.gpu_device_pool:
            if args.gpu_max_parallel_items > len(args.gpu_device_pool):
                args.gpu_max_parallel_items = len(args.gpu_device_pool)
            if args.gpu_max_parallel_items == 1 and len(args.gpu_device_pool) > 1:
                args.gpu_max_parallel_items = len(args.gpu_device_pool)
        if args.concurrent_limit > args.gpu_max_parallel_items:
            LOGGER.warning(
                "Reducing concurrent_limit %s → gpu_max_parallel_items %s.",
                args.concurrent_limit, args.gpu_max_parallel_items,
            )
            args.concurrent_limit = args.gpu_max_parallel_items


def _resolve_agent_md(
    args: argparse.Namespace, *, runner_root: Path, repo_root: Path,
) -> tuple[str, str]:
    """Find the agent guide markdown and return (path, prompt). If the
    user didn't ask for it, return ("", "")."""

    if not getattr(args, "use_agent_md_system_prompt", False):
        # Still resolve a path for logging/debug; just don't load.
        path = _agent_md_path(args, runner_root, repo_root)
        return str(path), ""

    path = _agent_md_path(args, runner_root, repo_root)
    if not path.exists():
        LOGGER.warning("--use_agent_md_system_prompt set, but %s missing.", path)
        return str(path), ""
    return str(path), path.read_text(encoding="utf-8").strip()


def _agent_md_path(args: argparse.Namespace, runner_root: Path, repo_root: Path) -> Path:
    explicit = getattr(args, "agent_md_file", None)
    if explicit:
        return Path(explicit).expanduser().resolve()
    variant = getattr(args, "agent_md_variant", "auto")
    gpu_enabled = bool(getattr(args, "allow_shell_gpu", False))
    candidates: list[str] = []
    if variant == "default":
        candidates = ["agent.md"]
    elif variant == "gpu":
        candidates = ["agent_gpu.md", "agent.md"]
    elif variant == "gpu_new":
        candidates = ["agent_gpu_new.md", "agent_gpu.md", "agent.md"]
    else:  # auto
        candidates = ["agent_gpu.md", "agent.md"] if gpu_enabled else ["agent.md"]
    for name in candidates:
        for root in (runner_root, repo_root):
            path = root / name
            if path.exists():
                return path
    return runner_root / candidates[0]
