"""Per-item workspace + env + outer-sandbox wiring.

The codex and claude harness runners both stage one workspace per
item, build a workspace env, and (optionally) wrap the inner CLI
invocation in a bwrap outer sandbox. That setup was duplicated word for
word across six runners; this module is the single source of truth.

Not in scope: the inner CLI command builder (codex `exec` flags,
claude `--print` flags). Those live in the harness runner files
because they differ between codex and claude.
"""

from __future__ import annotations

import argparse
import logging
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from omnicoding.benchmarks.common.sandbox import build_outer_sandbox_command
from omnicoding.benchmarks.common.workspace import (
    build_workspace_env,
    iter_binary_bind_paths,
    iter_gpu_device_paths,
    prepare_isolated_claude_home,
    prepare_isolated_codex_home,
)

LOGGER = logging.getLogger(__name__)


# Internal paths used inside the bwrap sandbox. The outer sandbox
# masks the workspace at these paths so the inner CLI sees stable
# locations regardless of the host workspace path.
INTERNAL_WORKSPACE_ROOT = Path("/tmp/codex-workspace")
INTERNAL_HOME_ROOT = Path("/tmp/codex-home")


@dataclass
class ItemWorkspace:
    """Per-item staged workspace + the env/path needed to invoke a CLI in it.

    `workspace_dir` is the host filesystem dir we mkdtemp'd.
    `internal_workspace` is what the CLI sees when wrapped in bwrap (the
        host workspace_dir is bind-mounted at INTERNAL_WORKSPACE_ROOT).
    `staged_paths` are workspace-relative paths to inputs the spec staged.
    `env` already has HOME, CODEX_HOME, XDG_*, PATH, GPU vars set.
    """

    workspace_dir: Path
    isolated_home: Path
    staged_paths: list[Path]
    artifacts_dir: Path
    output_message_path: Path
    env: dict[str, str]


def make_workspace(
    *,
    args: argparse.Namespace,
    item_id: str,
    cli_kind: str,
    binary_path: Optional[Path] = None,
) -> ItemWorkspace:
    """Stage a fresh per-item workspace under `args.workspace_root`.

    `cli_kind` ∈ {"codex", "claude"} picks the right isolated-home
    layout. `binary_path` is the codex/claude binary path; if its parent
    is a node installation, that root is added to PATH so the CLI's
    bundled Node can resolve.
    """

    workspace_dir = Path(
        tempfile.mkdtemp(
            prefix=f"{cli_kind}_item_{item_id}_",
            dir=str(args.workspace_root),
        )
    )
    artifacts_dir = workspace_dir / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    (workspace_dir / "inputs").mkdir(parents=True, exist_ok=True)

    if cli_kind == "codex":
        isolated_home = prepare_isolated_codex_home(workspace_dir)
    elif cli_kind == "claude":
        isolated_home = prepare_isolated_claude_home(workspace_dir, copy_credentials=False)
    else:
        raise ValueError(f"Unknown cli_kind: {cli_kind!r}")

    env = build_workspace_env(
        workspace_dir,
        shared_python_env=getattr(args, "shared_python_env", None),
        shared_python_first=True,
        isolated_home=isolated_home,
        binary_path=binary_path,
    )
    _apply_gpu_env(env, args)

    return ItemWorkspace(
        workspace_dir=workspace_dir,
        isolated_home=isolated_home,
        staged_paths=[],
        artifacts_dir=artifacts_dir,
        output_message_path=artifacts_dir / "last_message.txt",
        env=env,
    )


def _apply_gpu_env(env: dict[str, str], args: argparse.Namespace) -> None:
    if getattr(args, "allow_shell_gpu", False):
        visible = getattr(args, "gpu_visible_devices", None)
        if visible is not None:
            env["CUDA_VISIBLE_DEVICES"] = visible
            env["NVIDIA_VISIBLE_DEVICES"] = visible
        env.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true")
        env.setdefault("CUDA_MODULE_LOADING", "LAZY")
        if getattr(args, "gpu_soft_memory_fraction", None) is not None:
            fraction = f"{args.gpu_soft_memory_fraction:.4f}".rstrip("0").rstrip(".")
            env["CODEX_GPU_SOFT_MEMORY_FRACTION"] = fraction
            env.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", fraction)
    else:
        env["CUDA_VISIBLE_DEVICES"] = "-1"
        env["NVIDIA_VISIBLE_DEVICES"] = "void"


def assign_gpu_device(env: dict[str, str], device: str | None) -> None:
    """Override CUDA_VISIBLE_DEVICES for one item from the device pool."""

    if device is None:
        return
    env["CUDA_VISIBLE_DEVICES"] = device
    env["NVIDIA_VISIBLE_DEVICES"] = device


def cleanup_workspace(workspace: ItemWorkspace, *, keep: bool) -> None:
    if keep:
        return
    shutil.rmtree(workspace.workspace_dir, ignore_errors=True)


def iter_restricted_host_paths(args: argparse.Namespace) -> list[Path]:
    """Paths that the outer sandbox should mask from the inner CLI."""

    candidates: list[Path] = []
    for attr in ("repo_root", "dataset_root", "input_path", "workspace_root", "output_root"):
        value = getattr(args, attr, None)
        if value is None:
            continue
        path = Path(value).resolve()
        if path not in candidates:
            candidates.append(path)
    return candidates


def iter_hidden_host_prefixes(args: argparse.Namespace) -> list[Path]:
    if not getattr(args, "workspace_only_outer_sandbox", False):
        return []
    workspace_root = getattr(args, "workspace_root", None)
    if workspace_root is None:
        return []
    return [Path(workspace_root).resolve().parent]


def wrap_outer_sandbox(
    *,
    inner_command: list[str],
    workspace: ItemWorkspace,
    args: argparse.Namespace,
    binary_path: Path,
) -> list[str]:
    """Wrap inner_command in bwrap. Caller decides whether to use it
    based on `args.outer_sandbox`."""

    return build_outer_sandbox_command(
        inner_command=inner_command,
        workspace_dir=workspace.workspace_dir,
        isolated_home=workspace.isolated_home,
        internal_workspace_root=INTERNAL_WORKSPACE_ROOT,
        internal_home_root=INTERNAL_HOME_ROOT,
        binary_bind_paths=iter_binary_bind_paths(binary_path),
        restricted_host_paths=iter_restricted_host_paths(args),
        hidden_host_prefixes=iter_hidden_host_prefixes(args),
        gpu_device_paths=iter_gpu_device_paths() if getattr(args, "allow_shell_gpu", False) else [],
        extra_readonly_bind_paths=(
            [Path(args.shared_python_env).resolve()]
            if getattr(args, "shared_python_env", None)
            else []
        ),
    )


def workspace_paths_for_invocation(
    workspace: ItemWorkspace, *, outer_sandbox: bool,
) -> tuple[Path, Path]:
    """Return (cwd, output_message_path) the inner CLI should be told.

    With outer_sandbox the CLI sees `INTERNAL_WORKSPACE_ROOT`; without
    it, it sees the host workspace dir.
    """

    if outer_sandbox:
        return INTERNAL_WORKSPACE_ROOT, INTERNAL_WORKSPACE_ROOT / "artifacts" / "last_message.txt"
    return workspace.workspace_dir, workspace.output_message_path


def should_retry_without_outer_sandbox(
    return_code: int | None,
    timed_out: bool,
    stderr_text: str,
) -> bool:
    """The bubblewrap layer occasionally fails with `EROFS` or
    permission errors that are unrelated to the inner CLI. When that
    happens we re-run without outer sandbox so the item can still
    succeed."""

    if timed_out:
        return False
    if return_code in (None, 0):
        return False
    needles = ("bwrap:", "EROFS", "Permission denied", "Read-only file system")
    return any(needle in (stderr_text or "") for needle in needles)
