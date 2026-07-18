"""Bubblewrap sandbox command helpers for coding-agent runners."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

from omnicoding.benchmarks.common.workspace import mirror_parent_dirs_under_home


def _append_workspace_env(
    command: list[str], internal_home_root: Path, internal_workspace_root: Path
) -> None:
    workspace_tmp = internal_workspace_root / "tmp"
    command.extend(
        [
            "--setenv",
            "HOME",
            str(internal_home_root),
            "--setenv",
            "CODEX_HOME",
            str(internal_home_root / ".codex"),
            "--setenv",
            "XDG_CACHE_HOME",
            str(internal_home_root / ".cache"),
            "--setenv",
            "XDG_CONFIG_HOME",
            str(internal_home_root / ".config"),
            "--setenv",
            "XDG_DATA_HOME",
            str(internal_home_root / ".local" / "share"),
            "--setenv",
            "XDG_STATE_HOME",
            str(internal_home_root / ".local" / "state"),
            # Inside bwrap, `/` is read-only. Point TMPDIR at a writable
            # workspace-relative path so Claude Code's bash session state
            # (`${TMPDIR}/claude-${UID}/<encoded-cwd>/...`) doesn't blow up
            # with EROFS on the next shell tool call.
            "--setenv",
            "TMPDIR",
            str(workspace_tmp),
            "--setenv",
            "TMP",
            str(workspace_tmp),
            "--setenv",
            "TEMP",
            str(workspace_tmp),
        ]
    )


def _append_ml_cache_env(command: list[str], internal_workspace_root: Path) -> None:
    cache_root = internal_workspace_root / ".cache"
    command.extend(
        [
            "--setenv",
            "HF_HOME",
            str(cache_root / "huggingface"),
            "--setenv",
            "HUGGINGFACE_HUB_CACHE",
            str(cache_root / "huggingface" / "hub"),
            "--setenv",
            "TRANSFORMERS_CACHE",
            str(cache_root / "huggingface" / "transformers"),
            "--setenv",
            "HF_DATASETS_CACHE",
            str(cache_root / "huggingface" / "datasets"),
            "--setenv",
            "TORCH_HOME",
            str(cache_root / "torch"),
        ]
    )


def build_outer_sandbox_command(
    *,
    inner_command: list[str],
    workspace_dir: Path,
    isolated_home: Path,
    internal_workspace_root: Path,
    internal_home_root: Path,
    binary_bind_paths: Iterable[Path],
    restricted_host_paths: Iterable[Path],
    hidden_host_prefixes: Iterable[Path],
    gpu_device_paths: Iterable[Path] = (),
    extra_readonly_bind_paths: Iterable[Path] = (),
    include_ml_cache_env: bool = False,
) -> list[str]:
    mask_root = workspace_dir / ".sandbox_masks"
    empty_dir = mask_root / "empty_dir"
    empty_file = mask_root / "empty_file"
    empty_dir.mkdir(parents=True, exist_ok=True)
    empty_file.parent.mkdir(parents=True, exist_ok=True)
    empty_file.write_text("", encoding="utf-8")

    command = [
        "bwrap",
        "--die-with-parent",
        "--ro-bind",
        "/",
        "/",
        "--dev",
        "/dev",
        "--proc",
        "/proc",
        "--bind",
        "/tmp",
        "/tmp",
        "--bind",
        str(isolated_home),
        str(internal_home_root),
        "--dir",
        str(internal_workspace_root),
        "--bind",
        str(workspace_dir),
        str(internal_workspace_root),
        "--chdir",
        str(internal_workspace_root),
    ]
    _append_workspace_env(command, internal_home_root, internal_workspace_root)
    if include_ml_cache_env:
        _append_ml_cache_env(command, internal_workspace_root)

    for device_path in gpu_device_paths:
        command.extend(["--dev-bind-try", str(device_path), str(device_path)])

    for prefix in hidden_host_prefixes:
        if prefix.resolve() == workspace_dir.resolve():
            continue
        command.extend(["--bind", str(empty_dir), str(prefix)])

    for bind_path in binary_bind_paths:
        mirror_parent_dirs_under_home(isolated_home, bind_path)
        command.extend(["--ro-bind", str(bind_path), str(bind_path)])

    for path in restricted_host_paths:
        if path.resolve() == workspace_dir.resolve():
            continue
        mount_source = empty_dir if path.is_dir() else empty_file
        command.extend(["--bind", str(mount_source), str(path)])

    for path in extra_readonly_bind_paths:
        if not path.exists():
            continue
        command.extend(["--ro-bind", str(path), str(path)])

    command.append("--")
    command.extend(inner_command)
    return command
