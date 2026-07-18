"""Workspace, home, GPU, and runtime helpers for coding-agent runners."""

from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Any


def copy_file_if_exists(src: Path, dst: Path) -> None:
    if not src.exists():
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def reset_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def ensure_runner_home_dirs(isolated_home: Path) -> None:
    for dirname in (".cache", ".config", ".local", ".local/share", ".local/state"):
        (isolated_home / dirname).mkdir(parents=True, exist_ok=True)


def prepare_isolated_codex_home(workspace_dir: Path, *, copy_auth: bool = True) -> Path:
    isolated_home = workspace_dir / ".runner_home"
    codex_dir = isolated_home / ".codex"
    reset_dir(codex_dir)

    if copy_auth:
        source_codex_dir = Path.home() / ".codex"
        copy_file_if_exists(source_codex_dir / "auth.json", codex_dir / "auth.json")

    for dirname in ("memories", "cache", "log", "tmp"):
        (codex_dir / dirname).mkdir(parents=True, exist_ok=True)
    ensure_runner_home_dirs(isolated_home)
    return isolated_home


def prepare_isolated_claude_home(workspace_dir: Path, *, copy_credentials: bool = False) -> Path:
    isolated_home = workspace_dir / ".runner_home"
    claude_dir = isolated_home / ".claude"
    reset_dir(claude_dir)

    if copy_credentials:
        source_claude_dir = Path.home() / ".claude"
        copy_file_if_exists(source_claude_dir / ".credentials.json", claude_dir / ".credentials.json")

    for dirname in (".claude/tmp", ".claude/cache", ".claude/logs"):
        (isolated_home / dirname).mkdir(parents=True, exist_ok=True)
    ensure_runner_home_dirs(isolated_home)
    return isolated_home


def prepare_isolated_mixed_home(workspace_dir: Path) -> Path:
    isolated_home = prepare_isolated_claude_home(workspace_dir, copy_credentials=True)
    codex_dir = isolated_home / ".codex"
    reset_dir(codex_dir)

    source_codex_dir = Path.home() / ".codex"
    copy_file_if_exists(source_codex_dir / "auth.json", codex_dir / "auth.json")
    for dirname in ("memories", "cache", "log", "tmp"):
        (codex_dir / dirname).mkdir(parents=True, exist_ok=True)
    return isolated_home


def mirror_parent_dirs_under_home(isolated_home: Path, host_path: Path) -> None:
    home = Path.home().resolve()
    try:
        relative = host_path.resolve().relative_to(home)
    except ValueError:
        return
    (isolated_home / relative).parent.mkdir(parents=True, exist_ok=True)


def find_node_installation_root(path: Path) -> Path | None:
    resolved = path.expanduser().resolve()
    for parent in (resolved,) + tuple(resolved.parents):
        if (parent / "bin" / "node").exists() and (parent / "lib" / "node_modules").exists():
            return parent
    return None


def iter_binary_bind_paths(binary_path: Path) -> list[Path]:
    installation_root = find_node_installation_root(binary_path)
    if installation_root is not None:
        return [installation_root]
    return [binary_path]


def iter_gpu_device_paths() -> list[Path]:
    candidates = [
        Path("/dev/kfd"),
        Path("/dev/nvidiactl"),
        Path("/dev/nvidia-uvm"),
        Path("/dev/nvidia-uvm-tools"),
    ]
    candidates.extend(sorted(Path("/dev").glob("nvidia[0-9]*")))
    dri_root = Path("/dev/dri")
    if dri_root.exists():
        candidates.extend(sorted(dri_root.glob("card*")))
        candidates.extend(sorted(dri_root.glob("renderD*")))

    seen: set[str] = set()
    existing: list[Path] = []
    for path in candidates:
        path_str = str(path)
        if path_str in seen:
            continue
        seen.add(path_str)
        if path.exists():
            existing.append(path)
    return existing


def normalize_gpu_visible_devices(raw_value: str | None) -> str | None:
    if raw_value is None:
        return None
    normalized = raw_value.strip()
    if not normalized:
        return None
    if normalized.lower() in {"all", "any"}:
        return None
    return normalized


def parse_gpu_device_pool(raw_value: str | None) -> list[str]:
    if raw_value is None:
        return []

    devices: list[str] = []
    seen: set[str] = set()
    for part in raw_value.split(","):
        normalized = part.strip()
        if not normalized or normalized.lower() in {"all", "any"}:
            continue
        if normalized in seen:
            continue
        seen.add(normalized)
        devices.append(normalized)
    return devices


def _prepare_user_install_dirs(workspace_dir: Path) -> tuple[Path, list[Path]]:
    cache_root = workspace_dir / ".cache"
    py_user_base = workspace_dir / ".pyuserbase"
    npm_prefix = workspace_dir / ".npm-global"
    cargo_home = workspace_dir / ".cargo"
    rustup_home = workspace_dir / ".rustup"
    gem_home = workspace_dir / ".gem"
    bin_dirs = [
        py_user_base / "bin",
        npm_prefix / "bin",
        cargo_home / "bin",
        gem_home / "bin",
    ]

    for path in [
        cache_root,
        cache_root / "pip",
        cache_root / "uv",
        cache_root / "npm",
        cache_root / "pypoetry",
        py_user_base,
        npm_prefix,
        cargo_home,
        rustup_home,
        gem_home,
        *bin_dirs,
    ]:
        path.mkdir(parents=True, exist_ok=True)

    return cache_root, bin_dirs


_LEAKY_PARENT_CLAUDE_ENV_VARS = (
    # Claude Code CLI sets these inside its own session. If sbatch
    # `--export=ALL` propagates them to a child Claude run, the child believes
    # it is operating *inside* an existing Claude session: it tries to attach
    # to the parent's SSE port, recurses on `CLAUDE_CODE_EXECPATH`, and writes
    # its shell session state to `${TMPDIR}/claude-${UID}/...` which sits
    # outside the bwrap workspace and fails with EROFS.
    "CLAUDECODE",
    "CLAUDE_CODE_ENTRYPOINT",
    "CLAUDE_CODE_EXECPATH",
    "CLAUDE_CODE_SSE_PORT",
    "CLAUDE_CODE_OAUTH_TOKEN",
    # VS Code injects these when invoking us; not useful and may collide with
    # the child sandbox.
    "VSCODE_IPC_HOOK_CLI",
    "VSCODE_GIT_IPC_HANDLE",
)


def build_workspace_env(
    workspace_dir: Path,
    *,
    shared_python_env: str | None = None,
    shared_python_first: bool = False,
    isolated_home: Path | None = None,
    binary_path: Path | None = None,
    runtime_support_dir: Path | None = None,
    include_ml_cache: bool = False,
    provider_env_overrides: dict[str, Any] | None = None,
) -> dict[str, str]:
    env = os.environ.copy()
    for key in _LEAKY_PARENT_CLAUDE_ENV_VARS:
        env.pop(key, None)
    # Redirect TMPDIR into the per-run workspace. Otherwise tools that follow
    # TMPDIR (Claude Code's bash session state, npm/pip cache fallbacks, etc.)
    # write to the host TMPDIR which is not bind-mounted writable in the
    # bwrap outer sandbox.
    workspace_tmp = workspace_dir / "tmp"
    workspace_tmp.mkdir(parents=True, exist_ok=True)
    env["TMPDIR"] = str(workspace_tmp)
    env["TMP"] = str(workspace_tmp)
    env["TEMP"] = str(workspace_tmp)
    cache_root, bin_dirs = _prepare_user_install_dirs(workspace_dir)

    extra_bins = [str(path) for path in bin_dirs]
    if binary_path is not None:
        installation_root = find_node_installation_root(binary_path)
        if installation_root is not None:
            extra_bins.append(str((installation_root / "bin").resolve()))

    if shared_python_env:
        shared_env_root = Path(shared_python_env).expanduser().resolve()
        shared_env_bin = str((shared_env_root / "bin").resolve())
        if shared_python_first:
            extra_bins.insert(0, shared_env_bin)
        else:
            extra_bins.append(shared_env_bin)
        env["VIRTUAL_ENV"] = str(shared_env_root)

    inherited_path = env.get("PATH", "")
    env["PATH"] = os.pathsep.join(extra_bins + ([inherited_path] if inherited_path else []))

    if isolated_home is not None:
        env["HOME"] = str(isolated_home)
        env["CODEX_HOME"] = str(isolated_home / ".codex")
        env["XDG_CACHE_HOME"] = str(isolated_home / ".cache")
        env["XDG_CONFIG_HOME"] = str(isolated_home / ".config")
        env["XDG_DATA_HOME"] = str(isolated_home / ".local" / "share")
        env["XDG_STATE_HOME"] = str(isolated_home / ".local" / "state")
    else:
        env["XDG_CACHE_HOME"] = str(cache_root)

    py_user_base = workspace_dir / ".pyuserbase"
    npm_prefix = workspace_dir / ".npm-global"
    cargo_home = workspace_dir / ".cargo"
    rustup_home = workspace_dir / ".rustup"
    gem_home = workspace_dir / ".gem"

    env["PIP_CACHE_DIR"] = str(cache_root / "pip")
    env["PIP_DISABLE_PIP_VERSION_CHECK"] = "1"
    env["PIP_NO_INPUT"] = "1"
    env["PYTHONUSERBASE"] = str(py_user_base)
    env["UV_CACHE_DIR"] = str(cache_root / "uv")
    env["POETRY_CACHE_DIR"] = str(cache_root / "pypoetry")
    env["npm_config_cache"] = str(cache_root / "npm")
    env["npm_config_prefix"] = str(npm_prefix)
    env["CARGO_HOME"] = str(cargo_home)
    env["RUSTUP_HOME"] = str(rustup_home)
    env["GEM_HOME"] = str(gem_home)
    env["GEM_PATH"] = str(gem_home)

    if include_ml_cache:
        hf_home = cache_root / "huggingface"
        hf_hub = hf_home / "hub"
        transformers_cache = hf_home / "transformers"
        datasets_cache = hf_home / "datasets"
        for path in (hf_home, hf_hub, transformers_cache, datasets_cache, cache_root / "torch"):
            path.mkdir(parents=True, exist_ok=True)
        env["HF_HOME"] = str(hf_home)
        env["HUGGINGFACE_HUB_CACHE"] = str(hf_hub)
        env["TRANSFORMERS_CACHE"] = str(transformers_cache)
        env["HF_DATASETS_CACHE"] = str(datasets_cache)
        env["TORCH_HOME"] = str(cache_root / "torch")

    if runtime_support_dir is not None:
        existing_pythonpath = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = (
            f"{runtime_support_dir}:{existing_pythonpath}" if existing_pythonpath else str(runtime_support_dir)
        )

    if provider_env_overrides:
        env.update({str(key): str(value) for key, value in provider_env_overrides.items()})
    return env


def write_torch_gpu_runtime_support(workspace_dir: Path, *, env_var: str) -> Path:
    support_dir = workspace_dir / ".runtime_support"
    support_dir.mkdir(parents=True, exist_ok=True)
    sitecustomize_path = support_dir / "sitecustomize.py"
    sitecustomize_path.write_text(
        f"""
import os

fraction_value = os.environ.get({env_var!r}, "").strip()
if fraction_value:
    try:
        fraction = float(fraction_value)
    except ValueError:
        fraction = None
    if fraction is not None and 0.0 < fraction <= 1.0:
        try:
            import torch

            if torch.cuda.is_available():
                visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
                if visible:
                    device_count = len([part for part in visible.split(",") if part.strip() and part.strip() != "-1"])
                else:
                    device_count = torch.cuda.device_count()
                for device_index in range(device_count):
                    try:
                        torch.cuda.set_per_process_memory_fraction(fraction, device_index)
                    except Exception:
                        pass
        except Exception:
            pass
""".strip()
        + "\n",
        encoding="utf-8",
    )
    return support_dir
