"""Generic KIRA driver for the local-model benchmarks.

Mirrors ``run_bench_mini_swe.py`` and ``run_bench_opencode.py``: takes
``--bench`` to dispatch on a registered ``BenchSpec``, builds the codex-
style prompt, runs ``KiraAgent`` in a per-item ``/tmp`` workspace, and
writes ``results.json`` consumed by the wide-smoke analyzer.

For large runs, submit through a batch executor rather than running workers on
a login node.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import shutil
import sys
import tempfile
import time
import traceback
from pathlib import Path
from typing import Any, Optional

from omnicoding.benchmarks import specs  # noqa: E402
from omnicoding.harnesses.effort_policy import pick_reasoning_effort, step_limit_for_effort  # noqa: E402
# I/O + workspace helpers live in a sibling module so this driver stays
# under the 800-line cap. Re-export underscore aliases for back-compat
# with existing test imports (``from run_bench_kira import _foo``).
from omnicoding.harnesses.kira_io import (  # noqa: E402
    decode_subcall_images_to_dir as _decode_subcall_images_to_dir,
    setup_per_job_venv as _setup_per_job_venv,
    filter_by_source_indices as _filter_by_source_indices,
    load_existing_results as _load_existing_results,
    load_prior_rows as _load_prior_rows,
    archive_prior_row as _archive_prior_row,
    atomic_write_results as _atomic_write_results,
    sorted_rows as _sorted_rows,
)
import os as _os  # noqa: E402  — KIRA_PREV_EFFORT lookup below
from omnicoding.benchmarks.common.spec import (  # noqa: E402
    CONTINUE_RETRY_LIMIT,
    RELATIVE_PATH_HINT,
    BuildPromptCtx,
    ResultRowCtx,
    build_continue_prompt,
    current_gpu_device,
)
from omnicoding.benchmarks.common.runtime import assign_gpu_device  # noqa: E402
from omnicoding.benchmarks.common.workspace import parse_gpu_device_pool  # noqa: E402
from omnicoding.agents.kira import (  # noqa: E402
    AgentResult,
    EndpointPool,
    KiraAgent,
    SYSTEM_PROMPT,
    TOOLS,
    default_api_base,
    default_max_tool_reminders,
    messages_preview,
    parse_endpoints,
    resolve_provider,
    trajectory_to_dicts,
)
from omnicoding.agents.kira.endpoint_pool import EndpointSession  # noqa: E402
from omnicoding.agents.kira.shell import DEFAULT_MAX_OUTPUT_BYTES  # noqa: E402
from omnicoding.benchmarks.common.shared_python import normalize_shared_python_env
from omnicoding.paths import runtime_root

LOGGER = logging.getLogger("run_bench_kira")
HARNESS_VERSION = "1.0.0"
HARNESS_NAME = "kira"

REPO_ROOT = runtime_root()
WEB_SEARCH_BIN_DIR = os.environ.get("OMNICODING_WEB_SEARCH_BIN_DIR", "").strip()
# Bundled extra binaries for the worker shell. Currently just ripgrep
# (round-17: GPT-5.5 reaches for ``rg`` reflexively; without it on PATH
# the model fell back to ``grep -nR`` and lost a step). Add other
# self-contained tools here as needed; keep them static so the binary
# works on any compute node.
EXTRA_BIN_DIR = os.environ.get("OMNICODING_EXTRA_BIN_DIR", "").strip()
WEB_SEARCH_PROMPT_HINT = (
    "\n\nA web_search command is on PATH; use it for any fact-finding the "
    "staged inputs cannot answer. Usage:\n"
    "  web_search \"<query>\"             # markdown-formatted Tavily results\n"
    "  web_search \"<query>\" --max 8     # top 8 results\n"
    "  web_search \"<query>\" --json      # raw Tavily JSON\n"
    "Keep queries short and specific. Prefer quoting concrete entities."
)


def _build_extra_env() -> dict[str, str]:
    """Inject web_search + bundled tools (ripgrep) on PATH for the
    agent's bash, mirroring ``run_bench_mini_swe`` so the model sees the
    same shell tooling on every compute node."""
    cur_path = os.environ.get("PATH", "")
    parts = cur_path.split(os.pathsep)
    prepend = []
    for raw_dir in (WEB_SEARCH_BIN_DIR, EXTRA_BIN_DIR):
        if not raw_dir:
            continue
        directory = str(Path(raw_dir).expanduser().resolve())
        if directory not in parts:
            prepend.append(directory)
    if not prepend:
        return {}
    return {"PATH": os.pathsep.join(prepend + [cur_path])}


def _resolve_provider_defaults(args: argparse.Namespace) -> None:
    """Fill in api_base / api_key / max_tool_reminders based on the
    auto-detected provider when the caller didn't pass them. Mutates
    args in place so downstream code reads a single resolved namespace.

    Resolution order:
      1. CLI flag (``--api_base xxx``)
      2. Provider-aware env (e.g. OPENROUTER_API_KEY for openrouter/*)
      3. Generic env (OPENAI_API_BASE / OPENAI_API_KEY)
      4. Provider default (``kira.provider.default_api_base``)
      5. Local fallback (qwen → 127.0.0.1:8080/v1, key → 'local')
    """
    provider = resolve_provider(args.model_name, args.provider)
    args.provider = provider

    # Multi-endpoint pool: when set, takes precedence over --api_base.
    # Each item in the run is pinned (sticky) to one endpoint so the
    # agent's multi-step KV cache stays warm at that sglang instance.
    args.endpoint_pool = None
    if args.endpoints:
        args.endpoint_pool = parse_endpoints(args.endpoints)
        # Set api_base to the first endpoint as a sane single-endpoint
        # fallback (e.g. for the resolved-provider log line below).
        if args.api_base is None:
            args.api_base = args.endpoint_pool.endpoints[0].url

    if args.api_base is None:
        args.api_base = (
            os.environ.get("OPENAI_API_BASE")
            or default_api_base(provider)
            or ("http://127.0.0.1:8080/v1" if provider == "qwen" else None)
        )

    if args.api_key is None:
        if provider == "openrouter":
            args.api_key = os.environ.get("OPENROUTER_API_KEY") or ""
        else:
            args.api_key = os.environ.get("OPENAI_API_KEY") or "local"

    if args.max_tool_reminders is None:
        # Driver-side override: claude/codex/opencode/kira share
        # CONTINUE_RETRY_LIMIT for Qwen runs to keep the wide-smoke
        # analyzer dedupe-friendly. For non-Qwen, fall through to the
        # provider default (qwen=10, anthropic/openai=2, others=4).
        if provider == "qwen":
            args.max_tool_reminders = CONTINUE_RETRY_LIMIT
        else:
            args.max_tool_reminders = default_max_tool_reminders(provider)

    pool_desc = args.endpoint_pool.describe() if args.endpoint_pool else "(single)"
    LOGGER.info(
        "kira.driver provider=%s api_base=%s endpoint_pool=%s max_tool_reminders=%d "
        "enable_thinking=%s reasoning_effort=%s thinking_budget=%s",
        provider, args.api_base, pool_desc, args.max_tool_reminders,
        args.enable_thinking, args.reasoning_effort, args.thinking_budget_tokens,
    )
    if provider == "openrouter" and not args.api_key:
        raise SystemExit(
            "openrouter provider detected but no API key. Set "
            "OPENROUTER_API_KEY or pass --api_key."
        )


def _run_one(
    spec,
    item: dict[str, Any],
    idx: int,
    dataset_root: Path,
    args: argparse.Namespace,
    base_extra_env: dict[str, str],
    assigned_gpu: Optional[str] = None,
) -> dict[str, Any]:
    """Stage inputs into a tmp workspace, run the agent, score, and
    return the BenchSpec-shaped result row."""
    workspace = Path(tempfile.mkdtemp(prefix=f"kira_{spec.name}_{idx:04d}_", dir=args.workspace_root))
    error: str | None = None
    result: AgentResult | None = None
    env_setup_seconds = 0.0
    # Per-task env: web_search PATH plus, when --gpu_device_pool is in
    # use, the slot we acquired. assign_gpu_device is a no-op if None.
    extra_env = dict(base_extra_env)
    assign_gpu_device(extra_env, assigned_gpu)
    # Per-job venv clone (default ON): the model can pip install whatever
    # it needs into <workspace>/.venv without polluting the shared base.
    # Cloned once per item; torn down with the workspace.
    base_venv = Path(args.shared_python_env) if args.shared_python_env else None
    job_venv_path = base_venv
    if not args.no_per_job_venv and base_venv is not None:
        t0 = time.monotonic()
        job_venv_path = _setup_per_job_venv(workspace, base_venv)
        env_setup_seconds = time.monotonic() - t0
        # PATH order: overlay/bin (model's python+pip) → base/bin (the
        # baseline's whisper/etc., Python 3.10 shebangs) → inherited
        # PATH (which on many clusters also has a system or environment
        # ffmpeg/ffprobe — those stay reachable as a fallback because
        # base venv doesn't ship them as ``bin/ffmpeg`` symlinks).
        # Without base/bin in front, miniconda's whisper@Python 3.13
        # would win the lookup (verified in r17.6 soak). DON'T outright
        # scrub miniconda from PATH — the model genuinely needs ffmpeg
        # and the base venv only ships imageio_ffmpeg as a versioned
        # site-package binary, not a PATH-resolvable ``ffmpeg`` symlink.
        overlay_bin = str(job_venv_path / "bin")
        base_bin = str(base_venv / "bin")
        cur_path = extra_env.get("PATH") or os.environ.get("PATH", "")
        extra_env["PATH"] = os.pathsep.join([overlay_bin, base_bin, cur_path])
        extra_env["VIRTUAL_ENV"] = str(job_venv_path)
    # Sticky per-item endpoint: every LLM call within this _run_one
    # invocation goes to the same sglang so the multi-step KV cache
    # stays warm. With no pool, fall back to the legacy single api_base.
    # When the pool has ≥2 URLs, build a session so failover can rotate
    # mid-item if the sticky URL stalls / dies — preserves the rollout.
    pool: EndpointPool | None = getattr(args, "endpoint_pool", None)
    session: EndpointSession | None = None
    if pool is not None:
        # ``shard_offset`` shifts this shard's starting pool index so
        # parallel shards don't all dogpile pool[0] on item 0. Submitter
        # sets it to the shard number (0..N-1); pool's modulo wraps it.
        pool_idx = idx + getattr(args, "shard_offset", 0)
        session = EndpointSession(pool, idx=pool_idx)
        api_base = session.current_url
    else:
        api_base = args.api_base
    try:
        staged = spec.stage_inputs(item, dataset_root, workspace)
        ctx = BuildPromptCtx(
            item=item,
            staged_paths=list(staged),
            sandbox="workspace-write",
            allow_shell_network=True,
            allow_shell_gpu=True,
            shared_python_env=str(job_venv_path) if job_venv_path else None,
            disable_native_vision=False,
            extra_system_prompt="",
        )
        # Split-prompt assembly. The spec's static prefix
        # (workspace/network/tool-workflow rules) plus the kira
        # harness-runtime hints (WEB_SEARCH bash wrapper, relative-path
        # serializer warning) land in ``role=system`` so the LLM
        # provider's prompt cache holds them byte-identical across all
        # items. Only the per-item ``Available staged files`` +
        # Question + Options live in ``role=user``.
        system_prefix = (
            spec.build_system_prefix(ctx)
            + WEB_SEARCH_PROMPT_HINT
            + RELATIVE_PATH_HINT
        )
        user_question = spec.build_user_question(ctx)
        prompt = system_prefix + "\n\n" + user_question  # prompt.txt audit only
        LOGGER.info(
            "kira.prompt bench=%s idx=%d sys_prefix_len=%d user_q_len=%d",
            spec.name, idx, len(system_prefix), len(user_question),
        )
        (workspace / "artifacts").mkdir(parents=True, exist_ok=True)
        (workspace / "artifacts" / "prompt.txt").write_text(prompt, encoding="utf-8")

        # Per-item reasoning_effort policy. ``--effort_strategy=auto``
        # picks low/medium/high/xhigh from the item's ``Level`` field
        # (or random low/medium when unlabeled, attempt 1) or random
        # high/xhigh on attempt ≥ 2 (escalation). ``fixed`` keeps the
        # legacy behavior of using ``args.reasoning_effort`` verbatim.
        if args.effort_strategy == "auto":
            effort = pick_reasoning_effort(
                item, attempt=args.attempt, prev_effort=args.prev_effort or None,
            )
            LOGGER.info(
                "kira.driver effort=auto si=%s level=%s attempt=%d prev=%s → %s",
                item.get("__source_index__"), item.get("Level") or "<none>",
                args.attempt, args.prev_effort or "<none>", effort,
            )
        else:
            effort = args.reasoning_effort

        # Per-effort step_limit. ``--effort_strategy=auto`` derives it
        # from the chosen effort (low:40 medium:80 high:100 xhigh:100);
        # ``fixed`` keeps the CLI/env value verbatim.
        if args.effort_strategy == "auto":
            step_limit = step_limit_for_effort(effort, fallback=args.step_limit)
        else:
            step_limit = args.step_limit

        agent = KiraAgent(
            workspace=workspace,
            model_name=args.model_name,
            provider=args.provider,
            api_base=api_base,
            endpoint_session=session,
            continue_prompt=build_continue_prompt(spec),
            api_key=args.api_key,
            step_limit=step_limit,
            request_timeout_s=args.request_timeout,
            block_timeout_s=args.block_timeout,
            extra_env=extra_env,
            enable_thinking=args.enable_thinking,
            reasoning_effort=effort,
            thinking_budget_tokens=args.thinking_budget_tokens,
            max_tool_reminders=args.max_tool_reminders,
            enable_summarize=args.enable_summarize,
            temperature=args.temperature,
            top_p=args.top_p,
            seed=args.seed,
            # Land the sub-call log inside the workspace so it survives
            # alongside ``messages.json``; the post-run mirror block
            # below copies it to ``out_artifacts/`` for SFT consumers.
            image_subcall_log=workspace / "artifacts" / "image_subcalls.jsonl",
            image_read_mode=args.image_read_mode,
        )
        try:
            result = agent.run(user_question, system_prefix=system_prefix)
        except Exception as exc:  # noqa: BLE001
            error = f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}"
            LOGGER.exception("KiraAgent.run crashed for idx=%d", idx)

        final_text = result.final_text if result else ""
        prediction = spec.extract_prediction(final_text, item)
        is_correct = spec.is_correct(item, prediction)

        retry_attempts = list(result.retry_attempts) if result else []
        extra: dict[str, Any] = {
            "harness": HARNESS_NAME,
            "harness_version": HARNESS_VERSION,
            "model": args.model_name,
            "api_base": session.current_url if session else api_base,
            "endpoint_failovers": session.history() if session else [],
            "gpu_device_assigned": assigned_gpu or current_gpu_device(),
            "exit_reason": result.exit_reason if result else "preflight_error",
            "completed": bool(result and result.completed),
            "prompt_tokens": result.cumulative_prompt_tokens if result else 0,
            "completion_tokens": result.cumulative_completion_tokens if result else 0,
            "cached_tokens": result.cumulative_cached_tokens if result else 0,
            "reasoning_tokens": result.cumulative_reasoning_tokens if result else 0,
            "trajectory_steps": trajectory_to_dicts(result.trajectory) if result else [],
            "messages_preview": messages_preview(result.messages) if result else [],
            "retry_attempts": retry_attempts,
            "retry_count": len(retry_attempts),
            "error": error or (result.error if result else None),
            # Sampling / provider knobs that change the output distribution.
            # SFT-RL parity needs these per-row so a downstream replay or
            # rollout script can reproduce (or detect mismatch with) the
            # generation settings used for this trajectory.
            "reasoning_effort": effort,                        # actual effort used
            "reasoning_effort_strategy": args.effort_strategy,  # how it was picked
            "attempt": int(args.attempt),
            "step_limit_used": int(step_limit),
            "thinking_budget_tokens": args.thinking_budget_tokens,
            "enable_thinking": bool(args.enable_thinking),
            "enable_summarize": bool(args.enable_summarize),
            "step_limit": int(args.step_limit),
            "max_tool_reminders": int(args.max_tool_reminders or 0),
            "temperature": args.temperature,
            "top_p": args.top_p,
            "seed": args.seed,
            "n_summarizations": int(result.n_summarizations) if result else 0,
            "per_job_venv": (
                str(job_venv_path)
                if job_venv_path is not None and job_venv_path != base_venv
                else None
            ),
            "env_setup_seconds": round(env_setup_seconds, 2),
        }
        # Persist the full conversation under the workspace AND under
        # the per-item OUT dir. Workspace lives on compute-node /tmp
        # (not reachable from the login node), so a parallel copy in
        # OUT (on shared NFS) is what makes round-N debugging usable.
        if result is not None:
            artifacts_dir = workspace / "artifacts"
            artifacts_dir.mkdir(parents=True, exist_ok=True)
            # No top-level slice: per-turn observations are already capped
            # (shell 30 KB, image_read text bounded by sub-LLM). Slicing the
            # serialized JSON would chop mid-string and yield invalid JSON
            # that SFT loaders can't parse.
            messages_blob = json.dumps(result.messages, ensure_ascii=False, indent=2)
            (artifacts_dir / "messages.json").write_text(messages_blob, encoding="utf-8")
            # Per-trajectory dir keyed by ``__source_index__``, NOT by the
            # in-loop ``idx``. With the on-demand dispatcher (one slurm job
            # per item), every job sees ``pending=[item_for_si]`` so its
            # ``idx`` is always 0 — using idx would land all 8 jobs in
            # ``item_0000/`` and overwrite each other's messages/images.
            si = int(item.get("__source_index__", idx))
            out_artifacts = Path(args.output_dir).resolve() / f"item_{si:04d}"
            out_artifacts.mkdir(parents=True, exist_ok=True)
            (out_artifacts / "messages.json").write_text(messages_blob, encoding="utf-8")
            (out_artifacts / "final_text.txt").write_text(
                result.final_text, encoding="utf-8",
            )
            (out_artifacts / "trajectory.json").write_text(
                json.dumps(trajectory_to_dicts(result.trajectory), ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            if result.pre_summary_snapshots:
                (out_artifacts / "pre_summary_snapshots.json").write_text(
                    json.dumps(
                        result.pre_summary_snapshots, ensure_ascii=False, indent=2,
                    ),
                    encoding="utf-8",
                )
            # Mirror image_read sub-call log (raw image bytes + sub-LLM
            # request/response) so trajectories stay self-contained for
            # SFT/RL replay even after the workspace is wiped.
            src_img_log = artifacts_dir / "image_subcalls.jsonl"
            if src_img_log.exists():
                shutil.copy2(src_img_log, out_artifacts / "image_subcalls.jsonl")
                # Decode each successful image_b64 to a real file under
                # ``out_artifacts/images/`` so the multimodal SFT path can
                # reference paths that survive on shared NFS — /tmp gets
                # wiped post-run and was the only persistent home before.
                _decode_subcall_images_to_dir(
                    src_img_log, out_artifacts / "images",
                )

        # exit_reason="error" + "timeout" in the error string is the
        # only way the loop surfaces a request_timeout_s firing; mark
        # the row so the wide-smoke analyzer sees it the same way as
        # opencode/claude TimeoutExpired rows. result.error is None
        # for clean exits (task_complete, no_tool_calls, step_limit) —
        # coerce to "" before concatenating so we don't TypeError.
        err_blob = (error or "") + " " + ((result.error if result else None) or "")
        timed_out = bool(
            result and result.exit_reason == "error"
            and "timeout" in err_blob.lower()
        )
        row_ctx = ResultRowCtx(
            item=item,
            item_index=idx,
            prediction=prediction,
            is_correct=is_correct,
            # Round-12 BUG-K4: slice the LAST 8000 chars, not the
            # first. On long runs (50+ steps) the model's
            # ``<answer>...</answer>`` lands at the end of the
            # transcript; the first 8000 are early-trajectory ffprobe /
            # ls output and useless for post-hoc audit. extract_prediction
            # already runs on the FULL final_text so prediction is
            # unaffected; this only changes the debug field.
            raw_model_output=final_text[-8000:],
            tool_call_num=int(result.n_tool_calls) if result else 0,
            return_code=None,
            timed_out=timed_out,
            stdout_text="",
            stderr_text=error or "",
            workspace_dir=workspace,
            keep_workdirs=os.environ.get("KEEP_WORKDIRS") == "1",
            include_gold_fields=args.include_gold_fields_in_results,
            extra=extra,
        )
        return spec.result_row(row_ctx)
    finally:
        if os.environ.get("KEEP_WORKDIRS") != "1":
            shutil.rmtree(workspace, ignore_errors=True)


async def _run_one_async(
    spec,
    item: dict[str, Any],
    idx: int,
    dataset_root: Path,
    args: argparse.Namespace,
    base_extra_env: dict[str, str],
    semaphore: asyncio.Semaphore,
    gpu_slot_queue: Optional[asyncio.Queue],
) -> dict[str, Any]:
    async with semaphore:
        assigned_gpu: Optional[str] = None
        if gpu_slot_queue is not None:
            assigned_gpu = await gpu_slot_queue.get()
        try:
            return await asyncio.to_thread(
                _run_one, spec, item, idx, dataset_root, args, base_extra_env, assigned_gpu,
            )
        finally:
            if gpu_slot_queue is not None and assigned_gpu is not None:
                await gpu_slot_queue.put(assigned_gpu)


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bench", required=True, choices=specs.names())
    parser.add_argument("--input_file", required=True)
    parser.add_argument("--dataset_root", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument(
        "--model_name",
        default=os.environ.get("KIRA_MODEL_NAME", "openai/Qwen3.6-27B"),
        help="litellm model id. Examples: openai/Qwen3.6-27B (local sglang), "
             "openrouter/anthropic/claude-sonnet-4-5, openai/gpt-5.",
    )
    parser.add_argument(
        "--provider",
        default=os.environ.get("KIRA_PROVIDER", "auto"),
        choices=["auto", "qwen", "openai", "anthropic", "openrouter", "other"],
        help="Semantic model provider. Keep the LiteLLM routing prefix in "
             "--model_name and use this override for provider-specific chat "
             "behavior (for example: openai/... with --provider qwen).",
    )
    parser.add_argument(
        "--api_base", default=None,
        help="Single API base URL. Defaults: qwen → OPENAI_API_BASE env or "
             "http://127.0.0.1:8080/v1; openrouter → https://openrouter.ai/api/v1. "
             "Ignored when --endpoints is set.",
    )
    parser.add_argument(
        "--endpoints", default=os.environ.get("KIRA_ENDPOINTS") or "",
        help="Multi-endpoint pool with weighted sticky routing. Format: "
             "'URL=W,URL2=W2' or 'URL=W|URL2=W2' (use '|' inside SLURM "
             "--export to avoid comma splitting). Each item is pinned "
             "to one endpoint by (idx mod Σweights) so the multi-step "
             "KV cache stays warm. Weights should mirror each "
             "endpoint's --max-running-requests budget. Empty/unset → "
             "legacy single --api_base mode.",
    )
    parser.add_argument(
        "--shard_offset", type=int,
        default=int(os.environ.get("KIRA_SHARD_OFFSET", "0")),
        help="Offset added to per-item idx when picking the sticky "
             "endpoint. Set to the shard number (0..N-1) so parallel "
             "shards start on different endpoints — without this all "
             "shards dogpile pool[0] on item 0 and the second host "
             "stays idle for the longest item. No effect without "
             "--endpoints.",
    )
    parser.add_argument(
        "--api_key", default=None,
        help="API key. Defaults: openrouter → OPENROUTER_API_KEY env; "
             "qwen / openai → OPENAI_API_KEY env or 'local'.",
    )
    parser.add_argument("--step_limit", type=int, default=int(os.environ.get("KIRA_STEP_LIMIT", "120")))
    parser.add_argument("--request_timeout", type=int, default=int(os.environ.get("KIRA_REQUEST_TIMEOUT", "900")))
    parser.add_argument("--block_timeout", type=int, default=int(os.environ.get("KIRA_BLOCK_TIMEOUT", "600")),
                        help="Harness-side wall clock for stuck LLM calls.")
    parser.add_argument("--image_read_mode",
                        default=os.environ.get("KIRA_IMAGE_READ_MODE", "native"),
                        choices=["native", "sub_llm"],
                        help="native (default): image_read injects the actual "
                             "image into the main agent's conversation as a "
                             "follow-up user message. sub_llm: legacy path — a "
                             "separate vision LLM converts the image to text "
                             "and that text is the tool reply.")
    parser.add_argument("--enable_thinking", default=os.environ.get("KIRA_ENABLE_THINKING", "1") == "1",
                        action="store_true",
                        help="Qwen-only: send chat_template_kwargs.enable_thinking=true. "
                             "Default ON for the published collection recipe. "
                             "Ignored for non-Qwen providers.")
    parser.add_argument("--reasoning_effort", default=os.environ.get("KIRA_REASONING_EFFORT") or None,
                        help="OpenAI o-series only: low|medium|high. Used "
                             "verbatim when --effort_strategy=fixed; ignored "
                             "when --effort_strategy=auto.")
    parser.add_argument("--effort_strategy",
                        default=os.environ.get("KIRA_EFFORT_STRATEGY") or "fixed",
                        choices=["fixed", "auto"],
                        help="fixed (default): use --reasoning_effort verbatim. "
                             "auto: derive per-item from item['Level'] "
                             "({easy:low, medium:medium, hard:high}); when "
                             "Level is unset, random pick from {low,medium} "
                             "on attempt 1, {high,xhigh} on attempt ≥ 2 "
                             "(seeded by item id for reproducibility).")
    parser.add_argument("--attempt", type=int,
                        default=int(os.environ.get("KIRA_ATTEMPT") or "1"),
                        help="Trajectory attempt number (1=first run, "
                             "≥2=escalation). Drives effort_policy on auto.")
    parser.add_argument("--prev_effort",
                        default=os.environ.get("KIRA_PREV_EFFORT") or "",
                        help="Effort used on the previous attempt (used by "
                             "effort_policy when --attempt ≥ 2 to enforce "
                             "monotonic escalation: never picks ≤ prev). "
                             "Empty → legacy random {high, xhigh} fallback.")
    parser.add_argument("--thinking_budget_tokens", type=int,
                        default=int(os.environ.get("KIRA_THINKING_BUDGET", "0")) or None,
                        help="Anthropic only: extended-thinking budget tokens.")
    parser.add_argument("--max_tool_reminders", type=int,
                        default=int(os.environ.get("KIRA_MAX_TOOL_REMINDERS", "0")) or None,
                        help="0 / unset → use kira.provider.default_max_tool_reminders. "
                             "Driver default also applies CONTINUE_RETRY_LIMIT below if Qwen.")
    parser.add_argument("--enable_summarize", default=os.environ.get("KIRA_ENABLE_SUMMARIZE", "1") == "1",
                        action="store_true",
                        help="On context-overflow, run a summarizer LLM and retry once. "
                             "Default ON; set KIRA_ENABLE_SUMMARIZE=0 to surface raw error.")
    parser.add_argument("--temperature", type=float,
                        default=(float(os.environ.get("KIRA_TEMPERATURE")) if os.environ.get("KIRA_TEMPERATURE") else None),
                        help="Sampling temperature. None → provider default (sglang ~1.0, "
                             "OpenAI o-series fixed at 1.0). Set explicitly for "
                             "trajectory replay / RL rollout parity.")
    parser.add_argument("--top_p", type=float,
                        default=(float(os.environ.get("KIRA_TOP_P")) if os.environ.get("KIRA_TOP_P") else None),
                        help="Nucleus-sampling cutoff. None → provider default.")
    parser.add_argument("--seed", type=int,
                        default=(int(os.environ.get("KIRA_SEED")) if os.environ.get("KIRA_SEED") else None),
                        help="Sampling seed. None → unset. sglang honors it; OpenAI / "
                             "OpenRouter best-effort. Recorded in results.json regardless "
                             "so SFT data prep can dedup deterministically-seeded runs.")
    parser.add_argument("--max_items", type=int, default=1)
    parser.add_argument(
        "--source_indices", default=os.environ.get("KIRA_SOURCE_INDICES") or "",
        help="Comma-separated list of __source_index__ values; only items "
             "whose source_index is in the list run. Used by the on-demand "
             "single-item dispatcher (one slurm job per item). Applied on "
             "top of spec.filter_items so --max_items still bounds output.",
    )
    parser.add_argument("--workspace_root", default="/tmp",
                        help="Parent directory for per-item temp workspaces "
                             "(matches claude/codex --workspace_root).")
    parser.add_argument(
        "--shared_python_env",
        default=os.environ.get("OMNICODING_SHARED_PYTHON_ENV") or None,
        help="Optional base virtualenv exposed to the agent. When set, the "
             "default behavior creates an isolated per-item overlay.",
    )
    parser.add_argument("--concurrent_limit", type=int, default=1,
                        help="Max items processed in parallel (asyncio Semaphore).")
    parser.add_argument("--gpu_device_pool", default=None,
                        help="Comma-separated GPU indices for per-item assignment "
                             "(e.g. '0,1,2'); empty/None means use SLURM default.")
    parser.add_argument("--include_gold_fields_in_results", action="store_true")
    parser.add_argument(
        "--no_per_job_venv", action="store_true",
        help="Disable per-job venv clone. Default: each item gets its own "
             "writable copy of the configured base environment under "
             "<workspace>/.venv so "
             "the model can pip install without polluting the shared base. "
             "Disable if base venv is missing or you want to use the base "
             "directly (read-only since round-17.6).",
    )
    parser.add_argument(
        "--results_filename", default="results.json",
        help="Per-shard results file under --output_dir. Default 'results.json'. "
             "On-demand single-item dispatcher uses 'results.<source_index>.json' "
             "so concurrent slurm jobs writing to the same output dir do not race "
             "on the atomic results.json rename. dispatch_synthetic.py merges them.",
    )
    args = parser.parse_args()
    args.shared_python_env = normalize_shared_python_env(args.shared_python_env)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    _resolve_provider_defaults(args)

    spec = specs.get(args.bench)
    out_dir = Path(args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    # Run-level metadata: pinned once per output dir so SFT data prep can
    # reconstruct exactly what prompt/tools/sampling settings produced
    # each item's messages.json without scraping every row's `extra`.
    run_meta = {
        "harness": HARNESS_NAME,
        "harness_version": HARNESS_VERSION,
        "bench": args.bench,
        "model": args.model_name,
        "provider": args.provider,
        "api_base": args.api_base,
        "system_prompt": SYSTEM_PROMPT,
        "tools_spec": TOOLS,
        "step_limit": int(args.step_limit),
        "max_tool_reminders": int(args.max_tool_reminders or 0),
        "request_timeout_s": int(args.request_timeout),
        "block_timeout_s": int(args.block_timeout),
        "shell_max_output_bytes": int(DEFAULT_MAX_OUTPUT_BYTES),
        "enable_thinking": bool(args.enable_thinking),
        "enable_summarize": bool(args.enable_summarize),
        "reasoning_effort": args.reasoning_effort,
        "thinking_budget_tokens": args.thinking_budget_tokens,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "seed": args.seed,
    }
    (out_dir / "run_meta.json").write_text(
        json.dumps(run_meta, ensure_ascii=False, indent=2), encoding="utf-8",
    )

    items_all = spec.iterate_items(Path(args.input_file).resolve())
    items = spec.filter_items(items_all, args)
    items = _filter_by_source_indices(items, args.source_indices)
    dataset_root = Path(args.dataset_root).resolve()
    base_extra_env = _build_extra_env()

    LOGGER.info(
        "bench=%s loaded=%d after_filter=%d max_items=%s input=%s",
        args.bench, len(items_all), len(items), args.max_items, args.input_file,
    )
    LOGGER.info(
        "kira.driver step_limit=%d request_timeout=%ds api_base=%s concurrent_limit=%d",
        args.step_limit, args.request_timeout, args.api_base, args.concurrent_limit,
    )

    semaphore = asyncio.Semaphore(max(1, args.concurrent_limit))
    gpu_slot_queue: Optional[asyncio.Queue] = None
    pool = parse_gpu_device_pool(args.gpu_device_pool)
    if pool:
        gpu_slot_queue = asyncio.Queue()
        for device in pool:
            gpu_slot_queue.put_nowait(device)
        LOGGER.info("kira.driver gpu_device_pool=%s", pool)

    # Auto-resume: read any existing results file in out_dir and skip
    # items already scored without error. Errored rows are re-tried.
    # Cross-run reuse: point a new RUN_ROOT at the same OUT to inherit
    # the prior shard's results.
    existing = _load_existing_results(
        out_dir, args.results_filename, min_attempt=args.attempt,
    )
    # Prior-row map (any attempt). Used when an escalation pass writes a
    # fresh attempt-N row to splice the prior attempt into prior_attempts.
    all_prior = _load_prior_rows(out_dir, args.results_filename)
    rows_by_si: dict[Any, dict[str, Any]] = {}
    pending: list[tuple[int, dict[str, Any]]] = []
    for idx, item in enumerate(items):
        si = item.get("__source_index__", idx)
        prev = existing.get(si)
        if prev is not None and not prev.get("error"):
            rows_by_si[si] = prev
        else:
            pending.append((idx, item))
    LOGGER.info(
        "kira.driver resume out=%s skip=%d run=%d total=%d",
        out_dir, len(rows_by_si), len(pending), len(items),
    )
    # Persist what we already have so a hard kill before the first new
    # item finishes still leaves a coherent file.
    _atomic_write_results(out_dir, _sorted_rows(rows_by_si), args.results_filename)

    if not pending:
        LOGGER.info(
            "kira.driver wrote %s/%s (resumed=%d, ran=0)",
            out_dir, args.results_filename, len(rows_by_si),
        )
        return

    write_lock = asyncio.Lock()
    tasks = [
        asyncio.create_task(
            _run_one_async(spec, item, idx, dataset_root, args, base_extra_env, semaphore, gpu_slot_queue)
        )
        for idx, item in pending
    ]
    n_ran = 0
    for fut in asyncio.as_completed(tasks):
        row = await fut
        si = row.get("source_index")
        if si is not None:
            _archive_prior_row(row, all_prior.get(si))
            rows_by_si[si] = row
        n_ran += 1
        async with write_lock:
            _atomic_write_results(out_dir, _sorted_rows(rows_by_si), args.results_filename)
    LOGGER.info(
        "kira.driver wrote %s/%s (n=%d, resumed=%d, ran=%d)",
        out_dir, args.results_filename, len(rows_by_si), len(rows_by_si) - n_ran, n_ran,
    )


def cli_main() -> None:
    """Synchronous console-script wrapper around the async driver."""
    asyncio.run(main())


if __name__ == "__main__":
    cli_main()
