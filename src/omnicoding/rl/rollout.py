"""Custom Relax rollout function.

Wired in via ``--custom-generate-function-path
omnicoding.rl.rollout.generate``. Replaces the stock per-sample generator for
our multimodal+terminal-tool-use RL setup:

- Modal-side rollout fn POSTs ``(task_id, sampling_params)`` to the local
  cloudflared-tunneled coordinator.
- Coordinator runs one ``KiraAgent`` trajectory with kira's ``api_base`` set to
  the Modal SGLang endpoint (the same model being trained).
- Coordinator grades the trajectory locally and ships back ``messages``,
  ``final_text``, ``reward``, ``exit_reason``.
- This function tokenizes the returned messages, builds a token-level
  ``loss_mask`` (1 for assistant turns, 0 for system/user/tool observations),
  and populates ``Sample`` for Relax to consume downstream.

Per-trajectory wall-clock cap = ``request_timeout_s × (max_turns + 4)``,
mirrored on both sides; coordinator returns a ``"timeout"`` trajectory rather
than blocking.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import threading
import time
from typing import TYPE_CHECKING, Any

import httpx

from omnicoding.rl.tokenize import tokenize_trajectory, tokenize_trajectory_multimodal
from omnicoding.rl.secrets import load_coordinator_token

if TYPE_CHECKING:
    from relax.utils.types import Sample

LOG = logging.getLogger("relax_router.rollout")

# ─── runtime knobs (env-driven so the launcher controls them) ────────────────

# Optional dynamic URL discovery: every rollout call can re-read the
# coordinator URL from a configured Modal Dict instead of baking a tunnel URL
# into a container. A static environment URL remains the portable fallback.
#
# The Modal Dict round-trip is ~ms; a small TTL cache keeps us off the hot path.
_URL_CACHE: dict[str, Any] = {"url": None, "ts": 0.0}
_URL_CACHE_LOCK = threading.Lock()
_URL_CACHE_TTL_S = float(os.environ.get("ROLLOUT_URL_CACHE_TTL_S", "30"))
_STATE_DICT_NAME = os.environ.get("RELAX_ROUTER_STATE_DICT", "").strip()


def _modal_dict_url() -> str | None:
    """Best-effort fetch of the current coordinator URL from Modal Dict. Returns
    None on any error (Modal not installed, dict not found, key missing) so the
    caller can fall back to env."""
    if not _STATE_DICT_NAME:
        return None
    try:
        import modal  # noqa: PLC0415

        d = modal.Dict.from_name(_STATE_DICT_NAME, create_if_missing=False)
        v = d.get("coordinator_url")
        if v and isinstance(v, str):
            return v.strip().rstrip("/") or None
    except Exception as exc:  # noqa: BLE001
        LOG.debug("modal Dict url lookup failed: %s", exc)
    return None


def _coordinator_base() -> str:
    """Resolve the coordinator BASE URL (no path) with dynamic refresh via
    Modal Dict.

    Resolution order:
    1. In-process cache (TTL = ``ROLLOUT_URL_CACHE_TTL_S``, default 30 s).
    2. Modal Dict ``coordinator_url`` key (live; survives tunnel restarts).
    3. Env var ``ROLLOUT_COORDINATOR_PUBLIC_URL`` (set at @modal.enter from
       Modal Dict — covers the bootstrap and the "modal not available" cases).
    """
    now = time.time()
    with _URL_CACHE_LOCK:
        if _URL_CACHE["url"] and (now - _URL_CACHE["ts"]) < _URL_CACHE_TTL_S:
            return _URL_CACHE["url"]

    url = _modal_dict_url() or os.environ.get("ROLLOUT_COORDINATOR_PUBLIC_URL", "").strip()
    if not url:
        raise RuntimeError(
            "coordinator URL not found. Set ROLLOUT_COORDINATOR_PUBLIC_URL or "
            "configure RELAX_ROUTER_STATE_DICT with a 'coordinator_url' key."
        )
    url = url.rstrip("/")
    with _URL_CACHE_LOCK:
        _URL_CACHE["url"] = url
        _URL_CACHE["ts"] = now
    return url


def _sglang_base_url(args: Any) -> str:
    """Where kira hits inference. Prefer the env override (Modal-public web URL
    bound to the Relax SGLang proxy), falling back to the in-cluster Ray Serve
    address (works when coordinator and Relax are on the same network — not
    our setup, but useful for local-only smoke runs)."""
    override = os.environ.get("ROLLOUT_SGLANG_PUBLIC_URL", "").strip()
    if override:
        return override.rstrip("/") + ("" if override.rstrip("/").endswith("/v1") else "/v1")
    ip = getattr(args, "sglang_router_ip", None)
    port = getattr(args, "sglang_router_port", None)
    if not (ip and port):
        raise RuntimeError(
            "ROLLOUT_SGLANG_PUBLIC_URL not set and args.sglang_router_{ip,port} unavailable."
        )
    return f"http://{ip}:{port}/v1"


def _sglang_model_name(args: Any) -> str:
    """The 'model' field SGLang/Modal exposes. litellm needs a provider prefix
    (``openai/...`` for OpenAI-compatible endpoints like SGLang). We always tag
    as openai/* so kira's litellm.completion() routes correctly."""
    explicit = os.environ.get("ROLLOUT_SGLANG_MODEL")
    if explicit:
        return explicit if explicit.startswith("openai/") else f"openai/{explicit}"
    base = getattr(args, "hf_checkpoint", "default")
    return base if base.startswith("openai/") else f"openai/{base}"


# ─── tokenization + loss masking ─────────────────────────────────────────────


# ─── coordinator HTTP ────────────────────────────────────────────────────────


def _coerce_metadata(md: Any) -> dict:
    """Relax's StreamingDataset loads a JSON-string ``metadata`` parquet column
    as-is (no auto-decode). Coerce to dict for downstream `{**md}` usage."""
    if md is None:
        return {}
    if isinstance(md, dict):
        return md
    if isinstance(md, str):
        return json.loads(md) if md else {}
    raise TypeError(f"unexpected sample.metadata type {type(md).__name__}")


def _build_payload(sample: "Sample", sampling_params: dict, args: Any) -> dict:
    md = _coerce_metadata(sample.metadata)
    if "task_id" not in md:
        raise ValueError("sample.metadata.task_id missing — did you build the prompt-set parquet?")
    max_turns = int(os.environ.get("KIRA_MAX_TURNS", getattr(args, "max_turns", 30) or 30))
    request_timeout_s = int(os.environ.get("KIRA_REQUEST_TIMEOUT", "900"))
    block_timeout_s = int(os.environ.get("KIRA_BLOCK_TIMEOUT", "1200"))
    # Per-turn assistant-output cap (kira's chat-completions max_tokens). Decoupled
    # from Relax's --rollout-max-response-len, which is the TOTAL (multi-turn-summed)
    # response budget enforced post-hoc in Relax/relax/backends/megatron/data.py:692.
    # If we forwarded sampling_params["max_new_tokens"] (= rollout-max-response-len)
    # straight through, kira would let a single turn emit ~200K tokens — eating the
    # whole context budget on turn 1 and starving the remaining 99 turns.
    configured_max_tokens = os.environ.get("KIRA_MAX_TOKENS_PER_TURN", "").strip()
    max_tokens_per_turn = int(configured_max_tokens) if configured_max_tokens else 8192
    if not 1 <= max_tokens_per_turn <= 32768:
        raise ValueError("KIRA_MAX_TOKENS_PER_TURN must be between 1 and 32768")
    return {
        "task_id": md["task_id"],
        "n_samples": 1,
        "sglang_base_url": _sglang_base_url(args),
        "sglang_model_name": _sglang_model_name(args),
        "sampling_params": {
            "temperature": sampling_params.get("temperature"),
            "top_p": sampling_params.get("top_p"),
            "max_tokens": max_tokens_per_turn,
            "enable_thinking": True,
        },
        "max_turns": max_turns,
        "request_timeout_s": request_timeout_s,
        "block_timeout_s": block_timeout_s,
    }


# Each individual HTTP call is short — well under Cloudflare quick-tunnel's
# 100s origin-timeout. The kira trajectory itself takes minutes; we discover
# its result by polling a `job_id` returned from /rollout/submit.
_HTTP_PER_CALL_TIMEOUT_S = float(os.environ.get("ROLLOUT_HTTP_TIMEOUT_S", "60"))
_POLL_INTERVAL_S = float(os.environ.get("ROLLOUT_POLL_INTERVAL_S", "20"))


def _auth_headers() -> dict[str, str]:
    return {"Authorization": f"Bearer {load_coordinator_token()}"}


async def _submit_job(payload: dict) -> str:
    """POST /rollout/submit → return job_id. Returns in <1s normally."""
    url = _coordinator_base() + "/rollout/submit"
    LOG.info("POST %s task=%s n=%d", url, payload["task_id"], payload.get("n_samples"))
    async with httpx.AsyncClient(
        timeout=httpx.Timeout(_HTTP_PER_CALL_TIMEOUT_S, connect=30.0),
        headers=_auth_headers(),
    ) as client:
        r = await client.post(url, json=payload)
        r.raise_for_status()
        return r.json()["job_id"]


async def _poll_result(job_id: str, deadline_s: float) -> dict:
    """Poll GET /rollout/result/{job_id} until status != pending. Each poll
    is its own short HTTP call (no Cloudflare 524 risk). Returns the
    decoded RolloutResponse (the inner ``response`` field)."""
    url = _coordinator_base() + f"/rollout/result/{job_id}"
    end_at = time.monotonic() + deadline_s
    while True:
        if time.monotonic() >= end_at:
            raise TimeoutError(
                f"job {job_id} did not complete in {deadline_s:.0f}s; "
                "coordinator may still be processing — increase deadline or check coordinator log."
            )
        try:
            async with httpx.AsyncClient(
                timeout=httpx.Timeout(_HTTP_PER_CALL_TIMEOUT_S, connect=15.0),
                headers=_auth_headers(),
            ) as client:
                r = await client.get(url)
        except (httpx.HTTPError, httpx.TimeoutException) as exc:
            # Transient network blip on a poll — wait and retry. The task is
            # still running on the coordinator; we just couldn't reach it
            # this tick.
            LOG.warning("poll %s transient error: %s; retrying", job_id, exc)
            await asyncio.sleep(_POLL_INTERVAL_S)
            continue

        if r.status_code == 202:
            await asyncio.sleep(_POLL_INTERVAL_S)
            continue
        if r.status_code == 200:
            body = r.json()
            status = body.get("status")
            if status == "completed":
                return body["response"]
            if status == "error":
                raise RuntimeError(f"coordinator job error: {body.get('error')}")
            # Defensive: unknown status, treat as pending.
            await asyncio.sleep(_POLL_INTERVAL_S)
            continue
        r.raise_for_status()  # any other code: surface


async def _post_rollout(payload: dict) -> dict:
    """Submit + poll. Total wall clock cap mirrors coordinator's per-trajectory
    deadline (request_timeout_s × (max_turns + 4)) plus a small slack."""
    deadline = payload["request_timeout_s"] * (payload["max_turns"] + 4) + 60
    job_id = await _submit_job(payload)
    LOG.info(
        "submit ok task=%s job=%s; polling every %ss with %ss deadline",
        payload["task_id"], job_id, int(_POLL_INTERVAL_S), int(deadline),
    )
    return await _poll_result(job_id, deadline_s=deadline)


# ─── exit_reason → Sample.Status (lazy-mapped to keep import light) ──────────


def _status_for(exit_reason: str | None):
    """Map kira exit_reason → Relax ``Sample.Status``. Imported lazily so this
    module can be loaded for tokenize-only tests without Relax installed."""
    from relax.utils.types import Sample as _Sample  # noqa: PLC0415

    table = {
        "task_complete": _Sample.Status.COMPLETED,
        "step_limit": _Sample.Status.TRUNCATED,
        "no_tool_calls": _Sample.Status.FAILED,
        "context_overflow": _Sample.Status.TRUNCATED,
        "timeout": _Sample.Status.FAILED,
        "error": _Sample.Status.FAILED,
    }
    return table.get(exit_reason or "", _Sample.Status.COMPLETED)


def _failed_sample(sample: "Sample", reason: str) -> "Sample":
    """Return a Sample whose downstream-relevant tensors are all length-0.

    The Megatron actor's data pipeline expects ``len(loss_mask) == response_length``
    and assembles ``tokens`` / ``loss_masks`` into the rollout batch. Leaving
    ``tokens`` non-empty while ``response_length=0`` would produce mismatched
    shapes; zeroing all four keeps the invariant. ``Status.FAILED`` signals the
    actor to skip this sample for grad updates.
    """
    from relax.utils.types import Sample as _Sample  # noqa: PLC0415

    sample.tokens = []
    sample.loss_mask = []
    sample.response = ""
    sample.response_length = 0
    sample.reward = {
        "score": 0.0,
        "correctness": 0.0,
        "raw_acc": 0.0,
        "format": 0.0,
        "modality_match": 0.0,
        "p_bad_tool": 0.0,
        "removed": 1.0,
    }
    sample.remove_sample = True
    sample.status = _Sample.Status.FAILED
    sample.metadata = {
        **_coerce_metadata(sample.metadata),
        "rollout_router_error": reason,
        "rollout_outcome_reward": 0.0,
        "rollout_raw_outcome_reward": 0.0,
        "rollout_format_reward": 0.0,
        "rollout_reward_components": sample.reward,
    }
    return sample


# ─── public entry point ──────────────────────────────────────────────────────


async def generate(
    args: Any,
    sample: "Sample",
    sampling_params: dict,
    *_positional: Any,
    **_kwargs: Any,
) -> "Sample":
    """Custom Relax rollout function: one Sample → one kira trajectory.

    Relax's ``call_rollout_fn`` (relax/engine/rollout/base_types.py:20) does
    ``fn(*args, **kwargs, evaluation=evaluation)`` AND its caller already
    forwards ``evaluation`` inside ``args`` — so a literal ``evaluation``
    parameter on us collides ("got multiple values for argument 'evaluation'").
    We absorb everything into ``*_positional`` / ``**_kwargs`` and read the
    flag from whichever spot it landed.

    Train and eval go through the same kira coordinator — eval-only knobs
    (lower temperature, fewer turns) live in the Relax launch config, not here.
    """
    evaluation = bool(_kwargs.get("evaluation", _positional[0] if _positional else False))
    if evaluation:
        LOG.info("evaluation rollout for task=%s", _coerce_metadata(sample.metadata).get("task_id"))
    try:
        payload = _build_payload(sample, sampling_params, args)
    except Exception as exc:  # noqa: BLE001
        LOG.error("payload build failed: %s", exc)
        return _failed_sample(sample, f"payload: {exc}")

    try:
        resp = await _post_rollout(payload)
    except Exception as exc:  # noqa: BLE001
        LOG.error("coordinator POST failed task=%s: %s", payload["task_id"], exc)
        return _failed_sample(sample, f"coordinator: {exc}")

    trajectories = resp.get("trajectories") or []
    if not trajectories:
        return _failed_sample(sample, "coordinator returned no trajectories")
    traj = trajectories[0]

    # Lazy-import GenerateState so this module can be unit-tested without Relax.
    from relax.engine.rollout.sglang_rollout import GenerateState

    state = GenerateState(args)
    apply_kwargs = getattr(args, "apply_chat_template_kwargs", None) or {}
    # Prefer the multimodal path whenever a HF processor is available — kira's
    # ``image_read_mode="native"`` (default since 2026-04) injects images as
    # OpenAI ``image_url`` data: URLs in user-message content lists. The
    # text-only ``tokenize_trajectory`` would render those as
    # ``<|vision_start|><|image_pad|><|vision_end|>`` placeholders WITHOUT
    # populating ``Sample.multimodal_train_inputs`` — Megatron actor's vision
    # tower would then forward random embeddings at image_pad positions,
    # silently corrupting training. The multimodal path runs ONE processor
    # call on the full trajectory, getting both correctly expanded image_pad
    # tokens AND the pixel_values + image_grid_thw the trainer needs.
    multimodal_inputs = None
    multimodal_train_inputs = None
    if state.processor is not None:
        tokens, loss_mask, response_length, multimodal_inputs, multimodal_train_inputs = (
            tokenize_trajectory_multimodal(
                traj["messages"], state.tokenizer, state.processor,
                apply_chat_template_kwargs=apply_kwargs,
            )
        )
    else:
        tokens, loss_mask, response_length = tokenize_trajectory(
            traj["messages"], state.tokenizer, apply_chat_template_kwargs=apply_kwargs,
        )

    sample.tokens = tokens
    sample.loss_mask = loss_mask
    sample.response = traj.get("final_text") or ""
    sample.response_length = response_length
    if multimodal_inputs is not None:
        sample.multimodal_inputs = multimodal_inputs
    if multimodal_train_inputs is not None:
        sample.multimodal_train_inputs = multimodal_train_inputs
    reward_details = traj.get("reward_details") or {
        "score": float(traj.get("reward") or 0.0),
        "correctness": float(traj.get("outcome_reward") or 0.0),
        "raw_acc": float(traj.get("raw_outcome_reward") or 0.0),
        "format": float(traj.get("format_reward") or 0.0),
        "modality_match": float(traj.get("modality_match", 1.0) or 0.0),
        "p_bad_tool": float(traj.get("p_bad_tool") or 0.0),
        "removed": float(bool(traj.get("removed"))),
    }
    sample.reward = reward_details
    sample.status = _status_for(traj.get("exit_reason"))
    sample.remove_sample = bool(traj.get("removed"))
    sample.metadata = {
        **_coerce_metadata(sample.metadata),
        "rollout_exit_reason": traj.get("exit_reason"),
        "rollout_n_steps": traj.get("n_steps"),
        "rollout_n_tool_calls": traj.get("n_tool_calls"),
        "rollout_extracted_answer": traj.get("extracted_answer"),
        "rollout_outcome_reward": float(traj.get("outcome_reward") or 0.0),
        "rollout_raw_outcome_reward": float(traj.get("raw_outcome_reward") or 0.0),
        "rollout_format_reward": float(traj.get("format_reward") or 0.0),
        "rollout_modality_reward": float(traj.get("modality_reward") or 0.0),
        "rollout_bad_tool_reward": float(traj.get("bad_tool_reward") or 0.0),
        "rollout_modality_match": float(traj.get("modality_match", 1.0) or 0.0),
        "rollout_p_bad_tool": float(traj.get("p_bad_tool") or 0.0),
        "rollout_n_unparseable": int(traj.get("n_unparseable") or 0),
        "rollout_n_disallowed": int(traj.get("n_disallowed") or 0),
        "rollout_n_escape": int(traj.get("n_escape") or 0),
        "rollout_n_syntax_fail": int(traj.get("n_syntax_fail") or 0),
        "rollout_reward_components": reward_details,
        "rollout_remove_sample": bool(traj.get("removed")),
        "rollout_router_error": traj.get("error"),
    }

    LOG.info(
        "rollout done task=%s reward=%.2f (out=%.1f fmt=%.1f) tokens=%d resp_len=%d steps=%s exit=%s",
        payload["task_id"], reward_details["score"],
        float(traj.get("outcome_reward") or 0.0), float(traj.get("format_reward") or 0.0),
        len(tokens), response_length,
        traj.get("n_steps"), traj.get("exit_reason"),
    )
    return sample
