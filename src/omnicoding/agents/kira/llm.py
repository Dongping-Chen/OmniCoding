"""LLM call wrapper for KIRA: retry, block-timeout, error translation.

The agent loop should never see raw litellm errors — it gets a
``LLMResponse`` (a small dataclass mirroring the OpenAI shape we care
about) or one of three local exceptions:

  - ``ContextLengthExceededError`` — request too big; loop should
    summarize and retry. Raised from ``litellm``'s
    ``ContextWindowExceededError`` plus any provider-specific
    "context length" message string match.
  - ``OutputLengthExceededError`` — finish_reason == "length"; loop
    should re-prompt for a shorter answer. Carries the truncated
    content so the loop can show the user what we got.
  - ``BlockTimeoutError`` — call hung past the harness-side wall clock.
    Distinct from litellm's ``Timeout`` (which fires from the HTTP
    client side); this fires when the HTTP client itself locks up
    (rare but real on flaky proxy / sglang restarts).

Retry policy: tenacity, 5 exponential attempts (0.5 → 4 s), skip on
``AuthenticationError`` / ``BadRequestError`` / our local
``OutputLengthExceeded`` / ``ContextLengthExceeded``. Anything else —
rate-limit, transient 5xx, network blip — gets retried.

The block-timeout uses a ``ThreadPoolExecutor`` because the agent loop
is sync. We submit the litellm call to a worker thread and call
``future.result(timeout=block_timeout_s)``. If that fires, the worker
thread keeps running in the background until its own HTTP timeout
(``request_timeout_s``) trips it — we just abandon the future. This is
fine: the abandoned thread costs an idle socket for ``request_timeout_s -
block_timeout_s`` seconds at most. We don't try to cancel it because
litellm is not cancel-safe.
"""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FuturesTimeoutError
from dataclasses import dataclass
from typing import Any

import litellm
from litellm.exceptions import (
    AuthenticationError as LiteLLMAuthenticationError,
    BadRequestError,
    ContextWindowExceededError as LiteLLMContextWindowExceededError,
)
from tenacity import (
    retry,
    retry_if_exception_type,
    retry_if_not_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from omnicoding.agents.kira.provider import provider_kwargs, resolve_routed_provider

# OpenAI reasoning models (gpt-5.x line + the o-series) reject
# ``temperature`` / ``top_p`` outright on some backends (ChatGPT
# codex-router responds with HTTP 400 ``Unsupported parameter:
# temperature``; the OpenAI API quietly enforces 1.0). Treat these as
# "sampling-fixed" and skip forwarding the params even when the caller
# passed an explicit value — the trajectory's ``run_meta.json`` already
# records what was requested for SFT-RL audit.
_OPENAI_REASONING_HINTS = ("gpt-5", "o1-", "o3-", "o4-")


def _sampling_params_supported(model_name: str, provider: str | None = None) -> bool:
    routed = resolve_routed_provider(model_name, provider)
    if routed != "openai":
        return True
    n = model_name.lower()
    return not any(hint in n for hint in _OPENAI_REASONING_HINTS)

LOGGER = logging.getLogger("kira.llm")

DEFAULT_BLOCK_TIMEOUT_S = 600
DEFAULT_REQUEST_TIMEOUT_S = 900


class ContextLengthExceededError(RuntimeError):
    """Request payload exceeds the model's context window."""


class OutputLengthExceededError(RuntimeError):
    """Model returned finish_reason='length' (truncated mid-stream)."""

    def __init__(self, message: str, *, truncated_content: str = ""):
        super().__init__(message)
        self.truncated_content = truncated_content


class BlockTimeoutError(RuntimeError):
    """Harness-side wall-clock timer fired before litellm returned."""


@dataclass
class LLMResponse:
    content: str
    reasoning_content: str
    tool_calls: list[dict[str, Any]]
    finish_reason: str
    prompt_tokens: int
    completion_tokens: int
    # Subtotals from ``usage.prompt_tokens_details`` /
    # ``usage.completion_tokens_details``. Provider-dependent fields, so
    # they default to 0 when the response doesn't carry them. Tracked
    # separately so the driver can compute effective billing (cached
    # tokens charged at ~50% on OpenAI/OpenRouter) and surface the
    # internal-reasoning vs visible-completion split for debug.
    cached_tokens: int = 0
    reasoning_tokens: int = 0
    # Set True by ``KiraAgent._recover_if_needed`` after it pulls
    # ``tool_calls`` out of malformed ``<tool_call>`` XML in ``content``
    # (Qwen3.6 + sglang qwen3_coder failure mode). Surfaced on the saved
    # assistant message so SFT data prep can tell that the structured
    # ``tool_calls`` was synthesized from raw XML, and the original wire
    # output is what's still in ``content``.
    recovered_from_content: bool = False


# Exception classes that should NOT be retried — auth errors won't
# magically heal, bad requests won't heal, and our local exceptions are
# already terminal.
_DO_NOT_RETRY = (
    BadRequestError,
    LiteLLMAuthenticationError,
    ContextLengthExceededError,
    OutputLengthExceededError,
)


def _coerce_tool_calls_to_dicts(tool_calls: Any) -> list[dict[str, Any]]:
    """litellm returns Pydantic objects, sometimes plain dicts. Coerce
    to dicts so the rest of the pipeline only deals with one shape."""
    if not tool_calls:
        return []
    out: list[dict[str, Any]] = []
    for tc in tool_calls:
        if isinstance(tc, dict):
            fn = tc.get("function") or {}
            out.append({
                "id": tc.get("id"),
                "type": tc.get("type") or "function",
                "function": {
                    "name": fn.get("name") or "",
                    "arguments": fn.get("arguments"),
                },
            })
            continue
        fn = getattr(tc, "function", None)
        out.append({
            "id": getattr(tc, "id", None),
            "type": getattr(tc, "type", None) or "function",
            "function": {
                "name": getattr(fn, "name", "") or "",
                "arguments": getattr(fn, "arguments", None),
            },
        })
    return out


def _extract_token_subdetails(usage: Any) -> tuple[int, int]:
    """Return ``(cached_tokens, reasoning_tokens)`` from a litellm usage
    object, falling back to 0 when the field isn't present (provider
    didn't include it). litellm normalizes both Pydantic objects and
    plain dicts here, so we tolerate either."""
    if usage is None:
        return 0, 0

    def _lookup(obj: Any, key: str) -> Any:
        if isinstance(obj, dict):
            return obj.get(key)
        return getattr(obj, key, None)

    prompt_details = _lookup(usage, "prompt_tokens_details")
    cached = 0
    if prompt_details is not None:
        cached = int(_lookup(prompt_details, "cached_tokens") or 0)
    completion_details = _lookup(usage, "completion_tokens_details")
    reasoning = 0
    if completion_details is not None:
        reasoning = int(_lookup(completion_details, "reasoning_tokens") or 0)
    return cached, reasoning


def _parse_litellm_response(response: Any) -> LLMResponse:
    """Pull out content / tool_calls / finish_reason / token counts. The
    finish_reason check happens AFTER assembling the LLMResponse so the
    loop sees the truncated content even when finish_reason='length'."""
    choice = response.choices[0]
    message = choice.message
    usage = getattr(response, "usage", None)
    prompt_tokens = int(getattr(usage, "prompt_tokens", 0) or 0)
    completion_tokens = int(getattr(usage, "completion_tokens", 0) or 0)
    cached_tokens, reasoning_tokens = _extract_token_subdetails(usage)
    return LLMResponse(
        content=getattr(message, "content", "") or "",
        reasoning_content=getattr(message, "reasoning_content", "") or "",
        tool_calls=_coerce_tool_calls_to_dicts(getattr(message, "tool_calls", None)),
        finish_reason=getattr(choice, "finish_reason", "") or "",
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        cached_tokens=cached_tokens,
        reasoning_tokens=reasoning_tokens,
    )


def _raw_completion(**kwargs: Any) -> Any:
    """Single litellm.completion call. Translates ContextWindowExceeded
    into our local exception so the retry decorator skips it cleanly."""
    try:
        return litellm.completion(**kwargs)
    except LiteLLMContextWindowExceededError as exc:
        raise ContextLengthExceededError(str(exc)) from exc


@retry(
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=0.5, min=0.5, max=4),
    retry=(
        retry_if_exception_type(Exception)
        & retry_if_not_exception_type(_DO_NOT_RETRY)
    ),
    reraise=True,
)
def _completion_with_retry(**kwargs: Any) -> Any:
    return _raw_completion(**kwargs)


def _with_block_timeout(fn, timeout_s: float, **kwargs: Any) -> Any:
    """Run ``fn(**kwargs)`` in a worker thread and time out at
    ``timeout_s``. Worker thread is abandoned on timeout — see module
    docstring."""
    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(fn, **kwargs)
        try:
            return future.result(timeout=timeout_s)
        except FuturesTimeoutError as exc:
            LOGGER.warning("kira.llm block-timeout fired after %.0fs", timeout_s)
            raise BlockTimeoutError(f"LLM call blocked for {timeout_s}s") from exc


def _build_completion_kwargs(
    *,
    messages: list[dict[str, Any]],
    model_name: str,
    provider: str | None,
    api_base: str | None,
    api_key: str | None,
    tools: list[dict[str, Any]] | None,
    request_timeout_s: int,
    enable_thinking: bool,
    reasoning_effort: str | None,
    thinking_budget_tokens: int | None,
    temperature: float | None = None,
    top_p: float | None = None,
    max_tokens: int | None = None,
    seed: int | None = None,
) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "model": model_name,
        "messages": messages,
        "timeout": request_timeout_s,
        "drop_params": True,
    }
    if tools:
        kwargs["tools"] = tools
    if api_base:
        kwargs["api_base"] = api_base
    if api_key:
        kwargs["api_key"] = api_key
    # Sampling params are forwarded only when explicitly set by the caller
    # so the default behaviour (provider-default sampling) is unchanged
    # for runs that don't opt in. ``drop_params=True`` above filters
    # anything litellm itself recognises as unsupported, but the ChatGPT
    # codex-router backend rejects ``temperature``/``top_p`` for the
    # whole gpt-5.x + o-series line with an HTTP 400 that bypasses
    # litellm's drop list. ``_sampling_params_supported`` short-circuits
    # those models so kira keeps running instead of dying mid-trajectory.
    sampling_ok = _sampling_params_supported(model_name, provider)
    if temperature is not None:
        if sampling_ok:
            kwargs["temperature"] = temperature
        else:
            LOGGER.info(
                "kira.llm temperature=%s ignored for %s (reasoning-only model)",
                temperature, model_name,
            )
    if top_p is not None:
        if sampling_ok:
            kwargs["top_p"] = top_p
        else:
            LOGGER.info(
                "kira.llm top_p=%s ignored for %s (reasoning-only model)",
                top_p, model_name,
            )
    if max_tokens is not None:
        kwargs["max_tokens"] = max_tokens
    if seed is not None:
        kwargs["seed"] = seed
    extras = provider_kwargs(
        model_name=model_name,
        provider=provider,
        enable_thinking=enable_thinking,
        reasoning_effort=reasoning_effort,
        thinking_budget_tokens=thinking_budget_tokens,
    )
    kwargs.update(extras)
    return kwargs


def call_llm_with_tools(
    *,
    messages: list[dict[str, Any]],
    model_name: str,
    provider: str | None = None,
    api_base: str | None,
    api_key: str | None,
    tools: list[dict[str, Any]],
    request_timeout_s: int = DEFAULT_REQUEST_TIMEOUT_S,
    block_timeout_s: int = DEFAULT_BLOCK_TIMEOUT_S,
    enable_thinking: bool = True,
    reasoning_effort: str | None = None,
    thinking_budget_tokens: int | None = None,
    temperature: float | None = None,
    top_p: float | None = None,
    max_tokens: int | None = None,
    seed: int | None = None,
) -> LLMResponse:
    """Main agent-loop entrypoint. Calls litellm with retry + block timeout
    + error translation. Returns ``LLMResponse`` on success.

    Raises:
        ContextLengthExceededError: input too big.
        OutputLengthExceededError: finish_reason == 'length'.
        BlockTimeoutError: harness-side wall clock fired.
    """
    completion_kwargs = _build_completion_kwargs(
        messages=messages,
        model_name=model_name,
        provider=provider,
        api_base=api_base,
        api_key=api_key,
        tools=tools,
        request_timeout_s=request_timeout_s,
        enable_thinking=enable_thinking,
        reasoning_effort=reasoning_effort,
        thinking_budget_tokens=thinking_budget_tokens,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        seed=seed,
    )

    LOGGER.info(
        "kira.llm call model=%s msgs=%d tools=%d block_timeout=%ds",
        model_name, len(messages), len(tools or []), block_timeout_s,
    )
    response = _with_block_timeout(
        _completion_with_retry, timeout_s=block_timeout_s, **completion_kwargs,
    )
    parsed = _parse_litellm_response(response)
    if parsed.finish_reason == "length":
        LOGGER.warning("kira.llm finish_reason=length (truncated)")
        raise OutputLengthExceededError(
            "Response was truncated (finish_reason=length)",
            truncated_content=parsed.content,
        )
    return parsed


def call_llm_for_image(
    *,
    messages: list[dict[str, Any]],
    model_name: str,
    provider: str | None = None,
    api_base: str | None,
    api_key: str | None,
    request_timeout_s: int = DEFAULT_REQUEST_TIMEOUT_S,
    block_timeout_s: int = DEFAULT_BLOCK_TIMEOUT_S,
) -> str:
    """Multimodal image_read call. No tools, no thinking — we want a
    plain text description back. Same retry / block-timeout policy."""
    completion_kwargs = _build_completion_kwargs(
        messages=messages,
        model_name=model_name,
        provider=provider,
        api_base=api_base,
        api_key=api_key,
        tools=None,
        request_timeout_s=request_timeout_s,
        enable_thinking=False,
        reasoning_effort=None,
        thinking_budget_tokens=None,
    )
    LOGGER.info("kira.llm image_read call model=%s msgs=%d", model_name, len(messages))
    response = _with_block_timeout(
        _completion_with_retry, timeout_s=block_timeout_s, **completion_kwargs,
    )
    parsed = _parse_litellm_response(response)
    return parsed.content
