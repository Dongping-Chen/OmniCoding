"""Provider auto-detection for the KIRA harness.

KIRA can talk to several backends through litellm, each with its own
quirks:

  - **qwen** (sglang local) — needs ``extra_body.chat_template_kwargs.
    enable_thinking=true`` to flip thinking on, and the ``recovery.py``
    XML-salvage layer because Qwen3.6 sometimes emits half-broken
    ``<tool_call>`` blocks. Also frequently "bows out" mid-task, so the
    continue-prompt budget needs to be high (10).
  - **anthropic** — extended-thinking is ``extra_body.thinking={
    "type":"enabled","budget_tokens":N}``. No XML quirks. Bowing-out
    is rare (budget 2 is enough).
  - **openai** — ``reasoning_effort`` for o-series; otherwise plain.
    Bowing-out rare.
  - **openrouter** — pass-through to whatever model_name routes to;
    detect the underlying provider from the path segment after the
    ``openrouter/`` prefix (``openrouter/anthropic/claude-...`` →
    treat as anthropic for the kwargs we send, but the API base stays
    OpenRouter's).
  - **other** — generic OpenAI-compatible endpoint; send no provider-
    specific extras. drop_params=True covers anything we accidentally
    send.

The detection is by ``model_name`` substring — same heuristic litellm
itself uses. We expose:

  - ``detect_provider(model_name)`` → str
  - ``provider_kwargs(provider, ...)`` → dict (extra_body, etc.)
  - ``default_max_tool_reminders(provider)`` → int

Non-goals: validating that the model actually exists, or routing API
calls. Those stay in ``kira.llm``.
"""

from __future__ import annotations

import logging
from typing import Any

LOGGER = logging.getLogger("kira.provider")

# Substring → canonical provider. Ordering matters: ``openrouter/`` is
# checked before ``openai/`` because litellm uses ``openrouter/<vendor>/
# <model>`` and we want OpenRouter to win the routing decision.
_PROVIDER_HINTS: list[tuple[str, str]] = [
    ("openrouter/", "openrouter"),
    ("anthropic/", "anthropic"),
    ("claude", "anthropic"),
    ("qwen", "qwen"),
    ("code-x-sft", "qwen"),
    ("openai/", "openai"),
    ("gpt-", "openai"),
    ("o1-", "openai"),
    ("o3-", "openai"),
    ("o4-", "openai"),
]

PROVIDER_NAMES = frozenset({"qwen", "openai", "anthropic", "openrouter", "other"})


def detect_provider(model_name: str) -> str:
    """Return one of ``qwen`` / ``openai`` / ``anthropic`` / ``openrouter``
    / ``other``. Case-insensitive, substring-based."""
    if not model_name:
        return "other"
    n = model_name.lower()
    for hint, provider in _PROVIDER_HINTS:
        if hint in n:
            LOGGER.debug("kira.provider %s → %s (hint=%s)", model_name, provider, hint)
            return provider
    LOGGER.debug("kira.provider %s → other (no hint matched)", model_name)
    return "other"


def detect_routed_provider(model_name: str) -> str:
    """For ``openrouter/<vendor>/<model>``, return the *underlying* vendor.
    Otherwise same as ``detect_provider``. Used to decide which extra_body
    flavor to attach when the call goes through OpenRouter."""
    if not model_name:
        return "other"
    n = model_name.lower()
    if n.startswith("openrouter/"):
        rest = n[len("openrouter/"):]
        for hint, provider in _PROVIDER_HINTS:
            if hint == "openrouter/":
                continue
            if hint in rest:
                return provider
        return "other"
    return detect_provider(model_name)


def resolve_provider(model_name: str, provider: str | None = None) -> str:
    """Resolve an optional semantic-provider override.

    LiteLLM routing prefixes and model behavior are separate concerns for
    OpenAI-compatible local servers. For example,
    ``openai/shuaishuaicdp/Code-X-SFT-27B`` must keep its ``openai/`` routing
    prefix while using Qwen chat-template and multimodal-message behavior.
    """
    if provider in (None, "auto"):
        return detect_provider(model_name)
    normalized = provider.lower()
    if normalized not in PROVIDER_NAMES:
        allowed = ", ".join(sorted(PROVIDER_NAMES))
        raise ValueError(f"provider must be one of: {allowed}")
    return normalized


def resolve_routed_provider(model_name: str, provider: str | None = None) -> str:
    """Resolve behavior for a direct or OpenRouter-routed model."""
    selected = resolve_provider(model_name, provider)
    if selected == "openrouter":
        return detect_routed_provider(model_name)
    return selected


def provider_kwargs(
    *,
    model_name: str,
    provider: str | None = None,
    enable_thinking: bool = True,
    reasoning_effort: str | None = None,
    thinking_budget_tokens: int | None = None,
) -> dict[str, Any]:
    """Build the per-provider kwargs dict (extra_body / reasoning_effort)
    to splice into the ``litellm.completion`` call.

    Logic:
      - Qwen: ``extra_body.chat_template_kwargs.enable_thinking`` based
        on the flag.
      - Anthropic (direct or via OpenRouter): ``extra_body.thinking={
        "type":"enabled","budget_tokens":N}`` if ``thinking_budget_tokens``
        > 0; otherwise no extras (default Claude behavior).
      - OpenAI (direct or via OpenRouter): ``reasoning_effort`` if set
        (only meaningful for o-series; drop_params filters for others).
      - Anything else: nothing.

    Returns the dict to merge into completion_kwargs (caller does the
    merge to make logging easy — never mutates).
    """
    routed = resolve_routed_provider(model_name, provider)
    out: dict[str, Any] = {}

    if routed == "qwen":
        out["extra_body"] = {"chat_template_kwargs": {"enable_thinking": enable_thinking}}
    elif routed == "anthropic" and thinking_budget_tokens and thinking_budget_tokens > 0:
        out["extra_body"] = {
            "thinking": {"type": "enabled", "budget_tokens": int(thinking_budget_tokens)}
        }
    elif routed == "openai" and reasoning_effort:
        # Send reasoning_effort via ``extra_body`` rather than top-level
        # so litellm's ``responses_api_bridge_check`` does NOT trigger
        # the Responses-API bridge for gpt-5.x + tools combos. The
        # bridge round-trips through /responses (non-streaming), where
        # the chatgpt.com codex backend returns ``reasoning.summary=[]``
        # → reasoning_content is lost. Going through plain
        # /chat/completions lets the codex-router collect
        # ``reasoning_text.delta`` events from the upstream stream and
        # surface them as ``message.reasoning_content`` on the response.
        # Verified: top-level reasoning_effort → 0 reasoning_content
        # captured across 6 GPT-5.5 trajectories; extra_body path →
        # full CoT recovered.
        out["extra_body"] = {"reasoning_effort": reasoning_effort}

    LOGGER.debug("kira.provider_kwargs model=%s routed=%s out=%s", model_name, routed, out)
    return out


def default_max_tool_reminders(provider: str) -> int:
    """Per-provider default for the no-tool-call reminder budget. Qwen3.6
    routinely "bows out" mid-task — needs a generous budget. Frontier
    Anthropic / OpenAI models almost never need a reminder."""
    if provider == "qwen":
        return 10
    if provider in ("anthropic", "openai"):
        return 2
    return 4  # OpenRouter / generic — split the difference


def default_api_base(provider: str) -> str | None:
    """Default api_base for a provider when caller didn't pass one. Qwen
    has no real default (depends on which sglang host is up); OpenRouter
    has a single canonical base; OpenAI/Anthropic let the SDK pick."""
    if provider == "openrouter":
        return "https://openrouter.ai/api/v1"
    return None
