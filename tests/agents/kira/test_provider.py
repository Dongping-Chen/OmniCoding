"""Pin the provider auto-detect heuristics + kwargs.

Each test pins one model_name shape we expect to encounter so that if a
new provider surfaces we add a test, not silently mis-classify it."""

from __future__ import annotations

import pytest

from omnicoding.agents.kira.provider import (
    default_api_base,
    default_max_tool_reminders,
    detect_provider,
    detect_routed_provider,
    provider_kwargs,
    resolve_provider,
)


# ---------- detect_provider -----------------------------------------

@pytest.mark.parametrize("model_name,expected", [
    ("openai/Qwen3.6-27B", "qwen"),  # local sglang served-name; "qwen" wins
    ("Qwen3.6-27B", "qwen"),
    ("qwen-2-5-72b-instruct", "qwen"),
    ("shuaishuaicdp/Code-X-SFT-27B", "qwen"),
    ("openai/shuaishuaicdp/Code-X-SFT-27B", "qwen"),
    ("openrouter/anthropic/claude-sonnet-4-5", "openrouter"),
    ("openrouter/openai/gpt-5", "openrouter"),
    ("openrouter/qwen/qwen3-coder", "openrouter"),
    ("anthropic/claude-opus-4-7", "anthropic"),
    ("claude-haiku-4-5", "anthropic"),
    ("openai/gpt-5", "openai"),
    ("gpt-5-mini", "openai"),
    ("o1-preview", "openai"),
    ("o3-mini", "openai"),
    ("o4-mini", "openai"),
    ("mistral-large", "other"),
    ("", "other"),
])
def test_detect_provider(model_name: str, expected: str):
    assert detect_provider(model_name) == expected


# ---------- detect_routed_provider ----------------------------------

@pytest.mark.parametrize("model_name,expected", [
    # OpenRouter routing exposes the underlying vendor for kwargs flavor.
    ("openrouter/anthropic/claude-sonnet-4-5", "anthropic"),
    ("openrouter/openai/gpt-5", "openai"),
    ("openrouter/qwen/qwen3-coder", "qwen"),
    ("openrouter/mistral/mistral-large", "other"),
    # Non-OpenRouter falls back to the same heuristic as detect_provider.
    ("openai/Qwen3.6-27B", "qwen"),
    ("anthropic/claude-opus-4-7", "anthropic"),
])
def test_detect_routed_provider(model_name: str, expected: str):
    assert detect_routed_provider(model_name) == expected


# ---------- provider_kwargs -----------------------------------------

def test_provider_kwargs_qwen_thinking_on():
    out = provider_kwargs(model_name="openai/Qwen3.6-27B", enable_thinking=True)
    assert out == {
        "extra_body": {"chat_template_kwargs": {"enable_thinking": True}}
    }


def test_provider_kwargs_qwen_thinking_off():
    out = provider_kwargs(model_name="Qwen3.6-27B", enable_thinking=False)
    assert out == {
        "extra_body": {"chat_template_kwargs": {"enable_thinking": False}}
    }


def test_explicit_qwen_provider_preserves_litellm_routing_name():
    model_name = "openai/shuaishuaicdp/Code-X-SFT-27B"
    assert resolve_provider(model_name, "qwen") == "qwen"
    assert provider_kwargs(
        model_name=model_name,
        provider="qwen",
        enable_thinking=True,
    ) == {"extra_body": {"chat_template_kwargs": {"enable_thinking": True}}}


def test_provider_kwargs_anthropic_no_thinking_budget():
    """Default Anthropic call sends nothing extra — budget=None means
    Claude's default behavior (no extended thinking)."""
    out = provider_kwargs(model_name="claude-opus-4-7")
    assert out == {}


def test_provider_kwargs_anthropic_with_thinking_budget():
    out = provider_kwargs(model_name="claude-opus-4-7", thinking_budget_tokens=8000)
    assert out == {"extra_body": {"thinking": {"type": "enabled", "budget_tokens": 8000}}}


def test_provider_kwargs_openrouter_anthropic_uses_anthropic_thinking():
    """Routing through OpenRouter shouldn't strip the underlying vendor's
    thinking flavor — Anthropic via OpenRouter still wants
    ``extra_body.thinking``."""
    out = provider_kwargs(
        model_name="openrouter/anthropic/claude-sonnet-4-5",
        thinking_budget_tokens=4096,
    )
    assert out == {"extra_body": {"thinking": {"type": "enabled", "budget_tokens": 4096}}}


def test_provider_kwargs_openai_o_series_with_reasoning_effort():
    """Round-16: ``reasoning_effort`` is sent via ``extra_body``, not as a
    top-level kwarg, to avoid litellm's ``responses_api_bridge_check``
    routing gpt-5.x + tools through ``/responses`` (which loses CoT)."""
    out = provider_kwargs(model_name="o3-mini", reasoning_effort="high")
    assert out == {"extra_body": {"reasoning_effort": "high"}}


def test_provider_kwargs_openai_no_reasoning_effort():
    out = provider_kwargs(model_name="gpt-5")
    assert out == {}


def test_provider_kwargs_other_provider_returns_empty():
    out = provider_kwargs(model_name="mistral-large", enable_thinking=True)
    assert out == {}


# ---------- default_max_tool_reminders ------------------------------

@pytest.mark.parametrize("provider,expected", [
    ("qwen", 10),
    ("anthropic", 2),
    ("openai", 2),
    ("openrouter", 4),
    ("other", 4),
])
def test_default_max_tool_reminders(provider: str, expected: int):
    assert default_max_tool_reminders(provider) == expected


# ---------- default_api_base ----------------------------------------

def test_default_api_base_openrouter():
    assert default_api_base("openrouter") == "https://openrouter.ai/api/v1"


def test_default_api_base_qwen_returns_none():
    assert default_api_base("qwen") is None


def test_default_api_base_other_returns_none():
    assert default_api_base("openai") is None
    assert default_api_base("anthropic") is None
    assert default_api_base("other") is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
