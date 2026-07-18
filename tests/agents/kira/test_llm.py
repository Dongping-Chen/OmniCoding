"""Mocked-litellm tests for kira/llm.py.

Each test stubs out ``litellm.completion`` and asserts how the wrapper
treats the response: retry on transient errors, raise our local
exceptions on terminal failures, splice provider kwargs correctly.

Live tests (real sglang / OpenRouter) live in ``test_loop_live.py``."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from litellm.exceptions import (
    AuthenticationError as LiteLLMAuthenticationError,
    BadRequestError,
    ContextWindowExceededError as LiteLLMContextWindowExceededError,
)

from omnicoding.agents.kira import llm as kira_llm
from omnicoding.agents.kira.llm import (
    BlockTimeoutError,
    ContextLengthExceededError,
    OutputLengthExceededError,
    call_llm_for_image,
    call_llm_with_tools,
)


def _build_response(*, content="", tool_calls=None, finish_reason="stop", prompt=10, completion=20):
    """Make a MagicMock that quacks like a litellm completion response."""
    msg = MagicMock()
    msg.content = content
    msg.reasoning_content = ""
    msg.tool_calls = tool_calls
    choice = MagicMock(); choice.message = msg; choice.finish_reason = finish_reason
    usage = MagicMock(); usage.prompt_tokens = prompt; usage.completion_tokens = completion
    resp = MagicMock(); resp.choices = [choice]; resp.usage = usage
    return resp


# ---------- happy path ----------------------------------------------

def test_call_llm_with_tools_happy_path_returns_parsed_response():
    tc_obj = MagicMock()
    tc_obj.id = "call_1"
    tc_obj.type = "function"
    tc_obj.function = MagicMock()
    tc_obj.function.name = "execute_commands"
    tc_obj.function.arguments = '{"analysis":"a","plan":"b","commands":[]}'
    resp = _build_response(content="ok", tool_calls=[tc_obj], finish_reason="tool_calls")
    with patch.object(kira_llm.litellm, "completion", return_value=resp) as m:
        out = call_llm_with_tools(
            messages=[{"role": "user", "content": "hi"}],
            model_name="openai/Qwen3.6-27B",
            api_base="http://x:8080/v1",
            api_key="k",
            tools=[{"type": "function", "function": {"name": "execute_commands"}}],
        )
    assert out.content == "ok"
    assert out.finish_reason == "tool_calls"
    assert len(out.tool_calls) == 1
    assert out.tool_calls[0]["function"]["name"] == "execute_commands"
    assert out.prompt_tokens == 10 and out.completion_tokens == 20
    # Provider auto-detect: Qwen → enable_thinking extra_body.
    sent_kwargs = m.call_args.kwargs
    assert sent_kwargs["extra_body"] == {"chat_template_kwargs": {"enable_thinking": True}}


def test_call_llm_with_tools_forwards_explicit_max_tokens():
    resp = _build_response(content="ok")
    with patch.object(kira_llm.litellm, "completion", return_value=resp) as mocked:
        call_llm_with_tools(
            messages=[{"role": "user", "content": "hi"}],
            model_name="openai/Qwen3.6-27B",
            api_base="http://x:8080/v1",
            api_key="k",
            tools=[],
            max_tokens=4096,
        )

    assert mocked.call_args.kwargs["max_tokens"] == 4096


def test_code_x_uses_openai_route_with_qwen_behavior():
    resp = _build_response(content="ok")
    with patch.object(kira_llm.litellm, "completion", return_value=resp) as mocked:
        call_llm_with_tools(
            messages=[{"role": "user", "content": "hi"}],
            model_name="openai/shuaishuaicdp/Code-X-SFT-27B",
            provider="qwen",
            api_base="http://127.0.0.1:8080/v1",
            api_key="local",
            tools=[],
        )

    sent = mocked.call_args.kwargs
    assert sent["model"] == "openai/shuaishuaicdp/Code-X-SFT-27B"
    assert sent["extra_body"] == {
        "chat_template_kwargs": {"enable_thinking": True}
    }


def test_call_llm_with_tools_anthropic_skips_qwen_extra_body():
    resp = _build_response(content="ok")
    with patch.object(kira_llm.litellm, "completion", return_value=resp) as m:
        call_llm_with_tools(
            messages=[{"role": "user", "content": "hi"}],
            model_name="claude-opus-4-7",
            api_base=None, api_key=None, tools=[],
        )
    sent_kwargs = m.call_args.kwargs
    # No qwen-shape extra_body; no thinking_budget so no anthropic extra either.
    assert "extra_body" not in sent_kwargs


def test_call_llm_with_tools_openrouter_anthropic_uses_anthropic_thinking():
    resp = _build_response(content="ok")
    with patch.object(kira_llm.litellm, "completion", return_value=resp) as m:
        call_llm_with_tools(
            messages=[{"role": "user", "content": "hi"}],
            model_name="openrouter/anthropic/claude-sonnet-4-5",
            api_base="https://openrouter.ai/api/v1", api_key="sk-or",
            tools=[],
            thinking_budget_tokens=4096,
        )
    sent_kwargs = m.call_args.kwargs
    assert sent_kwargs["extra_body"] == {
        "thinking": {"type": "enabled", "budget_tokens": 4096}
    }


# ---------- error translation ---------------------------------------

def test_context_window_translates_to_local_context_exception():
    """litellm's ContextWindowExceededError must surface as our local
    type (so the retry decorator skips it AND the loop's summarize-and-
    retry path can catch one specific exception type)."""
    with patch.object(kira_llm.litellm, "completion",
                      side_effect=LiteLLMContextWindowExceededError(
                          message="too big", model="x", llm_provider="x")):
        with pytest.raises(ContextLengthExceededError):
            call_llm_with_tools(
                messages=[], model_name="openai/Qwen3.6-27B",
                api_base=None, api_key=None, tools=[],
                block_timeout_s=10,
            )


def test_finish_reason_length_raises_output_length_with_content():
    resp = _build_response(content="cut here", finish_reason="length")
    with patch.object(kira_llm.litellm, "completion", return_value=resp):
        with pytest.raises(OutputLengthExceededError) as exc_info:
            call_llm_with_tools(
                messages=[], model_name="openai/Qwen3.6-27B",
                api_base=None, api_key=None, tools=[],
            )
    assert exc_info.value.truncated_content == "cut here"


# ---------- retry policy --------------------------------------------

def test_transient_error_is_retried_until_success():
    """Generic RuntimeError (mimics rate-limit / 5xx that litellm wraps)
    should retry. Tenacity's exponential backoff is fast (0.5s min) so
    the test resolves in ~1s."""
    resp_ok = _build_response(content="finally")
    side_effects = [RuntimeError("transient"), RuntimeError("transient again"), resp_ok]
    with patch.object(kira_llm.litellm, "completion", side_effect=side_effects) as m:
        out = call_llm_with_tools(
            messages=[], model_name="openai/Qwen3.6-27B",
            api_base=None, api_key=None, tools=[],
            block_timeout_s=30,
        )
    assert out.content == "finally"
    assert m.call_count == 3


def test_authentication_error_is_not_retried():
    err = LiteLLMAuthenticationError(message="bad key", model="x", llm_provider="x")
    with patch.object(kira_llm.litellm, "completion", side_effect=err) as m:
        with pytest.raises(LiteLLMAuthenticationError):
            call_llm_with_tools(
                messages=[], model_name="openai/Qwen3.6-27B",
                api_base=None, api_key=None, tools=[],
                block_timeout_s=10,
            )
    assert m.call_count == 1


def test_bad_request_error_is_not_retried():
    err = BadRequestError(message="bad", model="x", llm_provider="x")
    with patch.object(kira_llm.litellm, "completion", side_effect=err) as m:
        with pytest.raises(BadRequestError):
            call_llm_with_tools(
                messages=[], model_name="openai/Qwen3.6-27B",
                api_base=None, api_key=None, tools=[],
                block_timeout_s=10,
            )
    assert m.call_count == 1


# ---------- block timeout -------------------------------------------

def test_block_timeout_fires_when_call_hangs():
    """Patch litellm to sleep longer than block_timeout. The wrapper
    should raise BlockTimeoutError, NOT crash with FuturesTimeoutError."""
    import time

    def slow_call(**_kw):
        time.sleep(2.0)
        return _build_response()

    with patch.object(kira_llm.litellm, "completion", side_effect=slow_call):
        with pytest.raises(BlockTimeoutError):
            call_llm_with_tools(
                messages=[], model_name="openai/Qwen3.6-27B",
                api_base=None, api_key=None, tools=[],
                block_timeout_s=1,  # < the 2s sleep
            )


# ---------- image call ----------------------------------------------

def test_call_llm_for_image_returns_text_no_tools_kwarg():
    resp = _build_response(content="A small purple cat.")
    with patch.object(kira_llm.litellm, "completion", return_value=resp) as m:
        out = call_llm_for_image(
            messages=[{"role": "user", "content": [
                {"type": "text", "text": "describe"},
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,xxx"}},
            ]}],
            model_name="openai/Qwen3.6-27B",
            api_base="http://x:8080/v1", api_key="k",
        )
    assert out == "A small purple cat."
    sent_kwargs = m.call_args.kwargs
    # Tools must NOT be sent for image read — purely a description call.
    assert "tools" not in sent_kwargs
    # enable_thinking is forced OFF for image (we want a fast description).
    assert sent_kwargs["extra_body"] == {"chat_template_kwargs": {"enable_thinking": False}}


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
