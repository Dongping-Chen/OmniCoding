"""Tests for the conversation summarizer.

Mocks the LLM call so the summarizer's input/output handling is
exercised without hitting a real endpoint."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from omnicoding.agents.kira import summarize as kira_summarize
from omnicoding.agents.kira.summarize import (
    _build_summary_prompt,
    _serialize_message_for_summary,
    summarize_conversation,
)
from omnicoding.agents.kira.tools import SYSTEM_PROMPT


def test_serialize_string_content():
    out = _serialize_message_for_summary({"role": "user", "content": "hello"})
    assert out == "<user> hello"


def test_serialize_multimodal_keeps_text_drops_image():
    """Image parts would be huge base64 — must be dropped."""
    msg = {
        "role": "user",
        "content": [
            {"type": "text", "text": "look at this"},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,XXXBIG"}},
        ],
    }
    out = _serialize_message_for_summary(msg)
    assert "look at this" in out
    assert "XXXBIG" not in out


def test_serialize_truncates_long_content():
    long = "x" * 10000
    out = _serialize_message_for_summary({"role": "tool", "content": long}, max_chars=200)
    assert len(out) < len(long)
    assert "[truncated]" in out


def test_serialize_includes_tool_calls():
    msg = {
        "role": "assistant",
        "content": "ok",
        "tool_calls": [{
            "id": "c1",
            "function": {"name": "execute_commands", "arguments": '{"k":1}'},
        }],
    }
    out = _serialize_message_for_summary(msg)
    assert "ok" in out
    assert "execute_commands" in out


def test_build_summary_prompt_skips_system_includes_task():
    msgs = [
        {"role": "system", "content": "system rules"},
        {"role": "user", "content": "first user msg"},
        {"role": "assistant", "content": "assistant reply"},
    ]
    out = _build_summary_prompt(msgs, original_instruction="THE_TASK")
    assert "system rules" not in out  # system msg dropped
    assert "first user msg" in out
    assert "assistant reply" in out
    assert "THE_TASK" in out


def test_summarize_conversation_returns_two_message_list():
    msgs = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": "do thing X"},
        {"role": "assistant", "content": "doing it"},
    ]
    with patch.object(kira_summarize, "call_llm_for_image",
                      return_value="Summary: agent did half of X.") as m:
        out = summarize_conversation(
            messages=msgs,
            original_instruction="do thing X",
            model_name="openai/Qwen3.6-27B",
            api_base="http://x:8080/v1",
            api_key="k",
        )
    assert len(out) == 2
    assert out[0]["role"] == "system"
    assert out[0]["content"] == SYSTEM_PROMPT
    assert out[1]["role"] == "user"
    assert "do thing X" in out[1]["content"]
    assert "Summary: agent did half of X." in out[1]["content"]
    # Caller should have asked the summarizer with no tools — the wrapper
    # we stub already enforces that, but assert the model_name made it
    # through so a wrong default can't sneak past.
    assert m.call_args.kwargs["model_name"] == "openai/Qwen3.6-27B"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
