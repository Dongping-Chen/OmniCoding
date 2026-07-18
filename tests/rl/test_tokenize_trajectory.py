"""Unit tests for ``omnicoding.rl.rollout.tokenize_trajectory``.

Set ``OMNICODING_TEST_TOKENIZER`` to a compatible local tokenizer snapshot to
enable these integration tests.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

QWEN_TOKENIZER_DIR = os.environ.get("OMNICODING_TEST_TOKENIZER", "")


@pytest.fixture(scope="module")
def tokenizer():
    if not QWEN_TOKENIZER_DIR or not Path(QWEN_TOKENIZER_DIR).is_dir():
        pytest.skip("set OMNICODING_TEST_TOKENIZER to run tokenizer integration tests")
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(QWEN_TOKENIZER_DIR)


def _import_func():
    # tokenize_trajectory lives in its own module that has no Relax dep, so
    # the test can run on a machine without Megatron / Relax installed.
    from omnicoding.rl.tokenize import tokenize_trajectory  # noqa: PLC0415
    return tokenize_trajectory


def test_empty(tokenizer):
    fn = _import_func()
    tokens, loss_mask, resp_len = fn([], tokenizer)
    assert tokens == [] and loss_mask == [] and resp_len == 0


def test_prompt_only(tokenizer):
    """No assistant turn yet — prompt tokens only, empty loss mask."""
    fn = _import_func()
    msgs = [
        {"role": "system", "content": "You are helpful."},
        {"role": "user", "content": "Hi."},
    ]
    tokens, loss_mask, resp_len = fn(msgs, tokenizer)
    assert len(tokens) > 0
    assert loss_mask == []
    assert resp_len == 0


def test_single_assistant_turn(tokenizer):
    """One assistant message → leading <|im_start|>assistant\\n marker has loss=0
    (deepeyes convention — observation), content + <|im_end|> get loss=1."""
    fn = _import_func()
    msgs = [
        {"role": "system", "content": "You are helpful."},
        {"role": "user", "content": "What is 2+2?"},
        {"role": "assistant", "content": "It is 4."},
    ]
    tokens, loss_mask, resp_len = fn(msgs, tokenizer)
    assert resp_len == len(loss_mask) > 0
    # Role-marker prefix is <|im_start|>assistant\n — should be 3 tokens of 0,
    # the rest are model output (loss=1).
    role_marker = tokenizer.encode("<|im_start|>assistant\n", add_special_tokens=False)
    assert loss_mask[: len(role_marker)] == [0] * len(role_marker), (
        f"expected role-marker prefix to be 0s, got {loss_mask[: len(role_marker)]}"
    )
    assert all(m == 1 for m in loss_mask[len(role_marker):]), (
        f"expected content tokens to be 1s, got {loss_mask[len(role_marker):]}"
    )
    # Tokens should round-trip-ish: decoding response tokens contains the assistant content.
    decoded = tokenizer.decode(tokens[-resp_len:])
    assert "4" in decoded


def test_multi_turn_alternating(tokenizer):
    """Assistant + tool + assistant → tool tokens get loss=0, assistant content+<|im_end|>
    gets loss=1 but the leading <|im_start|>assistant\\n role markers are masked 0."""
    fn = _import_func()
    msgs = [
        {"role": "system", "content": "You are helpful."},
        {"role": "user", "content": "use tool then answer"},
        {"role": "assistant", "content": "I'll check.", "tool_calls": [
            {"id": "c1", "type": "function", "function": {"name": "execute_commands", "arguments": '{"keystrokes":"echo 4"}'}}
        ]},
        {"role": "tool", "tool_call_id": "c1", "content": "4\n"},
        {"role": "assistant", "content": "<answer>4</answer>"},
    ]
    tokens, loss_mask, resp_len = fn(msgs, tokenizer)
    assert resp_len == len(loss_mask) > 0
    n_one = sum(loss_mask)
    n_zero = resp_len - n_one
    assert n_one > 0, "expected assistant turns to contribute loss=1 tokens"
    assert n_zero > 0, "expected tool turns to contribute loss=0 tokens"
    # With two assistant turns × 3 role-marker tokens (= 6) masked, we should have
    # noticeably fewer 1s than the all-1 convention — but still well over half of
    # tokens are non-zero observations.
    assert 0.1 < n_one / resp_len < 0.9


def test_response_length_matches_loss_mask(tokenizer):
    fn = _import_func()
    msgs = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "u"},
        {"role": "assistant", "content": "a1"},
        {"role": "tool", "tool_call_id": "x", "content": "obs"},
        {"role": "assistant", "content": "a2"},
    ]
    tokens, loss_mask, resp_len = fn(msgs, tokenizer)
    assert len(loss_mask) == resp_len
    # tokens contains prompt + response
    assert len(tokens) >= resp_len


def test_text_fallback_normalizes_openai_tool_arguments() -> None:
    class StrictMappingTokenizer:
        def apply_chat_template(
            self,
            messages,
            *,
            tokenize,
            add_generation_prompt,
            **kwargs,
        ):
            del tokenize, add_generation_prompt, kwargs
            rendered = []
            for message in messages:
                rendered.append(
                    f"<|im_start|>{message['role']}\n{message.get('content', '')}"
                )
                for tool_call in message.get("tool_calls") or []:
                    function = tool_call["function"]
                    arguments = function["arguments"]
                    assert isinstance(arguments, dict)
                    rendered.append(
                        f"<tool_call>{function['name']}"
                        f"{json.dumps(arguments, sort_keys=True)}</tool_call>"
                    )
                rendered.append("<|im_end|>\n")
            return "".join(rendered)

        def encode(self, text, *, add_special_tokens):
            del add_special_tokens
            return list(text.encode())

    messages = [
        {"role": "system", "content": "You are helpful."},
        {"role": "user", "content": "Use a tool."},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "c1",
                    "type": "function",
                    "function": {
                        "name": "execute_commands",
                        "arguments": '{"keystrokes":"echo 4"}',
                    },
                }
            ],
        },
        {"role": "tool", "tool_call_id": "c1", "content": "4\n"},
        {"role": "assistant", "content": "<answer>4</answer>"},
    ]

    fn = _import_func()
    tokens, loss_mask, response_length = fn(messages, StrictMappingTokenizer())

    assert len(tokens) > response_length == len(loss_mask) > 0
    assert isinstance(messages[2]["tool_calls"][0]["function"]["arguments"], str)
