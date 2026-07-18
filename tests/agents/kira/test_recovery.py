"""Recovery-layer tests pinning the malformed-Qwen-XML shapes seen
during round-10 smoke. If the actual rate of these in production drops
to zero after a future Qwen update, we delete recovery.py wholesale —
keep these tests so we know which branches are needed."""

from __future__ import annotations

import json

from omnicoding.agents.kira.recovery import recover_tool_calls

_DECLARED = {"execute_commands", "task_complete", "image_read"}


def test_recover_round10_socialomni_l2_shape():
    """Verbatim-shape from kira_smoke_20260426_001521/socialomni_l2:
    missing leading ``<`` before ``function=``, opens
    ``<parameter=analysis>`` then writes the rest of the args dict in
    JSON syntax inside that body, never closes the parameter."""
    content = (
        "<tool_call>\n"
        "function=execute_commands>\n"
        "<parameter=analysis>\n"
        "Starting the task. I need to determine the answer.\", "
        "\"plan\": \"1. ffprobe metadata\\n2. extract frame\", "
        "\"commands\": [{\"keystrokes\": \"ffprobe -v error inputs/v.mp4\", \"duration\": 5}]}\n"
        "</tool_call>"
    )
    out = recover_tool_calls(content, _DECLARED)
    assert len(out) == 1
    call = out[0]
    assert call["function"]["name"] == "execute_commands"
    args = json.loads(call["function"]["arguments"])
    assert args["plan"].startswith("1. ffprobe")
    assert args["commands"] == [{"keystrokes": "ffprobe -v error inputs/v.mp4", "duration": 5}]


def test_recover_with_proper_leading_lt_in_function_tag():
    """``<function=NAME>...`` (proper qwen3_coder shape, but the
    parameter body still spills into JSON). We accept this too because
    the recovery is permissive on the function tag."""
    content = (
        "<tool_call>\n<function=execute_commands>\n"
        "<parameter=analysis>\n"
        "x\", \"plan\": \"y\", \"commands\": [{\"keystrokes\": \"ls\"}]}\n"
        "</tool_call>"
    )
    out = recover_tool_calls(content, _DECLARED)
    assert out and out[0]["function"]["name"] == "execute_commands"


def test_recover_image_read_param_shape():
    content = (
        "<tool_call>\nfunction=image_read>\n"
        "<parameter=file_path>\n"
        "frames/f1.png\", \"image_read_instruction\": \"describe\"}\n"
        "</tool_call>"
    )
    out = recover_tool_calls(content, _DECLARED)
    assert out and out[0]["function"]["name"] == "image_read"
    args = json.loads(out[0]["function"]["arguments"])
    assert args["file_path"] == "frames/f1.png"
    assert args["image_read_instruction"] == "describe"


def test_recover_rejects_unknown_tool():
    content = (
        "<tool_call>\nfunction=eat_lunch>\n"
        "<parameter=meal>\nsoup\"}\n"
        "</tool_call>"
    )
    assert recover_tool_calls(content, _DECLARED) == []


def test_recover_returns_empty_when_no_tool_call_block():
    assert recover_tool_calls("plain prose with no tool call", _DECLARED) == []
    assert recover_tool_calls(None, _DECLARED) == []
    assert recover_tool_calls("", _DECLARED) == []


def test_recover_strips_trailing_close_parameter_tag():
    content = (
        "<tool_call>\nfunction=execute_commands>\n"
        "<parameter=analysis>\n"
        "x\", \"plan\": \"y\", \"commands\": [{\"keystrokes\": \"ls\"}]}\n"
        "</parameter>\n"
        "</tool_call>"
    )
    out = recover_tool_calls(content, _DECLARED)
    assert out and "commands" in json.loads(out[0]["function"]["arguments"])


def test_recover_handles_multiple_tool_call_blocks():
    content = (
        "<tool_call>\nfunction=execute_commands>\n"
        "<parameter=analysis>\nfirst\", \"plan\": \"a\", "
        "\"commands\": [{\"keystrokes\": \"echo 1\"}]}\n"
        "</tool_call>\n"
        "and then more text\n"
        "<tool_call>\nfunction=execute_commands>\n"
        "<parameter=analysis>\nsecond\", \"plan\": \"b\", "
        "\"commands\": [{\"keystrokes\": \"echo 2\"}]}\n"
        "</tool_call>"
    )
    out = recover_tool_calls(content, _DECLARED)
    assert len(out) == 2
    assert "echo 1" in out[0]["function"]["arguments"]
    assert "echo 2" in out[1]["function"]["arguments"]


def test_recover_rejects_unparseable_garbage_quietly():
    content = (
        "<tool_call>\nfunction=execute_commands>\n"
        "<parameter=analysis>\nthis is a complete sentence with no closing JSON.\n"
        "</tool_call>"
    )
    assert recover_tool_calls(content, _DECLARED) == []
