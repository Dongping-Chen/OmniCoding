"""Regression tests for native image_read conversion into ms-swift Agent IR."""
from __future__ import annotations

import base64
import json
from pathlib import Path

import pytest

from omnicoding.data.conversion import convert_one


_TINY_PNG = (
    b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01"
    b"\x08\x06\x00\x00\x00\x1f\x15\xc4\x89\x00\x00\x00\rIDATx\x9cc\xfc"
    b"\xff\xff?\x03\x00\x06\x05\x02\x80\xa3\xfeL\xab\x00\x00\x00\x00IEND"
    b"\xaeB`\x82"
)

_TOOLS = [
    {"type": "function", "function": {
        "name": "image_read",
        "description": "read image",
        "parameters": {"type": "object", "properties": {}}}},
    {"type": "function", "function": {
        "name": "task_complete",
        "description": "end task",
        "parameters": {"type": "object", "properties": {}}}},
]


def _image_url() -> str:
    return "data:image/png;base64," + base64.b64encode(_TINY_PNG).decode("ascii")


def _image_read_call() -> dict:
    return {
        "id": "call_ir",
        "type": "function",
        "function": {
            "name": "image_read",
            "arguments": json.dumps({
                "file_path": "frame.png",
                "image_read_instruction": "describe",
            }),
        },
    }


def test_native_image_user_payload_folds_into_tool_response(tmp_path: Path):
    """KIRA's OpenAI-safe shape is tool ack + user image. The ms-swift
    row must fold that user image into the preceding tool_response so
    qwen3_5 template encoding remains valid and train/serve content is
    one Qwen <tool_response> observation."""
    messages = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "look at frame.png"},
        {"role": "assistant", "content": "", "tool_calls": [_image_read_call()]},
        {"role": "tool", "tool_call_id": "call_ir",
         "content": "Loaded 'frame.png' (image/png, 88 bytes)."},
        {"role": "user", "content": [
            {"type": "text", "text": "image_read: 'frame.png'"},
            {"type": "image_url", "image_url": {"url": _image_url()}},
        ]},
        {"role": "assistant", "content": "<answer>A</answer>",
         "tool_calls": [{"id": "call_tc", "type": "function",
                         "function": {"name": "task_complete", "arguments": "{}"}}]},
        {"role": "tool", "tool_call_id": "call_tc", "content": ""},
    ]

    row = convert_one(
        messages=messages,
        tools_spec=_TOOLS,
        multimodal=True,
        images_out_dir=tmp_path / "images",
        item_tag="item_0000",
    )

    roles = [m["role"] for m in row["messages"]]
    assert roles[:5] == ["system", "user", "assistant", "tool_call", "tool_response"]
    assert "user" not in roles[4:5], f"native image leaked as separate user: {roles}"

    tool_response = row["messages"][4]
    assert "Loaded 'frame.png'" in tool_response["content"]
    assert "image_read: 'frame.png'" in tool_response["content"]
    assert "<image>" in tool_response["content"]
    assert row["images"] and len(row["images"]) == 1
    assert Path(row["images"][0]).exists()


def test_native_image_folds_into_matching_image_read_response_in_multi_tool_turn(tmp_path: Path):
    """If a model emits image_read plus another tool in one assistant
    turn, the native image must attach to the image_read observation,
    not blindly to the last tool_response in the block."""
    messages = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "look and finish"},
        {"role": "assistant", "content": "", "tool_calls": [
            _image_read_call(),
            {"id": "call_tc", "type": "function",
             "function": {"name": "task_complete", "arguments": "{}"}},
        ]},
        {"role": "tool", "tool_call_id": "call_ir",
         "content": "Loaded 'frame.png' (image/png, 88 bytes)."},
        {"role": "tool", "tool_call_id": "call_tc", "content": "executed"},
        {"role": "user", "content": [
            {"type": "text", "text": "image_read: 'frame.png'"},
            {"type": "image_url", "image_url": {"url": _image_url()}},
        ]},
        {"role": "assistant", "content": "<answer>A</answer>"},
    ]

    row = convert_one(
        messages=messages,
        tools_spec=_TOOLS,
        multimodal=True,
        images_out_dir=tmp_path / "images",
        item_tag="item_0002",
    )
    tool_responses = [m for m in row["messages"] if m["role"] == "tool_response"]
    assert len(tool_responses) >= 2
    assert "<image>" in tool_responses[0]["content"]
    assert "image_read: 'frame.png'" in tool_responses[0]["content"]
    assert "<image>" not in tool_responses[1]["content"]


def test_ordinary_multimodal_user_stays_user(tmp_path: Path):
    """Only the native image_read follow-up after a tool_response is
    folded. A real multimodal user turn remains role=user."""
    messages = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": [
            {"type": "text", "text": "What is in this image?"},
            {"type": "image_url", "image_url": {"url": _image_url()}},
        ]},
        {"role": "assistant", "content": "<answer>A</answer>"},
    ]

    row = convert_one(
        messages=messages,
        tools_spec=_TOOLS,
        multimodal=True,
        images_out_dir=tmp_path / "images",
        item_tag="item_0001",
    )

    assert row["messages"][1]["role"] == "user"
    assert "What is in this image?" in row["messages"][1]["content"]
    assert "<image>" in row["messages"][1]["content"]
    assert len(row["images"]) == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
