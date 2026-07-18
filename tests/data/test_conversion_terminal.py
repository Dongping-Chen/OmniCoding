"""Regression tests for the SFT-converter ``_ensure_terminal_task_complete``.

Round-17.11 BUG-X4 (2026-04-30): under FINAL_ANSWER_PROTOCOL, the model
often emits ``<answer>X</answer>`` as plain assistant text and the
harness exits with ``exit_reason=no_tool_calls`` without ever recording
a ``task_complete`` tool_call. The trajectory is functionally complete
(the grader extracts the wrapper from any role), but SFT data ended up
with two distinct terminal shapes:

  - shape A (task_complete called): ``... assistant + task_complete tool_call``
  - shape B (no_tool_calls exit):   ``... assistant<answer>X</answer>`` (no tool_call)

Training on heterogeneous tail shapes muddies the "how a successful run
ends" signal. Fix: ``_ensure_terminal_task_complete`` post-processes
the SFT row and synthesizes a terminal ``task_complete`` pair for any
trajectory whose last assistant turn contains ``<answer>`` but isn't
followed by a real ``task_complete`` call.

These tests pin:
  1. Synthesis fires on shape-B trajectories (the BUG-X4 case).
  2. Synthesis does NOT fire on shape-A trajectories (no double).
  3. Synthesis does NOT fire when there's no ``<answer>`` (failed run
     — let the filter drop it instead of canonicalizing).
"""
from __future__ import annotations

import json
import pytest

from omnicoding.data.conversion import convert_one


_TOOLS = [
    {"type": "function", "function": {
        "name": "execute_commands",
        "description": "shell",
        "parameters": {"type": "object", "properties": {}}}},
    {"type": "function", "function": {
        "name": "task_complete",
        "description": "end task",
        "parameters": {"type": "object", "properties": {}}}},
]


def _last_taskcomplete_count(row: dict) -> int:
    """Count of task_complete tool_calls in the converted row."""
    return sum(
        1 for m in row["messages"]
        if m["role"] == "tool_call"
        and json.loads(m["content"]).get("name") == "task_complete"
    )


def test_synthesize_when_no_tool_calls_exit():
    """Shape-B trajectory (plain-text answer, no task_complete) → must
    synthesize a terminal task_complete pair so the SFT row matches
    shape-A trajectories."""
    messages = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "Q?"},
        {"role": "assistant", "content": "",
         "tool_calls": [{"id": "c1",
                          "function": {"name": "execute_commands",
                                       "arguments": '{"keystrokes":"ls"}'}}]},
        {"role": "tool", "name": "c1", "content": "out"},
        {"role": "assistant", "content": "<answer>B</answer>"},
    ]
    row = convert_one(messages=messages, tools_spec=_TOOLS, multimodal=False)
    last_3 = row["messages"][-3:]
    roles = [m["role"] for m in last_3]
    assert roles == ["assistant", "tool_call", "tool_response"], (
        f"expected synthesized task_complete pair, got tail roles={roles}"
    )
    tc = json.loads(last_3[1]["content"])
    assert tc["name"] == "task_complete"
    assert "<answer>B</answer>" in last_3[0]["content"]
    assert _last_taskcomplete_count(row) == 1


def test_no_double_synthesis_when_task_complete_present():
    """Shape-A trajectory (model called task_complete) → must NOT
    synthesize a second task_complete (would create a malformed pair)."""
    messages = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "Q?"},
        {"role": "assistant", "content": "<answer>A</answer>",
         "tool_calls": [{"id": "tc",
                          "function": {"name": "task_complete",
                                       "arguments": "{}"}}]},
        {"role": "tool", "name": "tc", "content": ""},
    ]
    row = convert_one(messages=messages, tools_spec=_TOOLS, multimodal=False)
    assert _last_taskcomplete_count(row) == 1, (
        "must not double-synthesize task_complete when one is already there"
    )


def test_no_synthesis_without_answer_wrapper():
    """Trajectory without <answer> wrapper at all (model bowed out
    without producing a wrapper) → must NOT fabricate a task_complete.
    These trajectories should fail the upstream filter; canonicalizing
    them would silently smuggle wrong-shaped runs into SFT data."""
    messages = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "Q?"},
        {"role": "assistant", "content": "I give up"},
    ]
    row = convert_one(messages=messages, tools_spec=_TOOLS, multimodal=False)
    assert _last_taskcomplete_count(row) == 0, (
        "must not synthesize task_complete when no <answer> wrapper present"
    )


def test_synthesis_preserves_pre_answer_tool_turns():
    """Verify the synthesized task_complete is inserted AFTER the last
    assistant turn — pre-answer tool_call/tool_response sequences must
    stay in order, otherwise the SFT trajectory becomes nonsensical."""
    messages = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "Q?"},
        {"role": "assistant", "content": "",
         "tool_calls": [{"id": "ec1",
                          "function": {"name": "execute_commands",
                                       "arguments": '{"keystrokes":"a"}'}}]},
        {"role": "tool", "name": "ec1", "content": "first"},
        {"role": "assistant", "content": "",
         "tool_calls": [{"id": "ec2",
                          "function": {"name": "execute_commands",
                                       "arguments": '{"keystrokes":"b"}'}}]},
        {"role": "tool", "name": "ec2", "content": "second"},
        {"role": "assistant", "content": "<answer>C</answer>"},
    ]
    row = convert_one(messages=messages, tools_spec=_TOOLS, multimodal=False)
    msgs = row["messages"]
    # Find the assistant with <answer>; everything before it must be
    # in original order; synthesized task_complete must come after.
    answer_idx = next(i for i, m in enumerate(msgs)
                      if m["role"] == "assistant" and "<answer>C</answer>" in m["content"])
    # All real tool_responses (first, second) must precede answer.
    pre_tool_responses = [m["content"] for m in msgs[:answer_idx] if m["role"] == "tool_response"]
    assert "first" in pre_tool_responses, "first tool result must precede answer"
    assert "second" in pre_tool_responses, "second tool result must precede answer"
    # The synthesized task_complete is at answer_idx+1.
    assert msgs[answer_idx + 1]["role"] == "tool_call"
    tc = json.loads(msgs[answer_idx + 1]["content"])
    assert tc["name"] == "task_complete"


def test_collapses_consecutive_assistant_runs():
    """Round-17.11 BUG-X5: under FINAL_ANSWER_PROTOCOL the model often
    emits multiple plain-text ``<answer>X</answer>`` turns in a row,
    each separated by a kira continue-retry reminder. The user-handler
    drops those reminders (ms-swift can't pair tool_response → user →
    assistant), so consecutive assistant entries leak into out_msgs.
    ms-swift's chat template requires alternating roles → the
    converter must collapse runs of consecutive assistants to one
    (the LAST, since it carries the answer + sits adjacent to
    task_complete)."""
    messages = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "Q?"},
        {"role": "assistant", "content": "<answer>3</answer>"},
        # Continue-retry user message kira injects after no_tool_calls;
        # converter drops it explicitly.
        {"role": "user",
         "content": "Please continue solving the task. If you need more "
                    "evidence, run more tools. Once you have enough to "
                    "decide, your FINAL response must end with the answer "
                    "tag in the exact format the task requested..."},
        {"role": "assistant", "content": "<answer>3</answer>"},
        {"role": "user",
         "content": "Please continue solving the task. If you need more "
                    "evidence..."},
        {"role": "assistant", "content": "<answer>3</answer>",
         "tool_calls": [{"id": "tc",
                          "function": {"name": "task_complete",
                                       "arguments": "{}"}}]},
        {"role": "tool", "name": "tc", "content": ""},
    ]
    row = convert_one(messages=messages, tools_spec=_TOOLS, multimodal=False)
    msgs = row["messages"]
    # No two consecutive assistants in the output.
    consec = sum(
        1 for i in range(1, len(msgs))
        if msgs[i].get("role") == "assistant"
           and msgs[i - 1].get("role") == "assistant"
    )
    assert consec == 0, (
        f"converter emitted {consec} pairs of consecutive assistants; "
        f"trace: {[m['role'] for m in msgs]}"
    )
    # Final answer + task_complete + tool_response remain.
    assert _last_taskcomplete_count(row) == 1
    last_assist_idx = max(
        i for i, m in enumerate(msgs) if m.get("role") == "assistant"
    )
    assert "<answer>3</answer>" in msgs[last_assist_idx]["content"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
