"""Loop tests with mocked LLM + shell.

We never spawn real bash here (that's ``test_shell.py``) and never call
litellm here (that's ``test_llm.py``). The point is to pin the
dispatch decision-tree:

  1. happy execute_commands path
  2. task_complete-only double-confirm (2 calls → done)
  3. **Bug A**: task_complete + execute_commands in one turn → run
     commands, return checklist, set pending=True
  4. **Bug B**: image_read + task_complete in one turn → image_read,
     return checklist, set pending=True (NOT silently skip pending)
  5. continue-prompt budget on no_tool_calls turns
  6. context-overflow → summarize → retry
  7. output-overflow → re-prompt-shorter → retry
"""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from omnicoding.agents.kira import loop as kira_loop
from omnicoding.agents.kira.llm import (
    ContextLengthExceededError,
    LLMResponse,
    OutputLengthExceededError,
)
from omnicoding.agents.kira.loop import KiraAgent


def _make_resp(content: str = "", tool_calls: list[dict] | None = None,
               finish_reason: str = "tool_calls") -> LLMResponse:
    return LLMResponse(
        content=content,
        reasoning_content="",
        tool_calls=tool_calls or [],
        finish_reason=finish_reason,
        prompt_tokens=10,
        completion_tokens=20,
    )


def _exec_commands_call(*, analysis="a", plan="p", keystrokes="echo hi", duration=0.5) -> dict:
    return {
        "id": "call_exec_1",
        "type": "function",
        "function": {
            "name": "execute_commands",
            "arguments": json.dumps({
                "analysis": analysis, "plan": plan,
                "commands": [{"keystrokes": keystrokes, "duration": duration}],
            }),
        },
    }


def _task_complete_call(idx: int = 0) -> dict:
    return {
        "id": f"call_tc_{idx}",
        "type": "function",
        "function": {"name": "task_complete", "arguments": "{}"},
    }


def _image_read_call(*, file_path="img.png", instruction="describe") -> dict:
    return {
        "id": "call_ir_1",
        "type": "function",
        "function": {
            "name": "image_read",
            "arguments": json.dumps({
                "file_path": file_path,
                "image_read_instruction": instruction,
            }),
        },
    }


@pytest.fixture
def fake_shell():
    """Patch PersistentShell so the loop never spawns bash."""
    shell = MagicMock()
    shell.run.return_value = "[fake terminal output]"
    shell.__enter__ = MagicMock(return_value=shell)
    shell.__exit__ = MagicMock(return_value=None)
    with patch.object(kira_loop, "PersistentShell", return_value=shell):
        yield shell


def _build_agent(
    tmp_path,
    endpoint_session=None,
    max_sticky_retries: int = 0,
    image_read_mode: str = "sub_llm",
    max_answer_retries: int = 1,
) -> KiraAgent:
    """Default ``max_sticky_retries=0`` so legacy tests still see the
    old "first BlockTimeout → failover/raise" behaviour. Sticky-retry
    tests pass an explicit non-zero value below.

    ``image_read_mode`` defaults to ``sub_llm`` here (NOT the harness
    production default ``native``) because most legacy tests were
    written when sub_llm was the only path and ``patch read_image``;
    keeping legacy default keeps those tests pinning the legacy path's
    pending_completion semantics. Native-mode tests below pass
    ``image_read_mode="native"`` explicitly."""
    return KiraAgent(
        workspace=tmp_path,
        model_name="openai/Qwen3.6-27B",
        api_base="http://x:8080/v1",
        endpoint_session=endpoint_session,
        continue_prompt="Use a tool to keep going.",
        api_key="local",
        step_limit=10,
        max_sticky_retries=max_sticky_retries,
        image_read_mode=image_read_mode,
        max_answer_retries=max_answer_retries,
    )


# ---------- happy execute_commands ----------------------------------

def test_happy_execute_commands_then_task_complete_single_call(tmp_path, fake_shell):
    """Single-call task_complete: commands turn → one task_complete
    exits. The model's first commands turn echoes ``<answer>...</answer>``
    so the wrapper-preflight gate is satisfied and the lone
    ``task_complete`` call ends the run with no extra reminder turn."""
    fake_shell.run.return_value = "<answer>X</answer>"  # wrapper in terminal
    agent = _build_agent(tmp_path)
    responses = [
        _make_resp(tool_calls=[_exec_commands_call(keystrokes="echo '<answer>X</answer>'")]),
        _make_resp(tool_calls=[_task_complete_call(0)]),
    ]
    with patch.object(kira_loop, "call_llm_with_tools", side_effect=responses):
        result = agent.run("write the answer then finish")
    assert result.completed is True
    assert result.exit_reason == "task_complete"
    assert result.n_steps == 2  # one commands turn + one task_complete turn
    keystrokes_sent = [c.args[0] for c in fake_shell.run.call_args_list]
    assert "echo '<answer>X</answer>'" in keystrokes_sent


def test_task_complete_without_wrapper_sends_reminder_then_exits(tmp_path, fake_shell):
    """Single-call task_complete preflight: when the model fires
    task_complete BEFORE emitting <answer>X</answer> anywhere, the
    harness appends a short user reminder (NOT the legacy double-confirm
    checklist) and lets the model retry once. After the reminder the
    model emits the wrapper alongside another task_complete and the
    run ends. With max_answer_retries=1 (default), this is a 3-turn
    trajectory: bare-task_complete → wrapper-emit-task_complete → exit."""
    agent = _build_agent(tmp_path)  # default max_answer_retries=1
    responses = [
        _make_resp(tool_calls=[_task_complete_call(0)]),  # turn 1: forgot wrapper
        _make_resp(content="<answer>X</answer>",          # turn 2: wrap + task_complete
                   tool_calls=[_task_complete_call(1)]),
    ]
    with patch.object(kira_loop, "call_llm_with_tools", side_effect=responses):
        result = agent.run("answer the question")
    assert result.completed is True
    assert result.exit_reason == "task_complete"
    # Trajectory has the user reminder between the two task_complete turns.
    reminder = next(
        (m for m in agent.messages if m.get("role") == "user"
         and isinstance(m.get("content"), str)
         and "<answer>" in m["content"] and "no <answer>" in m["content"]),
        None,
    )
    assert reminder is not None, "reminder message missing"
    # The reminder must NOT carry the legacy [!] Checklist sentinel.
    assert "[!] Checklist" not in reminder["content"]
    # And the reminder is short — no checklist bullet list.
    assert len(reminder["content"]) < 600


def test_task_complete_with_wrapper_in_assistant_content_no_reminder(tmp_path, fake_shell):
    """Path-b: when the wrapper is emitted directly in the assistant's
    same-turn content alongside task_complete, the preflight passes
    and no reminder is sent. Single LLM turn → exit."""
    agent = _build_agent(tmp_path)
    responses = [
        _make_resp(content="My answer: <answer>42</answer>",
                   tool_calls=[_task_complete_call(0)]),
    ]
    with patch.object(kira_loop, "call_llm_with_tools", side_effect=responses):
        result = agent.run("Q?")
    assert result.completed is True
    assert result.n_steps == 1
    # No reminder in the trajectory.
    reminders = [
        m for m in agent.messages if m.get("role") == "user"
        and isinstance(m.get("content"), str)
        and "no <answer>" in m["content"]
    ]
    assert len(reminders) == 0


def test_task_complete_wrapper_retries_exhausted_exits_anyway(tmp_path, fake_shell):
    """If the model never emits the wrapper even after the reminder,
    the harness must still exit (bounded run). Default budget is 1 →
    after one reminder turn, second task_complete with no wrapper
    finalizes anyway. Predicted answer will be empty downstream — that
    is a model failure, not a harness loop."""
    agent = _build_agent(tmp_path)
    responses = [
        _make_resp(tool_calls=[_task_complete_call(0)]),  # forgot wrapper
        _make_resp(tool_calls=[_task_complete_call(1)]),  # still forgot
    ]
    with patch.object(kira_loop, "call_llm_with_tools", side_effect=responses):
        result = agent.run("Q?")
    assert result.completed is True
    assert result.exit_reason == "task_complete"
    # Two LLM turns total — reminder budget exhausted on the second.
    assert result.n_steps == 2


# ---------- task_complete + commands in one turn --------------------

def test_task_complete_with_commands_runs_then_exits(tmp_path, fake_shell):
    """Model emits BOTH execute_commands AND task_complete in one
    assistant turn. Expected: run the commands so the workspace state
    matches the agent's intent, then exit. Single-call: no checklist,
    no second confirmation round.

    Wrapper preflight bypassed via ``max_answer_retries=0`` — this
    test is about the dispatch flow, not the answer-wrapper gate."""
    agent = _build_agent(tmp_path, max_answer_retries=0)
    responses = [
        _make_resp(tool_calls=[
            _exec_commands_call(keystrokes="echo work_done"),
            _task_complete_call(0),
        ]),
    ]
    with patch.object(kira_loop, "call_llm_with_tools", side_effect=responses):
        result = agent.run("do the thing then finish")

    assert result.completed is True
    assert result.exit_reason == "task_complete"
    # Exactly 1 LLM turn — single-call task_complete.
    assert result.n_steps == 1

    # The exec_commands tool reply carries the terminal output (no
    # checklist wrapper any more). The model's commands ran.
    first_tool_msg = next(
        m for m in agent.messages
        if m.get("role") == "tool" and m.get("tool_call_id", "").startswith("call_exec_1")
    )
    assert "[!] Checklist" not in first_tool_msg["content"]
    assert "[fake terminal output]" in first_tool_msg["content"]


# ---------- image_read + task_complete in one turn ------------------

def test_image_read_with_task_complete_processes_image_then_exits(tmp_path, fake_shell):
    """image_read AND task_complete in one turn. Single-call semantics:
    the image_read STILL runs (so the trajectory keeps the description
    or the actual pixels in native mode), then the loop exits — no
    checklist, no second task_complete required.

    Wrapper preflight bypassed via ``max_answer_retries=0`` — this
    test pins the dispatch flow, not the answer-wrapper gate."""
    agent = _build_agent(tmp_path, max_answer_retries=0)  # default image_read_mode=sub_llm
    responses = [
        _make_resp(tool_calls=[
            _image_read_call(file_path="frame.png", instruction="describe"),
            _task_complete_call(0),
        ]),
    ]
    with patch.object(kira_loop, "read_image", return_value="image_read result for 'frame.png':\nA red box."), \
         patch.object(kira_loop, "call_llm_with_tools", side_effect=responses):
        result = agent.run("look at frame.png then finish")

    assert result.completed is True
    assert result.exit_reason == "task_complete"
    assert result.n_steps == 1

    image_tool_msg = next(
        m for m in agent.messages
        if m.get("role") == "tool" and m.get("tool_call_id", "").startswith("call_ir_1")
    )
    # No checklist in the tool reply — it's the bare image description.
    assert "[!] Checklist" not in image_tool_msg["content"]
    assert "A red box" in image_tool_msg["content"]


# ---------- native image_read mode ----------------------------------

# 1×1 transparent PNG, mirrors test_image_read.py's _TINY_PNG.
_TINY_PNG_BYTES = (
    b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01"
    b"\x08\x06\x00\x00\x00\x1f\x15\xc4\x89\x00\x00\x00\rIDATx\x9cc\xfc"
    b"\xff\xff?\x03\x00\x06\x05\x02\x80\xa3\xfeL\xab\x00\x00\x00\x00IEND"
    b"\xaeB`\x82"
)


def test_native_image_read_injects_user_message_with_image_url(tmp_path, fake_shell):
    """Contract: native mode must (1) ack the tool call with text in
    ``role=tool`` and (2) follow up with a ``role=user`` message whose
    content is a multimodal list including an ``image_url`` block.
    The main agent's NEXT call should see all three messages."""
    img = tmp_path / "frame.png"
    img.write_bytes(_TINY_PNG_BYTES)
    agent = _build_agent(tmp_path, image_read_mode="native", max_answer_retries=0)
    responses = [
        _make_resp(tool_calls=[_image_read_call(file_path="frame.png", instruction="describe scene")]),
        _make_resp(tool_calls=[_task_complete_call(0)]),
    ]
    with patch.object(kira_loop, "call_llm_with_tools", side_effect=responses):
        result = agent.run("look at frame.png")
    assert result.completed is True

    # Locate the assistant turn that called image_read, then the
    # following tool reply, then the harness-injected user message.
    msgs = agent.messages
    assistant_ir_idx = next(
        i for i, m in enumerate(msgs)
        if m.get("role") == "assistant"
        and any((tc.get("function") or {}).get("name") == "image_read"
                for tc in (m.get("tool_calls") or []))
    )
    tool_msg = msgs[assistant_ir_idx + 1]
    user_msg = msgs[assistant_ir_idx + 2]

    assert tool_msg["role"] == "tool"
    assert "frame.png" in tool_msg["content"]
    # Tool reply must NOT contain a base64 blob (lives in user msg).
    assert "base64" not in tool_msg["content"]

    assert user_msg["role"] == "user"
    parts = user_msg["content"]
    assert isinstance(parts, list)
    assert any(p.get("type") == "text" for p in parts)
    assert any(p.get("type") == "image_url" for p in parts)
    img_part = next(p for p in parts if p.get("type") == "image_url")
    assert img_part["image_url"]["url"].startswith("data:image/png;base64,")


def test_native_image_read_missing_file_no_user_message_appended(tmp_path, fake_shell):
    """When the file doesn't exist, the tool reply carries the ERROR
    string and NO user message is appended (no garbage image gets into
    the conversation)."""
    agent = _build_agent(tmp_path, image_read_mode="native", max_answer_retries=0)
    responses = [
        _make_resp(tool_calls=[_image_read_call(file_path="missing.png", instruction="describe")]),
        _make_resp(tool_calls=[_task_complete_call(0)]),
    ]
    with patch.object(kira_loop, "call_llm_with_tools", side_effect=responses):
        agent.run("look at missing.png")

    msgs = agent.messages
    assistant_ir_idx = next(
        i for i, m in enumerate(msgs)
        if m.get("role") == "assistant"
        and any((tc.get("function") or {}).get("name") == "image_read"
                for tc in (m.get("tool_calls") or []))
    )
    tool_msg = msgs[assistant_ir_idx + 1]
    after_msg = msgs[assistant_ir_idx + 2] if assistant_ir_idx + 2 < len(msgs) else {}

    assert tool_msg["role"] == "tool"
    assert tool_msg["content"].startswith("ERROR:") or "ERROR:" in tool_msg["content"]
    # The next message is whatever came AFTER (the next assistant turn or
    # whatever) — must NOT be the harness-injected user image.
    if after_msg.get("role") == "user":
        # If a user message DOES land here for some unrelated reason,
        # it must not contain image_url parts.
        c = after_msg.get("content")
        if isinstance(c, list):
            assert not any(p.get("type") == "image_url" for p in c)


def test_native_image_read_with_task_complete_processes_image_then_exits(tmp_path, fake_shell):
    """Native + image_read + task_complete in one turn (single-call):
    the image still gets decoded and injected as a follow-up user
    message (so the trajectory has the pixels), AND the loop exits.
    The role=tool reply is the bare native ack, no checklist."""
    img = tmp_path / "frame.png"
    img.write_bytes(_TINY_PNG_BYTES)
    agent = _build_agent(tmp_path, image_read_mode="native", max_answer_retries=0)
    responses = [
        _make_resp(tool_calls=[
            _image_read_call(file_path="frame.png", instruction="describe"),
            _task_complete_call(0),
        ]),
    ]
    with patch.object(kira_loop, "call_llm_with_tools", side_effect=responses):
        result = agent.run("look at frame.png then finish")
    assert result.completed is True
    assert result.exit_reason == "task_complete"
    assert result.n_steps == 1

    msgs = agent.messages
    image_tool_msg = next(
        m for m in msgs
        if m.get("role") == "tool" and m.get("tool_call_id", "").startswith("call_ir_1")
    )
    assert "[!] Checklist" not in image_tool_msg["content"]
    # The image rides through as a user message even in single-call exit.
    user_with_image = [
        m for m in msgs
        if m.get("role") == "user" and isinstance(m.get("content"), list)
        and any(p.get("type") == "image_url" for p in m["content"])
    ]
    assert len(user_with_image) == 1


def test_native_image_read_does_not_call_sub_llm(tmp_path, fake_shell):
    """Critical contract: in native mode, the legacy ``read_image``
    sub-LLM path must NOT be invoked (it would consume tokens and
    break the train/serve parity native mode is meant to fix)."""
    img = tmp_path / "frame.png"
    img.write_bytes(_TINY_PNG_BYTES)
    agent = _build_agent(tmp_path, image_read_mode="native", max_answer_retries=0)
    responses = [
        _make_resp(tool_calls=[_image_read_call(file_path="frame.png", instruction="describe")]),
        _make_resp(tool_calls=[_task_complete_call(0)]),
    ]
    with patch.object(
        kira_loop, "read_image",
        side_effect=AssertionError("native mode must not call sub_llm read_image"),
    ), patch.object(kira_loop, "call_llm_with_tools", side_effect=responses):
        result = agent.run("look at frame.png")
    assert result.completed is True


def test_qwen_api_safe_messages_fold_native_image_into_tool(tmp_path):
    """Qwen/sGLang can render multimodal tool content inside
    <tool_response>. The send-time view folds KIRA's GPT-safe
    tool->user(image) pair without mutating the saved trajectory."""
    agent = _build_agent(tmp_path, image_read_mode="native")
    agent.messages = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "look"},
        {"role": "assistant", "content": "", "tool_calls": [
            _image_read_call(file_path="frame.png", instruction="describe"),
            _task_complete_call(0),
        ]},
        {"role": "tool", "tool_call_id": "call_ir_1",
         "content": "Loaded 'frame.png' (image/png, 88 bytes)."},
        {"role": "tool", "tool_call_id": "call_tc_0", "content": "executed"},
        {"role": "user", "content": [
            {"type": "text", "text": "image_read: 'frame.png'"},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAA="}},
        ]},
    ]

    safe = agent._api_safe_messages()
    assert [m["role"] for m in safe] == ["system", "user", "assistant", "tool", "tool"]
    image_tool = next(m for m in safe if m.get("tool_call_id") == "call_ir_1")
    assert isinstance(image_tool["content"], list)
    assert image_tool["content"][0]["type"] == "text"
    assert "Loaded 'frame.png'" in image_tool["content"][0]["text"]
    assert "image_read: 'frame.png'" in image_tool["content"][0]["text"]
    assert any(p.get("type") == "image_url" for p in image_tool["content"])
    # The non-image tool response stays plain text.
    tc_tool = next(m for m in safe if m.get("tool_call_id") == "call_tc_0")
    assert tc_tool["content"] == "executed"
    # Internal trajectory is still GPT/OpenAI-safe and split.
    assert agent.messages[-1]["role"] == "user"


def test_code_x_provider_override_folds_native_image_into_tool(tmp_path):
    agent = KiraAgent(
        workspace=tmp_path,
        model_name="openai/shuaishuaicdp/Code-X-SFT-27B",
        provider="qwen",
        api_base="http://127.0.0.1:8080/v1",
        continue_prompt="Use a tool.",
        image_read_mode="native",
    )
    agent.messages = [
        {"role": "assistant", "content": "", "tool_calls": [
            _image_read_call(file_path="frame.png", instruction="describe"),
        ]},
        {
            "role": "tool",
            "tool_call_id": "call_ir_1",
            "content": "Loaded frame.png",
        },
        {"role": "user", "content": [
            {"type": "text", "text": "image_read: frame.png"},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAA="}},
        ]},
    ]

    safe = agent._api_safe_messages()
    assert [message["role"] for message in safe] == ["assistant", "tool"]
    assert any(
        part.get("type") == "image_url" for part in safe[-1]["content"]
    )


def test_openai_api_safe_messages_keep_native_image_as_user(tmp_path):
    """GPT/OpenAI must keep the split shape because image payloads are
    not valid tool returns for that path."""
    agent = KiraAgent(
        workspace=tmp_path,
        model_name="openai/gpt-5.5",
        api_base="http://router/v1",
        continue_prompt="Use a tool.",
        api_key="router",
        image_read_mode="native",
    )
    agent.messages = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "look"},
        {"role": "assistant", "content": "", "tool_calls": [
            _image_read_call(file_path="frame.png", instruction="describe"),
        ]},
        {"role": "tool", "tool_call_id": "call_ir_1",
         "content": "Loaded 'frame.png' (image/png, 88 bytes)."},
        {"role": "user", "content": [
            {"type": "text", "text": "image_read: 'frame.png'"},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAA="}},
        ]},
    ]

    safe = agent._api_safe_messages()
    assert [m["role"] for m in safe] == ["system", "user", "assistant", "tool", "user"]
    assert isinstance(safe[-1]["content"], list)
    assert isinstance(safe[-2]["content"], str)


# ---------- continue-prompt budget ----------------------------------

def test_no_tool_calls_triggers_reminder_until_budget_exhausted(tmp_path, fake_shell):
    agent = KiraAgent(
        workspace=tmp_path,
        model_name="claude-opus-4-7",  # default reminders=2
        api_base=None,
        continue_prompt="REMINDER: use a tool",
        api_key="k",
        step_limit=10,
    )
    assert agent.max_tool_reminders == 2
    responses = [_make_resp(content="just prose, no tool", tool_calls=[]) for _ in range(3)]
    with patch.object(kira_loop, "call_llm_with_tools", side_effect=responses):
        result = agent.run("do something")
    assert result.exit_reason == "no_tool_calls"
    # Budget=2 means the model gets 2 reminders, then we bow out on the 3rd empty turn.
    assert len(result.retry_attempts) == 2
    assert all(a["reason"] == "no_tool_calls" for a in result.retry_attempts)


def test_no_tool_calls_then_recovery_resets_reminder_counter(tmp_path, fake_shell):
    """If the model recovers (one good tool turn), the reminder budget
    refills so future bad streaks get the full retry budget again.

    Wrapper preflight bypassed (``max_answer_retries=0``) so the
    retry_attempts list contains only the no_tool_calls reminder, not
    the missing-answer-wrapper reminder from the new gate."""
    agent = _build_agent(tmp_path, max_answer_retries=0)  # default reminders=10
    responses = [
        _make_resp(tool_calls=[]),                             # bad → reminder 1
        _make_resp(tool_calls=[_exec_commands_call()]),        # good → reset
        _make_resp(tool_calls=[_task_complete_call(0)]),
    ]
    with patch.object(kira_loop, "call_llm_with_tools", side_effect=responses):
        result = agent.run("do thing")
    assert result.completed is True
    # Only the no_tool_calls reminder lands in retry_attempts; the
    # answer-wrapper preflight was disabled.
    assert len(result.retry_attempts) == 1
    assert result.retry_attempts[0].get("reason") == "no_tool_calls"


# ---------- context overflow ----------------------------------------

def test_context_overflow_triggers_summarize_then_retry(tmp_path, fake_shell):
    """First call overflows; summarizer fires; retry succeeds."""
    agent = _build_agent(tmp_path)
    success = _make_resp(tool_calls=[_exec_commands_call()])
    # First call: overflow. Then the loop's _handle_context_overflow
    # calls _call_llm again — that should succeed.
    side_effects = [
        ContextLengthExceededError("too big"),
        success,
        _make_resp(tool_calls=[_task_complete_call(0)]),
        _make_resp(tool_calls=[_task_complete_call(1)]),
    ]
    with patch.object(kira_loop, "call_llm_with_tools", side_effect=side_effects), \
         patch.object(kira_loop, "summarize_conversation",
                      return_value=[{"role": "system", "content": "system"},
                                    {"role": "user", "content": "summarized handoff"}]) as m_sum:
        result = agent.run("long task")
    assert result.completed is True
    assert result.n_summarizations == 1
    assert m_sum.call_count == 1
    # After summarization the messages reset to [system, user(handoff), ...recent turns].
    assert any("summarized handoff" in str(m.get("content", "")) for m in agent.messages)


def test_context_overflow_with_summarize_disabled_surfaces_error(tmp_path, fake_shell):
    agent = KiraAgent(
        workspace=tmp_path,
        model_name="openai/Qwen3.6-27B",
        api_base="http://x:8080/v1",
        continue_prompt="x",
        enable_summarize=False,
    )
    with patch.object(kira_loop, "call_llm_with_tools",
                      side_effect=ContextLengthExceededError("nope")):
        result = agent.run("oversized")
    assert result.exit_reason == "error"
    assert "ContextLengthExceeded" in (result.error or "")


# ---------- output overflow -----------------------------------------

def test_output_overflow_reprompts_for_shorter_response(tmp_path, fake_shell):
    """finish_reason='length' on first call → loop appends 'shorter
    response' user message and calls again. Second call succeeds."""
    agent = _build_agent(tmp_path)
    side_effects = [
        OutputLengthExceededError("truncated", truncated_content="cut..."),
        _make_resp(tool_calls=[_exec_commands_call()]),
        _make_resp(tool_calls=[_task_complete_call(0)]),
        _make_resp(tool_calls=[_task_complete_call(1)]),
    ]
    with patch.object(kira_loop, "call_llm_with_tools", side_effect=side_effects):
        result = agent.run("verbose task")
    assert result.completed is True
    # The shorter-response prompt must appear in messages.
    shorter_prompts = [m for m in agent.messages
                       if m.get("role") == "user"
                       and "shorter" in str(m.get("content", "")).lower()]
    assert shorter_prompts, "expected a 'shorter response' re-prompt"


# ---------- exit reasons --------------------------------------------

def test_step_limit_reached_returns_step_limit(tmp_path, fake_shell):
    agent = KiraAgent(
        workspace=tmp_path,
        model_name="openai/Qwen3.6-27B",
        api_base="http://x:8080/v1",
        continue_prompt="x",
        step_limit=3,
    )
    # Always returns execute_commands; loop never finishes.
    responses = [_make_resp(tool_calls=[_exec_commands_call()]) for _ in range(10)]
    with patch.object(kira_loop, "call_llm_with_tools", side_effect=responses):
        result = agent.run("infinite work")
    assert result.exit_reason == "step_limit"
    assert result.n_steps == 3


def test_provider_default_max_reminders_for_qwen_is_10(tmp_path):
    agent = KiraAgent(
        workspace=tmp_path, model_name="openai/Qwen3.6-27B",
        api_base="x", continue_prompt="c",
    )
    assert agent.max_tool_reminders == 10


def test_provider_default_max_reminders_for_anthropic_is_2(tmp_path):
    agent = KiraAgent(
        workspace=tmp_path, model_name="claude-opus-4-7",
        api_base=None, continue_prompt="c",
    )
    assert agent.max_tool_reminders == 2


def test_explicit_max_reminders_overrides_provider_default(tmp_path):
    agent = KiraAgent(
        workspace=tmp_path, model_name="openai/Qwen3.6-27B",
        api_base="x", continue_prompt="c", max_tool_reminders=5,
    )
    assert agent.max_tool_reminders == 5


# ---------- endpoint failover (multi-endpoint pool) -------------------------


def test_failover_on_block_timeout_swaps_endpoint_and_succeeds(tmp_path, fake_shell):
    """Simulate the production preempt scenario: first call BlockTimeoutErrors
    on host A, failover to host B, second call succeeds. Trajectory survives."""
    from omnicoding.agents.kira.endpoint_pool import EndpointSession, parse_endpoints
    from omnicoding.agents.kira.llm import BlockTimeoutError

    pool = parse_endpoints("http://host-a:8080/v1=1,http://host-b:8080/v1=1")
    session = EndpointSession(pool, idx=0)
    start_url = session.current_url
    agent = _build_agent(tmp_path, endpoint_session=session)

    # First call → simulated stuck endpoint (BlockTimeoutError).
    # Second call (after failover) → normal exec_commands.
    # Third → task_complete (× 2 for double-confirm).
    responses = [
        BlockTimeoutError("LLM call blocked for 600s"),
        _make_resp(tool_calls=[_exec_commands_call(keystrokes="echo hi")]),
        _make_resp(tool_calls=[_task_complete_call(0)]),
        _make_resp(tool_calls=[_task_complete_call(1)]),
    ]
    with patch.object(kira_loop, "call_llm_with_tools", side_effect=responses):
        result = agent.run("do the thing")

    assert result.exit_reason == "task_complete"
    assert session.current_url != start_url
    history = session.history()
    assert len(history) == 1
    assert history[0]["from"] == start_url
    assert history[0]["reason"] == "BlockTimeoutError"


def test_failover_on_api_connection_error(tmp_path, fake_shell):
    """sglang restart looks like APIConnectionError to litellm — also failover."""
    from omnicoding.agents.kira.endpoint_pool import EndpointSession, parse_endpoints
    import litellm.exceptions as le

    pool = parse_endpoints("a=1,b=1")
    session = EndpointSession(pool, idx=0)
    agent = _build_agent(tmp_path, endpoint_session=session)

    responses = [
        le.APIConnectionError(message="boom", llm_provider="openai", model="x"),
        _make_resp(tool_calls=[_exec_commands_call()]),
        _make_resp(tool_calls=[_task_complete_call(0)]),
        _make_resp(tool_calls=[_task_complete_call(1)]),
    ]
    with patch.object(kira_loop, "call_llm_with_tools", side_effect=responses):
        result = agent.run("ok")

    assert result.exit_reason == "task_complete"
    assert len(session.history()) == 1


def test_failover_exhausted_raises_through(tmp_path, fake_shell):
    """When both endpoints are dead and budget runs out, the error bubbles
    up rather than looping forever."""
    from omnicoding.agents.kira.endpoint_pool import EndpointSession, parse_endpoints
    from omnicoding.agents.kira.llm import BlockTimeoutError

    pool = parse_endpoints("a=1,b=1")
    # max_failovers=2 → 3rd call exhausts (budget hit), session raises through.
    session = EndpointSession(pool, idx=0, max_failovers=2, max_per_url=2)
    agent = _build_agent(tmp_path, endpoint_session=session)

    # Every call BlockTimeoutErrors — failover budget is 2 → raise on the 3rd.
    err = BlockTimeoutError("stuck")
    with patch.object(kira_loop, "call_llm_with_tools", side_effect=[err, err, err]):
        result = agent.run("hopeless")
    # Loop catches the surfaced error and records it as fatal exit.
    assert result.exit_reason in ("error", "preflight_error")
    assert result.error and "BlockTimeoutError" in result.error
    # We attempted exactly the failover budget before giving up.
    assert len(session.history()) == 2


def test_no_session_raises_directly_without_failover(tmp_path, fake_shell):
    """Single-endpoint mode (no session) still bubbles BlockTimeoutError."""
    from omnicoding.agents.kira.llm import BlockTimeoutError

    agent = _build_agent(tmp_path)  # endpoint_session=None
    with patch.object(kira_loop, "call_llm_with_tools",
                      side_effect=BlockTimeoutError("stuck")):
        result = agent.run("x")
    assert result.exit_reason == "error"
    assert result.error and "BlockTimeoutError" in result.error


def test_record_success_resets_after_recovery(tmp_path, fake_shell):
    """A successful call clears the per-URL try counter so a later
    failure on a DIFFERENT URL can come back to the recovered one."""
    from omnicoding.agents.kira.endpoint_pool import EndpointSession, parse_endpoints
    from omnicoding.agents.kira.llm import BlockTimeoutError

    pool = parse_endpoints("a=1,b=1")
    session = EndpointSession(pool, idx=0, max_per_url=1)
    agent = _build_agent(tmp_path, endpoint_session=session)
    start_url = session.current_url
    other_url = "b" if start_url == "a" else "a"

    responses = [
        BlockTimeoutError("stuck"),                         # a fails
        _make_resp(tool_calls=[_exec_commands_call()]),     # b ok → record_success
        _make_resp(tool_calls=[_task_complete_call(0)]),    # b ok
        _make_resp(tool_calls=[_task_complete_call(1)]),    # b ok
    ]
    with patch.object(kira_loop, "call_llm_with_tools", side_effect=responses):
        result = agent.run("ok")

    assert result.exit_reason == "task_complete"
    assert session.current_url == other_url
    # `a` was tried once (and that try is now permanent in _tries['a']=1);
    # `b`'s success reset its own counter to 0.
    stats = session.stats()
    assert stats["tries"][other_url] == 0


# ---------- sticky retry on BlockTimeout (preserves session_id cache) -----


def test_sticky_retry_on_block_timeout_stays_on_same_endpoint(tmp_path, fake_shell):
    """Default ``max_sticky_retries=3`` retries on the SAME endpoint when a
    BlockTimeout fires — so chatgpt's session_id cache stays warm. Failover
    only fires after the sticky budget is exhausted."""
    from omnicoding.agents.kira.endpoint_pool import EndpointSession, parse_endpoints
    from omnicoding.agents.kira.llm import BlockTimeoutError

    pool = parse_endpoints("http://a/v1=1,http://b/v1=1")
    session = EndpointSession(pool, idx=0)
    start_url = session.current_url
    agent = _build_agent(tmp_path, endpoint_session=session, max_sticky_retries=3)

    # 1 timeout absorbed by sticky retry → 2nd call succeeds on SAME url.
    responses = [
        BlockTimeoutError("transient"),
        _make_resp(tool_calls=[_exec_commands_call()]),
        _make_resp(tool_calls=[_task_complete_call(0)]),
        _make_resp(tool_calls=[_task_complete_call(1)]),
    ]
    with patch.object(kira_loop, "call_llm_with_tools", side_effect=responses):
        result = agent.run("transient hiccup")

    assert result.exit_reason == "task_complete"
    # No failover — kept the warm session.
    assert session.current_url == start_url
    assert session.history() == []


def test_sticky_budget_exhausted_falls_through_to_failover(tmp_path, fake_shell):
    """``max_sticky_retries`` consecutive BlockTimeouts → fall through to
    failover. Once on the new endpoint, sticky budget resets."""
    from omnicoding.agents.kira.endpoint_pool import EndpointSession, parse_endpoints
    from omnicoding.agents.kira.llm import BlockTimeoutError

    pool = parse_endpoints("http://a/v1=1,http://b/v1=1")
    session = EndpointSession(pool, idx=0)
    start_url = session.current_url
    agent = _build_agent(tmp_path, endpoint_session=session, max_sticky_retries=2)

    # 3 timeouts on `a`: first 2 absorbed by sticky budget, 3rd triggers
    # failover. Then 1 success on `b` to wrap up.
    responses = [
        BlockTimeoutError("stuck-1"),
        BlockTimeoutError("stuck-2"),
        BlockTimeoutError("stuck-3"),
        _make_resp(tool_calls=[_exec_commands_call()]),
        _make_resp(tool_calls=[_task_complete_call(0)]),
        _make_resp(tool_calls=[_task_complete_call(1)]),
    ]
    with patch.object(kira_loop, "call_llm_with_tools", side_effect=responses):
        result = agent.run("eventually moves on")

    assert result.exit_reason == "task_complete"
    assert session.current_url != start_url
    history = session.history()
    assert len(history) == 1
    assert history[0]["reason"] == "BlockTimeoutError"


def test_sticky_budget_resets_after_success(tmp_path, fake_shell):
    """A successful call mid-trajectory should reset the sticky budget so
    a later transient timeout can re-absorb without forcing failover."""
    from omnicoding.agents.kira.endpoint_pool import EndpointSession, parse_endpoints
    from omnicoding.agents.kira.llm import BlockTimeoutError

    pool = parse_endpoints("http://a/v1=1,http://b/v1=1")
    session = EndpointSession(pool, idx=0)
    start_url = session.current_url
    agent = _build_agent(tmp_path, endpoint_session=session, max_sticky_retries=1)

    responses = [
        BlockTimeoutError("t1"),                         # consumes budget → 0
        _make_resp(tool_calls=[_exec_commands_call()]),  # success: budget reset → 1
        BlockTimeoutError("t2"),                         # consumes budget → 0 (sticky retry)
        _make_resp(tool_calls=[_task_complete_call(0)]),
        _make_resp(tool_calls=[_task_complete_call(1)]),
    ]
    with patch.object(kira_loop, "call_llm_with_tools", side_effect=responses):
        result = agent.run("two transient hiccups")

    assert result.exit_reason == "task_complete"
    # Both timeouts absorbed in-place — never failed over.
    assert session.current_url == start_url
    assert session.history() == []


# ---------- 429 / RateLimit → failover (acct quota exhausted) -------------


def test_failover_on_rate_limit_error(tmp_path, fake_shell):
    """A 429 from chatgpt.com (account weekly quota exhausted) propagates
    to litellm as RateLimitError → kira must rotate to the next slot
    instead of dying. Without this, all items pinned to the over-quota
    account fail mid-batch."""
    from omnicoding.agents.kira.endpoint_pool import EndpointSession, parse_endpoints
    import litellm.exceptions as le

    pool = parse_endpoints("acct-A=1,acct-B=1")
    session = EndpointSession(pool, idx=0)
    agent = _build_agent(tmp_path, endpoint_session=session)

    responses = [
        le.RateLimitError(message="429 quota_exceeded", llm_provider="openai", model="gpt-5.5"),
        _make_resp(tool_calls=[_exec_commands_call()]),
        _make_resp(tool_calls=[_task_complete_call(0)]),
        _make_resp(tool_calls=[_task_complete_call(1)]),
    ]
    with patch.object(kira_loop, "call_llm_with_tools", side_effect=responses):
        result = agent.run("acct-A is over quota")

    assert result.exit_reason == "task_complete"
    history = session.history()
    assert len(history) == 1
    assert history[0]["reason"] == "RateLimitError"


def test_no_session_block_timeout_uses_sticky_then_raises(tmp_path, fake_shell):
    """Single-endpoint mode: sticky retries (no failover available) then
    the error bubbles up. Verifies sticky retry doesn't depend on a pool."""
    from omnicoding.agents.kira.llm import BlockTimeoutError

    agent = _build_agent(tmp_path, max_sticky_retries=2)
    err = BlockTimeoutError("perma stuck")
    with patch.object(kira_loop, "call_llm_with_tools",
                      side_effect=[err, err, err]):
        result = agent.run("hopeless single endpoint")
    assert result.exit_reason == "error"
    assert result.error and "BlockTimeoutError" in result.error


# ---------- BUG-X1: completion checklist explicitly offers two paths ------
#
# Failure mode (BUG-X1): on long videozerobench items the model would
# call task_complete with the answer in prose only — no
# ``<answer>...</answer>`` wrapper anywhere in the trajectory. The
# pre-fix completion checklist only suggested one recovery path
# (``echo '<answer>X</answer>'`` via execute_commands), so when the
# model's shell was hung from large-file ffmpeg/whisper operations it
# could not comply and gave up. Fix: surface a second path — write the
# wrapper directly in the next assistant turn's content alongside
# task_complete — and tell the model the grader does not read
# tool-call arguments. These tests pin the new wording so a refactor
# does not silently revert the message.


def test_system_prompt_defers_to_final_answer_protocol():
    """Round-17.11 (2026-04-30): the two-path answer contract moved
    out of kira-core SYSTEM_PROMPT into the cross-benchmark
    ``FINAL_ANSWER_PROTOCOL`` const in ``common/spec.py``. kira-core
    now only carves out the "final-answer turn may be plain text"
    exception so the strict tool-call-every-turn rule does not
    contradict FINAL_ANSWER_PROTOCOL.

    The actual two-path detail (plain text preferred, echo as
    backstop) is verified cross-bench in
    ``benchmarks/tests/test_unified_prompt.py:test_system_prefix_carries_final_answer_protocol``.
    """
    from omnicoding.agents.kira.tools import SYSTEM_PROMPT
    # kira-core must reference the protocol so the carve-out for
    # plain-text final answer is discoverable from role=system.
    assert "Final-answer protocol" in SYSTEM_PROMPT, (
        "kira-core must point at FINAL_ANSWER_PROTOCOL so the carve-out "
        "for plain-text final answer is discoverable from the start of "
        "role=system"
    )
    # The exception must be explicit — otherwise the "every turn must
    # call a tool" rule wins and the model never tries plain text.
    assert "exception" in SYSTEM_PROMPT.lower(), (
        "kira-core must explicitly call out the final-answer-turn "
        "exception to the every-turn tool-call rule"
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
