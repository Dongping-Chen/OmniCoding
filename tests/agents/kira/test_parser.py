"""Pin the parser's Qwen-shape compatibility branches.

Each test pins one observed Qwen3.6 quirk so we know which branch is
exercised. If a branch never fires in production after a smoke run we
delete the corresponding fallback per dongping-rule (no defensive code
that's never proven necessary).
"""

from __future__ import annotations

import json

import pytest

from omnicoding.agents.kira.parser import (
    Command,
    ImageReadRequest,
    parse_tool_calls,
    _decode_arguments,
    _normalize_commands_field,
)


def _make_call(name: str, arguments) -> dict:
    if not isinstance(arguments, str):
        arguments = json.dumps(arguments)
    return {
        "id": f"call_{name}",
        "type": "function",
        "function": {"name": name, "arguments": arguments},
    }


# ---------- _decode_arguments ----------------------------------------

def test_decode_arguments_dict_passthrough():
    assert _decode_arguments({"a": 1}) == {"a": 1}


def test_decode_arguments_clean_json_string():
    assert _decode_arguments('{"a": 1}') == {"a": 1}


def test_decode_arguments_qwen_trailing_extra_brace():
    """Qwen3.6 sometimes appends a stray ``}}`` to the args string."""
    assert _decode_arguments('{"a": 1}}') == {"a": 1}


def test_decode_arguments_empty_string_returns_empty_dict():
    assert _decode_arguments("") == {}


def test_decode_arguments_unparseable_returns_empty():
    assert _decode_arguments("not-json-at-all") == {}


def test_decode_arguments_truncated_string_repairs():
    """Whole arguments dict cut off mid-string — the bracket-balance
    repair recovers it (used to slip through ``_decode_arguments`` and
    silently lose the call). This is the same family of failure as
    truncated commands, one level up the JSON tree."""
    raw = '{"analysis": "look", "plan": "list", "commands": [{"keystrokes": "ls'
    out = _decode_arguments(raw)
    assert out == {
        "analysis": "look",
        "plan": "list",
        "commands": [{"keystrokes": "ls"}],
    }


def test_decode_arguments_unparseable_appends_warning():
    warnings: list[str] = []
    assert _decode_arguments("definitely not json {", warnings=warnings) == {}
    assert any("not valid JSON" in w for w in warnings)


# ---------- _normalize_commands_field --------------------------------

def test_normalize_commands_already_list_of_dicts():
    out = _normalize_commands_field([{"keystrokes": "ls", "duration": 1}])
    assert out == [{"keystrokes": "ls", "duration": 1}]


def test_normalize_commands_json_string():
    """qwen3_coder treats <parameter=commands> body as a string; harness
    must JSON-decode it back into a list."""
    raw = '[{"keystrokes": "pwd", "duration": 0.1}]'
    assert _normalize_commands_field(raw) == [{"keystrokes": "pwd", "duration": 0.1}]


def test_normalize_commands_single_dict_wrapped():
    """Qwen sometimes drops the array wrapper for a single command."""
    out = _normalize_commands_field({"keystrokes": "echo hi"})
    assert out == [{"keystrokes": "echo hi"}]


def test_normalize_commands_empty_string():
    assert _normalize_commands_field("") == []


def test_normalize_commands_unparseable_string():
    assert _normalize_commands_field("not-json") == []


def test_normalize_commands_unparseable_string_appends_warning():
    """When the harness can't recover the JSON, the model must learn it
    so it stops re-emitting the same broken call. Without this, kira
    used to silently send empty terminal output back, which Qwen3.6
    interpreted as 'command ran with no output' and looped forever
    (round-N kira/omnigaia smoke surfaced this — 10× same warning,
    zero recovery)."""
    warnings: list[str] = []
    assert _normalize_commands_field("not-json", warnings=warnings) == []
    assert warnings, "expected a model-facing warning on parse failure"
    assert any("not valid JSON" in w for w in warnings)


def test_normalize_commands_repairs_truncated_array():
    """Qwen sometimes emits an unfinished JSON array (closing brackets
    chopped off — finish_reason=length or sglang qwen3_coder XML-arg
    extraction trimming the tail). The parser repairs it so the loop
    can still execute the commands the model clearly intended, and
    warns the model so it tightens up next turn."""
    warnings: list[str] = []
    raw = '[{"keystrokes": "ls inputs/", "duration": 1}, {"keystrokes": "echo hi'
    out = _normalize_commands_field(raw, warnings=warnings)
    assert out == [
        {"keystrokes": "ls inputs/", "duration": 1},
        {"keystrokes": "echo hi"},
    ]
    assert any("auto-repaired" in w or "truncated" in w for w in warnings)


def test_normalize_commands_repairs_single_dict_truncated():
    """Same repair on the single-dict-no-array shape (also seen in
    kira/omnigaia smoke: ``{"keystrokes": "ffprobe ... | head -80``)."""
    raw = '{"keystrokes": "ffprobe inputs/x.mp4 | head -80'
    out = _normalize_commands_field(raw)
    assert out == [{"keystrokes": "ffprobe inputs/x.mp4 | head -80"}]


def test_normalize_commands_unsupported_type_warns():
    warnings: list[str] = []
    out = _normalize_commands_field(42, warnings=warnings)
    assert out == []
    assert any("must be a JSON array" in w for w in warnings)


def test_normalize_commands_string_with_literal_newlines():
    """Qwen3.6 emits multi-line bash inside <parameter=commands> bodies
    without escaping the newlines, so the resulting JSON string has
    literal newlines inside string values. ``strict=False`` lets the
    parser through — without it, the loop dropped every multi-line
    command (round 10 smoke surfaced this; previously every
    ``ffprobe ... 2>/dev/null`` and every multi-cmd batch failed)."""
    raw = '[{"keystrokes": "echo a\nb", "duration": 1}]'
    out = _normalize_commands_field(raw)
    assert out == [{"keystrokes": "echo a\nb", "duration": 1}]


def test_decode_arguments_with_literal_newline_inside_string():
    raw = '{"keystrokes": "line1\nline2"}'
    assert _decode_arguments(raw) == {"keystrokes": "line1\nline2"}


# ---------- parse_tool_calls happy paths -----------------------------

def test_parse_execute_commands_strict_openai_shape():
    args = {"analysis": "look around", "plan": "list dir", "commands": [
        {"keystrokes": "ls -la", "duration": 0.1},
    ]}
    parsed = parse_tool_calls([_make_call("execute_commands", args)])
    assert parsed.commands == [Command(keystrokes="ls -la", duration=0.1)]
    assert parsed.analysis == "look around"
    assert parsed.plan == "list dir"
    assert parsed.task_complete is False
    assert parsed.image_read is None
    assert parsed.warnings == []


def test_parse_execute_commands_qwen_commands_as_json_string():
    """The qwen3_coder XML form often serializes the array as a string."""
    qwen_args_str = json.dumps({
        "analysis": "qwen check",
        "plan": "echo 1",
        "commands": json.dumps([{"keystrokes": "echo 1", "duration": 0.5}]),
    })
    parsed = parse_tool_calls([_make_call("execute_commands", qwen_args_str)])
    assert parsed.commands == [Command(keystrokes="echo 1", duration=0.5)]


def test_parse_execute_commands_inlined_top_level_keystrokes():
    """Qwen sometimes emits keystrokes/duration at the top level instead
    of nested inside a commands array."""
    args = {"analysis": "x", "plan": "y", "keystrokes": "pwd", "duration": 0.1}
    parsed = parse_tool_calls([_make_call("execute_commands", args)])
    assert parsed.commands == [Command(keystrokes="pwd", duration=0.1)]


def test_parse_execute_commands_missing_duration_defaults():
    args = {"analysis": "", "plan": "", "commands": [{"keystrokes": "ls"}]}
    parsed = parse_tool_calls([_make_call("execute_commands", args)])
    assert parsed.commands == [Command(keystrokes="ls", duration=1.0)]


def test_parse_execute_commands_duration_caps_at_60():
    args = {"analysis": "", "plan": "", "commands": [{"keystrokes": "sleep 999", "duration": 9999}]}
    parsed = parse_tool_calls([_make_call("execute_commands", args)])
    assert parsed.commands[0].duration == 60.0


def test_parse_execute_commands_skips_entry_missing_keystrokes():
    args = {"analysis": "", "plan": "", "commands": [{"duration": 0.1}, {"keystrokes": "ok"}]}
    parsed = parse_tool_calls([_make_call("execute_commands", args)])
    assert parsed.commands == [Command(keystrokes="ok", duration=1.0)]


def test_parse_image_read_happy_path():
    args = {"file_path": "/tmp/foo.png", "image_read_instruction": "describe scene"}
    parsed = parse_tool_calls([_make_call("image_read", args)])
    assert parsed.image_read == ImageReadRequest(
        file_path="/tmp/foo.png", image_read_instruction="describe scene",
    )


def test_parse_image_read_missing_field_warns():
    args = {"file_path": "/tmp/foo.png"}
    parsed = parse_tool_calls([_make_call("image_read", args)])
    assert parsed.image_read is None
    assert any("image_read" in w for w in parsed.warnings)


def test_parse_task_complete_signals_completion():
    parsed = parse_tool_calls([_make_call("task_complete", {})])
    assert parsed.task_complete is True


def test_parse_unknown_tool_warns():
    parsed = parse_tool_calls([_make_call("eat_lunch", {})])
    assert any("eat_lunch" in w or "Unknown tool" in w for w in parsed.warnings)


def test_parse_no_tool_calls_warns():
    parsed = parse_tool_calls(None)
    assert parsed.warnings  # the "no tool calls" message


def test_parse_multiple_image_read_warns_on_extras():
    args = {"file_path": "/a.png", "image_read_instruction": "x"}
    parsed = parse_tool_calls([
        _make_call("image_read", args),
        _make_call("image_read", {**args, "file_path": "/b.png"}),
    ])
    assert parsed.image_read.file_path == "/a.png"
    assert any("Multiple image_read" in w for w in parsed.warnings)


def test_parse_multiple_execute_commands_concatenated():
    """Two ``execute_commands`` calls in one assistant turn — Qwen does
    this rarely, but the parser must concat rather than drop the second."""
    parsed = parse_tool_calls([
        _make_call("execute_commands", {
            "analysis": "first", "plan": "p1",
            "commands": [{"keystrokes": "echo a"}],
        }),
        _make_call("execute_commands", {
            "analysis": "second", "plan": "p2",
            "commands": [{"keystrokes": "echo b"}],
        }),
    ])
    assert [c.keystrokes for c in parsed.commands] == ["echo a", "echo b"]
    # First-wins semantics for analysis/plan keep the trajectory log clean.
    assert parsed.analysis == "first"
    assert parsed.plan == "p1"


# ---------- defensive branches ---------------------------------------

def test_decode_arguments_none_returns_empty():
    assert _decode_arguments(None) == {}


def test_decode_arguments_list_input_roundtrips_to_empty_dict():
    """litellm should never hand us a list as ``arguments``, but if it
    ever does (custom backend regression) we fall through to {}."""
    out = _decode_arguments([1, 2, 3])
    assert out == {}


def test_decode_arguments_int_input_returns_empty_dict():
    assert _decode_arguments(42) == {}


def test_decode_arguments_decoded_is_array_not_dict_warns():
    """Top-level JSON array is valid JSON but not what we want; warn the
    model so it re-emits an object."""
    warnings: list[str] = []
    out = _decode_arguments("[1,2,3]", warnings=warnings)
    assert out == {}
    assert any("not a JSON object" in w for w in warnings)


def test_repair_truncated_json_empty_returns_none():
    from omnicoding.agents.kira.parser import _repair_truncated_json
    assert _repair_truncated_json("") is None


def test_repair_truncated_json_balanced_invalid_returns_none():
    """Balanced brackets but still invalid JSON (e.g., bare word) — the
    repair has nothing to add, so it returns None."""
    from omnicoding.agents.kira.parser import _repair_truncated_json
    assert _repair_truncated_json("not-json") is None


def test_repair_truncated_json_mismatched_close_returns_none():
    from omnicoding.agents.kira.parser import _repair_truncated_json
    # Close-brace without matching open
    assert _repair_truncated_json("}") is None


def test_repair_truncated_json_trailing_comma_chopped():
    from omnicoding.agents.kira.parser import _repair_truncated_json
    out = _repair_truncated_json('[{"a": 1},')
    assert out == [{"a": 1}]


def test_repair_truncated_json_open_value_after_colon_returns_none():
    from omnicoding.agents.kira.parser import _repair_truncated_json
    # ``{"a":`` — value missing; can't safely close.
    assert _repair_truncated_json('{"a":') is None


def test_repair_truncated_json_handles_escape_inside_string():
    """``\\"`` inside a string value must NOT terminate the string early;
    the escape branch in the state machine covers this."""
    from omnicoding.agents.kira.parser import _repair_truncated_json
    out = _repair_truncated_json(r'{"k": "say \"hi\"')
    assert out == {"k": 'say "hi"'}


# ---------- _coerce_duration -----------------------------------------

def test_coerce_duration_bool_returns_default():
    """``True``/``False`` are subtypes of int — must NOT be silently
    treated as 1.0/0.0. Pinned because the isinstance check order is
    fragile."""
    from omnicoding.agents.kira.parser import _coerce_duration, DEFAULT_DURATION
    assert _coerce_duration(True) == DEFAULT_DURATION
    assert _coerce_duration(False) == DEFAULT_DURATION


def test_coerce_duration_string_numeric():
    """Qwen sometimes hands back ``duration`` as a string."""
    from omnicoding.agents.kira.parser import _coerce_duration
    assert _coerce_duration("2.5") == 2.5


def test_coerce_duration_string_unparseable_returns_default():
    from omnicoding.agents.kira.parser import _coerce_duration, DEFAULT_DURATION
    assert _coerce_duration("foo") == DEFAULT_DURATION


def test_coerce_duration_zero_or_negative_uses_default():
    from omnicoding.agents.kira.parser import _coerce_duration, DEFAULT_DURATION
    assert _coerce_duration(0) == DEFAULT_DURATION
    assert _coerce_duration(-5) == DEFAULT_DURATION


def test_coerce_duration_caps_at_60():
    from omnicoding.agents.kira.parser import _coerce_duration
    assert _coerce_duration(9999) == 60.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
