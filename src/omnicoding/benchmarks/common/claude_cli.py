"""Helpers for parsing Claude Code CLI output."""

from __future__ import annotations

import json
import re
from collections.abc import Iterable
from typing import Any

CLAUDE_BUILTIN_TOOL_NAMES = frozenset(
    {
        "AskUserQuestion",
        "Bash",
        "CronCreate",
        "CronDelete",
        "CronList",
        "Edit",
        "EnterPlanMode",
        "EnterWorktree",
        "ExitPlanMode",
        "ExitWorktree",
        "Glob",
        "Grep",
        "ListMcpResourcesTool",
        "NotebookEdit",
        "Read",
        "ReadMcpResourceTool",
        "ScheduleWakeup",
        "Skill",
        "Task",
        "TaskOutput",
        "TaskStop",
        "TodoWrite",
        "WebFetch",
        "WebSearch",
        "Write",
    }
)
CLAUDE_ISOLATED_DISABLED_TOOLS = frozenset({"ListMcpResourcesTool", "ReadMcpResourceTool", "Skill"})


def _normalize_tool_name(tool: str) -> str:
    return re.split(r"[(,\s]", str(tool).strip(), maxsplit=1)[0].strip()


def build_claude_isolation_flags(
    allowed_tools: Iterable[str] | None,
    *,
    bare: bool = False,
) -> list[str]:
    flags: list[str] = []
    if bare:
        flags.append("--bare")
    flags.extend(["--disable-slash-commands", "--strict-mcp-config"])

    tools: list[str] = []
    seen: set[str] = set()
    for tool in allowed_tools or []:
        name = _normalize_tool_name(tool)
        if name in seen:
            continue
        # `mcp__<server>__<tool>` is how Claude Code namespaces server-provided
        # tools; they are intentionally absent from CLAUDE_BUILTIN_TOOL_NAMES.
        is_mcp = name.startswith("mcp__")
        if not is_mcp and (
            name not in CLAUDE_BUILTIN_TOOL_NAMES
            or name in CLAUDE_ISOLATED_DISABLED_TOOLS
        ):
            continue
        seen.add(name)
        tools.append(name)
    if tools:
        flags.extend(["--tools", ",".join(tools)])
    return flags


def parse_claude_jsonl(stdout_text: str) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    for line in stdout_text.splitlines():
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict):
            events.append(obj)
    return events


def parse_claude_stream_json(stdout_text: str) -> tuple[list[dict[str, Any]], dict[str, Any] | None, int]:
    events = parse_claude_jsonl(stdout_text)
    result_event: dict[str, Any] | None = None
    tool_call_count = 0
    for event in events:
        if event.get("type") == "result":
            result_event = event
        if event.get("type") != "assistant":
            continue
        message = event.get("message", {})
        content = message.get("content", []) if isinstance(message, dict) else []
        if not isinstance(content, list):
            continue
        for block in content:
            if isinstance(block, dict) and block.get("type") == "tool_use":
                tool_call_count += 1
    return events, result_event, tool_call_count


def extract_claude_final_result(events: list[dict[str, Any]], stdout_text: str, stderr_text: str) -> str:
    """Recover the model's final visible answer from a Claude Code stream.

    Recovery chain:
      1. Last ``result`` event's ``result`` field — this is the
         canonical "final assistant message" from the CLI.
      2. Last ``assistant`` event's ``text`` content blocks.
      3. Last ``assistant`` event's ``thinking`` content blocks. Qwen3.6
         + thinking ON often closes the turn with the answer
         (``<answer>X</answer>`` or "Answer: X") inside the
         ``<think>...</think>`` block; the proxy translates that into
         an Anthropic ``thinking`` block, and Claude Code never
         re-emits it as ``text``. Without this fallback, the
         ``result`` event is empty, the text-block scan returns "",
         and the driver dumps raw stdout into the prediction
         extractor — which then either fails or latches onto random
         event-JSON bytes.
      4. Raw combined stdout/stderr (last resort).
    """
    for event in reversed(events):
        if event.get("type") == "result":
            result = event.get("result")
            if isinstance(result, str) and result.strip():
                return result.strip()

    last_thinking = ""
    for event in reversed(events):
        if event.get("type") != "assistant":
            continue
        message = event.get("message")
        if not isinstance(message, dict):
            continue
        content = message.get("content")
        if not isinstance(content, list):
            continue
        text_blocks = [
            block.get("text", "")
            for block in content
            if isinstance(block, dict) and block.get("type") == "text"
        ]
        joined = "\n".join(part for part in text_blocks if str(part).strip()).strip()
        if joined:
            return joined
        if not last_thinking:
            thinking_blocks = [
                block.get("thinking", "")
                for block in content
                if isinstance(block, dict) and block.get("type") == "thinking"
            ]
            joined_thinking = "\n".join(part for part in thinking_blocks if str(part).strip()).strip()
            if joined_thinking:
                last_thinking = joined_thinking

    if last_thinking:
        return last_thinking

    combined = "\n".join(part for part in [stdout_text.strip(), stderr_text.strip()] if part).strip()
    return combined or "Error: Claude Code produced no final message."


def walk_claude_full_transcript(events: list[dict[str, Any]]) -> str:
    """Concat every assistant ``text`` + ``thinking`` block across the
    event log into one debug-friendly transcript. Used for
    ``raw_model_output`` so the wide-smoke audit can scan the model's
    reasoning across long runs (round-12 BUG-C3 fix).

    Distinct from ``extract_claude_final_result`` which only returns
    the LAST visible turn — that's the right input for the spec
    extractor (final answer is at the end), but it loses the
    intermediate reasoning that's often where you'd find the model
    "narrating the answer in prose without the wrapper" pattern
    BUG-X1 chases.
    """
    parts: list[str] = []
    for event in events:
        if event.get("type") != "assistant":
            continue
        message = event.get("message")
        if not isinstance(message, dict):
            continue
        content = message.get("content")
        if not isinstance(content, list):
            continue
        for block in content:
            if not isinstance(block, dict):
                continue
            btype = block.get("type")
            if btype == "text":
                txt = block.get("text") or ""
                if str(txt).strip():
                    parts.append(str(txt))
            elif btype == "thinking":
                thk = block.get("thinking") or ""
                if str(thk).strip():
                    parts.append(str(thk))
    return "\n\n".join(parts)


def sum_claude_tokens(events: list[dict[str, Any]]) -> tuple[int, int]:
    """Sum prompt + completion tokens across every assistant turn in
    a Claude Code ``--output-format stream-json`` event log.

    Round-12 BUG-C1 fix: pre-fix the runner left ``prompt_tokens`` and
    ``completion_tokens`` as None on every claude row, so wide-smoke
    cost analysis was missing one harness entirely.

    Each ``type=assistant`` event carries a ``message.usage`` dict with
    ``input_tokens`` and ``output_tokens`` (the Claude SDK schema). We
    add ``cache_creation_input_tokens`` to the input side and
    ``cache_read_input_tokens`` is *also* counted as input (it's still
    billed, just at a discount — for total-tokens accounting we add
    it). The ``result`` event sometimes also has a top-level ``usage``;
    if present, prefer it (one authoritative value) over the per-turn
    sum to handle resumes that re-run earlier turns. Returns
    ``(prompt_tokens, completion_tokens)``; either may be 0 if the
    Claude session emitted no usage data (e.g. error before any LLM
    call).
    """
    # Prefer the result event's authoritative total when present.
    for event in reversed(events):
        if event.get("type") != "result":
            continue
        usage = event.get("usage")
        if isinstance(usage, dict):
            inp = (
                int(usage.get("input_tokens") or 0)
                + int(usage.get("cache_creation_input_tokens") or 0)
                + int(usage.get("cache_read_input_tokens") or 0)
            )
            out = int(usage.get("output_tokens") or 0)
            if inp or out:
                return inp, out
        break  # only the LAST result event is authoritative
    prompt_tokens = 0
    completion_tokens = 0
    for event in events:
        if event.get("type") != "assistant":
            continue
        message = event.get("message")
        if not isinstance(message, dict):
            continue
        usage = message.get("usage")
        if not isinstance(usage, dict):
            continue
        prompt_tokens += (
            int(usage.get("input_tokens") or 0)
            + int(usage.get("cache_creation_input_tokens") or 0)
            + int(usage.get("cache_read_input_tokens") or 0)
        )
        completion_tokens += int(usage.get("output_tokens") or 0)
    return prompt_tokens, completion_tokens


def count_claude_tool_calls(events: list[dict[str, Any]]) -> int:
    count = 0
    for event in events:
        if event.get("type") != "assistant":
            continue
        message = event.get("message")
        if not isinstance(message, dict):
            continue
        content = message.get("content")
        if not isinstance(content, list):
            continue
        for block in content:
            if isinstance(block, dict) and block.get("type") == "tool_use":
                count += 1
    return count


def count_claude_tool_calls_from_json(parsed: dict[str, Any]) -> int:
    count = 0
    messages = parsed.get("messages", [])
    for message in messages:
        if not isinstance(message, dict):
            continue
        content = message.get("content", [])
        if not isinstance(content, list):
            continue
        for block in content:
            if isinstance(block, dict) and block.get("type") == "tool_use":
                count += 1
    return count


def extract_text_from_claude_json(parsed: dict[str, Any]) -> str:
    result = parsed.get("result")
    if result is not None and str(result).strip():
        return str(result)
    messages = parsed.get("messages", [])
    last_thinking = ""
    for message in reversed(messages):
        if not isinstance(message, dict) or message.get("role") != "assistant":
            continue
        content = message.get("content", "")
        if isinstance(content, str):
            if content.strip():
                return content
            continue
        if isinstance(content, list):
            text_parts = [
                block.get("text", "")
                for block in content
                if isinstance(block, dict) and block.get("type") == "text"
            ]
            joined_text = "\n".join(part for part in text_parts if str(part).strip()).strip()
            if joined_text:
                return joined_text
            # Qwen3.6 + thinking ON commonly puts the final answer
            # inside the ``<think>`` block; the proxy emits that as a
            # Claude ``thinking`` content block. Save the most-recent
            # one as a last-ditch fallback so the answer survives.
            if not last_thinking:
                thinking_parts = [
                    block.get("thinking", "")
                    for block in content
                    if isinstance(block, dict) and block.get("type") == "thinking"
                ]
                joined_thinking = "\n".join(part for part in thinking_parts if str(part).strip()).strip()
                if joined_thinking:
                    last_thinking = joined_thinking
    return last_thinking or ""


def summarize_claude_stream_line(line: str) -> str:
    text = line.strip()
    if not text or not text.startswith("{"):
        return text
    try:
        event = json.loads(text)
    except json.JSONDecodeError:
        return text

    event_type = event.get("type", "unknown")
    if event_type == "result":
        result_text = str(event.get("result", ""))[:200]
        return f"result: {result_text}"
    if event_type == "assistant":
        message = event.get("message", {})
        content = message.get("content", []) if isinstance(message, dict) else []
        if isinstance(content, list):
            for block in content:
                if isinstance(block, dict) and block.get("type") == "text":
                    return f"assistant: {str(block.get('text', ''))[:200]}"
                if isinstance(block, dict) and block.get("type") == "tool_use":
                    tool_input = json.dumps(block.get("input", {}), ensure_ascii=False)[:100]
                    return f"tool_use: {block.get('name', '?')}({tool_input})"
        return "assistant message"
    return f"{event_type}: {text[:200]}"
