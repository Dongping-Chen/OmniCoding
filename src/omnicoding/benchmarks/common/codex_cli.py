"""Helpers for parsing Codex CLI stream output."""

from __future__ import annotations

import json
from typing import Any


def parse_codex_jsonl(stdout_text: str) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    for line in stdout_text.splitlines():
        line = line.strip()
        if not line or not line.startswith("{"):
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict):
            events.append(obj)
    return events


def count_codex_tool_calls(events: list[dict[str, Any]]) -> int:
    count = 0
    seen_call_ids: set[Any] = set()
    for event in events:
        event_type = str(event.get("type", ""))
        item = event.get("item") if isinstance(event.get("item"), dict) else {}
        item_type = str(item.get("type") or item.get("item_type") or "")
        if event_type == "item.started" and item_type == "command_execution":
            call_id = item.get("id") or f"{event_type}:{count}"
            if call_id not in seen_call_ids:
                seen_call_ids.add(call_id)
                count += 1
                continue
        if any(token in event_type for token in ("exec_command", "apply_patch", "web_search", "tool_call", "mcp_tool_call")):
            call_id = event.get("call_id") or event.get("id") or f"{event_type}:{count}"
            if call_id not in seen_call_ids:
                seen_call_ids.add(call_id)
                count += 1
                continue
        payload_text = json.dumps(event, ensure_ascii=False)
        if '"recipient_name":"functions.exec_command"' in payload_text or '"recipient_name":"functions.apply_patch"' in payload_text:
            count += 1
    return count


def extract_codex_last_assistant_text(events: list[dict[str, Any]]) -> str:
    """Return concatenated text of the last `agent_message`/`assistant_message`
    item from a Codex JSONL stream, or the empty string if none exists.

    When Codex's `--output-last-message <file>` produces an empty file (the
    model never emitted a final assistant turn before the harness gave up
    or hit a token cap), drivers used to fall back to the entire stdout
    JSONL stream — turning the prediction extractor's input into raw event
    JSON, which neither the OmniGAIA `<answer>...</answer>` regex nor
    SocialOmni's MCQ normalizer can read. This helper is the right
    intermediate fallback: still inside Codex's own conversation, but
    structurally selecting the model's last visible message.
    """
    last_text = ""
    for event in events:
        if str(event.get("type", "")) != "item.completed":
            continue
        item = event.get("item")
        if not isinstance(item, dict):
            continue
        item_type = str(item.get("item_type") or item.get("type") or "")
        if item_type not in {"assistant_message", "agent_message"}:
            continue
        text = item.get("text")
        if isinstance(text, str) and text.strip():
            last_text = text
    return last_text


def extract_codex_last_reasoning_text(events: list[dict[str, Any]]) -> str:
    """Return the LAST ``reasoning`` item's text from a Codex JSONL stream.

    Qwen3.6 with thinking ON often closes its turn with the final
    answer (``<answer>X</answer>`` or "Answer: X" prose) inside a
    ``<think>...</think>`` block. The proxy surfaces that as a
    Responses-API ``reasoning`` output_item rather than an
    ``assistant_message``; ``--output-last-message`` therefore writes
    an empty file and ``extract_codex_last_assistant_text`` returns
    "". This is the secondary fallback drivers should consult before
    dumping the entire stdout: the model committed an answer, just in
    the wrong slot.
    """
    last_text = ""
    for event in events:
        if str(event.get("type", "")) != "item.completed":
            continue
        item = event.get("item")
        if not isinstance(item, dict):
            continue
        item_type = str(item.get("item_type") or item.get("type") or "")
        if item_type != "reasoning":
            continue
        text = item.get("text")
        if isinstance(text, str) and text.strip():
            last_text = text
            continue
        # Some Codex builds nest reasoning text under content[].text.
        content = item.get("content")
        if isinstance(content, list):
            chunks = []
            for c in content:
                if isinstance(c, dict):
                    t = c.get("text")
                    if isinstance(t, str) and t.strip():
                        chunks.append(t)
            if chunks:
                last_text = "\n".join(chunks)
    return last_text


def extract_codex_token_usage(events: list[dict[str, Any]]) -> dict[str, int]:
    usage_totals = {
        "input_tokens": 0,
        "output_tokens": 0,
        "cached_input_tokens": 0,
    }
    for event in events:
        if str(event.get("type", "")) != "turn.completed":
            continue
        usage = event.get("usage", {})
        if not isinstance(usage, dict):
            continue
        for key in usage_totals:
            value = usage.get(key, 0)
            if isinstance(value, (int, float)):
                usage_totals[key] += int(value)
    usage_totals["total_tokens"] = usage_totals["input_tokens"] + usage_totals["output_tokens"]
    return usage_totals


def summarize_codex_event_line(line: str) -> str:
    try:
        event = json.loads(line)
    except json.JSONDecodeError:
        return line

    event_type = event.get("type", "unknown")
    if event_type == "turn.started":
        return "turn.started"
    if event_type == "turn.completed":
        usage = event.get("usage", {})
        return f"turn.completed input={usage.get('input_tokens', '?')} output={usage.get('output_tokens', '?')}"
    if event_type == "turn.failed":
        err = event.get("error", {})
        return f"turn.failed {err.get('message', '')}".strip()
    if event_type == "thread.started":
        return f"thread.started {event.get('thread_id', '')}".strip()
    if event_type == "error":
        return f"error {event.get('message', '')}".strip()

    item = event.get("item", {})
    item_type = item.get("item_type", item.get("type", "")) if isinstance(item, dict) else ""
    if event_type == "item.started":
        if item_type == "command_execution":
            return f"command.started {item.get('command', '')}".strip()
        return f"{item_type}.started".strip(".")
    if event_type == "item.completed":
        if item_type in {"assistant_message", "agent_message"}:
            text = (item.get("text", "") or "").replace("\n", " ").strip()
            return f"assistant {text[:200]}".strip()
        if item_type == "reasoning":
            text = (item.get("text", "") or "").replace("\n", " ").strip()
            return f"reasoning {text[:200]}".strip()
        if item_type == "command_execution":
            cmd = item.get("command", "")
            exit_code = item.get("exit_code", "?")
            return f"command.completed exit={exit_code} {cmd}".strip()
        if item_type == "web_search":
            return f"web_search.completed {json.dumps(item, ensure_ascii=False)[:200]}"
        return f"{item_type}.completed".strip(".")
    return f"{event_type} {json.dumps(event, ensure_ascii=False)[:200]}".strip()
