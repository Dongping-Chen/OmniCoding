"""Trajectory + message serializers.

Pulled out of ``kira.loop`` to keep that file under the 800-LOC limit.
Pure functions: take dicts/dataclasses, return JSON-serialisable shapes
the wide-smoke analyzer / SFT data prep consume.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from omnicoding.agents.kira.loop import StepRecord


def trajectory_to_dicts(traj: list["StepRecord"]) -> list[dict[str, Any]]:
    return [
        {
            "step": r.step,
            "analysis": r.analysis,
            "plan": r.plan,
            "n_commands": r.n_commands,
            "is_task_complete": r.is_task_complete,
            "is_image_read": r.is_image_read,
            "output_chars": r.output_chars,
            "cached_tokens": r.cached_tokens,
            "reasoning_tokens": r.reasoning_tokens,
            "prompt_tokens": r.prompt_tokens,
            "completion_tokens": r.completion_tokens,
        }
        for r in traj
    ]


def messages_preview(messages: list[dict[str, Any]], n_tail: int = 6) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for m in messages[-n_tail:]:
        cleaned = {
            k: v for k, v in m.items()
            if k in {"role", "content", "tool_call_id", "tool_calls", "reasoning_content"}
        }
        if isinstance(cleaned.get("content"), str):
            cleaned["content"] = cleaned["content"][:5000]
        if isinstance(cleaned.get("reasoning_content"), str):
            cleaned["reasoning_content"] = cleaned["reasoning_content"][:5000]
        if isinstance(cleaned.get("tool_calls"), list):
            cleaned["tool_calls"] = [_preview_tool_call(tc) for tc in cleaned["tool_calls"]]
        out.append(cleaned)
    return out


def _preview_tool_call(tc: dict[str, Any]) -> dict[str, Any]:
    fn = tc.get("function") or {}
    args = fn.get("arguments")
    return {
        "id": tc.get("id"),
        "function": {
            "name": fn.get("name"),
            "arguments": json.dumps(args)[:800] if args else "",
        },
    }
