"""Conversation summarization for context-overflow recovery.

When the agent's chat history grows past the model's context window,
we replace it with a single user message containing:

  1. The original task instruction (so the model still knows what to
     do).
  2. An LLM-written progress summary (what's been tried, what worked,
     current workspace state).
  3. A "continue from here" instruction.

This is a deliberately simpler design than upstream KIRA, which uses
harbor's full subagent infrastructure. We don't track subagent
trajectories or rollouts — we just compress, retry once, and let the
loop continue. If the compressed conversation ALSO overflows context,
the loop bows out (rare; the summary is bounded to ~500 tokens).

The summarizer call uses ``kira.llm`` so it shares retry / block-timeout
/ provider-detection with the main loop. No tools are passed.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from omnicoding.agents.kira.llm import call_llm_for_image  # reused: same "no tools, plain text" call shape
from omnicoding.agents.kira.tools import SYSTEM_PROMPT

LOGGER = logging.getLogger("kira.summarize")

_SUMMARY_INSTRUCTION = (
    "You are summarizing a coding agent's progress for a context-window "
    "handoff. Read the conversation below and produce a concise summary "
    "(target: 300-500 tokens) covering:\n"
    "  1. What the agent has accomplished so far (files created, "
    "commands that succeeded, key information discovered).\n"
    "  2. Current state of the workspace (cwd, important files / "
    "artifacts produced, anything that still needs cleanup).\n"
    "  3. What remains to finish the task.\n"
    "  4. Any specific values, IDs, paths, or numeric results the agent "
    "must remember to produce the final answer.\n"
    "Be specific — the agent will see only this summary, not the "
    "original conversation."
)


def _serialize_message_for_summary(msg: dict[str, Any], max_chars: int = 4000) -> str:
    role = msg.get("role", "?")
    content = msg.get("content")
    if isinstance(content, list):
        # Multimodal — keep only text parts to save tokens.
        text_parts = [c.get("text", "") for c in content if isinstance(c, dict) and c.get("type") == "text"]
        body = "\n".join(text_parts)
    elif isinstance(content, str):
        body = content
    else:
        body = ""
    if len(body) > max_chars:
        body = body[: max_chars // 2] + "\n... [truncated] ...\n" + body[-max_chars // 2:]
    tool_calls = msg.get("tool_calls")
    if tool_calls:
        tc_text = json.dumps(
            [{"name": (tc.get("function") or {}).get("name"),
              "args": (tc.get("function") or {}).get("arguments")}
             for tc in tool_calls],
            ensure_ascii=False,
        )[:max_chars]
        body = body + ("\n" if body else "") + f"[tool_calls: {tc_text}]"
    return f"<{role}> {body}"


def _build_summary_prompt(
    messages: list[dict[str, Any]],
    original_instruction: str,
) -> str:
    convo_lines: list[str] = []
    for msg in messages:
        if msg.get("role") == "system":
            continue  # system prompt is not informative for the summary
        convo_lines.append(_serialize_message_for_summary(msg))
    convo = "\n\n".join(convo_lines)
    return (
        f"{_SUMMARY_INSTRUCTION}\n\n"
        f"=== Original task ===\n{original_instruction}\n\n"
        f"=== Conversation ===\n{convo}"
    )


def summarize_conversation(
    *,
    messages: list[dict[str, Any]],
    original_instruction: str,
    model_name: str,
    provider: str | None = None,
    api_base: str | None,
    api_key: str | None,
    request_timeout_s: int = 600,
    block_timeout_s: int = 600,
) -> list[dict[str, Any]]:
    """Run the summarizer LLM and return the new (compressed) messages
    list: ``[system, user(original task + summary + continue prompt)]``.

    The caller is responsible for replacing ``KiraAgent.messages`` with
    this return value.
    """
    summary_prompt = _build_summary_prompt(messages, original_instruction)
    LOGGER.info(
        "kira.summarize compressing msgs=%d prompt_chars=%d",
        len(messages), len(summary_prompt),
    )
    summary_text = call_llm_for_image(
        messages=[{"role": "user", "content": summary_prompt}],
        model_name=model_name,
        provider=provider,
        api_base=api_base,
        api_key=api_key,
        request_timeout_s=request_timeout_s,
        block_timeout_s=block_timeout_s,
    )
    LOGGER.info("kira.summarize produced summary_chars=%d", len(summary_text))

    handoff = (
        f"=== Original task ===\n{original_instruction}\n\n"
        f"=== Progress summary (your earlier conversation was compressed) ===\n"
        f"{summary_text}\n\n"
        "Continue from this state. Use execute_commands to inspect or "
        "act, image_read for images, or task_complete (twice) to finish."
    )
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": handoff},
    ]
