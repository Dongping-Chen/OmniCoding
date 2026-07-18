"""KIRA agent loop.

One ``KiraAgent.run(prompt)`` call:
  1. Seed messages with system + user prompt.
  2. While not done:
     a. Call ``kira.llm.call_llm_with_tools`` (retry + block-timeout +
        provider-aware extra_body).
     b. Recover Qwen-shape malformed ``<tool_call>`` blocks.
     c. Parse + dispatch ``execute_commands`` / ``image_read`` /
        ``task_complete``.
     d. Append the tool-result message and recurse.
  3. Stop on ``task_complete`` (single call exits — no double-confirm),
     ``step_limit`` reached, ``no_tool_calls`` after max reminders, or
     unrecoverable error (auth, bad request, abandoned thread).

Resilience layer (R11 audit):
  - ``ContextLengthExceededError`` → summarize history → retry once.
  - ``OutputLengthExceededError`` → re-prompt for shorter response →
    retry once.
  - Transient errors (rate limit, 5xx, network) → tenacity 5x retry
    inside ``kira.llm`` (loop never sees those).

State is kept in ``KiraAgent.messages`` (OpenAI-shape chat history) and
``KiraAgent.trajectory`` (per-step records the driver writes back into
``results.json``).
"""

from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from omnicoding.agents.kira.image_read import (
    NativeImageReadResult,
    read_image,
    read_image_native,
)
from omnicoding.agents.kira.llm import (
    BlockTimeoutError,
    ContextLengthExceededError,
    LLMResponse,
    OutputLengthExceededError,
    call_llm_with_tools,
)
from omnicoding.agents.kira.endpoint_pool import EndpointSession
from omnicoding.agents.kira.parser import Command, ImageReadRequest, parse_tool_calls
from omnicoding.agents.kira.provider import default_max_tool_reminders, resolve_provider
from omnicoding.agents.kira.recovery import recover_tool_calls
from omnicoding.agents.kira.shell import PersistentShell
from omnicoding.agents.kira.summarize import summarize_conversation
from omnicoding.agents.kira.tools import SYSTEM_PROMPT, TOOLS

# Litellm exception classes that indicate the endpoint is dead, restarting,
# overloaded, or out of quota — kira routes these through
# EndpointSession.failover() and retries the same prompt on the NEXT URL in
# the pool. Anything not in this set (auth errors, bad request, context
# overflow) falls through to its existing handler.
#
# Note: ``BlockTimeoutError`` is intentionally NOT here. We retry it on the
# SAME endpoint first (sticky retry) so chatgpt's ``session_id`` affinity
# stays warm — bouncing across slots loses the prefix cache. Only after the
# sticky budget is exhausted do we fall through to failover.
import litellm.exceptions as _le

_FAILOVER_EXC = (
    _le.APIConnectionError,
    _le.InternalServerError,
    _le.ServiceUnavailableError,
    _le.BadGatewayError,
    _le.Timeout,
    _le.RateLimitError,
)

_DECLARED_TOOL_NAMES: set[str] = {t["function"]["name"] for t in TOOLS}

# Used by ``_finish_or_remind`` to gate single-call ``task_complete``
# exits on the presence of an ``<answer>...</answer>`` wrapper. Single
# pattern used across the agent loop and the spec extractor — keep them
# in sync if the wrapper format ever changes.
_ANSWER_WRAPPER_RE = re.compile(r"<answer>(.+?)</answer>", re.DOTALL | re.IGNORECASE)

LOGGER = logging.getLogger("kira.loop")


@dataclass
class StepRecord:
    step: int
    analysis: str
    plan: str
    n_commands: int
    is_task_complete: bool
    is_image_read: bool
    output_chars: int
    prompt_tokens: int
    completion_tokens: int
    # Provider subtotals (default 0 when not reported). cached_tokens
    # is the slice of prompt_tokens that hit the implicit prompt cache
    # (OpenAI / OpenRouter charge ~50% of nominal for those).
    # reasoning_tokens is the part of completion_tokens spent on the
    # internal CoT (OpenAI o-series / gpt-5+).
    cached_tokens: int = 0
    reasoning_tokens: int = 0


@dataclass
class AgentResult:
    final_text: str
    n_steps: int
    n_tool_calls: int
    completed: bool
    # task_complete | step_limit | no_tool_calls | error
    exit_reason: str
    messages: list[dict[str, Any]]
    trajectory: list[StepRecord]
    # Per-reminder records that mirror the claude/codex/opencode
    # `retry_attempts` schema so the wide-smoke analyzer can dedupe
    # across all four harnesses. One entry per fired continue-reminder;
    # capped by `max_tool_reminders`.
    retry_attempts: list[dict[str, Any]] = field(default_factory=list)
    error: str | None = None
    cumulative_prompt_tokens: int = 0
    cumulative_completion_tokens: int = 0
    cumulative_cached_tokens: int = 0
    cumulative_reasoning_tokens: int = 0
    n_summarizations: int = 0
    # One entry per summarize fire: the full pre-compression message
    # list and the post-compression handoff messages. Without this the
    # ``messages`` field above only shows the post-summary view, and
    # everything before the first summarize is lost — which makes SFT
    # data prep miss large stretches of trajectory.
    pre_summary_snapshots: list[dict[str, Any]] = field(default_factory=list)


def _serialize_assistant_message(resp: LLMResponse) -> dict[str, Any]:
    msg: dict[str, Any] = {"role": "assistant", "content": resp.content or ""}
    if resp.tool_calls:
        msg["tool_calls"] = resp.tool_calls
    if resp.reasoning_content:
        # Stored for the final-text walker; the API ignores extra keys.
        msg["reasoning_content"] = resp.reasoning_content
    # Underscore-prefixed keys are dropped by litellm when this message
    # is sent back on the next turn (drop_params=True) but are preserved
    # in the saved trajectory so SFT/RL data prep can tell genuine model
    # output apart from harness-rewritten / synthesized state.
    if resp.recovered_from_content:
        msg["_recovered_from_content"] = True
    if resp.finish_reason:
        msg["_finish_reason"] = resp.finish_reason
    return msg


def _message_has_image_part(msg: dict[str, Any]) -> bool:
    content = msg.get("content")
    if not isinstance(content, list):
        return False
    for part in content:
        if not isinstance(part, dict):
            continue
        if part.get("type") in {"image_url", "image"}:
            return True
        if "image_url" in part or "image" in part:
            return True
    return False


def _tool_call_name_by_id(assistant_msg: dict[str, Any]) -> dict[str, str]:
    out: dict[str, str] = {}
    for idx, tc in enumerate(assistant_msg.get("tool_calls") or []):
        fn = tc.get("function") or {}
        name = fn.get("name") or ""
        if not name:
            continue
        tool_call_id = tc.get("id") or f"call_kira_{idx}"
        out[tool_call_id] = name
    return out


def _find_tail_image_read_tool_index(messages: list[dict[str, Any]]) -> int | None:
    """Return the tool message in the current tail tool block that
    corresponds to an ``image_read`` tool_call, if any."""
    if not messages or messages[-1].get("role") != "tool":
        return None
    first_tool = len(messages) - 1
    while first_tool > 0 and messages[first_tool - 1].get("role") == "tool":
        first_tool -= 1
    assistant_idx = first_tool - 1
    if assistant_idx < 0 or messages[assistant_idx].get("role") != "assistant":
        return None
    by_id = _tool_call_name_by_id(messages[assistant_idx])
    for idx in range(len(messages) - 1, first_tool - 1, -1):
        tool_call_id = messages[idx].get("tool_call_id")
        if isinstance(tool_call_id, str) and by_id.get(tool_call_id) == "image_read":
            return idx
    # Older/proxied tool calls may lack stable IDs. If the assistant
    # made exactly one image_read and there is exactly one tool reply,
    # the mapping is still unambiguous.
    if len(messages) - first_tool == 1 and list(by_id.values()).count("image_read") == 1:
        return first_tool
    return None


def _merge_tool_content_with_user_image(
    tool_content: Any,
    user_content: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Build a multimodal tool content list for Qwen/sGLang.

    The first text part contains the previous tool ack plus the
    native-image prelude, with a trailing newline so Qwen's image pad
    appears on its own line inside ``<tool_response>``.
    """
    parts = [dict(p) for p in user_content if isinstance(p, dict)]
    if isinstance(tool_content, list):
        merged = [dict(p) for p in tool_content if isinstance(p, dict)]
    else:
        text = "" if tool_content is None else str(tool_content).rstrip()
        merged = [{"type": "text", "text": text}] if text else []
    if not parts:
        return merged
    if merged and merged[-1].get("type") == "text" and parts[0].get("type") == "text":
        prev = str(merged[-1].get("text") or "").rstrip()
        cur = str(parts[0].get("text") or "").strip()
        joined = "\n".join(p for p in (prev, cur) if p)
        if len(parts) > 1:
            joined += "\n"
        merged[-1]["text"] = joined
        merged.extend(parts[1:])
    else:
        if merged and merged[-1].get("type") == "text":
            merged[-1]["text"] = str(merged[-1].get("text") or "").rstrip() + "\n"
        merged.extend(parts)
    return merged


def _fold_native_image_read_messages_for_qwen(
    messages: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Qwen/sGLang send-time adapter for native ``image_read``.

    KIRA's stored trajectory stays OpenAI/GPT-compatible:
    ``assistant(image_read) -> tool(ack) -> user([text, image_url])``.
    Qwen's chat template can render multimodal ``role=tool`` content
    inside ``<tool_response>...</tool_response>``, which is the same
    layout used by the ms-swift SFT converter. This function produces
    that Qwen-only view without mutating the saved trajectory.
    """
    out: list[dict[str, Any]] = []
    for msg in messages:
        if msg.get("role") == "user" and _message_has_image_part(msg):
            target_idx = _find_tail_image_read_tool_index(out)
            content = msg.get("content")
            if target_idx is not None and isinstance(content, list):
                target = dict(out[target_idx])
                target["content"] = _merge_tool_content_with_user_image(
                    target.get("content"), content,
                )
                out[target_idx] = target
                continue
        out.append(dict(msg))
    return out


@dataclass
class KiraAgent:
    workspace: Path
    model_name: str
    api_base: str
    # Continue/reminder prompt injected when the model emits a turn with
    # no tool call. Required so the harness driver can pass the
    # spec-aware text from omnicoding.benchmarks.common.spec.build_continue_prompt(spec).
    continue_prompt: str
    provider: str | None = None
    api_key: str = "local"
    image_model_name: str | None = None
    step_limit: int = 80
    request_timeout_s: int = 900
    block_timeout_s: int = 600
    extra_env: dict[str, str] = field(default_factory=dict)
    enable_thinking: bool = True
    reasoning_effort: str | None = None
    thinking_budget_tokens: int | None = None
    # None → auto-detected from model_name. Provider-default is
    # ``default_max_tool_reminders(provider)``.
    max_tool_reminders: int | None = None
    # Set False to disable summarization on context overflow. Default:
    # summarize once then retry; if that also overflows, exit with
    # error.
    enable_summarize: bool = True
    # Per-item endpoint session for multi-endpoint runs. When set, every
    # api_base read in this agent goes through ``session.current_url``;
    # on transient failures the loop calls ``session.failover()`` and
    # retries the same prompt against the new URL. ``None`` → legacy
    # single-endpoint mode using the static ``api_base`` field.
    endpoint_session: "EndpointSession | None" = None
    # Sampling overrides forwarded to ``kira.llm.call_llm_with_tools``.
    # ``None`` → leave at provider default (sglang ~1.0 temperature,
    # OpenAI o-series fixed at 1.0). Set explicitly when collecting
    # trajectories you intend to RL-replay or seed-dedup.
    temperature: float | None = None
    top_p: float | None = None
    max_tokens: int | None = None
    seed: int | None = None
    # ``image_read`` JSONL log path — raw bytes + sub-LLM
    # request/response. SFT/RL replay needs the bytes; the main agent
    # message history only carries the sub-LLM's text description.
    image_subcall_log: "Path | None" = None
    # ``native``: decode image and inject into main conversation as a
    # follow-up user message (keeps train/serve byte-identical).
    # ``sub_llm``: legacy — separate vision call returns text.
    image_read_mode: str = "native"
    # Single-call ``task_complete`` preflight: if the trajectory has
    # no ``<answer>...</answer>`` wrapper, append a short user
    # reminder and let the model retry this many times before exiting
    # anyway with an empty prediction (bounded run).
    max_answer_retries: int = 1
    # Sticky retry budget on ``BlockTimeoutError`` — retry the timed-
    # out call on the SAME endpoint first (preserves cache affinity,
    # 80%+ prompt-cache hit). Falls through to ``failover()`` after
    # budget exhausts. Reset on successful call.
    max_sticky_retries: int = 3

    def __post_init__(self) -> None:
        self.workspace = Path(self.workspace).resolve()
        self.workspace.mkdir(parents=True, exist_ok=True)
        if self.image_model_name is None:
            self.image_model_name = self.model_name
        self._provider = resolve_provider(self.model_name, self.provider)
        if self.max_tool_reminders is None:
            self.max_tool_reminders = default_max_tool_reminders(self._provider)
        self.messages: list[dict[str, Any]] = []
        self.trajectory: list[StepRecord] = []
        self._tool_reminders_used = 0
        self._cum_prompt_tokens = 0
        self._cum_completion_tokens = 0
        self._cum_cached_tokens = 0
        self._cum_reasoning_tokens = 0
        self._n_tool_calls = 0
        self._n_summarizations = 0
        self._retry_attempts: list[dict[str, Any]] = []
        self._answer_retries_used = 0
        self._original_instruction = ""
        # One snapshot per summarize-fire: the full pre-compression
        # message list and the post-compression handoff. Surfaces on
        # ``AgentResult.pre_summary_snapshots`` so the driver can dump
        # them next to messages.json.
        self._pre_summary_snapshots: list[dict[str, Any]] = []
        LOGGER.info(
            "kira.loop init provider=%s model=%s max_reminders=%d step_limit=%d",
            self._provider, self.model_name, self.max_tool_reminders, self.step_limit,
        )

    # ---------- LLM call with resilience layer ---------------------

    def _accumulate(self, resp: LLMResponse) -> None:
        """Add this response's token counts to the run-level cumulatives.
        Centralised so both the happy path and the
        context-overflow / output-overflow retry paths stay in sync."""
        self._cum_prompt_tokens += resp.prompt_tokens
        self._cum_completion_tokens += resp.completion_tokens
        self._cum_cached_tokens += resp.cached_tokens
        self._cum_reasoning_tokens += resp.reasoning_tokens

    def _effective_api_base(self) -> str:
        """Current URL: pool's session URL when in multi-endpoint mode,
        else the static ``api_base`` field set at construction."""
        if self.endpoint_session is not None:
            return self.endpoint_session.current_url
        return self.api_base

    def _llm_call_with_resilience(self) -> LLMResponse:
        """One LLM round-trip with summarize-on-overflow + output-length
        retry, plus sticky retry / endpoint failover on transient errors.

        Two distinct error policies:
          1. ``BlockTimeoutError`` (harness wall-clock fired): retry on the
             SAME endpoint up to ``max_sticky_retries`` times — this
             preserves chatgpt's session_id cache affinity. Falls through
             to failover only after sticky budget is exhausted.
          2. ``_FAILOVER_EXC`` (5xx, network drop, rate limit): immediate
             failover to the next pool slot. These are not transient on
             the same endpoint — staying put would just re-fail.

        Sticky budget resets on every successful call so a recovered
        endpoint can absorb future timeouts again.
        """
        sticky_left = self.max_sticky_retries
        while True:
            try:
                resp = self._call_llm()
                self._accumulate(resp)
                if self.endpoint_session is not None:
                    self.endpoint_session.record_success()
                return resp
            except ContextLengthExceededError as exc:
                return self._handle_context_overflow(exc)
            except OutputLengthExceededError as exc:
                return self._handle_output_overflow(exc)
            except BlockTimeoutError as exc:
                if sticky_left > 0:
                    sticky_left -= 1
                    LOGGER.warning(
                        "kira.loop sticky retry on %s left=%d (BlockTimeout)",
                        self._effective_api_base(), sticky_left,
                    )
                    continue
                if self._failover_or_raise(exc):
                    sticky_left = self.max_sticky_retries
                    continue
                raise
            except _FAILOVER_EXC as exc:
                if self._failover_or_raise(exc):
                    sticky_left = self.max_sticky_retries
                    continue
                raise

    def _failover_or_raise(self, exc: Exception) -> bool:
        """Try to rotate to the next pool slot. Returns True if rotation
        succeeded (caller should retry), False if no session or budget is
        exhausted (caller should re-raise the exception).
        """
        if self.endpoint_session is None:
            return False
        step_hint = len(self.trajectory) + 1
        if self.endpoint_session.failover(
            reason=type(exc).__name__, step_hint=step_hint,
        ):
            LOGGER.warning(
                "kira.loop endpoint failover step=%d reason=%s -> %s",
                step_hint, type(exc).__name__,
                self.endpoint_session.current_url,
            )
            return True
        LOGGER.error(
            "kira.loop endpoint failover budget exhausted step=%d reason=%s",
            step_hint, type(exc).__name__,
        )
        return False

    def _call_llm(self) -> LLMResponse:
        LOGGER.info(
            "kira.loop LLM call step=%d msgs=%d api_base=%s",
            len(self.trajectory) + 1, len(self.messages), self._effective_api_base(),
        )
        return call_llm_with_tools(
            messages=self._api_safe_messages(),
            model_name=self.model_name,
            provider=self._provider,
            api_base=self._effective_api_base(),
            api_key=self.api_key,
            tools=TOOLS,
            request_timeout_s=self.request_timeout_s,
            block_timeout_s=self.block_timeout_s,
            enable_thinking=self.enable_thinking,
            reasoning_effort=self.reasoning_effort,
            thinking_budget_tokens=self.thinking_budget_tokens,
            temperature=self.temperature,
            top_p=self.top_p,
            max_tokens=self.max_tokens,
            seed=self.seed,
        )

    def _api_safe_messages(self) -> list[dict[str, Any]]:
        """Strip underscore-prefixed debug keys before sending to the
        LLM. The saved trajectory keeps these (``_finish_reason``,
        ``_recovered_from_content``, ``_synthetic_truncated``) so SFT
        data prep can detect harness-rewritten state, but they have no
        meaning to the model and some strict provider endpoints reject
        unknown message-body fields outright.

        Qwen/sGLang additionally gets a send-time view that folds native
        ``image_read`` payloads from KIRA's OpenAI-safe shape
        ``tool(ack) -> user(image)`` into the corresponding ``tool``
        message. GPT/OpenAI still receives the split shape because the
        OpenAI API does not accept image payloads as tool returns.
        ``self.messages`` is never mutated by this adapter, so saved
        trajectories stay portable and the SFT converter can perform the
        same fold explicitly.
        """
        messages = [
            {k: v for k, v in m.items() if not k.startswith("_")}
            for m in self.messages
        ]
        if self._provider == "qwen":
            messages = _fold_native_image_read_messages_for_qwen(messages)
        return messages

    def _handle_context_overflow(self, exc: ContextLengthExceededError) -> LLMResponse:
        if not self.enable_summarize:
            LOGGER.warning("kira.loop context overflow + summarize disabled; re-raising")
            raise exc
        LOGGER.warning(
            "kira.loop context overflow at msgs=%d → summarizing", len(self.messages),
        )
        # Snapshot the full pre-compression history BEFORE replacing
        # ``self.messages`` so SFT data prep can stitch the original
        # turns back in. ``summarize_conversation`` discards everything
        # except the system prompt and a synthesized user message
        # carrying the LLM-written summary, which would otherwise leave
        # `messages` looking like a 2-message conversation by the time
        # the run ends.
        pre_summary = [dict(m) for m in self.messages]
        self.messages = summarize_conversation(
            messages=self.messages,
            original_instruction=self._original_instruction,
            model_name=self.model_name,
            provider=self._provider,
            api_base=self._effective_api_base(),
            api_key=self.api_key,
            request_timeout_s=self.request_timeout_s,
            block_timeout_s=self.block_timeout_s,
        )
        self._n_summarizations += 1
        self._pre_summary_snapshots.append({
            "summarize_index": self._n_summarizations,
            "step_at_summarize": len(self.trajectory) + 1,
            "pre_summary_messages": pre_summary,
            "post_summary_messages": [dict(m) for m in self.messages],
        })
        # Retry once. If THAT overflows too, surface the original error.
        resp = self._call_llm()
        self._accumulate(resp)
        return resp

    def _handle_output_overflow(self, exc: OutputLengthExceededError) -> LLMResponse:
        LOGGER.warning(
            "kira.loop output overflow; truncated_chars=%d, re-prompting",
            len(exc.truncated_content or ""),
        )
        # Save the model's ACTUAL partial output rather than the literal
        # string ``"[truncated]"``. The old behaviour trained SFT on a
        # synthetic placeholder the model never produced; downstream
        # tools can opt into ignoring this turn via the
        # ``_synthetic_truncated`` flag if they want.
        self.messages.append({
            "role": "assistant",
            "content": exc.truncated_content or "",
            "_synthetic_truncated": True,
            "_finish_reason": "length",
        })
        self.messages.append({
            "role": "user",
            "content": (
                "Your previous response was truncated (too long). "
                "Try again with fewer / shorter commands. Do NOT repeat "
                "what you already wrote — pick the single most useful "
                "next action."
            ),
        })
        resp = self._call_llm()
        self._accumulate(resp)
        return resp

    # ---------- agent step dispatch --------------------------------

    def _exec_commands(self, shell: PersistentShell, commands: list[Command]) -> str:
        if not commands:
            return ""
        outputs: list[str] = []
        for cmd in commands:
            LOGGER.info(
                "kira.loop exec keystrokes=%r duration=%.2f",
                cmd.keystrokes[:200], cmd.duration,
            )
            outputs.append(shell.run(cmd.keystrokes, cmd.duration))
        return "\n".join(outputs)

    def _exec_image_read(self, req: ImageReadRequest) -> str:
        return read_image(
            file_path=req.file_path,
            instruction=req.image_read_instruction,
            workspace=self.workspace,
            model_name=self.image_model_name or self.model_name,
            provider=self._provider,
            api_base=self._effective_api_base(),
            api_key=self.api_key,
            request_timeout_s=self.request_timeout_s,
            block_timeout_s=self.block_timeout_s,
            subcall_log_path=self.image_subcall_log,
        )

    def _append_tool_results(
        self,
        raw_calls: list[dict[str, Any]],
        main_result: str,
        warnings: list[str],
    ) -> None:
        """OpenAI requires one ``role=tool`` reply per tool_call. We
        attach the actual observation to the FIRST call and a stub
        ``"executed"`` to the rest, so multi-call assistant turns
        validate cleanly."""
        if not raw_calls:
            if warnings:
                self.messages.append({"role": "user", "content": "\n".join(warnings)})
            return
        warning_blob = ("\n".join(warnings) + "\n\n") if warnings else ""
        first_content = (warning_blob + main_result).rstrip()
        for idx, tc in enumerate(raw_calls):
            tool_call_id = tc.get("id") or f"call_kira_{idx}"
            content = first_content if idx == 0 else "executed"
            self.messages.append({
                "role": "tool",
                "tool_call_id": tool_call_id,
                "content": content or "executed",
            })

    def _recover_if_needed(self, resp: LLMResponse) -> None:
        """Qwen3.6 emits malformed ``<tool_call>`` XML in content that
        sglang's qwen3_coder parser silently rejects (leaves
        ``tool_calls=[]``). Recover those before deciding the model
        bowed out. Mutates ``resp.tool_calls`` in place and sets
        ``resp.recovered_from_content=True`` so the saved assistant
        message carries a flag SFT data prep can use to disambiguate
        synthesized vs. native structured tool_calls."""
        if resp.tool_calls:
            return
        recovered = recover_tool_calls(resp.content, _DECLARED_TOOL_NAMES)
        if not recovered and resp.reasoning_content:
            recovered = recover_tool_calls(resp.reasoning_content, _DECLARED_TOOL_NAMES)
        if recovered:
            LOGGER.info("kira.loop recovered %d tool calls from content", len(recovered))
            resp.tool_calls = recovered
            resp.recovered_from_content = True

    def _no_tool_calls_step(self) -> tuple[bool, str]:
        """Model emitted a turn with no tool call after recovery. Either
        re-prompt with the continue-reminder or bow out."""
        if self._tool_reminders_used >= (self.max_tool_reminders or 0):
            LOGGER.info(
                "kira.loop no tool calls after %d reminders; stopping",
                self._tool_reminders_used,
            )
            return True, "no_tool_calls"
        self._tool_reminders_used += 1
        self._retry_attempts.append({
            "attempt": self._tool_reminders_used,
            "step": len(self.trajectory) + 1,
            "reason": "no_tool_calls",
            "prompt": self.continue_prompt,
            "messages_at_attempt": len(self.messages),
            "timed_out": False,
        })
        self.messages.append({"role": "user", "content": self.continue_prompt})
        LOGGER.info(
            "kira.loop reminded model %d/%d to use a tool",
            self._tool_reminders_used, self.max_tool_reminders,
        )
        return False, ""

    def _step(self, shell: PersistentShell) -> tuple[bool, str]:
        """One agent step. Returns (done, exit_reason)."""
        resp = self._llm_call_with_resilience()
        self._recover_if_needed(resp)

        assistant_msg = _serialize_assistant_message(resp)
        self.messages.append(assistant_msg)
        parsed = parse_tool_calls(resp.tool_calls)

        if not resp.tool_calls:
            return self._no_tool_calls_step()

        self._n_tool_calls += len(resp.tool_calls)
        # The model recovered — re-arm the reminder budget.
        self._tool_reminders_used = 0

        if parsed.image_read is not None:
            return self._dispatch_image_read(parsed, resp)

        if parsed.task_complete and parsed.commands:
            return self._dispatch_task_complete_with_commands(parsed, shell, resp)

        if parsed.task_complete:
            return self._dispatch_task_complete_only(parsed, shell, resp)

        return self._dispatch_execute_commands(parsed, shell, resp)

    # ---------- dispatch sub-paths ---------------------------------

    def _finish_or_remind(self) -> tuple[bool, str]:
        """Single-call task_complete exit gate. Walk the trajectory; if
        an ``<answer>...</answer>`` wrapper is anywhere in the
        assistant + tool text (so either via path-a shell echo or
        path-b assistant content), exit normally. Otherwise burn one
        budget unit, append a short ``role=user`` reminder telling the
        model to emit the wrapper, and continue the loop. After the
        budget runs out, exit anyway — runs must be bounded.

        Why a budgeted reminder and not the legacy double-confirm
        checklist: the old checklist ran on EVERY task_complete (heavy
        ~700-char tool reply, always a second LLM turn). This fires
        only when needed, costs one short user message, and short-
        circuits as soon as the wrapper appears."""
        text = _walk_final_text(self.messages)
        if _ANSWER_WRAPPER_RE.search(text):
            return True, "task_complete"
        if self._answer_retries_used >= self.max_answer_retries:
            LOGGER.info(
                "kira.loop task_complete; no <answer> wrapper but "
                "max_answer_retries=%d exhausted, exiting with empty pred",
                self.max_answer_retries,
            )
            return True, "task_complete"
        self._answer_retries_used += 1
        reminder = (
            "task_complete called but no <answer>...</answer> wrapper is in your "
            "trajectory yet. The grader extracts your prediction from that wrapper, "
            "so without it the run is scored as no answer. Preferred fix: in your "
            "NEXT assistant turn, emit a plain-text message (no tool call) whose "
            "entire content is exactly `<answer>YOUR_ANSWER</answer>` — and only "
            "after that lands, call task_complete in the turn AFTER. Backstop: "
            "an `execute_commands` shell `echo '<answer>YOUR_ANSWER</answer>'` "
            "also gets parsed, but plain text saves a tool call and avoids "
            "shell-quoting hazards (apostrophes, `<`, `$` in the answer)."
        )
        self._retry_attempts.append({
            "attempt": self._answer_retries_used,
            "step": len(self.trajectory),
            "reason": "missing_answer_wrapper",
            "messages_at_attempt": len(self.messages),
        })
        self.messages.append({"role": "user", "content": reminder})
        LOGGER.info(
            "kira.loop reminded model to emit <answer> wrapper "
            "(retry %d/%d)",
            self._answer_retries_used, self.max_answer_retries,
        )
        return False, ""

    def _dispatch_image_read(self, parsed, resp: LLMResponse) -> tuple[bool, str]:
        """image_read in this turn. Mode-bifurcated: ``native`` (default)
        decodes the image locally and appends a follow-up user message
        with the actual pixels; ``sub_llm`` delegates to a vision LLM
        and returns its text description as the tool reply.

        If the model called ``task_complete`` in the SAME turn, we
        process the image first (so the trajectory keeps the pixels)
        and then exit. The model has already committed to finishing —
        single-call task_complete semantics apply across all dispatch
        paths."""
        if self.image_read_mode == "native":
            return self._dispatch_image_read_native(parsed, resp)
        return self._dispatch_image_read_sub_llm(parsed, resp)

    def _dispatch_image_read_sub_llm(self, parsed, resp: LLMResponse) -> tuple[bool, str]:
        """Legacy sub-LLM path — vision call returns a text description
        which becomes the tool reply. No image gets into the main
        conversation."""
        result = self._exec_image_read(parsed.image_read)
        self._record(parsed, output_chars=len(result), is_image_read=True, resp=resp)
        self._append_tool_results(resp.tool_calls, result, parsed.warnings)
        if parsed.task_complete:
            return self._finish_or_remind()
        return False, ""

    def _dispatch_image_read_native(self, parsed, resp: LLMResponse) -> tuple[bool, str]:
        """Native path — decode the image and append a follow-up
        ``user`` message containing the multimodal content blocks. The
        ``role=tool`` reply gets a short ack so OpenAI's tool-call
        pairing is satisfied; the image itself rides in the user
        message that lands directly after."""
        nr: NativeImageReadResult = read_image_native(
            file_path=parsed.image_read.file_path,
            instruction=parsed.image_read.image_read_instruction,
            workspace=self.workspace,
            subcall_log_path=self.image_subcall_log,
        )
        self._record(parsed, output_chars=len(nr.tool_text), is_image_read=True, resp=resp)
        self._append_tool_results(resp.tool_calls, nr.tool_text, parsed.warnings)
        # Inject the actual image bytes as a fresh user turn. On read
        # failure (missing file etc.) ``user_content`` is None and the
        # tool ack already carries an ``ERROR:`` string the model can
        # react to.
        if nr.user_content is not None:
            self.messages.append({"role": "user", "content": nr.user_content})
        if parsed.task_complete:
            return self._finish_or_remind()
        return False, ""

    def _dispatch_task_complete_with_commands(
        self, parsed, shell: PersistentShell, resp: LLMResponse,
    ) -> tuple[bool, str]:
        """Model called BOTH execute_commands AND task_complete in one
        turn. Run the commands so the workspace ends in the
        agent-intended final state, append the terminal output as the
        tool reply, then exit. Single-call task_complete: no
        confirmation round."""
        terminal_output = self._exec_commands(shell, parsed.commands)
        self._record(parsed, output_chars=len(terminal_output), is_image_read=False, resp=resp)
        self._append_tool_results(resp.tool_calls, terminal_output, parsed.warnings)
        return self._finish_or_remind()

    def _dispatch_task_complete_only(
        self, parsed, shell: PersistentShell, resp: LLMResponse,
    ) -> tuple[bool, str]:
        """Model called only ``task_complete``. Run a tiny ``pwd && ls``
        snapshot so the trajectory's last tool reply has SOMETHING (the
        OpenAI tool-call/reply pairing requirement), then exit."""
        terminal_output = shell.run("pwd && ls", duration=1.0)
        self._record(parsed, output_chars=len(terminal_output), is_image_read=False, resp=resp)
        self._append_tool_results(resp.tool_calls, terminal_output, parsed.warnings)
        return self._finish_or_remind()

    def _dispatch_execute_commands(
        self, parsed, shell: PersistentShell, resp: LLMResponse,
    ) -> tuple[bool, str]:
        terminal_output = self._exec_commands(shell, parsed.commands)
        self._record(parsed, output_chars=len(terminal_output), is_image_read=False, resp=resp)
        self._append_tool_results(resp.tool_calls, terminal_output, parsed.warnings)
        return False, ""

    # ---------- bookkeeping ----------------------------------------

    def _record(self, parsed, output_chars: int, is_image_read: bool, resp: LLMResponse) -> None:
        self.trajectory.append(StepRecord(
            step=len(self.trajectory) + 1,
            analysis=parsed.analysis,
            plan=parsed.plan,
            n_commands=len(parsed.commands),
            is_task_complete=parsed.task_complete,
            is_image_read=is_image_read,
            output_chars=output_chars,
            prompt_tokens=resp.prompt_tokens,
            completion_tokens=resp.completion_tokens,
            cached_tokens=resp.cached_tokens,
            reasoning_tokens=resp.reasoning_tokens,
        ))

    def run(self, instruction: str, *, system_prefix: str = "") -> AgentResult:
        """Seed the conversation and run the loop.

        ``instruction`` lands in ``role=user`` — it should be just the
        per-item question (Question + Options + staged-files list).

        ``system_prefix`` is appended to the harness ``SYSTEM_PROMPT``
        in ``role=system`` and is meant for the static benchmark
        instructions that don't change item-to-item (workspace rules,
        web_search hint, sandbox mode, tool-workflow rules). Splitting
        them out lets the LLM provider's prompt cache hold the long
        prefix across items in a dispatch run instead of re-encoding
        ~5 KB of identical text per item. Empty by default — legacy
        callers that pass the full prompt as ``instruction`` keep
        working unchanged."""
        self._original_instruction = instruction
        system_content = SYSTEM_PROMPT
        if system_prefix:
            system_content = system_content + "\n\n" + system_prefix
        self.messages = [
            {"role": "system", "content": system_content},
            {"role": "user", "content": instruction},
        ]
        exit_reason = "step_limit"
        error: str | None = None
        start = time.monotonic()
        with PersistentShell(cwd=self.workspace, env=self.extra_env) as shell:
            for _ in range(self.step_limit):
                try:
                    done, reason = self._step(shell)
                except (ContextLengthExceededError, BlockTimeoutError) as exc:
                    LOGGER.exception("kira.loop fatal: %s", type(exc).__name__)
                    error = f"{type(exc).__name__}: {exc}"
                    exit_reason = "error"
                    break
                except Exception as exc:  # noqa: BLE001
                    LOGGER.exception("kira.loop step crashed")
                    error = f"{type(exc).__name__}: {exc}"
                    exit_reason = "error"
                    break
                if done:
                    exit_reason = reason
                    break
        elapsed = time.monotonic() - start
        final_text = _walk_final_text(self.messages)
        LOGGER.info(
            "kira.loop done steps=%d tool_calls=%d reason=%s elapsed=%.1fs summaries=%d",
            len(self.trajectory), self._n_tool_calls, exit_reason, elapsed,
            self._n_summarizations,
        )
        return AgentResult(
            final_text=final_text,
            n_steps=len(self.trajectory),
            n_tool_calls=self._n_tool_calls,
            completed=(exit_reason == "task_complete"),
            exit_reason=exit_reason,
            messages=self.messages,
            trajectory=self.trajectory,
            retry_attempts=self._retry_attempts,
            error=error,
            cumulative_prompt_tokens=self._cum_prompt_tokens,
            cumulative_completion_tokens=self._cum_completion_tokens,
            cumulative_cached_tokens=self._cum_cached_tokens,
            cumulative_reasoning_tokens=self._cum_reasoning_tokens,
            n_summarizations=self._n_summarizations,
            pre_summary_snapshots=self._pre_summary_snapshots,
        )


def _walk_final_text(messages: list[dict[str, Any]]) -> str:
    """Concat assistant + tool message text + assistant reasoning. Skip
    user/system to avoid pulling the prompt's example answer back in.
    Mirrors `run_bench_mini_swe._final_text` so the spec extractor sees
    the same shape."""
    parts: list[str] = []
    for msg in messages:
        role = msg.get("role")
        if role not in {"assistant", "tool"}:
            continue
        content = msg.get("content")
        text = ""
        if isinstance(content, str):
            text = content
        elif isinstance(content, list):
            text = "\n".join(
                c.get("text") or "" for c in content
                if isinstance(c, dict) and c.get("type") == "text"
            )
        if text.strip():
            parts.append(text)
        rc = msg.get("reasoning_content")
        if isinstance(rc, str) and rc.strip():
            parts.append(rc)
    return "\n\n".join(parts)


from omnicoding.agents.kira.serialize import messages_preview, trajectory_to_dicts  # noqa: E402,F401
