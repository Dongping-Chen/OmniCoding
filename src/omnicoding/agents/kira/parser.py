"""Tool-call parsing tolerant of Qwen3.6's qwen3_coder XML quirks.

Qwen3.6 + sglang's ``--tool-call-parser qwen3_coder`` emits each tool
call's ``arguments`` as the parameter values seen inside
``<parameter=name>...</parameter>`` XML bodies, JSON-decoded
best-effort. Three Qwen quirks we have to absorb on the harness side:

1. Whole ``arguments`` string sometimes has a trailing extra ``}}``
   (documented in agent.md round 8 / proxy/responses_translate
   _normalize_function_arguments). Strip the trailing junk before
   parsing.

2. The nested ``commands`` array in ``execute_commands`` arrives as a
   JSON string (qwen3_coder's permissive type handler hands the raw
   ``<parameter=commands>`` body back as text). Re-decode it.

3. A single-command call sometimes drops the array wrapper and arrives
   as a single dict (``commands={"keystrokes": "ls"}``) or even
   inlined at the top level (no ``commands`` key, ``keystrokes`` lives
   alongside ``analysis``/``plan``). Both shapes get normalized to
   ``[{"keystrokes": ..., "duration": ...}]``.

Out: a normalized ``ParsedToolCalls`` with three lists:
  - ``commands``: ``list[Command]`` from ``execute_commands``
  - ``image_read``: ``ImageReadRequest | None``
  - ``task_complete``: ``bool``
plus ``analysis``, ``plan``, and any ``warning`` text the harness
should feed back to the model.
"""

from __future__ import annotations

import json
import logging
import os
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

LOGGER = logging.getLogger("kira.parser")

DEFAULT_DURATION = 1.0
MAX_DURATION = 60.0

# When set, every JSON-decode failure is dumped verbatim to this
# directory so we can diagnose Qwen's exact malformed output offline
# without re-running. Each dump is ``<role>_<ts>_<uuid>.txt`` with
# the raw value as-is. Empty / unset disables dumping.
_DEBUG_DUMP_DIR_ENV = "KIRA_PARSE_DEBUG_DIR"


def _maybe_dump_failed_blob(role: str, raw: str, exc: Exception) -> None:
    """Write the full failed string to disk when ``KIRA_PARSE_DEBUG_DIR``
    is set. Best-effort — never raises (the parser must keep running)."""
    dest = os.environ.get(_DEBUG_DUMP_DIR_ENV)
    if not dest:
        return
    try:
        d = Path(dest)
        d.mkdir(parents=True, exist_ok=True)
        fn = d / f"{role}_{int(time.time() * 1000)}_{uuid.uuid4().hex[:8]}.txt"
        fn.write_text(
            f"# kira.parser failure role={role} exc={type(exc).__name__}: {exc}\n"
            f"# len={len(raw)}\n"
            f"---RAW BEGIN---\n{raw}\n---RAW END---\n",
            encoding="utf-8",
        )
    except Exception:  # noqa: BLE001
        # Debug-only path; never block the loop on dump errors.
        pass


def _repair_truncated_json(s: str) -> Any:
    """Best-effort recovery for JSON that was cut off mid-value (Qwen
    sometimes finishes a tool call before closing every bracket/brace
    when finish_reason=length, and sglang's qwen3_coder XML extraction
    occasionally drops the trailing ``}}]`` when its tag-boundary
    heuristic fires early).

    Walks the string with a tiny state machine, tracks open
    ``"``/``[``/``{`` it never sees closed, and tries closing them in
    LIFO order. Returns the parsed value on success, or ``None`` if no
    repair makes it valid. Conservative — never introduces content,
    only appends closing characters.

    Returns ``None`` for empty input (caller already handles empty).
    """
    if not s:
        return None
    stack: list[str] = []
    in_str = False
    escape = False
    i = 0
    last_nonws_was_comma = False
    last_significant: str | None = None
    for ch in s:
        if in_str:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_str = False
            i += 1
            continue
        if ch == '"':
            in_str = True
        elif ch in "[{":
            stack.append("]" if ch == "[" else "}")
        elif ch in "]}":
            if stack and stack[-1] == ch:
                stack.pop()
            else:
                # Mismatched close — give up; can't safely repair.
                return None
        if not ch.isspace():
            last_nonws_was_comma = ch == ","
            last_significant = ch
        i += 1
    if not stack and not in_str:
        # No opens to close — string is balanced but still failed parse;
        # nothing for us to do.
        return None
    closing = ""
    if in_str:
        closing += '"'
    # If the last meaningful char was a trailing comma or an open
    # ``,`` followed by whitespace — drop it; otherwise json.loads will
    # reject the trailing comma. We rebuild ``s`` without that comma.
    candidate = s
    if last_nonws_was_comma:
        # Find last comma and chop it.
        idx = candidate.rfind(",")
        if idx >= 0:
            candidate = candidate[:idx]
    # If we just closed a string, the last value is complete; otherwise
    # we may be mid-token (e.g., ``"keystrokes": "ls`` without closing
    # quote — the closing ``"`` from in_str above handles that). For
    # an open object with a key but no value (``{"a":``), we also bail.
    if last_significant in (":", "{", "["):
        # ``{"key":`` — value missing; closing braces would yield invalid JSON.
        return None
    candidate += closing + "".join(reversed(stack))
    try:
        return json.loads(candidate, strict=False)
    except json.JSONDecodeError:
        return None


@dataclass(frozen=True)
class Command:
    keystrokes: str
    duration: float


@dataclass(frozen=True)
class ImageReadRequest:
    file_path: str
    image_read_instruction: str


@dataclass
class ParsedToolCalls:
    commands: list[Command] = field(default_factory=list)
    image_read: ImageReadRequest | None = None
    task_complete: bool = False
    analysis: str = ""
    plan: str = ""
    warnings: list[str] = field(default_factory=list)
    raw_calls: list[dict[str, Any]] = field(default_factory=list)


def _decode_arguments(raw: Any, warnings: list[str] | None = None) -> dict[str, Any]:
    """Decode a tool-call ``arguments`` blob into a dict. Tolerates Qwen's
    trailing extra brace bug (mirrors proxy/responses_translate
    _normalize_function_arguments) and falls back to the bracket-balancing
    repair when the trailing-byte trim isn't enough.

    When ``warnings`` is supplied, model-facing decode errors are
    appended for the loop to surface back to the model.
    """
    if raw is None:
        return {}
    if isinstance(raw, dict):
        return raw
    if not isinstance(raw, str):
        # OpenAI tool_calls always have ``arguments`` as a JSON string.
        # ``kira.llm._coerce_tool_calls_to_dicts`` enforces this shape
        # before we get here. If we ever see a list/int/etc., the
        # upstream contract is broken — log + drop.
        LOGGER.warning(
            "kira.parser arguments has unexpected type %s; dropping",
            type(raw).__name__,
        )
        return {}
    s = raw.strip()
    if not s:
        return {}
    # ``strict=False`` lets newlines/tabs/etc inside string bodies pass —
    # Qwen3.6's qwen3_coder emits multi-line bash literally (no
    # ``\n`` escape) inside ``<parameter>`` bodies, and the proxy hands
    # those bodies back verbatim. Strict JSON would reject them.
    try:
        decoded = json.loads(s, strict=False)
    except json.JSONDecodeError as exc:
        decoded = None
        pos = getattr(exc, "pos", None)
        if pos and 0 < pos <= len(s):
            try:
                decoded = json.loads(s[:pos], strict=False)
                LOGGER.info("kira.parser truncated %d trailing bytes from arguments", len(s) - pos)
            except json.JSONDecodeError:
                decoded = None
        if decoded is None:
            repaired = _repair_truncated_json(s)
            if repaired is not None:
                LOGGER.info("kira.parser arguments JSON was truncated; repaired")
                decoded = repaired
        if decoded is None:
            LOGGER.warning(
                "kira.parser failed to decode arguments (len=%d): %s",
                len(s), s[:200],
            )
            _maybe_dump_failed_blob("arguments", s, exc)
            if warnings is not None:
                warnings.append(
                    f"ERROR: Your tool-call `arguments` were not valid JSON "
                    f"(json decoder error at byte {exc.pos}: {exc.msg}). "
                    f"This call was dropped — re-emit it with a complete, "
                    f"valid JSON object."
                )
            return {}
    if isinstance(decoded, dict):
        return decoded
    LOGGER.warning("kira.parser arguments is not a dict (got %s): %s", type(decoded).__name__, s[:200])
    if warnings is not None:
        warnings.append(
            f"ERROR: Your tool-call `arguments` decoded to a "
            f"{type(decoded).__name__}, not a JSON object. The arguments "
            f'must be a single object like `{{"analysis": ..., "plan": '
            f'..., "commands": [...]}}`.'
        )
    return {}


def _coerce_duration(raw: Any) -> float:
    if isinstance(raw, bool):
        return DEFAULT_DURATION
    if isinstance(raw, (int, float)):
        d = float(raw)
    elif isinstance(raw, str) and raw.strip():
        try:
            d = float(raw.strip())
        except ValueError:
            return DEFAULT_DURATION
    else:
        return DEFAULT_DURATION
    if d <= 0:
        return DEFAULT_DURATION
    return min(d, MAX_DURATION)


def _normalize_commands_field(raw: Any, warnings: list[str] | None = None) -> list[dict[str, Any]]:
    """Turn whatever Qwen handed back as ``commands`` into a flat list of
    dicts. Returns ``[]`` if the value is unrecoverable.

    When ``warnings`` is supplied, any model-facing failure (unparseable
    JSON that even the repair pass can't fix; unsupported value type) is
    appended so the loop can surface it to the model via the next
    tool_result. The legacy single-arg signature is preserved for the
    existing test suite.
    """
    if raw is None:
        return []
    if isinstance(raw, str):
        s = raw.strip()
        if not s:
            return []
        try:
            raw = json.loads(s, strict=False)
        except json.JSONDecodeError as exc:
            repaired = _repair_truncated_json(s)
            if repaired is not None:
                LOGGER.info(
                    "kira.parser commands JSON was truncated; repaired (added closing chars)"
                )
                if warnings is not None:
                    warnings.append(
                        "Your `commands` argument JSON was truncated and the "
                        "harness auto-repaired it by closing trailing "
                        "strings/brackets. Emit complete JSON next turn so "
                        "the repair is not needed (repair may misinterpret "
                        "your intent)."
                    )
                raw = repaired
            else:
                LOGGER.warning(
                    "kira.parser commands is unparseable string (len=%d): %s",
                    len(s), s[:200],
                )
                _maybe_dump_failed_blob("commands", s, exc)
                if warnings is not None:
                    warnings.append(
                        f"ERROR: Your `commands` argument was not valid JSON "
                        f"(json decoder error at byte {exc.pos}: {exc.msg}). "
                        f"It must be a JSON array like "
                        f'`[{{"keystrokes": "ls", "duration": 1}}]`. '
                        f"No commands were executed this turn — re-emit the "
                        f"call with valid JSON."
                    )
                return []
    if isinstance(raw, dict):
        return [raw]
    if isinstance(raw, list):
        return [c for c in raw if isinstance(c, dict)]
    LOGGER.warning("kira.parser commands has unsupported type %s", type(raw).__name__)
    if warnings is not None:
        warnings.append(
            f"ERROR: Your `commands` argument was a {type(raw).__name__}; "
            f"it must be a JSON array of "
            f'{{"keystrokes": ..., "duration": ...}} objects.'
        )
    return []


def _commands_from_execute_args(
    args: dict[str, Any],
    warnings: list[str] | None = None,
) -> list[Command]:
    """Pull command(s) out of an ``execute_commands`` args dict.

    Round-12: the canonical schema is now flat — ``keystrokes`` +
    ``duration`` at the top level, one command per call. The legacy
    ``commands: [{keystrokes, duration}, ...]`` array shape is still
    accepted for backward compatibility (older models / saved
    trajectories). The flat shape avoids Qwen3.6 + qwen3_coder's
    nested-JSON serialisation quirks (``\\"`` shell escapes inside JSON,
    stray ``key=value`` after a closed string) that previously dropped
    whole calls.
    """
    items: list[dict[str, Any]] = []
    if "keystrokes" in args or "command" in args:
        # Canonical (round-12) flat shape — one command at the top level.
        items.append(args)
    raw_commands = args.get("commands")
    if raw_commands is not None:
        # Legacy nested-array shape; keep parsing it so older trajectories
        # still load. Concatenated after the flat command if both exist
        # (rare; only happens when a model splices both shapes).
        items.extend(_normalize_commands_field(raw_commands, warnings=warnings))
    if not items and warnings is not None:
        warnings.append(
            "ERROR: Your `execute_commands` call had neither `keystrokes` "
            "nor `commands`. Provide `keystrokes` (the shell text to run) "
            "and optionally `duration` (timeout in seconds)."
        )
    out: list[Command] = []
    for entry in items:
        ks = entry.get("keystrokes")
        if not isinstance(ks, str) or not ks:
            ks = entry.get("command")
        if not isinstance(ks, str) or not ks:
            LOGGER.warning("kira.parser command missing keystrokes: %s", entry)
            if warnings is not None:
                warnings.append(
                    "ERROR: An execute_commands call was missing the "
                    "`keystrokes` field; that call was skipped. Each call "
                    'must include `keystrokes` (the shell text to run) and '
                    'optionally `duration` (timeout in seconds).'
                )
            continue
        out.append(Command(keystrokes=ks, duration=_coerce_duration(entry.get("duration"))))
    return out


def _image_read_from_args(args: dict[str, Any]) -> ImageReadRequest | None:
    fp = args.get("file_path")
    instr = args.get("image_read_instruction")
    if not isinstance(fp, str) or not fp.strip():
        return None
    if not isinstance(instr, str) or not instr.strip():
        return None
    return ImageReadRequest(file_path=fp.strip(), image_read_instruction=instr.strip())


def parse_tool_calls(raw_tool_calls: list[dict[str, Any]] | None) -> ParsedToolCalls:
    """Walk a list of OpenAI-shape tool calls and collapse them into a
    single ``ParsedToolCalls``. Multiple ``execute_commands`` calls in
    one assistant turn (rare) are concatenated."""
    parsed = ParsedToolCalls(raw_calls=list(raw_tool_calls or []))
    if not raw_tool_calls:
        parsed.warnings.append(
            "Your response contained no tool calls. Use execute_commands "
            "to run a command, image_read to inspect an image, or "
            "task_complete to finish."
        )
        return parsed

    for call in raw_tool_calls:
        fn = call.get("function") or {}
        name = fn.get("name") or ""
        args = _decode_arguments(fn.get("arguments"), warnings=parsed.warnings)

        if name == "execute_commands":
            if not parsed.analysis and isinstance(args.get("analysis"), str):
                parsed.analysis = args["analysis"]
            if not parsed.plan and isinstance(args.get("plan"), str):
                parsed.plan = args["plan"]
            parsed.commands.extend(
                _commands_from_execute_args(args, warnings=parsed.warnings)
            )

        elif name == "task_complete":
            parsed.task_complete = True

        elif name == "image_read":
            req = _image_read_from_args(args)
            if req is None:
                parsed.warnings.append(
                    "image_read requires both 'file_path' and "
                    "'image_read_instruction' (non-empty strings)."
                )
            elif parsed.image_read is None:
                parsed.image_read = req
            else:
                parsed.warnings.append(
                    "Multiple image_read calls in one turn are not "
                    "supported; only the first was honored."
                )

        else:
            parsed.warnings.append(
                f"Unknown tool '{name}'. Use execute_commands, "
                "task_complete, or image_read."
            )
            LOGGER.warning("kira.parser unknown tool name=%s", name)

    return parsed
