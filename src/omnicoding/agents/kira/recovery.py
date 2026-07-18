"""Recover tool calls from malformed Qwen3.6 ``<tool_call>`` XML.

When KIRA's tool spec includes a nested array (``execute_commands``'s
``commands: [{keystrokes, duration}]``), Qwen3.6 frequently emits a
mishmash of qwen3_coder XML and qwen25 JSON like::

    <tool_call>
    function=execute_commands>           ← missing leading ``<``
    <parameter=analysis>
    Starting the task...", "plan": "...", "commands": [{...}]}
    </tool_call>                          ← closes <tool_call> directly,
                                            never closes <parameter>

sglang's ``--tool-call-parser qwen3_coder`` requires well-formed
``<function=NAME>...</function>`` and ``<parameter=K>V</parameter>``,
so it bails and the whole block lands in ``content`` with
``tool_calls = []``. Without recovery, every KIRA agent step gets 0
tool calls and the run aborts immediately.

Reading the structure literally:

  - ``<tool_call>`` opens the call ✅
  - ``function=NAME>`` (sometimes with a leading ``<``) names the tool
  - ``<parameter=FIRST_PARAM>`` opens the first parameter
  - The remainder is supposed to be that parameter's body, but the model
    actually wrote the *entire arguments dict* there in JSON syntax,
    starting with the first parameter's value. So we can reconstruct a
    valid arguments JSON by prepending ``{"FIRST_PARAM": "`` and
    re-parsing.

The recovery is conservative: we only accept the result if (a) the
function name is in ``declared_tools`` (caller-supplied allow-list),
and (b) the prepended JSON parses.

Does NOT replace the proxy salvage (which covers think-leak — XML
inside ``<think>`` reasoning_content). That salvage already runs first;
this recovery layer fires only when ``resp.tool_calls`` is still empty
and the assistant content has a ``<tool_call>`` block sglang couldn't
parse.
"""

from __future__ import annotations

import json
import logging
import re
import uuid
from typing import Any

LOGGER = logging.getLogger("kira.recovery")

_TOOL_CALL_RE = re.compile(r"<tool_call>(.*?)</tool_call>", re.DOTALL)
_FUNCTION_RE = re.compile(r"<?function=([^>\s]+)>", re.DOTALL)
_FIRST_PARAM_RE = re.compile(r"<parameter=([^>]+)>\s*", re.DOTALL)


def _strip_open_param_tags(body: str) -> str:
    """If the body has stray ``<parameter=NAME>`` tags within it (Qwen
    occasionally opens a second parameter mid-JSON), drop them so the
    JSON is contiguous. Conservative — only strip BARE tags, never the
    text around them."""
    return re.sub(r"<parameter=[^>]+>\s*", "", body)


def _recover_one(block: str, declared_tools: set[str]) -> dict[str, Any] | None:
    fn = _FUNCTION_RE.search(block)
    if not fn:
        return None
    name = fn.group(1).strip()
    if name not in declared_tools:
        LOGGER.debug("kira.recovery skipping unknown tool name=%r", name)
        return None
    rest = block[fn.end():].strip()
    param_m = _FIRST_PARAM_RE.search(rest)
    if not param_m:
        return None
    param_name = param_m.group(1).strip()
    inner = rest[param_m.end():].rstrip()
    # Strip a stray ``</parameter>`` that occasionally appears between the
    # last value and ``</tool_call>``.
    if inner.endswith("</parameter>"):
        inner = inner[: -len("</parameter>")].rstrip()
    inner = _strip_open_param_tags(inner)
    candidate = '{"' + param_name + '": "' + inner
    try:
        args = json.loads(candidate, strict=False)
    except json.JSONDecodeError as exc:
        LOGGER.warning(
            "kira.recovery JSON reconstruction failed for %s: %s | head=%s",
            name, exc, candidate[:200],
        )
        return None
    if not isinstance(args, dict):
        return None
    LOGGER.info("kira.recovery recovered tool=%s params=%s", name, list(args.keys()))
    return {
        "id": f"call_recovered_{uuid.uuid4().hex[:16]}",
        "type": "function",
        "function": {
            "name": name,
            "arguments": json.dumps(args, ensure_ascii=False),
        },
    }


def recover_tool_calls(text: str | None, declared_tools: set[str]) -> list[dict[str, Any]]:
    """Scan ``text`` for one or more ``<tool_call>...</tool_call>`` blocks
    that sglang's qwen3_coder parser failed to convert into native
    ``tool_calls``. Reconstruct each as an OpenAI-shape function call.
    Returns the list of recovered calls (possibly empty)."""
    if not text or "<tool_call>" not in text:
        return []
    out: list[dict[str, Any]] = []
    for m in _TOOL_CALL_RE.finditer(text):
        recovered = _recover_one(m.group(1), declared_tools)
        if recovered is not None:
            out.append(recovered)
    return out
