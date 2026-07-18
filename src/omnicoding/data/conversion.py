"""Convert a kira ``messages.json`` into ms-swift Agent JSONL.

ms-swift expects (per docs/source_en/Customization/Custom-dataset.md):
  - top-level ``tools`` JSON string with OpenAI-shape function specs
  - ``messages`` with explicit roles: system / user / assistant /
    tool_call / tool_response (or tool). At training time the
    agent_template (e.g. ``--agent_template qwen3_5``) renders these
    into the model's native XML format; pre-rendering would
    double-render.
  - ``images`` / ``audios`` / ``videos`` top-level lists pair with
    ``<image>`` / ``<audio>`` / ``<video>`` placeholders in content.

Pick the right ``agent_template`` for your target model:
  - ``qwen3_5`` → Qwen3.5 / Qwen3.6 (hybrid thinking; tools rendered
    as JSON list inside ``<tools>``).
  - ``qwen3_coder`` → Qwen3-Coder-30B-A3B-Instruct (non-thinking;
    tools rendered as XML tag tree).
  - ``qwen3_thinking`` → Qwen3-2507 Thinking variants.
The tool-call XML format itself (``<tool_call><function=N>...``) is
shared across all three; the differences are in the system block's
tool description and whether the model emits ``<think>``.

Per kira-trajectory IO:
  in   <out_dir>/run_meta.json
       <out_dir>/results.json
       <out_dir>/item_NNNN/messages.json
       <out_dir>/item_NNNN/image_subcalls.jsonl   (optional, for --multimodal)
  out  ms-swift JSONL with one row per item

Modes:
  default (text-only, parity-preserving):
      kira's text-only agent NEVER sees raw images at inference — only
      the text description that ``image_read`` returns. SFT data
      preserves this: tool_response contains the text description, no
      ``<image>`` tags.

  --multimodal (end-to-end multimodal training):
      For training a multimodal model that sees the image directly. Two
      kira run shapes are recognized:

        - native (``image_read_mode="native"``, default since 2026-04-29):
          the main agent's OpenAI-compatible conversation contains a
          short ``role=tool`` ack followed by a ``role=user`` message
          whose content is a list of ``[text, image_url]`` parts (the
          bytes in a ``data:`` URL). The converter decodes each
          ``image_url`` to a file under
          ``out_images_dir/<item>/img_<NNN>.<ext>``, replaces the part
          with an ``<image>`` placeholder, and folds that text/image
          payload back into the preceding ms-swift ``tool_response``.
          This keeps GPT collection legal (OpenAI cannot return images
          from a tool) while training Qwen through ms-swift's native
          Agent IR, where tool responses are rendered as Qwen
          ``<tool_response>`` user blocks.

        - legacy sub_llm (pre-2026-04-29 runs): ``role=tool`` content
          starts with ``image_read result for '<path>':`` and carries
          the sub-LLM text description. The converter picks up the
          source bytes from ``image_subcalls.jsonl`` (``image_b64``
          field) when present, or from a pre-decoded
          ``out/item_NNNN/images/<basename>`` directory if the harness
          mirrored the bytes.

      Both shapes produce an ms-swift Agent row with ``<image>`` tags
      paired against the top-level ``images`` list in placement order.

Filters & quality (stripped before SFT):
  - assistant turns flagged ``_synthetic_truncated`` (literal
    ``[truncated]`` would otherwise become a target).
  - empty assistant turns with no content + no tool_calls (the
    ``no_tool_calls`` bow-out the harness reminded against).

Usage:
  # text-only single item
  python convert_kira_to_msswift.py \\
      --in_dir   <out>/item_0000 --run_meta <out>/run_meta.json \\
      --out      sft.jsonl

  # batch + multimodal
  python convert_kira_to_msswift.py \\
      --batch_dir <out> --out sft.jsonl --multimodal \\
      --out_images_dir /tmp/sft_images
"""

from __future__ import annotations

import argparse
import base64
import json
import logging
import re
from pathlib import Path
from typing import Any

LOGGER = logging.getLogger("convert_kira_to_msswift")

CHECKLIST_SENTINEL = "[!] Checklist"
# Stable marker from ``common.spec.build_continue_prompt`` — the kira
# loop injects this user-role message when the model produces a turn
# with no tool_call (the no_tool_call reminder budget). Mid-stream user
# messages confuse ms-swift's swift-backend qwen3_5 template (it expects
# alternating user/tool → assistant rounds; a `tool_response → user →
# assistant` triplet trips ``response_role: "user"``). We drop these
# kira-injected reminders during conversion — they're harness-side
# nudges, not real user input, and the model's reply still lands as the
# next assistant turn alongside the surviving conversation.
KIRA_CONTINUE_PROMPT_MARKER = "Please continue solving the task"
IMAGE_TOOL_PREFIX_RE = re.compile(
    r"^image_read result for '([^']+)':\n(.*)$", re.DOTALL,
)
# Native-mode role=user content carries a ``data:image/<mime>;base64,<b64>``
# block emitted by ``kira.image_read.read_image_native``. Capture mime +
# bytes so we can dump the image to disk and substitute an ``<image>``
# placeholder for ms-swift.
DATA_URL_RE = re.compile(
    r"^data:(image/[a-zA-Z0-9+.-]+);base64,(.+)$", re.DOTALL,
)
ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", re.DOTALL | re.IGNORECASE)


# ---------- helpers ------------------------------------------------

def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    out: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return out


def _decode_args(raw: Any) -> dict[str, Any] | None:
    """Best-effort JSON args parse. ``None`` means we couldn't recover —
    caller drops the tool_call rather than poisoning the SFT row with a
    synthesized ``_raw_arguments`` key (which the model would learn to
    emit at inference and then misrender through qwen3_5 agent template).

    Repair attempts on JSONDecodeError, in order:
      1. Strip trailing comma before ``}`` / ``]`` (common GPT-5.5 mistake).
      2. Append a closing ``"`` if quote count is odd (truncated string).
      3. Append closing ``}``s/``]``s if balance is off (truncated obj).
    Each repair is independent; first one that parses wins.
    """
    if isinstance(raw, dict):
        return raw
    if not isinstance(raw, str):
        return None
    s = raw.strip()
    if not s:
        return None
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        pass
    for repaired in _json_repair_candidates(s):
        try:
            obj = json.loads(repaired)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict):
            return obj
    return None


def _json_repair_candidates(s: str) -> list[str]:
    """Generate a small set of structurally-plausible repairs."""
    out = []
    # 1. Strip trailing commas.
    no_trailing = re.sub(r",\s*([}\]])", r"\1", s)
    if no_trailing != s:
        out.append(no_trailing)
    # 2. Close an unterminated string at end-of-input.
    if s.count('"') % 2 != 0:
        out.append(s + '"')
        out.append(s + '"}')
    # 3. Close unbalanced braces/brackets.
    n_open_obj = s.count("{") - s.count("}")
    n_open_arr = s.count("[") - s.count("]")
    if n_open_obj > 0 or n_open_arr > 0:
        out.append(s + ("]" * max(0, n_open_arr)) + ("}" * max(0, n_open_obj)))
    return out


# ---------- core conversion ----------------------------------------

def _convert_assistant(
    msg: dict[str, Any],
    fill_empty_think: bool = True,
) -> list[dict[str, Any]]:
    """One kira assistant turn → 0+ ms-swift messages.

    Skip wholly empty turns (no content, no tool_calls) — those are the
    bow-out responses the harness reminded against; including them
    teaches the model to bow out, which is the opposite of what we want.

    Otherwise:
      - reasoning_content + content → ``role:assistant``. When the
        source had ``reasoning_content``, wrap as
        ``<think>\\n{reasoning}\\n</think>\\n\\n{content}``. When it
        didn't and ``fill_empty_think=True`` (default), prepend an
        empty ``<think>\\n\\n</think>\\n\\n`` placeholder — Qwen3.6's
        chat template forces every assistant turn to start with a
        ``<think>...</think>`` block, so ms-swift's swift backend (which
        does NOT auto-insert the placeholder for thinking-mode SFT)
        would otherwise produce a train-time format that drifts from
        what the model emits at inference. With the placeholder
        explicit in the data, every variant — swift backend, jinja
        backend, sglang inference — sees the same exact tokens.
      - each tool_call → ``role:tool_call`` with content =
        ``{"name": ..., "arguments": {...}}`` JSON string. ms-swift
        concatenates consecutive assistant + tool_call entries during
        rendering, matching kira's "one assistant turn" semantics.
    """
    if msg.get("_synthetic_truncated"):
        return []
    content = msg.get("content") or ""
    rc = msg.get("reasoning_content") or ""
    tcs = msg.get("tool_calls") or []
    if not (isinstance(content, str) and content.strip()) and not tcs and not (isinstance(rc, str) and rc.strip()):
        return []
    out: list[dict[str, Any]] = []
    asst_content = content if isinstance(content, str) else ""
    if isinstance(rc, str) and rc.strip():
        asst_content = f"<think>\n{rc.strip()}\n</think>\n\n{asst_content}"
    elif fill_empty_think:
        asst_content = f"<think>\n\n</think>\n\n{asst_content}"
    if asst_content.strip() or not tcs:
        # Always emit assistant (even empty) when there are tool_calls,
        # so the framework knows the boundary; or when content is the
        # actual final answer with no tool_call.
        out.append({"role": "assistant", "content": asst_content})
    for tc in tcs:
        fn = tc.get("function") or {}
        name = fn.get("name") or ""
        args = _decode_args(fn.get("arguments"))
        if args is None:
            LOGGER.info(
                "convert dropping tool_call name=%s with un-parseable args", name,
            )
            continue
        out.append({
            "role": "tool_call",
            "content": json.dumps({"name": name, "arguments": args}, ensure_ascii=False),
        })
    return out


def _convert_tool_reply(
    msg: dict[str, Any],
    image_subcalls_by_path: dict[str, dict[str, Any]] | None,
    multimodal: bool,
    images_collector: list[str] | None,
    images_out_dir: Path | None,
    item_tag: str,
    item_images_dir: Path | None,
) -> dict[str, Any] | None:
    """Render one ``role=tool`` message. Multimodal mode replaces the
    text description from ``image_read`` with an ``<image>`` tag and
    appends the image path to ``images_collector``.

    Image-bytes resolution order (cheapest first):
      1. Pre-decoded file under ``item_images_dir`` (from
         ``run_bench_kira._decode_subcall_images_to_dir``) — best path,
         no JSON parse, no base64 decode.
      2. Decode from ``image_subcalls_by_path`` JSONL record into a
         fresh file under ``images_out_dir/<item_tag>/``.
      3. Skip image substitution; emit the text description as-is.
    """
    content = msg.get("content")
    if not isinstance(content, str) or not content.strip():
        return None
    if not multimodal:
        return {"role": "tool_response", "content": content}
    m = IMAGE_TOOL_PREFIX_RE.match(content)
    if not m:
        return {"role": "tool_response", "content": content}
    file_path, desc = m.group(1), m.group(2)
    img_path = _resolve_image_path(
        file_path, image_subcalls_by_path, images_out_dir, item_tag, item_images_dir,
    )
    if img_path is None or images_collector is None:
        return {"role": "tool_response", "content": content}
    images_collector.append(str(img_path))
    return {
        "role": "tool_response",
        "content": f"<image>\nimage_read result for '{file_path}':\n{desc}",
    }


def _resolve_image_path(
    file_path: str,
    image_subcalls_by_path: dict[str, dict[str, Any]] | None,
    images_out_dir: Path | None,
    item_tag: str,
    item_images_dir: Path | None,
) -> Path | None:
    """Return a real path to the image bytes, preferring a pre-decoded
    file from the kira run over re-decoding from JSONL. None means we
    have no bytes for this path and the caller should emit text-only."""
    stem = Path(file_path).stem
    if item_images_dir is not None and item_images_dir.is_dir():
        for ext in ("jpg", "jpeg", "png", "gif", "webp"):
            candidate = item_images_dir / f"{stem}.{ext}"
            if candidate.exists():
                return candidate
    if image_subcalls_by_path is None or images_out_dir is None:
        return None
    rec = image_subcalls_by_path.get(file_path)
    if not rec or not rec.get("image_b64"):
        return None
    ext = (rec.get("mime") or "image/png").split("/")[-1]
    if ext == "jpeg":
        ext = "jpg"
    target = images_out_dir / item_tag / f"{stem}.{ext}"
    target.parent.mkdir(parents=True, exist_ok=True)
    if not target.exists():
        target.write_bytes(base64.b64decode(rec["image_b64"]))
    return target


def _convert_user_native_multimodal(
    msg: dict[str, Any],
    multimodal: bool,
    images_collector: list[str],
    images_out_dir: Path | None,
    item_tag: str,
) -> dict[str, Any] | None:
    """Convert a kira ``role=user`` message whose content is a list of
    multimodal parts (the shape ``read_image_native`` produces).
    Returns a flat user-shaped message
    ``{"role": "user", "content": "<text>\\n<image>\\n..."}`` and
    side-effects ``images_collector`` with the per-image file paths.

    The native harness path emits ``role=user`` right after the
    ``image_read`` tool reply, with content like::

        [
          {"type": "text", "text": "image_read result for '<path>' (image/jpeg, ...). You requested: ..."},
          {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,<b64>"}},
        ]

    Multiple ``image_url`` parts are honoured (one ``<image>`` per
    image, in order). When ``multimodal=False`` images are dropped and
    the text parts are concatenated. The caller decides whether this
    user-shaped payload remains a real user turn (ordinary multimodal
    user input) or is folded into the preceding ``tool_response``
    (native ``image_read`` observation).

    Returns ``None`` on:
      - non-list content (caller should fall through to legacy str
        handling)
      - empty content (drop the message)

    Filenames are deterministic: ``<item_tag>/img_<NNN>.<ext>`` where
    ``NNN`` is the running index of images_collector. This keeps repeated
    runs over the same trajectory writing identical paths.
    """
    content = msg.get("content")
    if not isinstance(content, list):
        return None
    text_parts: list[str] = []
    new_image_paths: list[str] = []
    for part in content:
        if not isinstance(part, dict):
            continue
        ptype = part.get("type")
        if ptype == "text":
            t = part.get("text") or ""
            if isinstance(t, str) and t.strip():
                text_parts.append(t)
        elif ptype == "image_url":
            url = (part.get("image_url") or {}).get("url") or ""
            m = DATA_URL_RE.match(url)
            if not m:
                LOGGER.warning(
                    "convert: skipping image_url with non-data URL "
                    "(item=%s, prefix=%r)", item_tag, url[:32],
                )
                continue
            mime, b64 = m.group(1), m.group(2)
            ext = mime.split("/")[-1]
            if ext == "jpeg":
                ext = "jpg"
            if multimodal and images_out_dir is not None:
                idx = len(images_collector) + len(new_image_paths)
                target = images_out_dir / item_tag / f"img_{idx:03d}.{ext}"
                target.parent.mkdir(parents=True, exist_ok=True)
                if not target.exists():
                    target.write_bytes(base64.b64decode(b64))
                new_image_paths.append(str(target))
    if not text_parts and not new_image_paths:
        return None
    text = "\n".join(text_parts)
    if multimodal and new_image_paths:
        # ms-swift convention: each ``<image>`` tag pairs with the
        # n-th entry in the top-level ``images`` list, so the order
        # placeholders appear in content == order of paths in the list.
        text = (text + "\n" if text else "") + "\n".join(["<image>"] * len(new_image_paths))
        images_collector.extend(new_image_paths)
    return {"role": "user", "content": text}


def _append_native_multimodal_to_tool_response(
    tool_response_msg: dict[str, Any],
    rendered_user_msg: dict[str, Any],
) -> None:
    """Fold a native ``image_read`` follow-up user payload into the
    preceding ms-swift ``tool_response``.

    KIRA keeps the OpenAI/GPT wire shape in ``messages.json``:
    ``assistant(tool_call=image_read) -> tool(ack) -> user(image)``.
    ms-swift's Agent IR must instead keep the whole observation in the
    tool-response slot so its swift backend can pair
    ``user/tool -> assistant`` rounds and the qwen3_5 agent template can
    render one Qwen-native ``<tool_response>`` user block.
    """
    prev = tool_response_msg.get("content") or ""
    cur = rendered_user_msg.get("content") or ""
    if not isinstance(prev, str):
        prev = str(prev)
    if not isinstance(cur, str):
        cur = str(cur)
    parts = [p for p in (prev.rstrip(), cur.strip()) if p]
    tool_response_msg["content"] = "\n".join(parts)


def _tool_call_name(msg: dict[str, Any]) -> str:
    try:
        tc = json.loads(msg.get("content") or "{}")
    except json.JSONDecodeError:
        return ""
    name = tc.get("name")
    return name if isinstance(name, str) else ""


def _find_last_image_read_tool_response_index(out_msgs: list[dict[str, Any]]) -> int | None:
    """Find the tail ``tool_response`` corresponding to an ``image_read`` call.

    ms-swift Agent rows keep parallel tool calls as:
    ``assistant, tool_call*, tool_response*``. Native KIRA image payloads
    are appended after the whole OpenAI tool-reply block, so when the
    assistant emitted e.g. ``image_read`` + ``task_complete`` together,
    blindly folding into the immediately previous tool_response would
    attach the image to ``task_complete``. Pair by ordinal instead.
    """
    if not out_msgs or out_msgs[-1].get("role") != "tool_response":
        return None
    resp_start = len(out_msgs) - 1
    while resp_start > 0 and out_msgs[resp_start - 1].get("role") == "tool_response":
        resp_start -= 1
    call_end = resp_start - 1
    if call_end < 0 or out_msgs[call_end].get("role") != "tool_call":
        return None
    call_start = call_end
    while call_start > 0 and out_msgs[call_start - 1].get("role") == "tool_call":
        call_start -= 1
    n_pairs = min(len(out_msgs) - resp_start, call_end - call_start + 1)
    for offset in range(n_pairs - 1, -1, -1):
        if _tool_call_name(out_msgs[call_start + offset]) == "image_read":
            return resp_start + offset
    return None


_PLACEHOLDERS = {"", "...", "X", "FINAL_ANSWER"}


def _last_answer(messages: list[dict[str, Any]]) -> str:
    """Last meaningful <answer>...</answer> across the trajectory.

    Skips:
      - the system message (kira's prompt seeds ``<answer>X</answer>``
        and ``<answer>...</answer>`` as format examples).
      - tool replies whose body is the harness double-confirm
        checklist (it echoes the original instruction text, which in
        turn quotes the same placeholder examples).
      - placeholder payloads (``X``, ``...``, empty) — the spec
        extractor's output is what counts; format examples are not.
    """
    last = ""
    for m in messages:
        role = m.get("role")
        if role == "system":
            continue
        c = m.get("content") or ""
        if not isinstance(c, str):
            continue
        if role == "tool" and CHECKLIST_SENTINEL in c:
            continue
        for match in ANSWER_RE.findall(c):
            payload = match.strip()
            if payload and payload not in _PLACEHOLDERS:
                last = payload
    return last


def _canonicalize_answer(
    out_msgs: list[dict[str, Any]],
    answer: str,
) -> None:
    """If the trained model needs to emit ``<answer>X</answer>`` in
    assistant content (the path-b form in kira's system prompt), but
    the harvested trajectory only carries the answer inside a
    ``tool_response`` (path-a, ``echo '<answer>X</answer>'``), inject
    it into every ``task_complete`` assistant turn that doesn't already
    have an answer in its content. Mutates ``out_msgs`` in place.

    Without this, GPT-5.5 trajectories that picked path-a teach the SFT
    model to emit empty assistant content next to ``task_complete`` —
    the answer is in the (loss-masked) tool_response, never in the
    assistant target. Canonicalizing puts it into the loss-active
    region so the trained model learns to emit the wrapper directly.
    """
    if not answer:
        return
    wrapped = f"<answer>{answer}</answer>"
    # Walk and find tool_call entries whose content is task_complete;
    # the immediately preceding assistant entry (if any) should carry
    # the answer in its content.
    for idx, msg in enumerate(out_msgs):
        if msg.get("role") != "tool_call":
            continue
        try:
            tc = json.loads(msg.get("content") or "{}")
        except json.JSONDecodeError:
            continue
        if tc.get("name") != "task_complete":
            continue
        # Look backwards for an adjacent assistant entry; insert one
        # if missing. Stop at the first non-{assistant,tool_call} role
        # we encounter walking back.
        ins = idx
        while ins > 0 and out_msgs[ins - 1].get("role") in ("assistant", "tool_call"):
            ins -= 1
            if out_msgs[ins].get("role") == "assistant":
                break
        if ins < idx and out_msgs[ins].get("role") == "assistant":
            asst = out_msgs[ins]
            existing = asst.get("content") or ""
            if "<answer>" not in existing:
                # Strip trailing whitespace before appending so the
                # final ``<think>...</think>\n\n<answer>...`` boundary
                # exactly matches Qwen3.6's chat template (a
                # ``<think>\n\n</think>\n\n`` placeholder plus a
                # spurious ``\n`` would otherwise add an extra blank
                # line that drifts inference vs train rendering).
                trimmed = existing.rstrip()
                asst["content"] = (trimmed + "\n\n" if trimmed else "") + wrapped
        else:
            # No assistant turn before this task_complete — synthesize one.
            out_msgs.insert(idx, {"role": "assistant", "content": wrapped})


def _collapse_consecutive_assistants(out_msgs: list[dict[str, Any]]) -> None:
    """Merge runs of consecutive ``role=assistant`` messages into one.

    Why this is needed: the kira harness emits a ``role=user`` continue-
    retry reminder after every ``no_tool_calls`` turn (so the model has
    a chance to call task_complete properly). The converter's user
    handler intentionally DROPS those reminders (ms-swift's swift
    backend can't pair ``tool_response → user → assistant``). The
    drop leaves multiple back-to-back assistant turns in ``out_msgs``,
    which ms-swift also can't train on (chat template requires
    alternating roles).

    Fix: collapse consecutive assistant turns to the LAST one — it
    carries the final answer wrapper and is the turn closest to
    task_complete (real or synthesized). Mutates ``out_msgs`` in place.

    Edge case: an assistant turn that had ``tool_calls`` is split by
    ``_convert_assistant`` into ``[assistant content, tool_call,
    tool_call, ...]`` — those turns are NEVER consecutive with another
    plain-text assistant in out_msgs (a tool_call entry separates them),
    so this only collapses runs of plain-text-only assistants.
    """
    if not out_msgs:
        return
    new: list[dict[str, Any]] = []
    for m in out_msgs:
        if (
            m.get("role") == "assistant"
            and new
            and new[-1].get("role") == "assistant"
        ):
            # Replace previous with current — last wins.
            new[-1] = m
        else:
            new.append(m)
    out_msgs.clear()
    out_msgs.extend(new)


def _ensure_terminal_task_complete(out_msgs: list[dict[str, Any]]) -> None:
    """If the trajectory's last meaningful turn is a plain-text
    assistant message containing ``<answer>...</answer>`` but NO
    following ``task_complete`` tool_call, synthesize one.

    Why this matters: under the FINAL_ANSWER_PROTOCOL (round 17.11)
    the model legitimately emits the answer as plain assistant text
    and the harness exits with ``exit_reason=no_tool_calls`` before
    any task_complete is recorded. The grader is happy (extracts
    wrapper from any role), but SFT data ends up with two distinct
    terminal shapes:
      shape A (task_complete called): ... assistant + task_complete
      shape B (no_tool_calls exit):   ... assistant<answer>X</answer>
    Training on mixed shapes muddies the "how a successful run ends"
    signal. Synthesize a terminal task_complete pair to canonicalize
    everything to shape A.

    Mutates ``out_msgs`` in place. Does nothing if:
      - the trajectory already ends with task_complete tool_call (or
        the tool_response that follows it),
      - the last assistant turn doesn't contain ``<answer>``.
    """
    if not out_msgs:
        return
    # Walk backward to find the last assistant turn and inspect.
    last_assistant_idx = None
    for i in range(len(out_msgs) - 1, -1, -1):
        if out_msgs[i].get("role") == "assistant":
            last_assistant_idx = i
            break
    if last_assistant_idx is None:
        return
    asst_content = out_msgs[last_assistant_idx].get("content") or ""
    if not isinstance(asst_content, str) or "<answer>" not in asst_content:
        return
    # Is there already a task_complete tool_call AFTER this assistant
    # turn? If so, nothing to synthesize.
    for j in range(last_assistant_idx + 1, len(out_msgs)):
        msg = out_msgs[j]
        if msg.get("role") != "tool_call":
            continue
        try:
            tc = json.loads(msg.get("content") or "{}")
        except json.JSONDecodeError:
            continue
        if tc.get("name") == "task_complete":
            return
    # Append synthetic task_complete pair right after the assistant.
    # Insert position: directly after the assistant turn so any tool
    # turns that came BEFORE the answer (e.g. lingering image_read
    # follow-up) stay where they are.
    insert_at = last_assistant_idx + 1
    synth_call = {
        "role": "tool_call",
        "content": json.dumps({"name": "task_complete", "arguments": {}},
                              ensure_ascii=False),
    }
    synth_resp = {
        "role": "tool_response",
        "content": "",  # task_complete has no meaningful return payload
    }
    out_msgs.insert(insert_at, synth_call)
    out_msgs.insert(insert_at + 1, synth_resp)


def convert_one(
    *,
    messages: list[dict[str, Any]],
    tools_spec: list[dict[str, Any]],
    image_subcalls: list[dict[str, Any]] | None = None,
    multimodal: bool = True,
    images_out_dir: Path | None = None,
    item_tag: str = "item",
    item_images_dir: Path | None = None,
    canonicalize_answer: bool = True,
    fill_empty_think: bool = True,
) -> dict[str, Any]:
    """Convert one kira ``messages.json`` to an ms-swift Agent row.

    ``item_images_dir`` (optional): a per-item directory of pre-decoded
    images created by ``run_bench_kira._decode_subcall_images_to_dir``.
    When present, image substitution prefers these files over re-decoding
    from ``image_subcalls`` JSONL — saves a base64 round-trip and lets
    the SFT data reference paths that already live alongside the
    trajectory artifacts.
    """
    sub_by_path = (
        {r.get("file_path_arg"): r for r in image_subcalls if r.get("status") == "ok"}
        if image_subcalls else None
    )
    images_collector: list[str] = []

    out_msgs: list[dict[str, Any]] = []
    seen_initial_user = False
    for msg in messages:
        role = msg.get("role")
        if role == "system":
            sys_content = msg.get("content") or ""
            if isinstance(sys_content, str) and sys_content.strip():
                out_msgs.append({"role": "system", "content": sys_content})
        elif role == "user":
            content = msg.get("content")
            # Native ``image_read`` emits a list-content user message
            # right after the role=tool ack. Convert each ``image_url``
            # part to a per-item file + ``<image>`` placeholder so
            # ms-swift's multimodal pipeline (or downstream Qwen-VL SFT)
            # can consume it. Falls through to legacy string handling
            # only when the harness produced a plain text message.
            if isinstance(content, list):
                rendered = _convert_user_native_multimodal(
                    msg, multimodal, images_collector,
                    images_out_dir, item_tag,
                )
                if rendered is not None:
                    target_idx = _find_last_image_read_tool_response_index(out_msgs)
                    if target_idx is not None:
                        _append_native_multimodal_to_tool_response(out_msgs[target_idx], rendered)
                    else:
                        out_msgs.append(rendered)
                        seen_initial_user = True
                continue
            if not (isinstance(content, str) and content.strip()):
                continue
            if seen_initial_user and KIRA_CONTINUE_PROMPT_MARKER in content:
                # kira no_tool_call reminder — drop. ms-swift's swift
                # backend can't pair `tool_response → user → assistant`.
                LOGGER.info("convert dropping kira continue-prompt user-msg (item=%s)", item_tag)
                continue
            out_msgs.append({"role": "user", "content": content})
            seen_initial_user = True
        elif role == "assistant":
            out_msgs.extend(_convert_assistant(msg, fill_empty_think=fill_empty_think))
        elif role == "tool":
            rendered = _convert_tool_reply(
                msg, sub_by_path, multimodal, images_collector,
                images_out_dir, item_tag, item_images_dir,
            )
            if rendered is not None:
                out_msgs.append(rendered)

    if canonicalize_answer:
        # Round-17.11: collapse consecutive assistant turns BEFORE the
        # other canonicalization steps so they operate on a clean
        # alternating-role list. (See ``_collapse_consecutive_assistants``
        # for the full why — kira continue-retry reminders dropped by the
        # user handler leave back-to-back assistants which ms-swift can't
        # train on.)
        _collapse_consecutive_assistants(out_msgs)
        # Pull the last <answer>...</answer> from the full source
        # trajectory (any role) and inject it into assistant content
        # next to each task_complete tool_call if missing.
        ans = _last_answer(messages)
        _canonicalize_answer(out_msgs, ans)
        # Round-17.11 (2026-04-30): under FINAL_ANSWER_PROTOCOL, the
        # model often emits ``<answer>X</answer>`` as plain assistant
        # text and the harness exits via ``no_tool_calls`` without
        # ever recording a ``task_complete`` tool_call. The trajectory
        # is functionally complete (the grader extracts the wrapper
        # from anywhere in the trajectory) but the SFT row has
        # heterogeneous tail shape: some end with ``task_complete``,
        # others end with a plain assistant text. Trains messy.
        # Synthesize a terminal task_complete to canonicalize.
        _ensure_terminal_task_complete(out_msgs)

    row: dict[str, Any] = {
        "messages": out_msgs,
        "tools": json.dumps(tools_spec, ensure_ascii=False),
    }
    if multimodal and images_collector:
        row["images"] = images_collector
    return row


# ---------- driver --------------------------------------------------

def _process_item(
    item_dir: Path,
    run_meta: dict[str, Any],
    multimodal: bool,
    images_out_dir: Path | None,
    canonicalize_answer: bool = True,
    fill_empty_think: bool = True,
) -> dict[str, Any] | None:
    msg_path = item_dir / "messages.json"
    if not msg_path.exists():
        return None
    messages = json.loads(msg_path.read_text(encoding="utf-8"))
    sub = _load_jsonl(item_dir / "image_subcalls.jsonl")
    return convert_one(
        messages=messages,
        tools_spec=run_meta["tools_spec"],
        image_subcalls=sub,
        multimodal=multimodal,
        images_out_dir=images_out_dir,
        item_tag=item_dir.name,
        item_images_dir=item_dir / "images",
        canonicalize_answer=canonicalize_answer,
        fill_empty_think=fill_empty_think,
    )


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--in_dir", type=Path, help="Single per-item dir (e.g. out/item_0000).")
    g.add_argument("--batch_dir", type=Path, help="Run output dir; converts every out/item_*/.")
    p.add_argument("--run_meta", type=Path, default=None,
                   help="Path to run_meta.json. Defaults to <in_dir>/../run_meta.json or "
                        "<batch_dir>/run_meta.json.")
    p.add_argument("--out", required=True, type=Path, help="Output JSONL path.")
    p.add_argument("--multimodal", action="store_true",
                   help="Replace text image_read descriptions with <image> tags + dump "
                        "the actual images for ms-swift's multimodal pipeline.")
    p.add_argument("--out_images_dir", type=Path, default=None,
                   help="Where to dump decoded images when --multimodal is set. "
                        "Defaults to <out>.parent / 'images'.")
    p.add_argument("--no_canonicalize_answer", action="store_true",
                   help="Don't inject <answer>X</answer> into task_complete assistant "
                        "content when the model emitted it via shell echo (path-a). "
                        "Default ON keeps SFT loss covering the answer.")
    p.add_argument("--no_fill_empty_think", action="store_true",
                   help="Don't prepend empty <think>\\n\\n</think>\\n\\n to assistant "
                        "turns missing reasoning_content. Default ON matches Qwen3.6's "
                        "chat template (every assistant turn opens with a think block).")
    args = p.parse_args(argv)

    if args.run_meta is None:
        if args.in_dir:
            args.run_meta = args.in_dir.parent / "run_meta.json"
        else:
            args.run_meta = args.batch_dir / "run_meta.json"
    run_meta = json.loads(args.run_meta.read_text(encoding="utf-8"))

    if args.multimodal and args.out_images_dir is None:
        args.out_images_dir = args.out.resolve().parent / "images"

    canonicalize = not args.no_canonicalize_answer
    fill_empty_think = not args.no_fill_empty_think
    rows: list[dict[str, Any]] = []
    if args.in_dir:
        row = _process_item(args.in_dir, run_meta, args.multimodal, args.out_images_dir,
                            canonicalize_answer=canonicalize,
                            fill_empty_think=fill_empty_think)
        if row:
            rows.append(row)
    else:
        for item_dir in sorted(args.batch_dir.glob("item_*")):
            if not item_dir.is_dir():
                continue
            row = _process_item(item_dir, run_meta, args.multimodal, args.out_images_dir,
                                canonicalize_answer=canonicalize,
                                fill_empty_think=fill_empty_think)
            if row:
                rows.append(row)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"wrote {args.out}: {len(rows)} rows")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
