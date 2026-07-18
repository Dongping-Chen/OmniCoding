"""Multi-turn chat → ``(tokens, loss_mask, response_length)`` tokenizer.

Two paths:

- ``tokenize_trajectory`` (legacy, text-only) — used for unit tests and as
  fallback when no HF ``processor`` is available. Walks message-prefix encodings
  and diffs token-id lengths to attribute deltas to a turn.

- ``tokenize_trajectory_multimodal`` (production for VL models) — does ONE
  processor call on the full trajectory (avoiding O(N) processor invocations on
  every prefix), then walks the resulting ``input_ids`` to build the loss mask
  by special-token boundary detection. The chat template is fully responsible
  for image_pad expansion (driven by ``image_grid_thw`` in
  ``multimodal_train_inputs``); we never need to redo image preprocessing.

Convention (mirrors ``examples/deepeyes/rollout.py``):
- prompt = leading messages up to (not incl.) the first assistant turn
- response = first assistant onward
- within an assistant turn: leading ``<|im_start|>assistant\\n`` role marker is
  treated as observation (loss=0); model-emitted content + ``<|im_end|>`` get
  loss=1
- all non-assistant message tokens (system / user / tool / inter-message
  scaffolding newlines) get loss=0
- image_pad / vision_start / vision_end tokens fall inside ``role=tool``
  segments after ``fold_native_image_read_messages_for_qwen`` runs (kira's
  native ``image_read`` stores ``assistant(image_read) → tool(ack) →
  user([text, image_url])`` but we fold to ``assistant(image_read) →
  tool([ack+text+image_url])`` to match the ms-swift SFT layout). Qwen's chat
  template renders that tool segment as ``<|im_start|>user\\n<tool_response>
  …<|vision_start|><|image_pad|>…<|vision_end|></tool_response><|im_end|>``;
  every token in the segment inherits loss=0 — what we want, since pixel
  observations are model input not output
- ``len(loss_mask) == response_length``
"""

from __future__ import annotations

import base64
import json
import logging
import re
from io import BytesIO
from typing import Any

LOG = logging.getLogger("relax_router.tokenize")

# Qwen3 family special-token IDs. Pulled at runtime from the tokenizer; cached
# per-tokenizer (keyed by ``id(tokenizer)``) so we pay the lookup once.
_SPECIALS_CACHE: dict[int, dict[str, int]] = {}

# Role token → name. Only ``assistant`` carries trainable loss; everything else
# is observation. Kira's ``role=tool`` messages get rendered by the Qwen chat
# template as ``<|im_start|>user\n<tool_response>...`` so we never see ``tool``
# as the post-im_start token.
_TRAIN_ROLE_NAME = "assistant"

# Placeholder tag → role-token-id key in the cache, used for fast lookup
_ROLE_NAMES = ("system", "user", "assistant", "tool")


def _specials(tokenizer) -> dict[str, int]:
    key = id(tokenizer)
    if key in _SPECIALS_CACHE:
        return _SPECIALS_CACHE[key]
    out: dict[str, int] = {}
    out["im_start"] = tokenizer.convert_tokens_to_ids("<|im_start|>")
    out["im_end"] = tokenizer.convert_tokens_to_ids("<|im_end|>")
    nl_ids = tokenizer.encode("\n", add_special_tokens=False)
    out["newline"] = nl_ids[0] if len(nl_ids) == 1 else -1
    # Role token ids — encode just the role name (no leading <|im_start|>)
    # because tokenizer encodes them as standalone tokens after the special token.
    for role in _ROLE_NAMES:
        ids = tokenizer.encode(role, add_special_tokens=False)
        if len(ids) != 1:
            LOG.warning("role %r encodes to %d tokens (expected 1); "
                        "loss mask boundary walk may misalign", role, len(ids))
        out[f"role_{role}"] = ids[0] if ids else -1
    _SPECIALS_CACHE[key] = out
    return out


def _message_boundaries(input_ids: list[int], specials: dict[str, int]) -> list[int]:
    """Return positions that look like real ``<|im_start|>{role}\n`` markers."""
    im_start = specials["im_start"]
    newline = specials.get("newline", -1)
    role_ids = {
        specials[f"role_{role}"]
        for role in _ROLE_NAMES
        if specials.get(f"role_{role}", -1) != -1
    }
    return [
        i
        for i, tok in enumerate(input_ids)
        if tok == im_start
        and i + 2 < len(input_ids)
        and input_ids[i + 1] in role_ids
        and input_ids[i + 2] == newline
    ]


# ─── kira message preprocessor (data: URLs → PIL.Image) ──────────────────────


_DATA_URL_RE = re.compile(r"^data:([^;]+);base64,(.+)$", re.DOTALL)


def _decode_data_url(url: str):
    """Decode a ``data:image/...;base64,...`` URL into a PIL.Image. Returns
    None on any decode failure (caller drops the image part)."""
    from PIL import Image  # noqa: PLC0415 — heavy import, lazy

    m = _DATA_URL_RE.match(url)
    if not m:
        return None
    try:
        raw = base64.b64decode(m.group(2))
    except (ValueError, base64.binascii.Error):
        return None
    try:
        img = Image.open(BytesIO(raw))
        img.load()  # force decode now so a corrupt JPEG fails here, not later
    except Exception as exc:  # noqa: BLE001
        LOG.warning("dropped corrupt inline image: %s", exc)
        return None
    if img.mode != "RGB":
        img = img.convert("RGB")
    return img


# ─── Qwen-format fold (verbatim port of kira/loop.py) ────────────────────────
# These helpers preserve the audited Kira folding behavior. They remain local
# to the tokenizer so the Relax-side module can load without importing the
# full agent runtime.
#
# Why we need it: kira stores trajectories in OpenAI shape
#   ``assistant(image_read tool_call) → tool(ack text) → user([text, image_url])``
# but at SGLang send-time kira folds them into Qwen + ms-swift's layout
#   ``assistant(image_read tool_call) → tool([text+image_url])``  ← image inside tool reply
# which is what the chat template renders as
# ``<|im_start|>user\n<tool_response>{ack+text}{vision_pad}</tool_response><|im_end|>``.
# ms-swift's SFT converter produces the same layout, so RL must too — otherwise
# the model sees a different surface form at train vs serve.
#
# Parity guard: tests/test_tokenize_multimodal::test_fold_parity_with_kira_reference
# imports the packaged Kira reference and checks representative trajectories.


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


# Public alias — preserves backward-compat with earlier test naming
fold_native_image_read_messages_for_qwen = _fold_native_image_read_messages_for_qwen


# ─── OpenAI/Kira message normalization ──────────────────────────────────────


def _normalize_tool_call_arguments(messages: list[dict]) -> list[dict]:
    """Return a copy with tool-call arguments normalized to mappings.

    Kira and the OpenAI API store ``function.arguments`` as a JSON string,
    while current Qwen chat templates iterate over a mapping. Invalid or
    non-mapping argument payloads become an empty mapping, matching the
    existing multimodal preprocessing behavior.
    """
    rewritten: list[dict] = []
    for msg in messages:
        new_msg = dict(msg)
        tool_calls = new_msg.get("tool_calls")
        if isinstance(tool_calls, list):
            normalized_calls = []
            for tool_call in tool_calls:
                if not isinstance(tool_call, dict):
                    normalized_calls.append(tool_call)
                    continue
                function = (
                    tool_call.get("function")
                    if isinstance(tool_call.get("function"), dict)
                    else None
                )
                if function and "arguments" in function:
                    arguments = function.get("arguments")
                    if isinstance(arguments, str):
                        try:
                            arguments = json.loads(arguments) if arguments else {}
                        except json.JSONDecodeError:
                            arguments = {}
                    if not isinstance(arguments, dict):
                        arguments = {}
                    function = {**function, "arguments": arguments}
                    tool_call = {**tool_call, "function": function}
                normalized_calls.append(tool_call)
            new_msg["tool_calls"] = normalized_calls
        rewritten.append(new_msg)
    return rewritten


# ─── kira → relax preprocessor (data: URLs → PIL.Image, normalize tool_calls) ─


def _kira_to_relax_messages(messages: list[dict]) -> tuple[list[dict], list]:
    """Pipeline:
      1. Fold native ``image_read`` user-turns into preceding tool replies
         (Qwen + ms-swift layout — see ``fold_native_image_read_messages_for_qwen``).
      2. Normalize assistant ``tool_calls[*].function.arguments`` from JSON
         strings to dicts (Qwen chat template requires dicts; kira / OpenAI
         spec store them as JSON strings).
      3. Convert any remaining ``image_url`` content parts into ``{"type":
         "image", "image": <PIL>}`` (decode ``data:`` URLs, keep http URLs
         as strings for downstream processor to resolve).

    Returns ``(rewritten_messages, ordered_pil_images)``. The PIL list is
    fed positionally to ``processor(text=..., images=PIL_list)`` in the same
    order the chat template emits ``<|image_pad|>`` placeholders.
    """
    folded = _normalize_tool_call_arguments(
        fold_native_image_read_messages_for_qwen(messages)
    )

    images: list = []
    rewritten: list[dict] = []
    for msg in folded:
        new_msg = dict(msg)
        # Convert image_url content parts. After the fold, images live inside
        # ``role=tool`` content lists; pre-fold standalone user turns with
        # images (e.g. dataset-provided multimodal prompts) are also handled.
        content = new_msg.get("content")
        if isinstance(content, list):
            new_parts: list = []
            for part in content:
                if not isinstance(part, dict):
                    new_parts.append(part)
                    continue
                if part.get("type") == "image_url":
                    url = (part.get("image_url") or {}).get("url") or ""
                    if url.startswith("data:"):
                        pil = _decode_data_url(url)
                        if pil is not None:
                            images.append(pil)
                            new_parts.append({"type": "image", "image": pil})
                        # else: drop part silently (corrupt data URL)
                    elif url:
                        # http(s) or file URL — relax/utils/multimodal/image_utils.py
                        # ``load_image_from_path`` handles these. We don't fetch
                        # locally; the processor downstream resolves it.
                        new_parts.append({"type": "image", "image": url})
                    # else: empty URL → drop
                elif part.get("type") == "image":
                    img = part.get("image")
                    if img is not None:
                        images.append(img)
                    new_parts.append(part)
                else:
                    new_parts.append(part)
            new_msg["content"] = new_parts
        rewritten.append(new_msg)
    return rewritten, images


# ─── legacy text-only tokenize (kept for tests + non-VL fallback) ────────────


_ROLE_MARKER_CACHE: dict[int, list[int]] = {}


def _assistant_role_marker_ids(tokenizer) -> list[int]:
    key = id(tokenizer)
    if key in _ROLE_MARKER_CACHE:
        return _ROLE_MARKER_CACHE[key]
    ids = tokenizer.encode("<|im_start|>assistant\n", add_special_tokens=False)
    _ROLE_MARKER_CACHE[key] = ids
    return ids


def _encode_prefix(messages: list[dict], tokenizer, apply_kwargs: dict | None = None) -> list[int]:
    if not messages:
        return []
    apply_kwargs = apply_kwargs or {}
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
        **apply_kwargs,
    )
    return tokenizer.encode(text, add_special_tokens=False)


def tokenize_trajectory(
    messages: list[dict],
    tokenizer,
    apply_chat_template_kwargs: dict | None = None,
) -> tuple[list[int], list[int], int]:
    """Text-only multi-turn tokenizer (legacy / fallback)."""
    if not messages:
        return [], [], 0

    messages = _normalize_tool_call_arguments(messages)
    first_asst = next((i for i, m in enumerate(messages) if m["role"] == "assistant"), None)
    if first_asst is None:
        return _encode_prefix(messages, tokenizer, apply_chat_template_kwargs), [], 0

    prompt_msgs = messages[:first_asst]
    response_msgs = messages[first_asst:]
    prompt_ids = _encode_prefix(prompt_msgs, tokenizer, apply_chat_template_kwargs)
    full_ids = _encode_prefix(messages, tokenizer, apply_chat_template_kwargs)

    if full_ids[: len(prompt_ids)] != prompt_ids:
        LOG.warning("prompt prefix mismatch — chat template behaves differently with assistants present")
        prompt_ids = full_ids[: len(prompt_ids)]

    response_tokens = full_ids[len(prompt_ids) :]
    role_marker = _assistant_role_marker_ids(tokenizer)
    role_marker_len = len(role_marker)

    loss_mask: list[int] = []
    cumulative = list(prompt_msgs)
    cursor = len(prompt_ids)
    for msg in response_msgs:
        cumulative.append(msg)
        cum_ids = _encode_prefix(cumulative, tokenizer, apply_chat_template_kwargs)
        if len(cum_ids) < cursor:
            LOG.warning("non-monotonic chat template render at role=%s; treating as observation", msg["role"])
            delta = 0
            cursor_advance = 0
        else:
            delta = len(cum_ids) - cursor
            cursor_advance = delta

        if msg["role"] == "assistant" and delta >= role_marker_len:
            delta_ids = cum_ids[cursor : cursor + role_marker_len]
            if delta_ids == role_marker:
                loss_mask.extend([0] * role_marker_len + [1] * (delta - role_marker_len))
            else:
                loss_mask.extend([1] * delta)
        else:
            loss_mask.extend([1 if msg["role"] == "assistant" else 0] * delta)

        cursor += cursor_advance

    if len(loss_mask) != len(response_tokens):
        if len(loss_mask) > len(response_tokens):
            loss_mask = loss_mask[: len(response_tokens)]
        else:
            loss_mask.extend([0] * (len(response_tokens) - len(loss_mask)))

    return prompt_ids + response_tokens, loss_mask, len(response_tokens)


# ─── multimodal tokenize ─────────────────────────────────────────────────────


def _build_loss_mask_from_ids(input_ids: list[int], specials: dict[str, int]) -> list[int]:
    """Walk ``input_ids`` and build a loss mask matching its length.

    Algorithm:
    1. Find every ``<|im_start|>`` position — these are message segment starts.
    2. For each segment, the role token sits at ``start + 1`` (e.g. ``assistant``
       at id 74455). Role marker is exactly 3 tokens: ``<|im_start|>``,
       ``{role}``, ``\\n``.
    3. Find ``<|im_end|>`` within the segment.
    4. If role == assistant: tokens from ``start+3`` through ``im_end_pos``
       (inclusive) get loss=1; everything else (role marker, tokens between
       segments, non-assistant content + their image_pad) stays 0.

    Length of returned mask equals ``len(input_ids)``.
    """
    n = len(input_ids)
    mask = [0] * n
    if n == 0:
        return mask

    im_end = specials["im_end"]
    asst_role = specials["role_assistant"]

    boundaries = _message_boundaries(input_ids, specials)
    if not boundaries:
        return mask

    boundaries.append(n)  # sentinel
    for k in range(len(boundaries) - 1):
        start = boundaries[k]
        next_start = boundaries[k + 1]
        if start + 1 >= n:
            continue
        role_tok = input_ids[start + 1]
        if role_tok != asst_role:
            continue
        # Find <|im_end|> in [start, next_start)
        im_end_pos = -1
        for j in range(start, next_start):
            if input_ids[j] == im_end:
                im_end_pos = j
                break
        if im_end_pos < 0:
            # Truncated mid-assistant; mask through end of segment.
            im_end_pos = next_start - 1
        # Role marker = 3 tokens: positions start, start+1, start+2 stay 0.
        # Content + closing <|im_end|> = 1.
        for i in range(start + 3, im_end_pos + 1):
            if i < n:
                mask[i] = 1
    return mask


def _find_first_assistant_pos(input_ids: list[int], specials: dict[str, int]) -> int | None:
    """Return the position of the first ``<|im_start|>`` whose role is
    assistant. Used to split prompt vs response."""
    asst_role = specials["role_assistant"]
    for i in _message_boundaries(input_ids, specials):
        if input_ids[i + 1] == asst_role:
            return i
    return None


def tokenize_trajectory_multimodal(
    messages: list[dict],
    tokenizer,
    processor,
    apply_chat_template_kwargs: dict | None = None,
) -> tuple[list[int], list[int], int, dict | None, dict | None]:
    """Multimodal (Qwen3-VL) trajectory tokenizer.

    Steps:
    1. Preprocess kira messages: (a) ``fold_native_image_read_messages_for_qwen``
       moves images out of standalone user turns into the matching ``image_read``
       tool reply (Qwen + ms-swift SFT layout); (b) decode ``data:`` image URLs
       into PIL.Image; (c) normalize assistant ``tool_calls`` arguments from
       JSON strings to dicts so the Jinja template renders.
    2. ``processor.apply_chat_template(rewritten, tokenize=False)`` → text with
       unexpanded ``<|image_pad|>`` placeholders inside
       ``<tool_response>...</tool_response>`` blocks.
    3. ``processor(text=..., images=[PIL,...])`` → ``input_ids`` with
       proper image-pad expansion (count derived from ``image_grid_thw`` per
       image) plus the ``multimodal_train_inputs`` dict (``pixel_values``,
       ``image_grid_thw``, etc.) that the Megatron actor consumes.
    4. Build loss mask by walking ``input_ids`` between ``<|im_start|>`` /
       ``<|im_end|>`` boundaries; assign loss=1 only inside assistant content
       (tool segments — including their image_pad tokens — inherit loss=0).

    Returns:
        ``(tokens, loss_mask, response_length, multimodal_inputs, multimodal_train_inputs)``

        - ``tokens``: full sequence (prompt + response), len = total input_ids.
        - ``loss_mask``: response-only mask, len = response_length.
        - ``response_length``: tokens from first assistant ``<|im_start|>``
          through the end of input_ids.
        - ``multimodal_inputs``: ``{"images": [PIL,...]}`` (None if no images).
          Used for partial-rollout resume so we don't re-decode data URLs.
        - ``multimodal_train_inputs``: processor outputs minus
          ``input_ids``/``attention_mask`` (None if no images). Megatron actor
          forward-passes these through the vision tower.
    """
    if not messages:
        return [], [], 0, None, None

    apply_kwargs = apply_chat_template_kwargs or {}
    rewritten, pil_images = _kira_to_relax_messages(messages)

    # Step 2: render text (template handles vision_start/vision_end placeholders)
    text = processor.apply_chat_template(
        rewritten, tokenize=False, add_generation_prompt=False, **apply_kwargs,
    )

    # Step 3: processor expands <|image_pad|> based on each image's grid_thw
    proc_kwargs: dict[str, Any] = {
        "text": text,
        "return_tensors": None,
        "return_mm_token_type_ids": False,
    }
    if pil_images:
        proc_kwargs["images"] = pil_images
    out = processor(**proc_kwargs)

    raw_ids = out["input_ids"]
    if hasattr(raw_ids, "tolist"):
        raw_ids = raw_ids.tolist()
    # processor returns either [seq] (return_tensors=None) or [[seq]] depending
    # on version; flatten the outer batch dim if present.
    if raw_ids and isinstance(raw_ids[0], list):
        input_ids = list(raw_ids[0])
    else:
        input_ids = list(raw_ids)

    # Build train-input dict (everything except input_ids/attention_mask)
    multimodal_train_inputs: dict | None = None
    if pil_images:
        multimodal_train_inputs = {
            k: v
            for k, v in out.items()
            if k not in ("input_ids", "attention_mask", "mm_token_type_ids")
        } or None
    multimodal_inputs = {"images": pil_images} if pil_images else None

    specials = _specials(tokenizer)

    # Step 4: loss mask — walk full input_ids, then slice to response side
    full_mask = _build_loss_mask_from_ids(input_ids, specials)

    # Split prompt vs response at first assistant <|im_start|>
    first_asst_pos = _find_first_assistant_pos(input_ids, specials)
    if first_asst_pos is None:
        # No assistant in trajectory → all prompt, no response.
        return input_ids, [], 0, multimodal_inputs, multimodal_train_inputs

    response_length = len(input_ids) - first_asst_pos
    loss_mask = full_mask[first_asst_pos:]
    return input_ids, loss_mask, response_length, multimodal_inputs, multimodal_train_inputs
