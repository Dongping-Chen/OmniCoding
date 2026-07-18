"""KIRA's image_read tool — two modes.

``read_image`` (sub-LLM mode): base64 → multimodal LLM → text
description. Used historically when the main agent runs a text-only
provider, or when keeping the image-token bill off the main session's
context budget matters more than feeding raw pixels to the agent.

``read_image_native`` (default): base64 → return image content blocks
that the dispatch loop attaches as a follow-up ``user`` message right
after the ``role=tool`` ack. The main agent then sees the actual image
in its conversation history. No sub-LLM call. Used when the main
agent is itself multimodal (gpt-5.x, qwen 2.5/3.x VL, …) — it removes
the train/serve gap where SFT data has rendered images but the
inference harness substitutes a description. Required for native
multimodal SFT / RL.

Both modes go through the same path resolver + base64 reader, both
honor ``subcall_log_path`` for SFT/RL replay (raw bytes + request
shape), and both return a string starting with ``ERROR:`` on failure
so a single bad image read doesn't kill the run.
"""

from __future__ import annotations

import base64
import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from omnicoding.agents.kira.llm import call_llm_for_image

LOGGER = logging.getLogger("kira.image_read")

_MIME_BY_EXT = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".gif": "image/gif",
    ".webp": "image/webp",
}


class ImageReadError(RuntimeError):
    pass


def _resolve_image_path(file_path: str, workspace: Path) -> Path:
    p = Path(file_path)
    if p.is_absolute():
        return p
    return (workspace / p).resolve()


def _read_image_b64(path: Path) -> tuple[str, str]:
    if not path.exists():
        raise ImageReadError(f"image not found: {path}")
    if not path.is_file():
        raise ImageReadError(f"not a regular file: {path}")
    ext = path.suffix.lower()
    mime = _MIME_BY_EXT.get(ext)
    if mime is None:
        raise ImageReadError(
            f"unsupported image extension '{ext}'. Convert to PNG and "
            f"call image_read on the converted file."
        )
    raw = path.read_bytes()
    return mime, base64.b64encode(raw).decode("ascii")


def _append_subcall_log(path: Path, record: dict[str, Any]) -> None:
    """Best-effort JSONL append. Never raises — the agent run must keep
    going even if the log dir is unwritable."""
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    except Exception as exc:  # noqa: BLE001
        LOGGER.warning("image_read subcall log write failed (%s): %s", path, exc)


def read_image(
    *,
    file_path: str,
    instruction: str,
    workspace: Path,
    model_name: str,
    provider: str | None = None,
    api_base: str | None = None,
    api_key: str | None = None,
    request_timeout_s: int = 600,
    block_timeout_s: int = 600,
    subcall_log_path: Path | None = None,
) -> str:
    """Resolve the file, base64-encode, ship to the multimodal LLM, and
    return the text description (or an error string starting with
    ``ERROR:`` so the agent gets feedback rather than crashing the
    whole run).

    When ``subcall_log_path`` is supplied, append a JSONL record
    containing the original arguments, raw image base64, sub-LLM
    request shape, and response text. Records are always written —
    success, input-validation failure, and sub-LLM-call failure all
    show up — so SFT/RL replay can reconstruct exactly what the
    multimodal sub-LLM saw and produced.
    """
    started = time.time()
    log_record: dict[str, Any] = {
        "ts": started,
        "file_path_arg": file_path,
        "instruction": instruction,
        "model_name": model_name,
        "api_base": api_base,
    }
    try:
        path = _resolve_image_path(file_path, workspace)
        mime, b64 = _read_image_b64(path)
    except ImageReadError as exc:
        LOGGER.warning("image_read input rejected: %s", exc)
        if subcall_log_path is not None:
            log_record.update({
                "resolved_path": None,
                "mime": None,
                "image_b64": None,
                "image_bytes": 0,
                "status": "input_rejected",
                "error": str(exc),
                "response_text": None,
                "elapsed_s": time.time() - started,
            })
            _append_subcall_log(subcall_log_path, log_record)
        return f"ERROR: {exc}"

    # Chat-completions multimodal shape: works directly for Qwen sglang
    # and Anthropic, and is also what litellm + codex-router expect on
    # the wire — the router translates this to the gpt-5.x Responses-API
    # ``input_image`` shape internally.
    messages = [{
        "role": "user",
        "content": [
            {"type": "text", "text": instruction},
            {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}},
        ],
    }]

    LOGGER.info(
        "image_read mime=%s bytes=%d model=%s",
        mime, len(b64) // 4 * 3, model_name,
    )
    log_record.update({
        "resolved_path": str(path),
        "mime": mime,
        "image_b64": b64,
        "image_bytes": len(b64) // 4 * 3,
        "request_messages": messages,
    })
    try:
        text = call_llm_for_image(
            messages=messages,
            model_name=model_name,
            provider=provider,
            api_base=api_base,
            api_key=api_key,
            request_timeout_s=request_timeout_s,
            block_timeout_s=block_timeout_s,
        )
    except Exception as exc:  # noqa: BLE001 — surface to agent as ERROR
        LOGGER.warning("image_read LLM call failed: %s", exc)
        if subcall_log_path is not None:
            log_record.update({
                "status": "llm_call_failed",
                "error": f"{type(exc).__name__}: {exc}",
                "response_text": None,
                "elapsed_s": time.time() - started,
            })
            _append_subcall_log(subcall_log_path, log_record)
        return f"ERROR: image_read LLM call failed: {exc}"

    if not isinstance(text, str):
        text = str(text)
    if subcall_log_path is not None:
        log_record.update({
            "status": "ok",
            "error": None,
            "response_text": text,
            "elapsed_s": time.time() - started,
        })
        _append_subcall_log(subcall_log_path, log_record)
    return f"image_read result for '{file_path}':\n{text}"


@dataclass
class NativeImageReadResult:
    """Output of ``read_image_native``.

    - ``tool_text`` is what the dispatch loop should put in the
      ``role=tool`` reply (short ack, or an ``ERROR:`` string if the
      read failed).
    - ``user_content`` is the multimodal content list to attach as a
      follow-up ``role=user`` message. ``None`` on error → no image
      should be injected.
    - ``file_path_arg`` and ``resolved_path`` are echoed for the
      dispatcher's record/log path.
    - ``mime`` and ``image_bytes`` are populated on success and let the
      dispatcher size telemetry without re-decoding.
    """

    tool_text: str
    user_content: list[dict[str, Any]] | None
    file_path_arg: str
    resolved_path: str | None
    mime: str | None
    image_bytes: int


def read_image_native(
    *,
    file_path: str,
    instruction: str,
    workspace: Path,
    subcall_log_path: Path | None = None,
) -> NativeImageReadResult:
    """Read an image from disk and prepare it for direct injection into
    the main agent's conversation. Does NOT make any LLM call.

    Returns a ``NativeImageReadResult``. The dispatch loop is expected
    to:
      1. Append ``tool_text`` as the ``role=tool`` reply for the
         original ``image_read`` tool_call (so OpenAI's call/response
         pairing is satisfied).
      2. Append ``user_content`` as a follow-up ``role=user`` message
         so the main agent sees the actual pixels on its next turn.

    The ``user_content`` shape is the standard chat-completions
    multimodal one (``[{"type":"text",...},{"type":"image_url",...}]``)
    that litellm + codex-router and Anthropic both accept. The
    ``image_url`` is a ``data:`` URL holding the original bytes.

    On failure (missing file, unsupported extension, etc.) the ``ERROR:
    ...`` text is placed in ``tool_text`` and ``user_content`` is set
    to ``None`` so no broken image gets injected — the agent gets the
    same recoverable feedback as the sub-LLM path.

    The ``subcall_log`` JSONL record is written with ``mode="native"``
    and no ``response_text`` field (there is no LLM response to
    record); SFT/RL replay still finds the raw bytes here.
    """
    started = time.time()
    log_record: dict[str, Any] = {
        "ts": started,
        "file_path_arg": file_path,
        "instruction": instruction,
        "mode": "native",
    }
    try:
        path = _resolve_image_path(file_path, workspace)
        mime, b64 = _read_image_b64(path)
    except ImageReadError as exc:
        LOGGER.warning("image_read input rejected (native): %s", exc)
        if subcall_log_path is not None:
            log_record.update({
                "resolved_path": None,
                "mime": None,
                "image_b64": None,
                "image_bytes": 0,
                "status": "input_rejected",
                "error": str(exc),
                "elapsed_s": time.time() - started,
            })
            _append_subcall_log(subcall_log_path, log_record)
        return NativeImageReadResult(
            tool_text=f"ERROR: {exc}",
            user_content=None,
            file_path_arg=file_path,
            resolved_path=None,
            mime=None,
            image_bytes=0,
        )

    # Decode-once to get exact byte count; b64 length / 4 * 3 over-counts
    # by the padding bytes (1-2 bytes per image). Tests pin exact bytes.
    image_bytes = len(base64.b64decode(b64))
    LOGGER.info(
        "image_read native mime=%s bytes=%d path=%s",
        mime, image_bytes, path,
    )
    if subcall_log_path is not None:
        log_record.update({
            "resolved_path": str(path),
            "mime": mime,
            "image_b64": b64,
            "image_bytes": image_bytes,
            "status": "ok",
            "error": None,
            "elapsed_s": time.time() - started,
        })
        _append_subcall_log(subcall_log_path, log_record)

    # Keep the prelude tight — every extra token here repeats per
    # image_read call and the model already has its own instruction in
    # the prior assistant turn, so nudging it again is pure waste.
    user_content: list[dict[str, Any]] = [
        {"type": "text", "text": f"image_read: '{file_path}'"},
        {
            "type": "image_url",
            "image_url": {"url": f"data:{mime};base64,{b64}"},
        },
    ]
    tool_text = f"Loaded '{file_path}' ({mime}, {image_bytes} bytes)."
    return NativeImageReadResult(
        tool_text=tool_text,
        user_content=user_content,
        file_path_arg=file_path,
        resolved_path=str(path),
        mime=mime,
        image_bytes=image_bytes,
    )
