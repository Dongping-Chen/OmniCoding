"""image_read static helpers + sub-LLM (mocked) and native paths."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from omnicoding.agents.kira import image_read as kira_image_read
from omnicoding.agents.kira.image_read import (
    ImageReadError,
    NativeImageReadResult,
    _read_image_b64,
    _resolve_image_path,
    read_image,
    read_image_native,
)


# 1×1 transparent PNG (88 bytes)
_TINY_PNG = (
    b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01"
    b"\x08\x06\x00\x00\x00\x1f\x15\xc4\x89\x00\x00\x00\rIDATx\x9cc\xfc"
    b"\xff\xff?\x03\x00\x06\x05\x02\x80\xa3\xfeL\xab\x00\x00\x00\x00IEND"
    b"\xaeB`\x82"
)


# ---------- _resolve_image_path -------------------------------------

def test_resolve_absolute_path_unchanged(tmp_path: Path):
    abs_path = tmp_path / "x.png"
    out = _resolve_image_path(str(abs_path), workspace=Path("/elsewhere"))
    assert out == abs_path


def test_resolve_relative_path_against_workspace(tmp_path: Path):
    out = _resolve_image_path("img/foo.png", workspace=tmp_path)
    assert out == (tmp_path / "img" / "foo.png").resolve()


# ---------- _read_image_b64 -----------------------------------------

def test_read_b64_png_returns_mime_and_b64(tmp_path: Path):
    p = tmp_path / "tiny.png"
    p.write_bytes(_TINY_PNG)
    mime, b64 = _read_image_b64(p)
    assert mime == "image/png"
    assert len(b64) > 0
    # b64 of ~70 bytes is ~96 chars; sanity check
    assert len(b64) > 60


def test_read_b64_missing_file_raises():
    with pytest.raises(ImageReadError, match="not found"):
        _read_image_b64(Path("/nonexistent/never.png"))


def test_read_b64_unsupported_extension_raises(tmp_path: Path):
    p = tmp_path / "doc.txt"
    p.write_bytes(b"hello")
    with pytest.raises(ImageReadError, match="unsupported image extension"):
        _read_image_b64(p)


def test_read_b64_directory_raises(tmp_path: Path):
    sub = tmp_path / "subdir.png"  # name has png ext but is a dir
    sub.mkdir()
    with pytest.raises(ImageReadError, match="not a regular file"):
        _read_image_b64(sub)


# ---------- read_image (full path with mocked LLM) ------------------

def test_read_image_happy_path(tmp_path: Path):
    p = tmp_path / "scene.png"
    p.write_bytes(_TINY_PNG)
    with patch.object(kira_image_read, "call_llm_for_image",
                      return_value="A red square.") as m:
        out = read_image(
            file_path="scene.png",
            instruction="describe",
            workspace=tmp_path,
            model_name="openai/Qwen3.6-27B",
            api_base="http://x:8080/v1",
            api_key="k",
        )
    assert "A red square." in out
    assert "image_read result for 'scene.png'" in out
    # Multimodal content was assembled correctly.
    sent_messages = m.call_args.kwargs["messages"]
    assert sent_messages[0]["role"] == "user"
    parts = sent_messages[0]["content"]
    assert any(p.get("type") == "image_url" for p in parts)
    assert any(p.get("type") == "text" for p in parts)


def test_read_image_missing_file_returns_error_string(tmp_path: Path):
    out = read_image(
        file_path="missing.png",
        instruction="describe",
        workspace=tmp_path,
        model_name="x",
    )
    assert out.startswith("ERROR:")
    assert "not found" in out


def test_read_image_llm_failure_returns_error_string(tmp_path: Path):
    p = tmp_path / "x.png"
    p.write_bytes(_TINY_PNG)
    with patch.object(kira_image_read, "call_llm_for_image",
                      side_effect=RuntimeError("503 Service Unavailable")):
        out = read_image(
            file_path="x.png",
            instruction="describe",
            workspace=tmp_path,
            model_name="x",
        )
    assert out.startswith("ERROR:")
    assert "503" in out


def test_read_image_unsupported_extension_returns_error_string(tmp_path: Path):
    p = tmp_path / "doc.bmp"
    p.write_bytes(b"\x00\x00")
    out = read_image(
        file_path="doc.bmp",
        instruction="describe",
        workspace=tmp_path,
        model_name="x",
    )
    assert out.startswith("ERROR:")
    assert "unsupported" in out.lower()


# ---------- read_image_native (no LLM call, image goes to caller) ----

def test_read_image_native_happy_path(tmp_path: Path):
    p = tmp_path / "scene.png"
    p.write_bytes(_TINY_PNG)
    out = read_image_native(
        file_path="scene.png",
        instruction="Describe the image.",
        workspace=tmp_path,
    )
    assert isinstance(out, NativeImageReadResult)
    assert out.user_content is not None
    # Standard chat-completions multimodal shape: one text + one image_url.
    assert len(out.user_content) == 2
    assert out.user_content[0]["type"] == "text"
    assert "scene.png" in out.user_content[0]["text"]
    assert out.user_content[1]["type"] == "image_url"
    url = out.user_content[1]["image_url"]["url"]
    assert url.startswith("data:image/png;base64,")
    # Sanity: the embedded base64 decodes back to the original bytes.
    import base64 as _b64
    encoded = url.split(",", 1)[1]
    assert _b64.b64decode(encoded) == _TINY_PNG
    # Tool ack mentions the path so the model has something coherent
    # before the image arrives.
    assert "scene.png" in out.tool_text
    assert out.mime == "image/png"
    assert out.image_bytes == len(_TINY_PNG)
    assert out.resolved_path == str((tmp_path / "scene.png").resolve())


def test_read_image_native_prelude_is_short(tmp_path: Path):
    """Lock the trimmed prelude — every extra token repeats per
    image_read call and the model already has its instruction in the
    prior assistant turn. Keep tool_text + user text under a token
    budget instead of growing back into a paragraph next refactor."""
    p = tmp_path / "scene.png"
    p.write_bytes(_TINY_PNG)
    out = read_image_native(
        file_path="scene.png",
        instruction="Describe in detail what you see, every color, every object.",
        workspace=tmp_path,
    )
    # Tool ack ≤ 80 chars (path + mime + bytes).
    assert len(out.tool_text) <= 80, f"tool_text too long: {out.tool_text!r}"
    # User text ≤ 80 chars (path identifier only — instruction already
    # rides in the prior assistant tool_call).
    user_text = out.user_content[0]["text"]
    assert len(user_text) <= 80, f"user text too long: {user_text!r}"
    # Instruction MUST NOT be re-injected in the user message: the
    # model wrote it itself one turn ago.
    assert "Describe in detail" not in user_text, (
        f"instruction should not be echoed: {user_text!r}"
    )


def test_read_image_native_missing_file_returns_error(tmp_path: Path):
    out = read_image_native(
        file_path="missing.png",
        instruction="describe",
        workspace=tmp_path,
    )
    assert out.user_content is None
    assert out.tool_text.startswith("ERROR:")
    assert "not found" in out.tool_text
    assert out.resolved_path is None
    assert out.mime is None


def test_read_image_native_unsupported_extension_returns_error(tmp_path: Path):
    p = tmp_path / "doc.bmp"
    p.write_bytes(b"\x00\x00")
    out = read_image_native(
        file_path="doc.bmp",
        instruction="describe",
        workspace=tmp_path,
    )
    assert out.user_content is None
    assert out.tool_text.startswith("ERROR:")
    assert "unsupported" in out.tool_text.lower()


def test_read_image_native_does_not_call_llm(tmp_path: Path):
    """Critical contract: native mode must NOT consume any LLM tokens."""
    p = tmp_path / "x.png"
    p.write_bytes(_TINY_PNG)
    with patch.object(
        kira_image_read, "call_llm_for_image",
        side_effect=AssertionError("native mode must not call sub-LLM"),
    ):
        out = read_image_native(
            file_path="x.png",
            instruction="describe",
            workspace=tmp_path,
        )
    assert out.user_content is not None  # success despite no LLM mock


def test_read_image_native_subcall_log_records_bytes(tmp_path: Path):
    p = tmp_path / "y.png"
    p.write_bytes(_TINY_PNG)
    log = tmp_path / "subcalls.jsonl"
    out = read_image_native(
        file_path="y.png",
        instruction="describe",
        workspace=tmp_path,
        subcall_log_path=log,
    )
    assert out.user_content is not None
    rec = json.loads(log.read_text().strip())
    assert rec["mode"] == "native"
    assert rec["status"] == "ok"
    assert rec["mime"] == "image/png"
    assert rec["image_bytes"] == len(_TINY_PNG)
    assert rec["image_b64"]   # bytes captured for SFT/RL replay
    # No LLM response field — there is no LLM call.
    assert "response_text" not in rec or rec.get("response_text") is None


def test_read_image_native_subcall_log_records_failures(tmp_path: Path):
    log = tmp_path / "subcalls.jsonl"
    out = read_image_native(
        file_path="missing.png",
        instruction="describe",
        workspace=tmp_path,
        subcall_log_path=log,
    )
    assert out.user_content is None
    rec = json.loads(log.read_text().strip())
    assert rec["status"] == "input_rejected"
    assert "not found" in rec["error"]
    assert rec["image_b64"] is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
