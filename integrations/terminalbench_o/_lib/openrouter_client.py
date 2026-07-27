"""Thin OpenRouter client used by all TerminalBench-O graders.

- Reads OPENROUTER_API_KEY / OPENROUTER_BASE_URL from the process environment.
- Default model: openai/gpt-5.4-mini  (cheap, multimodal, good enough).
- Provides text + vision helpers, plus a JSON-mode helper.
"""
from __future__ import annotations

import base64
import json
import mimetypes
import os
import time
from pathlib import Path
from typing import Any, Iterable

# --------------------------------------------------------------------------- #
# OpenAI SDK (works with OpenRouter base_url)
# --------------------------------------------------------------------------- #
def _timeout_seconds() -> float:
    raw = (
        os.environ.get("CLAW_BENCH_OPENAI_TIMEOUT_SEC")
        or os.environ.get("CLAW_BENCH_JUDGE_TIMEOUT_SEC")
        or "90"
    )
    try:
        return max(1.0, float(raw))
    except ValueError:
        return 90.0


def _client(api_key: str | None = None, base_url: str | None = None):
    from openai import OpenAI
    return OpenAI(
        api_key=api_key or os.environ["OPENROUTER_API_KEY"],
        base_url=base_url or os.environ.get(
            "OPENROUTER_BASE_URL",
            "https://openrouter.ai/api/v1",
        ),
        timeout=_timeout_seconds(),
    )


DEFAULT_MODEL = os.environ.get("CLAW_BENCH_AGENT_MODEL", "openai/gpt-5.4-mini")


# --------------------------------------------------------------------------- #
# Image encoding helper
# --------------------------------------------------------------------------- #
def _img_data_url(path: str | Path) -> str:
    p = Path(path)
    mime = mimetypes.guess_type(str(p))[0] or "image/jpeg"
    b64 = base64.b64encode(p.read_bytes()).decode()
    return f"data:{mime};base64,{b64}"


# --------------------------------------------------------------------------- #
# Public helpers
# --------------------------------------------------------------------------- #
def chat(
    messages: list[dict],
    model: str | None = None,
    temperature: float = 0.2,
    max_tokens: int = 2048,
    response_format: dict | None = None,
    retries: int = 3,
    backoff: float = 2.0,
    api_key: str | None = None,
    base_url: str | None = None,
) -> str:
    """Plain chat completion. Returns the assistant's text content."""
    cli = _client(api_key=api_key, base_url=base_url)
    last_err = None
    for attempt in range(retries):
        try:
            kwargs: dict[str, Any] = dict(
                model=model or DEFAULT_MODEL,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            if response_format:
                kwargs["response_format"] = response_format
            r = cli.chat.completions.create(**kwargs)
            return r.choices[0].message.content or ""
        except Exception as e:
            last_err = e
            if attempt + 1 < retries:
                time.sleep(backoff ** attempt)
    raise RuntimeError(f"chat failed after {retries} attempts: {last_err}")


def chat_json(messages: list[dict], **kw) -> Any:
    """Chat with JSON response_format; returns parsed object."""
    def _parse(raw: str) -> Any:
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            # last-resort: strip code fences / language hints
            s = raw.strip().strip("`")
            if s.startswith("json"):
                s = s[4:].lstrip()
            try:
                return json.loads(s)
            except json.JSONDecodeError:
                if "{" in s and "}" in s:
                    try:
                        return json.loads(s[s.find("{"):s.rfind("}") + 1])
                    except json.JSONDecodeError:
                        pass
                lowered = s.strip().lower()
                if lowered in {"yes", "no"}:
                    return {"answer": lowered}
                raise

    try:
        raw = chat(messages, response_format={"type": "json_object"}, **kw)
        return _parse(raw)
    except Exception:
        # Some OpenRouter providers/models do not support response_format.
        # Fall back to a normal chat call and rely on the prompt's strict
        # JSON instruction plus the parser above.
        raw = chat(messages, **kw)
        return _parse(raw)


def vision(
    prompt: str,
    images: Iterable[str | Path],
    model: str | None = None,
    detail: str = "low",
    **kw,
) -> str:
    """Single-shot vision call: one user message with text + N images."""
    content = [{"type": "text", "text": prompt}]
    for img in images:
        content.append({
            "type": "image_url",
            "image_url": {"url": _img_data_url(img), "detail": detail},
        })
    return chat([{"role": "user", "content": content}], model=model, **kw)


def vision_json(prompt: str, images: Iterable[str | Path], **kw) -> Any:
    raw = vision(prompt, images, **kw)
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        s = raw.strip().strip("`")
        if s.startswith("json"):
            s = s[4:].lstrip()
        return json.loads(s)


# --------------------------------------------------------------------------- #
# Smoke test
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    print("model:", DEFAULT_MODEL)
    out = chat([{"role": "user", "content": "Reply with the single word: pong"}], max_tokens=32)
    print("response:", repr(out))
