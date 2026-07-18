"""Tavily web search with multi-key rotation.

Reads one key from ``$TAVILY_API_KEY`` or multiple keys from the explicitly
configured ``$TAVILY_KEYS_FILE`` (one per line; blank/`#`-prefixed lines are
ignored). On HTTP 429 / quota error the current key is retired (this process
only) and the next non-blacklisted key is tried; raise ``TavilyExhausted`` if
every key is dead.
"""

from __future__ import annotations

import json
import os
import random
import threading
from pathlib import Path
from typing import Any

import requests

TAVILY_ENDPOINT = "https://api.tavily.com/search"
DEFAULT_TIMEOUT_S = 30.0

_lock = threading.Lock()
_blacklist: set[str] = set()


class TavilyError(RuntimeError):
    pass


class TavilyExhausted(TavilyError):
    """All known Tavily API keys returned quota / auth errors."""


def _load_keys() -> list[str]:
    env_key = os.environ.get("TAVILY_API_KEY", "").strip()
    if env_key:
        return [env_key]

    configured_path = os.environ.get("TAVILY_KEYS_FILE", "").strip()
    if not configured_path:
        raise TavilyError(
            "no Tavily keys configured: set TAVILY_API_KEY or TAVILY_KEYS_FILE"
        )
    path = Path(configured_path).expanduser()
    if not path.is_file():
        raise TavilyError(f"TAVILY_KEYS_FILE does not exist or is not a file: {path}")
    keys: list[str] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        # Strip inline `# owner` annotations after the key.
        if "#" in line:
            line = line.split("#", 1)[0].strip()
        if line:
            keys.append(line)
    if not keys:
        raise TavilyError(f"no usable keys parsed from {path}")
    return keys


def _is_quota_error(status: int, body: str) -> bool:
    if status in (401, 402, 403, 429):
        return True
    body_lower = body.lower()
    return any(
        marker in body_lower
        for marker in ("quota", "rate limit", "exceeded", "unauthorized", "invalid api key")
    )


def search(
    query: str,
    *,
    max_results: int = 5,
    search_depth: str = "basic",
    include_answer: bool = True,
    include_raw_content: bool = False,
    timeout_s: float = DEFAULT_TIMEOUT_S,
) -> dict[str, Any]:
    """Call Tavily ``/search`` with key rotation on quota errors.

    Returns the raw JSON response from Tavily on success.
    """
    keys = _load_keys()
    with _lock:
        live = [k for k in keys if k not in _blacklist]
    if not live:
        raise TavilyExhausted("all Tavily keys blacklisted in this process")
    random.shuffle(live)
    last_err: str | None = None
    for key in live:
        payload = {
            "api_key": key,
            "query": query,
            "max_results": max_results,
            "search_depth": search_depth,
            "include_answer": include_answer,
            "include_raw_content": include_raw_content,
        }
        try:
            r = requests.post(TAVILY_ENDPOINT, json=payload, timeout=timeout_s)
        except requests.RequestException as exc:
            last_err = f"network: {exc}"
            continue
        if r.status_code == 200:
            return r.json()
        if _is_quota_error(r.status_code, r.text):
            with _lock:
                _blacklist.add(key)
            last_err = f"key dead ({r.status_code}): {r.text[:200]}"
            continue
        # Non-quota error (e.g. 400 bad request): bubble up — no point trying other keys.
        raise TavilyError(f"Tavily {r.status_code}: {r.text[:500]}")
    raise TavilyExhausted(f"all keys failed; last={last_err}")


def format_markdown(payload: dict[str, Any]) -> str:
    """Render Tavily payload into a compact markdown block for the model."""
    lines: list[str] = []
    q = payload.get("query") or ""
    if q:
        lines.append(f"# Web search: {q}")
    answer = payload.get("answer")
    if answer:
        lines.append("")
        lines.append("## Answer")
        lines.append(str(answer).strip())
    results = payload.get("results") or []
    if results:
        lines.append("")
        lines.append("## Results")
        for i, hit in enumerate(results, 1):
            title = (hit.get("title") or "").strip() or "(no title)"
            url = (hit.get("url") or "").strip()
            content = (hit.get("content") or "").strip()
            score = hit.get("score")
            lines.append(f"{i}. **{title}**")
            if url:
                lines.append(f"   {url}")
            if score is not None:
                lines.append(f"   score: {score:.3f}")
            if content:
                snippet = content if len(content) <= 600 else content[:600].rstrip() + "…"
                lines.append(f"   > {snippet}")
    if not results and not answer:
        lines.append("(no results)")
    return "\n".join(lines).strip() + "\n"


def search_text(query: str, **kwargs: Any) -> str:
    """Convenience: search + markdown format."""
    return format_markdown(search(query, **kwargs))


if __name__ == "__main__":
    import sys

    args = sys.argv[1:]
    if not args:
        print("usage: tavily_search.py <query> [max_results]", file=sys.stderr)
        sys.exit(2)
    q = args[0]
    n = int(args[1]) if len(args) > 1 else 5
    try:
        print(search_text(q, max_results=n))
    except TavilyError as exc:
        print(f"tavily error: {exc}", file=sys.stderr)
        sys.exit(1)
