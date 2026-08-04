"""Tavily web search with multi-key rotation.

Reads one key from ``$TAVILY_API_KEY`` or multiple keys from the explicitly
configured ``$TAVILY_KEYS_FILE`` (one per line; blank/`#`-prefixed lines are
ignored). Permanent authentication/quota failures retire a key for this
process. HTTP 429 rate limits instead honor ``Retry-After`` with a temporary
cooldown. A key is leased to at most one upstream request at a time so a local
burst cannot stampede the same development key.
"""

from __future__ import annotations

import json
import os
import random
import threading
import time
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Any

import requests

TAVILY_ENDPOINT = "https://api.tavily.com/search"
DEFAULT_TIMEOUT_S = 30.0

_lock = threading.Lock()
_state_changed = threading.Condition(_lock)
_blacklist: set[str] = set()
_cooldown_until: dict[str, float] = {}
_in_flight: set[str] = set()

DEFAULT_RATE_LIMIT_COOLDOWN_S = 60.0
MIN_KEY_START_INTERVAL_S = 0.6
MAX_TOTAL_TIMEOUT_S = 40.0


class TavilyError(RuntimeError):
    pass


class TavilyExhausted(TavilyError):
    """All known Tavily API keys returned quota / auth errors."""


class TavilyRateLimited(TavilyError):
    """All usable Tavily keys are temporarily rate-limited."""


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
    if status in (401, 402, 403):
        return True
    body_lower = body.lower()
    return any(
        marker in body_lower
        for marker in (
            "quota",
            "usage limit",
            "credit limit",
            "monthly limit",
            "plan limit",
            "unauthorized",
            "invalid api key",
        )
    )


def _rate_limit_cooldown_s(headers: Any) -> float:
    raw_value = headers.get("retry-after") if headers is not None else None
    if raw_value is None:
        raw_value = headers.get("Retry-After") if headers is not None else None
    if raw_value is None:
        return DEFAULT_RATE_LIMIT_COOLDOWN_S
    try:
        return max(0.0, float(raw_value))
    except (TypeError, ValueError):
        try:
            retry_at = parsedate_to_datetime(str(raw_value))
            if retry_at.tzinfo is None:
                retry_at = retry_at.replace(tzinfo=timezone.utc)
            return max(0.0, (retry_at - datetime.now(timezone.utc)).total_seconds())
        except (TypeError, ValueError, OverflowError):
            return DEFAULT_RATE_LIMIT_COOLDOWN_S


def _acquire_key(
    keys: list[str],
    attempted: set[str],
    *,
    deadline: float,
) -> str:
    with _state_changed:
        while True:
            now = time.monotonic()
            remaining = deadline - now
            if remaining <= 0:
                raise TavilyRateLimited("timed out waiting for an available Tavily key")
            live = [key for key in keys if key not in _blacklist and key not in attempted]
            if not live:
                raise TavilyExhausted("all Tavily keys exhausted for this request")
            available = [
                key
                for key in live
                if key not in _in_flight and _cooldown_until.get(key, 0.0) <= now
            ]
            if available:
                random.shuffle(available)
                key = available[0]
                _in_flight.add(key)
                _cooldown_until[key] = max(
                    _cooldown_until.get(key, 0.0),
                    now + MIN_KEY_START_INTERVAL_S,
                )
                return key

            cooling = [
                _cooldown_until.get(key, 0.0) - now
                for key in live
                if key not in _in_flight and _cooldown_until.get(key, 0.0) > now
            ]
            if len(cooling) == len(live):
                earliest = min(cooling)
                if earliest > remaining:
                    retry_after = max(1, int(earliest + 0.999))
                    raise TavilyRateLimited(
                        f"all Tavily keys temporarily rate-limited; retry after {retry_after}s"
                    )
                _state_changed.wait(timeout=earliest)
                continue

            wait_s = min(remaining, 1.0)
            if cooling:
                wait_s = min(wait_s, min(cooling))
            _state_changed.wait(timeout=wait_s)


def _release_key(
    key: str,
    *,
    permanent: bool = False,
    cooldown_s: float | None = None,
) -> None:
    with _state_changed:
        _in_flight.discard(key)
        if permanent:
            _blacklist.add(key)
            _cooldown_until.pop(key, None)
        elif cooldown_s is not None:
            _cooldown_until[key] = max(
                _cooldown_until.get(key, 0.0),
                time.monotonic() + cooldown_s,
            )
        _state_changed.notify_all()


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
    keys = list(dict.fromkeys(_load_keys()))
    attempted: set[str] = set()
    last_err: str | None = None
    saw_rate_limit = False
    deadline = time.monotonic() + min(max(timeout_s, 0.1), MAX_TOTAL_TIMEOUT_S)
    while len(attempted) < len(keys):
        try:
            key = _acquire_key(keys, attempted, deadline=deadline)
        except TavilyExhausted:
            break
        attempted.add(key)
        payload = {
            "api_key": key,
            "query": query,
            "max_results": max_results,
            "search_depth": search_depth,
            "include_answer": include_answer,
            "include_raw_content": include_raw_content,
        }
        permanent = False
        cooldown_s: float | None = None
        try:
            remaining_s = deadline - time.monotonic()
            if remaining_s <= 0:
                raise TavilyRateLimited("Tavily request deadline expired before dispatch")
            try:
                r = requests.post(TAVILY_ENDPOINT, json=payload, timeout=remaining_s)
            except requests.RequestException as exc:
                last_err = f"network: {exc}"
                continue
            if r.status_code == 200:
                return r.json()
            if _is_quota_error(r.status_code, r.text):
                permanent = True
                last_err = f"key dead ({r.status_code}): {r.text[:200]}"
                continue
            if r.status_code == 429:
                cooldown_s = _rate_limit_cooldown_s(r.headers)
                saw_rate_limit = True
                last_err = f"key rate-limited ({cooldown_s:.0f}s): {r.text[:200]}"
                continue
            # Non-quota error (e.g. 400 bad request): trying another key cannot help.
            raise TavilyError(f"Tavily {r.status_code}: {r.text[:500]}")
        finally:
            _release_key(key, permanent=permanent, cooldown_s=cooldown_s)
    if saw_rate_limit:
        raise TavilyRateLimited(f"all available Tavily keys rate-limited; last={last_err}")
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
