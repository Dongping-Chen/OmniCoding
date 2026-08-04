from __future__ import annotations

import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from email.utils import format_datetime

import pytest

from omnicoding.tools import tavily_search


class _Response:
    def __init__(
        self,
        status_code: int,
        *,
        text: str = "",
        payload: dict | None = None,
        headers: dict[str, str] | None = None,
    ) -> None:
        self.status_code = status_code
        self.text = text
        self._payload = payload or {}
        self.headers = headers or {}

    def json(self) -> dict:
        return self._payload


@pytest.fixture(autouse=True)
def _reset_tavily_key_state():
    with tavily_search._state_changed:
        tavily_search._blacklist.clear()
        tavily_search._cooldown_until.clear()
        tavily_search._in_flight.clear()
    yield
    with tavily_search._state_changed:
        tavily_search._blacklist.clear()
        tavily_search._cooldown_until.clear()
        tavily_search._in_flight.clear()


def test_tavily_api_key_does_not_depend_on_package_location(monkeypatch) -> None:
    monkeypatch.setenv("TAVILY_API_KEY", "test-key")
    monkeypatch.delenv("TAVILY_KEYS_FILE", raising=False)

    assert tavily_search._load_keys() == ["test-key"]


def test_tavily_requires_explicit_configuration(monkeypatch) -> None:
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)
    monkeypatch.delenv("TAVILY_KEYS_FILE", raising=False)

    with pytest.raises(tavily_search.TavilyError, match="set TAVILY_API_KEY"):
        tavily_search._load_keys()


def test_tavily_rate_limit_temporarily_cools_key_instead_of_blacklisting(
    monkeypatch,
) -> None:
    keys = ["rate-limited-key", "working-key"]
    responses = iter(
        [
            _Response(
                429,
                text="Your request has been blocked due to excessive requests.",
                headers={"retry-after": "60"},
            ),
            _Response(200, payload={"results": [{"title": "ok"}]}),
        ]
    )
    monkeypatch.setattr(tavily_search, "_load_keys", lambda: keys)
    monkeypatch.setattr(tavily_search.random, "shuffle", lambda values: None)
    monkeypatch.setattr(tavily_search.requests, "post", lambda *args, **kwargs: next(responses))

    payload = tavily_search.search("test")

    assert payload == {"results": [{"title": "ok"}]}
    assert "rate-limited-key" not in tavily_search._blacklist
    assert tavily_search._cooldown_until["rate-limited-key"] > time.monotonic() + 55


def test_tavily_leases_a_key_to_one_upstream_request_at_a_time(monkeypatch) -> None:
    state_lock = threading.Lock()
    active = 0
    max_active = 0

    def post(*args, **kwargs):
        nonlocal active, max_active
        with state_lock:
            active += 1
            max_active = max(max_active, active)
        time.sleep(0.05)
        with state_lock:
            active -= 1
        return _Response(200, payload={"results": []})

    monkeypatch.setattr(tavily_search, "_load_keys", lambda: ["shared-key"])
    monkeypatch.setattr(tavily_search.requests, "post", post)

    started = time.monotonic()
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(tavily_search.search, ("first", "second")))

    assert results == [{"results": []}, {"results": []}]
    assert max_active == 1
    assert time.monotonic() - started >= 0.55
    assert tavily_search._in_flight == set()


def test_tavily_waits_for_short_key_cooldown(monkeypatch) -> None:
    monkeypatch.setattr(tavily_search, "_load_keys", lambda: ["cooling-key"])
    monkeypatch.setattr(
        tavily_search.requests,
        "post",
        lambda *args, **kwargs: _Response(200, payload={"results": []}),
    )
    tavily_search._cooldown_until["cooling-key"] = time.monotonic() + 0.1

    started = time.monotonic()
    assert tavily_search.search("test", timeout_s=1) == {"results": []}

    assert time.monotonic() - started >= 0.08


def test_tavily_releases_key_after_unexpected_upstream_exception(monkeypatch) -> None:
    monkeypatch.setattr(tavily_search, "_load_keys", lambda: ["working-key"])
    monkeypatch.setattr(
        tavily_search.requests,
        "post",
        lambda *args, **kwargs: (_ for _ in ()).throw(ValueError("adapter failure")),
    )

    with pytest.raises(ValueError, match="adapter failure"):
        tavily_search.search("test")

    assert tavily_search._in_flight == set()


def test_tavily_permanently_blacklists_auth_failure_and_rotates(monkeypatch) -> None:
    responses = iter(
        [
            _Response(401, text="invalid api key"),
            _Response(200, payload={"results": []}),
        ]
    )
    monkeypatch.setattr(tavily_search, "_load_keys", lambda: ["dead-key", "working-key"])
    monkeypatch.setattr(tavily_search.random, "shuffle", lambda values: None)
    monkeypatch.setattr(tavily_search.requests, "post", lambda *args, **kwargs: next(responses))

    assert tavily_search.search("test") == {"results": []}
    assert tavily_search._blacklist == {"dead-key"}


def test_tavily_retry_after_accepts_seconds_and_http_date() -> None:
    assert tavily_search._rate_limit_cooldown_s({"Retry-After": "3600"}) == 3600

    retry_at = datetime.now(timezone.utc) + timedelta(seconds=120)
    parsed = tavily_search._rate_limit_cooldown_s(
        {"Retry-After": format_datetime(retry_at, usegmt=True)}
    )

    assert 118 <= parsed <= 120


def test_tavily_attempts_share_one_end_to_end_deadline(monkeypatch) -> None:
    observed_timeouts: list[float] = []

    def fail(*args, **kwargs):
        observed_timeouts.append(kwargs["timeout"])
        time.sleep(0.02)
        raise tavily_search.requests.ConnectionError("network down")

    monkeypatch.setattr(tavily_search, "_load_keys", lambda: ["first", "second"])
    monkeypatch.setattr(tavily_search.random, "shuffle", lambda values: None)
    monkeypatch.setattr(tavily_search.requests, "post", fail)

    with pytest.raises(tavily_search.TavilyExhausted):
        tavily_search.search("test", timeout_s=0.2)

    assert len(observed_timeouts) == 2
    assert 0 < observed_timeouts[1] < observed_timeouts[0] <= 0.2
    assert tavily_search._in_flight == set()
