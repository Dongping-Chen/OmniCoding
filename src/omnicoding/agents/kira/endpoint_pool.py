"""Multi-endpoint pool for the kira runner with sticky, weighted routing
plus per-item failover.

Why this exists
---------------
A single sglang server caps `--max-running-requests` at some number (e.g. 32
for a multi-GPU inference host). Once we shard a bench into many parallel kira jobs, each
making sequential per-step LLM calls, the queue depth at the proxy hits 50 and
50−32 = 18 requests sit idle in the queue every step. We can fan out across
multiple SGLang instances to lift the ceiling.

But naïve random routing wastes the long shared system prompt + accumulating
tool-call history that sglang's RadixAttention caches per request — bouncing
the same item between two servers means each server warms up its KV from
scratch every step. That throws away the >100k cached prompt tokens this loop
relies on.

So routing is **sticky per item**: every LLM call inside `_run_one(idx=N)` goes
to the same endpoint until that endpoint fails. Different items distribute
across endpoints by weight (matching each endpoint's `--max-running-requests`
budget) so total in-flight load balances proportionally.

When a per-call exception indicates the endpoint is down or stuck (network
errors, ``BlockTimeoutError`` from the harness wall clock, sglang restart),
:class:`EndpointSession` rotates the item to a different endpoint mid-run.
The agent re-issues the same prompt against the new server — KV cache cold,
but the trajectory survives. Without this, a single 600s stall would discard
50+ steps of accumulated rollout.

Spec format
-----------
``URL=W,URL2=W2`` — weights default to 1 if omitted. Examples:

    http://host-a:8080/v1=32,http://host-b:8080/v1=32   # equal capacity
    http://host-a:8080/v1=4,http://host-b:8080/v1=2     # unequal capacity
    http://127.0.0.1:8080/v1                            # single endpoint, weight=1

Indexing
--------
``pick_for_index(N)`` returns the URL by interleaved round-robin over a
sequence of total ``Σ weight`` slots. Interleaved (Bresenham-style) rather than
``[A]*4 + [B]*2`` so the first few items spread across servers instead of all
landing on the heaviest one.
"""
from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Iterable, List, Optional

LOGGER = logging.getLogger("kira.endpoint_pool")


@dataclass(frozen=True)
class Endpoint:
    url: str
    weight: int


class EndpointPool:
    """Sticky, weighted endpoint chooser. Use ``pick_for_index`` per item."""

    def __init__(self, endpoints: Iterable[Endpoint]):
        eps = list(endpoints)
        if not eps:
            raise ValueError("EndpointPool requires at least one endpoint")
        if any(ep.weight <= 0 for ep in eps):
            raise ValueError("endpoint weights must be positive integers")
        self.endpoints: List[Endpoint] = eps
        self._sequence: List[str] = _bresenham_interleave(eps)

    @property
    def total_weight(self) -> int:
        return len(self._sequence)

    def urls(self) -> List[str]:
        """Distinct URLs in the pool, preserving spec order."""
        seen = set()
        ordered: list[str] = []
        for ep in self.endpoints:
            if ep.url not in seen:
                seen.add(ep.url)
                ordered.append(ep.url)
        return ordered

    def pick_for_index(self, idx: int) -> str:
        return self._sequence[idx % self.total_weight]

    def describe(self) -> str:
        parts = [f"{ep.url}={ep.weight}" for ep in self.endpoints]
        return ",".join(parts)


def parse_endpoints(spec: str) -> EndpointPool:
    """Parse ``"URL=W,URL2=W2"`` into an :class:`EndpointPool`.

    Accepts either ``,`` or ``|`` as the endpoint separator. Use ``|`` when
    passing through interfaces that already overload comma — notably SLURM's
    ``sbatch --export=A=1,B=2`` parser, which would otherwise split a single
    multi-endpoint value mid-string and silently drop everything after the
    first comma.
    """
    out: List[Endpoint] = []
    # Split on either delimiter; both can mix in the same spec.
    raw_parts = [p for chunk in spec.split("|") for p in chunk.split(",")]
    for raw in raw_parts:
        part = raw.strip()
        if not part:
            continue
        if "=" in part:
            url, w = part.rsplit("=", 1)
            try:
                weight = int(w.strip())
            except ValueError as exc:
                raise ValueError(f"endpoint weight must be int, got {w!r}") from exc
        else:
            url, weight = part, 1
        out.append(Endpoint(url=url.strip(), weight=weight))
    return EndpointPool(out)


def _bresenham_interleave(endpoints: List[Endpoint]) -> List[str]:
    """Build the per-slot URL sequence so adjacent slots span endpoints.

    For weights ``[4, 2]`` returns ``[A, A, B, A, A, B]`` (not ``[A,A,A,A,B,B]``).
    Standard largest-remainder schedule: at each slot pick the endpoint whose
    next ideal target is smallest.
    """
    total = sum(ep.weight for ep in endpoints)
    targets = [0.0 for _ in endpoints]
    steps = [total / ep.weight for ep in endpoints]
    seq: List[str] = []
    for _ in range(total):
        j = min(range(len(endpoints)), key=lambda i: targets[i])
        seq.append(endpoints[j].url)
        targets[j] += steps[j]
    return seq


@dataclass
class FailoverEvent:
    """One endpoint rotation. Surfaced in result rows for analysis."""
    from_url: str
    to_url: str
    reason: str
    step_hint: Optional[int] = None  # which loop step triggered it (best-effort)
    timestamp: float = field(default_factory=time.time)


class EndpointSession:
    """Per-item endpoint state with failover.

    One session per ``_run_one`` invocation. The session starts pinned to
    ``pool.pick_for_index(idx)`` so the first call lands on the deterministic
    sticky URL (warming RadixAttention KV with the system prompt). On a
    transient endpoint error, ``failover(reason)`` rotates to a different URL
    in the pool and returns ``True``; the caller retries the same prompt
    against the new URL. When the budget is exhausted, returns ``False`` and
    the caller bubbles the error up.

    Budget semantics
    ----------------
    * ``max_failovers`` caps total rotations across the item lifetime
      (default = 2 × pool size, allowing one full ring + a recycle pass).
    * Each URL is tried at most ``max_per_url`` times before being skipped
      until the round-robin wraps around. This prevents a single dead
      endpoint from absorbing the entire failover budget.
    * After a successful call, ``record_success()`` resets the per-URL
      try-count for the *current* URL only, so subsequent failures on
      OTHER endpoints don't re-enable a known-bad one.
    """

    def __init__(
        self,
        pool: EndpointPool,
        idx: int,
        max_failovers: Optional[int] = None,
        max_per_url: int = 2,
    ):
        self._pool = pool
        self._idx = idx
        self._urls = pool.urls()
        if not self._urls:
            raise ValueError("EndpointPool has no urls")
        self._max_failovers = (
            max_failovers if max_failovers is not None else 2 * len(self._urls)
        )
        self._max_per_url = max_per_url
        self._current_url = pool.pick_for_index(idx)
        self._tries: dict[str, int] = {self._current_url: 0}
        self._failovers: list[FailoverEvent] = []
        self._lock = threading.Lock()

    @property
    def current_url(self) -> str:
        return self._current_url

    def record_success(self) -> None:
        """Reset the try-count for the URL that just succeeded."""
        with self._lock:
            self._tries[self._current_url] = 0

    def failover(self, reason: str, step_hint: Optional[int] = None) -> bool:
        """Rotate to the next eligible URL. Returns False when exhausted."""
        with self._lock:
            self._tries[self._current_url] = self._tries.get(self._current_url, 0) + 1
            if len(self._failovers) >= self._max_failovers:
                LOGGER.warning(
                    "kira.endpoint_pool failover budget exhausted (idx=%d total=%d)",
                    self._idx, len(self._failovers),
                )
                return False
            # Round-robin from the current URL; pick the next OTHER one
            # whose try-count is below the per-URL cap.
            start = self._urls.index(self._current_url)
            n = len(self._urls)
            for off in range(1, n + 1):
                cand = self._urls[(start + off) % n]
                if cand == self._current_url:
                    continue
                if self._tries.get(cand, 0) < self._max_per_url:
                    ev = FailoverEvent(
                        from_url=self._current_url, to_url=cand,
                        reason=reason, step_hint=step_hint,
                    )
                    LOGGER.warning(
                        "kira.endpoint_pool idx=%d failover %s -> %s reason=%s step=%s",
                        self._idx, ev.from_url, ev.to_url, reason, step_hint,
                    )
                    self._failovers.append(ev)
                    self._current_url = cand
                    self._tries.setdefault(cand, 0)
                    return True
            LOGGER.warning(
                "kira.endpoint_pool idx=%d all urls exceeded per-url cap (%d) reason=%s",
                self._idx, self._max_per_url, reason,
            )
            return False

    def history(self) -> list[dict]:
        """JSON-serialisable summary for the result row."""
        return [
            {
                "from": e.from_url, "to": e.to_url,
                "reason": e.reason, "step": e.step_hint,
                "ts": e.timestamp,
            }
            for e in self._failovers
        ]

    def stats(self) -> dict:
        return {
            "idx": self._idx,
            "current_url": self._current_url,
            "tries": dict(self._tries),
            "failovers": len(self._failovers),
        }
