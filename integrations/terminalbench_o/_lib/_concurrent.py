"""Tiny concurrency helper for LLM/VLM judge calls.

All graders run sequentially today, which makes axes_yes_no_score and
mllm_video_axes the dominant grader cost (N_axes × ~1s, or
N_frames × N_axes for video). This helper lets the _lib judges fan out
calls in parallel via a thread pool while preserving result ordering.

Tunable via env:
    CLAW_BENCH_JUDGE_CONCURRENCY  -- max workers (default 8, clamped to [1,32])

The helper is intentionally minimal — just `parallel_map` returning results
in input order. Threads are appropriate because the work is I/O-bound on
network calls; the underlying `chat()` / `vision()` already retry/backoff,
so we don't add extra retry logic here.
"""
from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor
from typing import Callable, Iterable, List, TypeVar

T = TypeVar("T")
R = TypeVar("R")


def _max_workers() -> int:
    raw = os.environ.get("CLAW_BENCH_JUDGE_CONCURRENCY", "8")
    try:
        n = int(raw)
    except ValueError:
        n = 8
    return max(1, min(32, n))


def parallel_map(
    fn: Callable[[T], R],
    items: Iterable[T],
    max_workers: int | None = None,
) -> List[R]:
    """Apply `fn` to each item concurrently. Results returned in input order.

    Exceptions raised by `fn` propagate; callers should wrap individual
    work units in their own try/except if they want partial failure
    tolerance (which most existing judge helpers already do).
    """
    items_list = list(items)
    if not items_list:
        return []
    workers = max_workers or _max_workers()
    if workers <= 1 or len(items_list) <= 1:
        return [fn(it) for it in items_list]
    with ThreadPoolExecutor(max_workers=min(workers, len(items_list))) as ex:
        return list(ex.map(fn, items_list))
