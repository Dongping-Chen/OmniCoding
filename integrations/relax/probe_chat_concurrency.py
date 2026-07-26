"""Measure request overlap against an OpenAI-compatible chat endpoint.

This is intentionally independent of Relax so the same payload can be sent to
the SGLang router directly and through Relax's Ray Serve proxy.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import time
from dataclasses import asdict, dataclass
from typing import Any

import httpx


@dataclass
class RequestResult:
    index: int
    started_s: float
    finished_s: float
    latency_s: float
    status_code: int
    completion_tokens: int
    error: str | None = None


def _max_overlap(results: list[RequestResult]) -> int:
    events: list[tuple[float, int]] = []
    for result in results:
        if result.status_code == 200:
            events.append((result.started_s, 1))
            events.append((result.finished_s, -1))
    active = 0
    maximum = 0
    for _, delta in sorted(events, key=lambda item: (item[0], -item[1])):
        active += delta
        maximum = max(maximum, active)
    return maximum


async def _run(args: argparse.Namespace) -> int:
    endpoint = args.base_url.rstrip("/") + "/v1/chat/completions"
    started = time.perf_counter()
    semaphore = asyncio.Semaphore(args.concurrency)

    async with httpx.AsyncClient(
        timeout=httpx.Timeout(args.timeout, connect=30.0),
        limits=httpx.Limits(
            max_connections=args.concurrency,
            max_keepalive_connections=args.concurrency,
        ),
    ) as client:

        async def one(index: int) -> RequestResult:
            async with semaphore:
                request_started = time.perf_counter()
                payload: dict[str, Any] = {
                    "model": args.model,
                    "messages": [
                        {
                            "role": "user",
                            "content": f"{args.prompt}\nRequest id: {index}",
                        }
                    ],
                    "temperature": args.temperature,
                    "max_tokens": args.max_tokens,
                    "stream": False,
                }
                try:
                    response = await client.post(endpoint, json=payload)
                    request_finished = time.perf_counter()
                    completion_tokens = 0
                    if response.status_code == 200:
                        completion_tokens = int(
                            (response.json().get("usage") or {}).get("completion_tokens") or 0
                        )
                    return RequestResult(
                        index=index,
                        started_s=request_started - started,
                        finished_s=request_finished - started,
                        latency_s=request_finished - request_started,
                        status_code=response.status_code,
                        completion_tokens=completion_tokens,
                        error=None if response.status_code == 200 else response.text[:500],
                    )
                except Exception as exc:  # noqa: BLE001
                    request_finished = time.perf_counter()
                    return RequestResult(
                        index=index,
                        started_s=request_started - started,
                        finished_s=request_finished - started,
                        latency_s=request_finished - request_started,
                        status_code=0,
                        completion_tokens=0,
                        error=f"{type(exc).__name__}: {exc}",
                    )

        results = await asyncio.gather(*(one(index) for index in range(args.requests)))

    wall_s = time.perf_counter() - started
    successful = [result for result in results if result.status_code == 200]
    summary = {
        "endpoint": endpoint,
        "model": args.model,
        "requests": args.requests,
        "concurrency": args.concurrency,
        "successful": len(successful),
        "failed": args.requests - len(successful),
        "wall_s": wall_s,
        "sum_success_latency_s": sum(result.latency_s for result in successful),
        "effective_overlap": (
            sum(result.latency_s for result in successful) / wall_s if wall_s else 0.0
        ),
        "max_observed_overlap": _max_overlap(results),
        "latency_p50_s": (
            statistics.median(result.latency_s for result in successful) if successful else None
        ),
        "completion_tokens": sum(result.completion_tokens for result in successful),
        "results": [asdict(result) for result in results],
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if len(successful) == args.requests else 1


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--requests", type=int, default=8)
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--timeout", type=float, default=600.0)
    parser.add_argument(
        "--prompt",
        default="Write a numbered list of 100 short observations about batching.",
    )
    args = parser.parse_args()
    if args.requests < 1 or args.concurrency < 1:
        parser.error("--requests and --concurrency must be positive")
    return asyncio.run(_run(args))


if __name__ == "__main__":
    raise SystemExit(main())
