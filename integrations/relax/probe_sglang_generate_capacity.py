"""Measure native SGLang generation concurrency without loading a tokenizer.

The stock serving benchmark reloads Torch and the tokenizer for every point in
a sweep.  On the shared filesystem that adds roughly 50 seconds per point.
This probe sends exact-length token-id prompts to ``/generate`` using only the
Python standard library, so the measured interval is model work rather than
client startup.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import statistics
import threading
import time
import urllib.request
from pathlib import Path
from typing import Any


def _post_json(url: str, payload: dict[str, Any], timeout: float) -> dict[str, Any]:
    body = json.dumps(payload, separators=(",", ":")).encode()
    request = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json", "Connection": "close"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read())


def _post_empty(url: str, timeout: float) -> None:
    request = urllib.request.Request(url, data=b"", method="POST")
    with urllib.request.urlopen(request, timeout=timeout):
        pass


def _percentile(values: list[float], quantile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, int(len(ordered) * quantile + 0.999) - 1))
    return ordered[index]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--input-length", type=int, required=True)
    parser.add_argument("--output-length", type=int, required=True)
    parser.add_argument("--concurrency", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--timeout", type=float, default=1800)
    parser.add_argument("--seed-offset", type=int, default=0)
    args = parser.parse_args()
    if min(args.input_length, args.output_length, args.concurrency) < 1:
        parser.error("input length, output length, and concurrency must be positive")

    base_url = args.base_url.rstrip("/")
    # Low vocabulary IDs avoid multimodal placeholder tokens. Rotating this
    # 900-token pattern gives requests different prefixes, preventing radix
    # cache reuse from inflating capacity or throughput.
    token_pattern = list(range(100, 1000))
    barrier = threading.Barrier(args.concurrency)

    def run_one(index: int) -> dict[str, Any]:
        offset = (args.seed_offset + index * 113) % len(token_pattern)
        rotated = token_pattern[offset:] + token_pattern[:offset]
        repeats = (args.input_length + len(rotated) - 1) // len(rotated)
        input_ids = (rotated * repeats)[: args.input_length]
        payload = {
            "input_ids": input_ids,
            "sampling_params": {
                "max_new_tokens": args.output_length,
                "temperature": 0,
                "ignore_eos": True,
            },
            "stream": False,
        }
        barrier.wait()
        started = time.perf_counter()
        try:
            response = _post_json(
                f"{base_url}/generate",
                payload,
                timeout=args.timeout,
            )
            latency = time.perf_counter() - started
            metadata = response.get("meta_info") or {}
            return {
                "index": index,
                "latency_s": latency,
                "prompt_tokens": metadata.get("prompt_tokens"),
                "completion_tokens": metadata.get("completion_tokens"),
                "queue_time_s": metadata.get("queue_time"),
                "server_e2e_latency_s": metadata.get("e2e_latency"),
                "dp_rank": metadata.get("dp_rank"),
                "total_retractions": metadata.get("total_retractions"),
                "finish_reason": metadata.get("finish_reason"),
                "error": None,
            }
        except Exception as exc:  # noqa: BLE001
            return {
                "index": index,
                "latency_s": time.perf_counter() - started,
                "error": f"{type(exc).__name__}: {exc}",
            }

    # Flush before starting the clock. This makes every point a conservative
    # no-prefix-reuse measurement.
    _post_empty(f"{base_url}/flush_cache", timeout=60)
    started = time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor(
        max_workers=args.concurrency
    ) as executor:
        results = list(executor.map(run_one, range(args.concurrency)))
    duration = time.perf_counter() - started

    successes = [item for item in results if not item["error"]]
    latencies = [float(item["latency_s"]) for item in successes]
    prompt_tokens = sum(int(item.get("prompt_tokens") or 0) for item in successes)
    completion_tokens = sum(
        int(item.get("completion_tokens") or 0) for item in successes
    )
    record = {
        "input_length": args.input_length,
        "output_length": args.output_length,
        "concurrency": args.concurrency,
        "duration": duration,
        "completed": len(successes),
        "failed": len(results) - len(successes),
        "total_input_tokens": prompt_tokens,
        "total_output_tokens": completion_tokens,
        "request_throughput": len(successes) / duration,
        "input_throughput": prompt_tokens / duration,
        "output_throughput": completion_tokens / duration,
        "mean_e2e_latency_ms": (
            statistics.fmean(latencies) * 1000 if latencies else None
        ),
        "p90_e2e_latency_ms": (
            _percentile(latencies, 0.90) * 1000 if latencies else None
        ),
        "p99_e2e_latency_ms": (
            _percentile(latencies, 0.99) * 1000 if latencies else None
        ),
        "max_concurrent_requests": args.concurrency,
        "dp_ranks": {
            str(rank): sum(item.get("dp_rank") == rank for item in successes)
            for rank in sorted(
                {
                    item.get("dp_rank")
                    for item in successes
                    if item.get("dp_rank") is not None
                }
            )
        },
        "max_queue_time_s": max(
            (float(item.get("queue_time_s") or 0) for item in successes),
            default=0,
        ),
        "total_retractions": sum(
            int(item.get("total_retractions") or 0) for item in successes
        ),
        "requests": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(record, sort_keys=True) + "\n")
    print(json.dumps({key: value for key, value in record.items() if key != "requests"}))
    return 0 if not record["failed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
