"""Poll the SGLang Prometheus endpoint during one serving benchmark.

The serving benchmark's client-side concurrency includes requests waiting at
the HTTP layer.  This poller records the scheduler-side running and queued
request counts so a capacity sweep can distinguish accepted concurrency from
requests that actually fit in the KV cache at the same time.
"""

from __future__ import annotations

import argparse
import json
import signal
import time
import urllib.request
from pathlib import Path


METRICS = {
    "sglang:num_running_reqs": ("running_requests", "sum"),
    "sglang:num_queue_reqs": ("queued_requests", "sum"),
    "sglang:token_usage": ("token_usage", "max"),
    "sglang:full_token_usage": ("full_token_usage", "max"),
}


def _parse_metrics(payload: str) -> dict[str, float]:
    values: dict[str, list[float]] = {name: [] for name in METRICS}
    for raw_line in payload.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        metric_and_labels, separator, raw_value = line.rpartition(" ")
        if not separator:
            continue
        metric_name = metric_and_labels.split("{", 1)[0]
        if metric_name not in METRICS:
            continue
        # Tensor- and pipeline-parallel ranks report the same scheduler state.
        # Keep one representative (TP0/PP0) per data-parallel replica, then
        # aggregate across the remaining DP-labelled series.
        labels = metric_and_labels.split("{", 1)[1] if "{" in metric_and_labels else ""
        if 'tp_rank="' in labels and 'tp_rank="0"' not in labels:
            continue
        if 'pp_rank="' in labels and 'pp_rank="0"' not in labels:
            continue
        try:
            values[metric_name].append(float(raw_value))
        except ValueError:
            continue

    result: dict[str, float] = {}
    for metric_name, (output_name, aggregation) in METRICS.items():
        samples = values[metric_name]
        if not samples:
            continue
        result[output_name] = sum(samples) if aggregation == "sum" else max(samples)
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--interval", type=float, default=0.1)
    args = parser.parse_args()
    if args.interval <= 0:
        parser.error("--interval must be positive")

    stopping = False

    def stop(_signum: int, _frame: object) -> None:
        nonlocal stopping
        stopping = True

    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    metrics_url = args.base_url.rstrip("/") + "/metrics"
    started = time.monotonic()
    with args.output.open("w") as output:
        while not stopping:
            record: dict[str, object] = {
                "elapsed_s": time.monotonic() - started,
                "wall_time_s": time.time(),
            }
            try:
                with urllib.request.urlopen(metrics_url, timeout=2) as response:
                    payload = response.read().decode("utf-8", errors="replace")
                record.update(_parse_metrics(payload))
            except Exception as exc:  # noqa: BLE001
                record["error"] = f"{type(exc).__name__}: {exc}"
            output.write(json.dumps(record, sort_keys=True) + "\n")
            output.flush()
            time.sleep(args.interval)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
