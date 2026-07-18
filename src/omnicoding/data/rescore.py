"""Post-hoc rescore an existing kira run with the current spec code.

Use case
--------
The runner persists ``results.json`` rows with ``predicted`` / ``is_correct``
computed at item-end time using whatever extractor was on disk *then*. When
extractors get more lenient (e.g. VZB BUG-X1 fix added a "Final Answer:" tail
fallback), older runs miss out — the trajectory is fine but the score is
stale.

This script walks a RUN_ROOT, replays ``spec.extract_prediction`` +
``spec.is_correct`` over the saved ``final_text.txt`` for each item, and
writes a sibling ``results_rescored.json`` next to each shard's
``results.json``. The originals are never touched — the live runner can keep
appending to ``results.json`` while we rescore safely.

Skips items whose original ``error`` is set (extractor bug fixes don't
unbreak Connection-error rollouts) and items missing ``final_text.txt``.

Usage
-----
    .venv_harness/bin/python local_model/scripts/rescore_kira.py \\
        --run_root local_model/outputs/dual_kira_resume_20260427_101501

Reports per-bench: items rescored, predicted-flip, is_correct-flip
(None/False → True), is_correct-flip (True → False).
"""
from __future__ import annotations

import argparse
import json
import logging
from collections import Counter
from pathlib import Path
from typing import Any

from omnicoding.benchmarks import specs

LOGGER = logging.getLogger("rescore_kira")


def _load_shard_inputs(run_root: Path, bench: str, shard_idx: str) -> list[dict[str, Any]]:
    """Re-read the per-shard input JSON used to seed this run."""
    p = run_root / "_shards" / bench / f"shard_{shard_idx}.json"
    if not p.exists():
        return []
    raw = json.loads(p.read_text(encoding="utf-8"))
    if isinstance(raw, list):
        return raw
    if isinstance(raw, dict) and isinstance(raw.get("data"), list):
        return raw["data"]
    return []


def _items_by_source_index(items: list[dict[str, Any]]) -> dict[Any, dict[str, Any]]:
    out: dict[Any, dict[str, Any]] = {}
    for i, item in enumerate(items):
        si = item.get("__source_index__", i)
        out[si] = item
    return out


def _rescore_shard(spec, results_path: Path, items_by_si: dict[Any, dict]) -> dict:
    """Replay extract+score for every row that has a saved final_text.txt.
    Writes sibling ``results_rescored.json``. Returns counters."""
    rows = json.loads(results_path.read_text(encoding="utf-8"))
    shard_dir = results_path.parent
    counters = Counter()
    out_rows: list[dict[str, Any]] = []

    for r in rows:
        new = dict(r)
        si = r.get("source_index")
        counters["total"] += 1
        if r.get("error"):
            counters["skip_errored"] += 1
            out_rows.append(new)
            continue
        item = items_by_si.get(si)
        if item is None:
            counters["skip_no_item"] += 1
            out_rows.append(new)
            continue
        # Each item dir is item_NNNN/. Use source_index as 4-digit id.
        # Some specs use string ids; fall back to None then.
        item_dir = None
        if isinstance(si, int):
            item_dir = shard_dir / f"item_{si:04d}"
        if item_dir is None or not (item_dir / "final_text.txt").exists():
            counters["skip_no_final_text"] += 1
            out_rows.append(new)
            continue

        final_text = (item_dir / "final_text.txt").read_text(encoding="utf-8", errors="replace")
        old_pred = r.get("predicted_option") or r.get("predicted_answer") or r.get("predicted")
        old_ic = r.get("is_correct")

        try:
            new_pred = spec.extract_prediction(final_text, item)
            new_ic = spec.is_correct(item, new_pred)
        except Exception as exc:  # noqa: BLE001
            counters["rescore_error"] += 1
            new["rescore_error"] = f"{type(exc).__name__}: {exc}"
            out_rows.append(new)
            continue

        # Persist the updated values into whichever predicted_* field
        # the spec actually uses, mirroring run_bench_kira's write.
        if "predicted_option" in r:
            new["predicted_option"] = new_pred
        elif "predicted_answer" in r:
            new["predicted_answer"] = new_pred
        elif "predicted" in r:
            new["predicted"] = new_pred
        else:
            new["predicted"] = new_pred
        new["is_correct"] = new_ic

        if old_pred != new_pred:
            counters["predicted_flipped"] += 1
        if old_ic != new_ic:
            counters["is_correct_changed"] += 1
            if not old_ic and new_ic is True:
                counters["recovered_to_correct"] += 1
            elif old_ic is True and not new_ic:
                counters["regressed_from_correct"] += 1
        out_rows.append(new)

    out_path = results_path.with_name("results_rescored.json")
    tmp = out_path.with_name(out_path.name + ".tmp")
    tmp.write_text(json.dumps(out_rows, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(out_path)
    return counters


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_root", required=True,
                        help="Output dir of a prior kira run "
                             "(contains <bench>_kira/shard_NN/results.json + _shards/).")
    parser.add_argument("--benches", nargs="*", default=None,
                        help="Restrict to specific benches; default = all that have shards.")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    run_root = Path(args.run_root).resolve()
    if not run_root.exists():
        raise SystemExit(f"run_root not found: {run_root}")

    # Discover bench dirs.
    bench_dirs = sorted([p for p in run_root.iterdir()
                         if p.is_dir() and p.name.endswith("_kira")])
    if args.benches:
        bench_dirs = [p for p in bench_dirs
                      if p.name.replace("_kira", "") in args.benches]

    grand_total = Counter()
    for bench_dir in bench_dirs:
        bench = bench_dir.name.replace("_kira", "")
        try:
            spec = specs.get(bench)
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning("[%s] skip: cannot resolve spec (%s)", bench, exc)
            continue

        shards = sorted([p for p in bench_dir.iterdir()
                         if p.is_dir() and p.name.startswith("shard_")])
        bench_total = Counter()
        for shard in shards:
            results_path = shard / "results.json"
            if not results_path.exists():
                continue
            shard_idx = shard.name.replace("shard_", "")
            inputs = _load_shard_inputs(run_root, bench, shard_idx)
            by_si = _items_by_source_index(inputs)
            cnt = _rescore_shard(spec, results_path, by_si)
            bench_total += cnt
        LOGGER.info(
            "[%s] total=%d  predicted_flipped=%d  ic_changed=%d  recovered=%d  regressed=%d  "
            "skip_err=%d  skip_no_item=%d  skip_no_text=%d  rescore_err=%d",
            bench,
            bench_total["total"],
            bench_total["predicted_flipped"],
            bench_total["is_correct_changed"],
            bench_total["recovered_to_correct"],
            bench_total["regressed_from_correct"],
            bench_total["skip_errored"],
            bench_total["skip_no_item"],
            bench_total["skip_no_final_text"],
            bench_total["rescore_error"],
        )
        grand_total += bench_total

    LOGGER.info(
        "[ALL] total=%d  predicted_flipped=%d  ic_changed=%d  recovered=%d  regressed=%d",
        grand_total["total"],
        grand_total["predicted_flipped"],
        grand_total["is_correct_changed"],
        grand_total["recovered_to_correct"],
        grand_total["regressed_from_correct"],
    )


if __name__ == "__main__":
    main()
