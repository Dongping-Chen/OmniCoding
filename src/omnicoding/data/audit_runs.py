#!/usr/bin/env python3
"""Audit kira×bench result rows across one-or-more output dirs.

Usage:
    audit_kira_runs.py <out_root> [<out_root> ...]

Aggregates results.json from each ``<out_root>/<bench>_kira*/shard_*/`` and
reports rows / exit_reason / correctness / token usage per bench. Resolves
the per-bench prediction field correctly so MCQ vs free-text rows are
both audited:

    lvomnibench / socialomni_l1 / socialomni_l2 → ``predicted_option``
    videozerobench                              → ``level3_answer``
    omnigaia                                    → ``predicted_answer``

For each bench it also prints up to ``--show_wrong`` representative
incorrect rows (pred + gold side-by-side) and ``--show_no_tool`` rows
that exited ``no_tool_calls`` so a glance reveals whether the model
silently skipped tools or genuinely answered wrong.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from collections import Counter
from pathlib import Path

PRED_FIELD = {
    "lvomnibench": "predicted_option",
    "socialomni_l1": "predicted_option",
    "socialomni_l2": "predicted_option",
    "videozerobench": "level3_answer",
    "omnigaia": "predicted_answer",
}

# Field name in INPUT FILE used as gold for each bench. Names match the
# field the spec's ``_is_correct`` compares against; kira's
# --include_gold_fields_in_results is off by default so we recover gold
# by joining row → input item.
GOLD_FIELD = {
    "lvomnibench": "correct_option",   # MCQ letter; ``answer`` is free-text label
    "socialomni_l1": "correct_answer",
    "socialomni_l2": None,             # split-per-question shape; gold lookup not wired
    "videozerobench": "answer",
    "omnigaia": "answer",
}

# Unique key joining row → input item. Each entry is the field name
# in the INPUT FILE; ROW_JOIN_KEY is the corresponding row field
# (often different — e.g. socialomni inputs have ``id`` while rows use
# ``question_id``).
JOIN_KEY = {
    "lvomnibench": ("video_id", "question"),
    "socialomni_l1": "id",
    "socialomni_l2": None,
    "videozerobench": "question_id",
    "omnigaia": "id",
}

ROW_JOIN_KEY = {
    "lvomnibench": ("video_id", "question"),
    "socialomni_l1": "question_id",   # row field name differs from input
    "socialomni_l2": None,
    "videozerobench": "question_id",
    "omnigaia": "id",
}

# Default input file paths (relative to OmniCoding root) for ``--inputs auto``.
DEFAULT_INPUTS = {
    "lvomnibench": "LVOmniBench/data/subsets/first100/data.json",
    "socialomni_l1": "SocialOmni/data/subsets/level_1_first100.json",
    "socialomni_l2": "SocialOmni/data/subsets/level_2_first100.json",
    "videozerobench": "coding_agent_benchmarks_1/benchmarks/videozerobench/VideoZeroBench_500_v0.json",
    "omnigaia": "coding_agent_benchmarks_1/benchmarks/omnigaia/data/test_metadata.json",
}

# Benches whose spec emits is_correct=None by design (require LLM-as-judge).
LLM_JUDGE_BENCHES = {"omnigaia"}

BENCHES = list(PRED_FIELD)


def collect_rows(out_root: str, bench: str) -> list[dict]:
    """All shard_*/results.json rows under any <bench>_kira* subdir."""
    rows: list[dict] = []
    for results_path in sorted(
        glob.glob(f"{out_root}/{bench}_kira*/shard_*/results.json")
    ):
        try:
            rows.extend(json.load(open(results_path)))
        except Exception as exc:
            print(
                f"[warn] could not read {results_path}: {exc}", file=sys.stderr
            )
    return rows


def _row_join_value(row: dict, bench: str):
    key = ROW_JOIN_KEY[bench]
    if key is None:
        return None
    if isinstance(key, tuple):
        return tuple(row.get(k) for k in key)
    return row.get(key)


def _build_gold_index(input_path: Path, bench: str) -> dict | None:
    if JOIN_KEY[bench] is None or GOLD_FIELD[bench] is None:
        return None
    items = json.loads(input_path.read_text())
    if isinstance(items, dict) and "data" in items:
        items = items["data"]
    key = JOIN_KEY[bench]
    field = GOLD_FIELD[bench]
    out: dict = {}
    for item in items:
        if isinstance(key, tuple):
            k = tuple(item.get(name) for name in key)
        else:
            k = item.get(key)
        if k is None or (isinstance(k, tuple) and any(v is None for v in k)):
            continue
        out[k] = item.get(field)
    return out


def gold_for_row(row: dict, bench: str, gold_index: dict | None) -> str | None:
    """Best-effort gold extraction. Prefers the input-file join; falls
    back to in-row fields when ``--include_gold_fields_in_results`` was
    on for the run."""
    if gold_index is not None and JOIN_KEY[bench] is not None:
        v = gold_index.get(_row_join_value(row, bench))
        if v not in (None, ""):
            return v
    field = GOLD_FIELD.get(bench)
    if field and row.get(field) not in (None, ""):
        return row[field]
    for k in ("correct_option", "correct_answer", "label", "answer"):
        if row.get(k):
            return row[k]
    return None


def audit_bench(rows: list[dict], bench: str, gold_index: dict | None, *, show_wrong: int, show_no_tool: int) -> dict:
    pred_key = PRED_FIELD[bench]
    n = len(rows)
    if n == 0:
        return {"n": 0}
    exits = Counter(r.get("exit_reason", "?") for r in rows)
    no_pred = sum(1 for r in rows if not r.get(pred_key))
    is_correct_vals = [r.get("is_correct") for r in rows]
    correct = sum(1 for v in is_correct_vals if v is True)
    incorrect = sum(1 for v in is_correct_vals if v is False)
    skipped = sum(1 for v in is_correct_vals if v is None)
    avg_tc = sum(int(r.get("tool_call_num") or 0) for r in rows) / n
    avg_pt = sum(int(r.get("prompt_tokens") or 0) for r in rows) / n
    avg_ct = sum(int(r.get("completion_tokens") or 0) for r in rows) / n
    cached = sum(int(r.get("cached_tokens") or 0) for r in rows) / n

    print(f"\n=== {bench} (pred_key={pred_key}) ===")
    print(f"  rows                : {n}")
    print(f"  exit_reasons        : {dict(exits)}")
    print(f"  empty prediction    : {no_pred}")
    if bench in LLM_JUDGE_BENCHES:
        print(f"  scoring             : LLM-as-judge required (spec emits is_correct=None by design)")
    if (correct + incorrect):
        print(
            f"  correct/incorrect/none = {correct}/{incorrect}/{skipped}  "
            f"accuracy = {100*correct/n:.1f}% (over n={n})  "
            f"scored = {100*correct/(correct+incorrect):.1f}% (over {correct+incorrect})"
        )
    else:
        print(f"  correct/incorrect/none = {correct}/{incorrect}/{skipped}  (no scored rows)")
    print(
        f"  avg tool_calls={avg_tc:.1f}  avg prompt_tokens={avg_pt:.0f}  "
        f"avg completion_tokens={avg_ct:.0f}  avg cached_tokens={cached:.0f}"
    )

    def _row_id(r: dict):
        return (
            r.get("question_id")
            or r.get("source_question_id")
            or r.get("id")
            or r.get("source_index")
        )

    if show_wrong > 0:
        wrong = [r for r in rows if r.get("is_correct") is False]
        if wrong:
            print(f"  -- {min(show_wrong, len(wrong))} incorrect samples --")
            for r in wrong[:show_wrong]:
                pred = r.get(pred_key)
                gold = gold_for_row(r, bench, gold_index)
                exit_r = r.get("exit_reason", "?")
                tc = r.get("tool_call_num", 0)
                print(
                    f"     id={_row_id(r)} exit={exit_r} steps={tc} "
                    f"pred={str(pred)[:80]!r} gold={str(gold)[:80]!r}"
                )

    if show_no_tool > 0:
        nt = [r for r in rows if r.get("exit_reason") == "no_tool_calls"]
        if nt:
            print(f"  -- {min(show_no_tool, len(nt))} no_tool_calls samples --")
            for r in nt[:show_no_tool]:
                pred = r.get(pred_key)
                gold = gold_for_row(r, bench, gold_index)
                tc = r.get("tool_call_num", 0)
                print(
                    f"     id={_row_id(r)} steps={tc} "
                    f"pred={str(pred)[:80]!r} gold={str(gold)[:80]!r} "
                    f"is_correct={r.get('is_correct')}"
                )

    return {
        "n": n,
        "correct": correct,
        "incorrect": incorrect,
        "skipped": skipped,
        "exits": dict(exits),
        "empty_pred": no_pred,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("out_roots", nargs="+", help="One or more output dirs to aggregate")
    ap.add_argument("--bench", action="append", choices=BENCHES, help="Limit to specific benches; repeat to add more.")
    ap.add_argument("--show_wrong", type=int, default=0, help="Print N incorrect samples per bench")
    ap.add_argument("--show_no_tool", type=int, default=0, help="Print N no_tool_calls samples per bench")
    ap.add_argument("--benchmark-data-root", type=Path,
                    help="Root containing the benchmark files listed by DEFAULT_INPUTS. "
                         "Required for gold lookup; omit together with --no_gold.")
    ap.add_argument("--no_gold", action="store_true", help="Skip loading inputs for gold lookup (faster, less detailed).")
    args = ap.parse_args()

    benches = args.bench or BENCHES
    if not args.no_gold and args.benchmark_data_root is None:
        ap.error("--benchmark-data-root is required unless --no_gold is set")

    gold_indices: dict[str, dict | None] = {}
    if not args.no_gold:
        for bench in benches:
            inp = args.benchmark_data_root / DEFAULT_INPUTS[bench]
            try:
                gold_indices[bench] = _build_gold_index(inp, bench) if inp.exists() else None
            except Exception as exc:
                print(f"[warn] gold lookup failed for {bench}: {exc}", file=sys.stderr)
                gold_indices[bench] = None

    summary: dict[str, dict] = {}
    for bench in benches:
        rows: list[dict] = []
        for root in args.out_roots:
            rows.extend(collect_rows(root, bench))
        summary[bench] = audit_bench(rows, bench, gold_indices.get(bench), show_wrong=args.show_wrong, show_no_tool=args.show_no_tool)

    print("\n=== overall ===")
    total_n = sum(s.get("n", 0) for s in summary.values())
    total_c = sum(s.get("correct", 0) for s in summary.values())
    total_i = sum(s.get("incorrect", 0) for s in summary.values())
    total_s = sum(s.get("skipped", 0) for s in summary.values())
    print(f"  rows total={total_n}  correct={total_c}  incorrect={total_i}  unscored={total_s}")
    if total_c + total_i:
        print(f"  scored accuracy = {100*total_c/(total_c+total_i):.1f}%  (over {total_c+total_i} scored)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
