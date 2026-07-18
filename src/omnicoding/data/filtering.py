"""Filter kira trajectories down to ones the model got right, then
emit ms-swift Agent JSONL via convert_kira_to_msswift.

Correctness layers (cheap → expensive):
  1. exit_reason == 'task_complete' AND error is None
  2. predicted_answer non-empty after extract_answer(...)
  3. Ground-truth match against the staged dataset's ``ground_truth``
     list. Supports two answer-types:
       - mcq: normalized exact match (case/punct-insensitive); matches
         "A", "a)", "A. text", "A) text", "[A]" etc. via the same
         family of permutations the SFT dataset already enumerates.
       - open: case-insensitive normalized inclusion (predicted is a
         substring of any GT, or any GT is a substring of predicted,
         after stripping whitespace/punct). For real-world open-text
         grading you'd plug in an LLM judge here; we keep this as a
         reproducible heuristic the user can audit.

Optional LLM-judge (``--judge_model openai/gpt-5.5`` etc.): when set,
items that fail the heuristic still get a second-chance grading via
the same litellm path kira uses. The judge prompt asks for ``YES`` /
``NO`` only, parsed strictly.

Outputs:
  <out_dir>/sft_train.jsonl       ms-swift JSONL (correct items only)
  <out_dir>/filter_report.json    per-item verdict + reason
  <out_dir>/images/<item>/...     decoded images (when --multimodal)

Usage:
  python filter_correct_trajectories.py \\
      --batch_dir /tmp/kira_sft_v2_*/out \\
      --items_file /tmp/kira_sft_v2_*/dataset/items.json \\
      --out_dir   /tmp/kira_sft_v2_*/sft \\
      --multimodal
"""

from __future__ import annotations

import argparse
import json
import re
import string
import sys
from pathlib import Path
from typing import Any

from omnicoding.data.conversion import convert_one

# Match the omnigaia spec's extractor (last <answer>X</answer>) so the
# filter reproduces what the harness scored, even if results.json
# doesn't carry predicted_answer for some reason.
_ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", re.DOTALL | re.IGNORECASE)


def _extract_last_answer(text: str) -> str:
    if not text:
        return ""
    matches = _ANSWER_RE.findall(text)
    return matches[-1].strip() if matches else ""


# ---------- correctness ---------------------------------------------

_PUNCT_TBL = str.maketrans("", "", string.punctuation)


def _normalize(s: str) -> str:
    s = (s or "").strip().lower()
    s = s.translate(_PUNCT_TBL)
    s = re.sub(r"\s+", " ", s)
    return s.strip()


def is_correct(
    *,
    answer_type: str,
    predicted: str,
    ground_truths: list[str] | None,
) -> tuple[bool, str]:
    """Return (correct, reason). ``answer_type`` is mcq | open | None."""
    if not predicted:
        return False, "empty_prediction"
    if not ground_truths:
        return False, "no_ground_truth"
    pred_n = _normalize(predicted)
    if not pred_n:
        return False, "prediction_empty_after_normalize"
    gt_n = [_normalize(g) for g in ground_truths if isinstance(g, str)]
    if answer_type == "mcq":
        # MCQ ground_truth lists already enumerate every formatting
        # variant the dataset accepts. Exact normalized match.
        for g in gt_n:
            if pred_n == g:
                return True, "exact_match"
        # Single-letter pred ("A") vs full-text gt — try first token.
        first_tok = pred_n.split(" ", 1)[0]
        if first_tok and any(g.startswith(first_tok + " ") or g == first_tok for g in gt_n):
            return True, "first_token_match"
        return False, "no_match_mcq"
    # open: substring inclusion either way
    for g in gt_n:
        if not g:
            continue
        if g == pred_n or g in pred_n or pred_n in g:
            return True, f"open_substring(g_in_p={g in pred_n})"
    return False, "no_match_open"


# ---------- I/O -----------------------------------------------------

def _load_items_by_id(items_path: Path) -> dict[str, dict[str, Any]]:
    items = json.loads(items_path.read_text(encoding="utf-8"))
    return {it["id"]: it for it in items}


def _load_results(out_dir: Path) -> list[dict[str, Any]]:
    p = out_dir / "results.json"
    if not p.exists():
        return []
    return json.loads(p.read_text(encoding="utf-8"))


def _load_messages(item_dir: Path) -> list[dict[str, Any]]:
    p = item_dir / "messages.json"
    if not p.exists():
        return []
    return json.loads(p.read_text(encoding="utf-8"))


def _load_image_subcalls(item_dir: Path) -> list[dict[str, Any]]:
    p = item_dir / "image_subcalls.jsonl"
    if not p.exists():
        return []
    out: list[dict[str, Any]] = []
    for line in p.read_text(encoding="utf-8").splitlines():
        if line.strip():
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return out


# ---------- pass/fail decision -------------------------------------

def _decide_pass(
    *,
    r: dict[str, Any],
    exit_reason: str | None,
    ok: bool,
    predicted: str,
    rescue_on: bool,
    correctness_reason: str,
) -> tuple[bool, str]:
    """Decide pass/fail for one trajectory. Two flavors:

    Strict (rescue_on=False):
        Pass iff ``exit_reason == 'task_complete'`` AND no error AND
        prediction matches ground_truth.

    Rescue (default, rescue_on=True):
        Pass iff prediction matches ground_truth — regardless of
        exit_reason. ``error`` exits still count as long as the model
        emitted a correct ``<answer>`` before crashing. The trajectory
        rendered through ms-swift will skip the synthetic-truncated
        turn; the loss-active region still teaches the right answer.
    """
    if rescue_on:
        if ok:
            return True, correctness_reason + ";rescue=on" if exit_reason != "task_complete" else correctness_reason
        return False, correctness_reason
    bad_exit = bool(r.get("error")) or exit_reason != "task_complete"
    if bad_exit:
        return False, f"bad_exit({exit_reason})"
    return ok, correctness_reason


# ---------- driver --------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--batch_dir", required=True, type=Path,
                   help="Run output dir (parent of item_NNNN/ + results.json + run_meta.json).")
    p.add_argument("--items_file", required=True, type=Path,
                   help="Staged items.json (with answer_type + ground_truth per row).")
    p.add_argument("--out_dir", required=True, type=Path,
                   help="Where sft_train.jsonl and filter_report.json land.")
    p.add_argument("--multimodal", action="store_true", default=True,
                   help="(default) Replace image_read text descriptions with "
                        "<image> tags and reference decoded images from "
                        "<batch_dir>/item_NNNN/images/ (created by run_bench_kira). "
                        "Required for SFT-training a vision-language model. "
                        "Pass --text_only to disable.")
    p.add_argument("--text_only", dest="multimodal", action="store_false",
                   help="Skip image substitution; write tool_responses verbatim. "
                        "Use only when training a text-only model.")
    p.add_argument("--keep_failed", action="store_true",
                   help="Also include failed trajectories in the JSONL "
                        "(marked via row['_filter_passed']=false). Useful for "
                        "DPO 'rejected' construction or audit; default OFF.")
    p.add_argument("--no_canonicalize_answer", action="store_true",
                   help="Don't inject <answer> into task_complete assistant content "
                        "when GPT chose path-a (shell echo). Default ON.")
    p.add_argument("--no_fill_empty_think", action="store_true",
                   help="Don't prepend empty <think>\\n\\n</think>\\n\\n on assistant "
                        "turns missing reasoning_content. Default ON matches Qwen3.6 chat template.")
    p.add_argument("--no_rescue_partial", action="store_true",
                   help="Disable partial-trajectory rescue. Default: rescue ON — accept "
                        "rows whose exit_reason is step_limit/no_tool_calls/error if a "
                        "<answer> can be extracted and matches ground_truth. Without this "
                        "rescue, every batch loses a sizable tail (long trajectories that "
                        "hit step_limit but already wrote the right answer mid-trajectory).")
    args = p.parse_args(argv)

    items_by_id = _load_items_by_id(args.items_file)
    results = _load_results(args.batch_dir)
    if not results:
        print(f"[filter] no results.json under {args.batch_dir}; nothing to do", file=sys.stderr)
        return 1
    run_meta = json.loads((args.batch_dir / "run_meta.json").read_text(encoding="utf-8"))

    args.out_dir.mkdir(parents=True, exist_ok=True)
    images_root = args.out_dir / "images" if args.multimodal else None

    rows_kept: list[dict[str, Any]] = []
    report: list[dict[str, Any]] = []
    n_row_errors = 0

    for r in results:
        # Per-row try/except so one malformed row (bad messages.json,
        # convert_one assertion failure, image decode crash, etc.) can't
        # take down the whole batch. The verdict is still recorded with
        # the error reason so downstream auditing sees what was dropped.
        try:
            item_id = r.get("id") if isinstance(r, dict) else None
            si = r.get("source_index") if isinstance(r, dict) else None
            item_dir = args.batch_dir / f"item_{int(si):04d}" if isinstance(si, int) else None
            gold = items_by_id.get(item_id) or {}
            answer_type = (gold.get("answer_type") or "").lower()
            gts = gold.get("ground_truth") or []

            exit_reason = r.get("exit_reason") if isinstance(r, dict) else None
            rescue_on = not args.no_rescue_partial
            # Predicted answer: harness-extracted first, fallback to scanning
            # messages.json. The fallback also catches rescued-tail trajectories
            # where exit_reason ≠ task_complete but the model wrote <answer>.
            predicted = (r.get("predicted_answer") or "").strip() if isinstance(r, dict) else ""
            if not predicted and item_dir and item_dir.exists():
                msgs = _load_messages(item_dir)
                joined = "\n".join(
                    (m.get("content") or "") if isinstance(m.get("content"), str) else ""
                    for m in msgs if m.get("role") in ("assistant", "tool")
                )
                predicted = _extract_last_answer(joined)
            ok, correctness_reason = is_correct(
                answer_type=answer_type, predicted=predicted, ground_truths=gts,
            )
            passed, reason = _decide_pass(
                r=r if isinstance(r, dict) else {}, exit_reason=exit_reason, ok=ok,
                predicted=predicted, rescue_on=rescue_on,
                correctness_reason=correctness_reason,
            )

            verdict = {
                "id": item_id,
                "source_index": si,
                "answer_type": answer_type,
                "predicted": predicted,
                "ground_truth_head": (gts or [None])[0],
                "exit_reason": exit_reason,
                "tool_call_num": r.get("tool_call_num") if isinstance(r, dict) else None,
                "passed": passed,
                "reason": reason,
            }
            report.append(verdict)

            if not (passed or args.keep_failed):
                continue
            if not item_dir or not item_dir.exists():
                verdict["reason"] = (verdict["reason"] or "") + ";no_item_dir"
                continue

            msgs = _load_messages(item_dir)
            sub = _load_image_subcalls(item_dir) if args.multimodal else None
            row = convert_one(
                messages=msgs,
                tools_spec=run_meta["tools_spec"],
                image_subcalls=sub,
                multimodal=args.multimodal,
                images_out_dir=images_root,
                item_tag=item_dir.name,
                item_images_dir=(item_dir / "images") if args.multimodal else None,
                canonicalize_answer=not args.no_canonicalize_answer,
                fill_empty_think=not args.no_fill_empty_think,
            )
            if not passed and args.keep_failed:
                row["_filter_passed"] = False
                row["_filter_reason"] = verdict["reason"]
            rows_kept.append(row)
        except Exception as exc:
            n_row_errors += 1
            si = r.get("source_index") if isinstance(r, dict) else None
            item_id = r.get("id") if isinstance(r, dict) else None
            print(f"[filter] row error si={si} id={item_id}: "
                  f"{type(exc).__name__}: {exc}", file=sys.stderr)
            report.append({
                "id": item_id, "source_index": si,
                "passed": False, "reason": f"row_error:{type(exc).__name__}:{exc}",
            })

    # Write outputs
    sft_path = args.out_dir / "sft_train.jsonl"
    with sft_path.open("w", encoding="utf-8") as f:
        for row in rows_kept:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    report_path = args.out_dir / "filter_report.json"
    summary = {
        "n_total": len(report),
        "n_passed": sum(1 for v in report if v["passed"]),
        "n_row_errors": n_row_errors,
        "by_answer_type": {},
    }
    for v in report:
        a = v.get("answer_type") or "?"
        s = summary["by_answer_type"].setdefault(a, {"total": 0, "passed": 0})
        s["total"] += 1
        if v["passed"]:
            s["passed"] += 1
    report_path.write_text(json.dumps({"summary": summary, "items": report},
                                      ensure_ascii=False, indent=2))

    print(f"[filter] {summary['n_passed']}/{summary['n_total']} passed "
          f"({100*summary['n_passed']/max(summary['n_total'],1):.1f}%)")
    for a, s in summary["by_answer_type"].items():
        print(f"  {a}: {s['passed']}/{s['total']}")
    print(f"[filter] wrote {sft_path}, {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
