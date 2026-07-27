#!/usr/bin/env python3
"""Grader for T104a — Dashcam Drive Brief."""

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import os
import sys as _sys
from pathlib import Path as _Path
_REPO_ROOT = _Path(__file__).resolve().parents[2]
_sys.path.insert(0, str(_REPO_ROOT / "_lib"))
try:
    from open_vocab_judge import axes_yes_no_score
except Exception:
    axes_yes_no_score = None

EVENT_VOCAB = [
    "hard_brake", "hard_accel",
    "sharp_turn_left", "sharp_turn_right",
    "near_stop", "cruise",
]

EXPECTED_SEQS = ["seq_A", "seq_B", "seq_C", "seq_D", "seq_E"]


# ---------- loaders ----------

def load_gt(path: Path) -> Dict[str, Any]:
    with open(path) as f:
        return json.load(f)


def load_csv(path: Path) -> List[Dict[str, str]]:
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def load_json(path: Path) -> Any:
    with open(path) as f:
        return json.load(f)


def load_text(path: Path) -> str:
    with open(path) as f:
        return f.read()


# ---------- helpers ----------

def safe_float(s: Any) -> float:
    try:
        return float(s)
    except (TypeError, ValueError):
        return float("nan")


def safe_int(s: Any) -> int:
    try:
        return int(float(s))
    except (TypeError, ValueError):
        return 0


def interval_iou(a_start: int, a_end: int, b_start: int, b_end: int) -> float:
    """IoU between two inclusive integer frame intervals."""
    inter = max(0, min(a_end, b_end) - max(a_start, b_start) + 1)
    union = (a_end - a_start + 1) + (b_end - b_start + 1) - inter
    return inter / union if union > 0 else 0.0


# ---------- metric: distance_mape ----------

def metric_distance_mape(csv_rows: List[Dict[str, str]], gt: Dict[str, Any]) -> float:
    """Mean per-seq absolute percent error on total_distance_m."""
    by_id = {r.get("seq_id", "").strip(): r for r in csv_rows}
    errs = []
    for s in gt["sequences"]:
        sid = s["seq_id"]
        gt_d = float(s["total_distance_m"])
        if sid not in by_id:
            errs.append(1.0)
            continue
        pred = safe_float(by_id[sid].get("total_distance_m"))
        denom = max(abs(gt_d), 1e-3)
        errs.append(abs(pred - gt_d) / denom)
    return sum(errs) / len(errs) if errs else 1.0


# ---------- metric: speed_mape ----------

def metric_speed_mape(csv_rows: List[Dict[str, str]], gt: Dict[str, Any]) -> float:
    """Combined avg+max speed MAPE."""
    by_id = {r.get("seq_id", "").strip(): r for r in csv_rows}
    errs = []
    for s in gt["sequences"]:
        sid = s["seq_id"]
        if sid not in by_id:
            errs.append(1.0)
            errs.append(1.0)
            continue
        for field in ("avg_speed_kmh", "max_speed_kmh"):
            gt_v = float(s[field])
            pred = safe_float(by_id[sid].get(field))
            denom = max(abs(gt_v), 1e-3)
            errs.append(abs(pred - gt_v) / denom)
    return sum(errs) / len(errs) if errs else 1.0


# ---------- metric: event count accuracy ----------

def metric_event_count_accuracy(events_json: Dict[str, Any], gt: Dict[str, Any]) -> float:
    """Mean per-seq |pred_count - gt_count| / max(1, gt_count)."""
    pred_by_id = {s.get("seq_id"): s.get("events", []) for s in events_json.get("sequences", [])}
    errs = []
    for s in gt["sequences"]:
        sid = s["seq_id"]
        gt_count = len(s.get("events", []))
        pred_count = len(pred_by_id.get(sid, []))
        denom = max(1, gt_count)
        errs.append(abs(pred_count - gt_count) / denom)
    return sum(errs) / len(errs) if errs else 1.0


# ---------- metric: event recall / precision ----------

def _match_events(
    a_events: List[Dict[str, Any]],
    b_events: List[Dict[str, Any]],
    iou_thresh: float = 0.4,
) -> int:
    """For each event in `a`, return count of those that have at least one
    event in `b` with same type and IoU >= threshold."""
    matched = 0
    for ev in a_events:
        et = ev.get("event_type")
        try:
            s, e = int(ev.get("frame_start")), int(ev.get("frame_end"))
        except (TypeError, ValueError):
            continue
        if e < s:
            continue
        for cand in b_events:
            if cand.get("event_type") != et:
                continue
            try:
                cs, ce = int(cand.get("frame_start")), int(cand.get("frame_end"))
            except (TypeError, ValueError):
                continue
            if ce < cs:
                continue
            if interval_iou(s, e, cs, ce) >= iou_thresh:
                matched += 1
                break
    return matched


def metric_event_recall(events_json: Dict[str, Any], gt: Dict[str, Any]) -> float:
    pred_by_id = {s.get("seq_id"): s.get("events", []) for s in events_json.get("sequences", [])}
    total_gt = 0
    total_match = 0
    for s in gt["sequences"]:
        sid = s["seq_id"]
        gt_evs = s.get("events", [])
        pred_evs = pred_by_id.get(sid, [])
        total_gt += len(gt_evs)
        total_match += _match_events(gt_evs, pred_evs)
    return total_match / total_gt if total_gt > 0 else 1.0


def metric_event_precision(events_json: Dict[str, Any], gt: Dict[str, Any]) -> float:
    gt_by_id = {s["seq_id"]: s.get("events", []) for s in gt["sequences"]}
    total_pred = 0
    total_match = 0
    for s in events_json.get("sequences", []):
        sid = s.get("seq_id")
        pred_evs = s.get("events", [])
        gt_evs = gt_by_id.get(sid, [])
        # Only count predictions whose event_type is in the vocab
        valid_preds = [e for e in pred_evs if e.get("event_type") in EVENT_VOCAB]
        total_pred += len(valid_preds)
        total_match += _match_events(valid_preds, gt_evs)
    return total_match / total_pred if total_pred > 0 else 0.0


# ---------- metric: aggregate consistency ----------

def metric_aggregate_consistency(
    csv_rows: List[Dict[str, str]],
    events_json: Dict[str, Any],
    aggregate: Dict[str, Any],
) -> float:
    """Aggregate fields within ±2% of sum of per-seq stats."""
    if not csv_rows:
        return 0.0
    sum_dist_m = sum(safe_float(r.get("total_distance_m")) for r in csv_rows)
    sum_dur_sec = sum(safe_float(r.get("duration_sec")) for r in csv_rows)
    sum_events = sum(len(s.get("events", [])) for s in events_json.get("sequences", []))

    # Pred aggregate
    pred_dist_km = safe_float(aggregate.get("total_distance_km"))
    pred_dur_min = safe_float(aggregate.get("total_duration_min"))
    pred_total_events = safe_int(aggregate.get("total_events"))
    pred_dominant = aggregate.get("dominant_event_type")

    # Compute expected dominant from CSV+events_json (count occurrences)
    type_counts: Dict[str, int] = {}
    for s in events_json.get("sequences", []):
        for ev in s.get("events", []):
            et = ev.get("event_type")
            if et:
                type_counts[et] = type_counts.get(et, 0) + 1
    expected_dominant = (
        max(type_counts.items(), key=lambda kv: kv[1])[0] if type_counts else None
    )

    def within_pct(pred: float, expect: float, pct: float) -> bool:
        if expect == 0:
            return abs(pred) <= 0.02  # near zero
        return abs(pred - expect) / max(abs(expect), 1e-9) <= pct

    ok_dist = within_pct(pred_dist_km, sum_dist_m / 1000.0, 0.02)
    ok_dur = within_pct(pred_dur_min, sum_dur_sec / 60.0, 0.02)
    ok_evt = pred_total_events == sum_events
    ok_dom = (expected_dominant is None) or (pred_dominant == expected_dominant)

    return 1.0 if (ok_dist and ok_dur and ok_evt and ok_dom) else 0.0


# ---------- metric: cross_doc_consistency ----------

def metric_cross_doc_consistency(
    csv_rows: List[Dict[str, str]],
    events_json: Dict[str, Any],
) -> float:
    """csv has 5 rows; events.json has 5 sequences; csv.n_events == len(events)."""
    if len(csv_rows) != 5:
        return 0.0
    pred_seqs = events_json.get("sequences", [])
    if len(pred_seqs) != 5:
        return 0.0
    csv_ids = sorted(r.get("seq_id", "") for r in csv_rows)
    if csv_ids != EXPECTED_SEQS:
        return 0.0
    json_ids = sorted(s.get("seq_id", "") for s in pred_seqs)
    if json_ids != EXPECTED_SEQS:
        return 0.0
    pred_by_id = {s.get("seq_id"): s.get("events", []) for s in pred_seqs}
    for r in csv_rows:
        sid = r.get("seq_id")
        try:
            n_events_csv = int(r.get("n_events"))
        except (TypeError, ValueError):
            return 0.0
        if n_events_csv != len(pred_by_id.get(sid, [])):
            return 0.0
    return 1.0


# ---------- metric: brief quality ----------

def metric_brief_quality(brief: str, events_json: Dict[str, Any]) -> float:
    wc = len(brief.split())
    if wc > 420:
        return 0.0
    if wc < 80:
        return 0.0
    if axes_yes_no_score is None or os.environ.get("CLAW_BENCH_SKIP_VLM") == "1":
        score = 0.0
        if "##" in brief: score += 0.2
        seq_named = sum(1 for s in ["seq_A","seq_B","seq_C","seq_D","seq_E"] if s in brief)
        if seq_named >= 3: score += 0.3
        if any(k in brief.lower() for k in ["riskiest","most risk","most dangerous","highest risk"]):
            score += 0.2
        if any(k in brief.lower() for k in ["recommend","suggest","should","propose"]):
            score += 0.15
        if any(k in brief for k in ["km","m/s","kmh","seconds","frames"]):
            score += 0.15
        return min(1.0, score)
    axes = [
        ("identifies_riskiest_seq",
         "Does the brief explicitly identify the riskiest sequence by seq_id and justify the choice with concrete reference to events from EVENTS_JSON?"),
        ("references_events_and_seqs",
         "Does the brief name at least 3 different seq_ids (seq_A..seq_E) AND reference at least 2 specific event_types (e.g. cruise, hard_brake, sharp_turn_left) tied to those sequences?"),
        ("quantitative_facts",
         "Does the brief cite at least 2 quantitative facts (distance in km/m, speed in km/h or m/s, duration in seconds, or specific frame indices/event counts) sourced from the data?"),
        ("actionable_recommendation",
         "Does the brief include at least one concrete recommendation tied to a specific sequence or event pattern (not generic 'drive safely')?"),
        ("structure_and_length",
         "Is the brief structured with ≥1 ## header, 80-400 words, coherent prose (not keyword soup or numeric dump)?"),
    ]
    payload = (
        f"EVENTS_JSON:\n{json.dumps(events_json, ensure_ascii=False)[:4000]}\n\n"
        f"DRIVE_BRIEF.md:\n{brief[:6000]}"
    )
    res = axes_yes_no_score(payload, axes, threshold=0.65)
    return float(res.get("score", 0.0))


# ---------- driver ----------

def main() -> None:
    parser = argparse.ArgumentParser(description="Grade T104a Dashcam Drive Brief")
    parser.add_argument("--output", type=Path, required=True, help="Path to output dir")
    parser.add_argument("--gt", type=Path, required=True, help="Path to GT JSON")
    parser.add_argument("--fixtures", type=Path, required=False, help="Fixtures dir (unused)")
    args = parser.parse_args()

    out = args.output
    try:
        csv_rows = load_csv(out / "drive_stats.csv")
        events_json = load_json(out / "events.json")
        aggregate = load_json(out / "aggregate_stats.json")
        brief = load_text(out / "drive_brief.md")
        gt = load_gt(args.gt)
    except FileNotFoundError as e:
        print(json.dumps({"pass": False, "stage": "gating", "errors": [f"missing required file: {e}"]}, indent=2))
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(json.dumps({"pass": False, "stage": "gating", "errors": [f"invalid JSON: {e}"]}, indent=2))
        sys.exit(1)

    results = {
        "distance_mape":         metric_distance_mape(csv_rows, gt),
        "speed_mape":            metric_speed_mape(csv_rows, gt),
        "event_count_accuracy":  metric_event_count_accuracy(events_json, gt),
        "event_type_recall":     metric_event_recall(events_json, gt),
        "event_type_precision":  metric_event_precision(events_json, gt),
        "aggregate_consistency": metric_aggregate_consistency(csv_rows, events_json, aggregate),
        "cross_doc_consistency": metric_cross_doc_consistency(csv_rows, events_json),
        "brief_quality":         metric_brief_quality(brief, events_json),
    }

    # Threshold + comparison direction
    # "lte" = score must be <= threshold (lower is better, e.g. error rates)
    # "gte" = score must be >= threshold (higher is better)
    thresholds: Dict[str, Tuple[str, float]] = {
        "distance_mape":         ("lte", 0.10),
        "speed_mape":            ("lte", 0.15),
        "event_count_accuracy":  ("lte", 0.20),
        "event_type_recall":     ("gte", 0.55),
        "event_type_precision":  ("gte", 0.55),
        "aggregate_consistency": ("gte", 1.0),
        "cross_doc_consistency": ("gte", 1.0),
        "brief_quality":         ("gte", 0.65),
    }

    weights = {k: 1.0 / len(thresholds) for k in thresholds}
    dims = {}
    score = 0.0
    passed_all = True
    for k, v in results.items():
        direction, t = thresholds[k]
        ok = (v <= t) if direction == "lte" else (v >= t)
        passed_all &= ok
        if direction == "lte":
            credit = max(0.0, min(1.0, t / max(v, 1e-9)))
        else:
            credit = max(0.0, min(1.0, v / max(t, 1e-9)))
        score += weights[k] * credit
        dims[k] = {
            "value": round(v, 4),
            "threshold": t,
            "direction": direction,
            "ok": ok,
            "weight": weights[k],
        }

    print(json.dumps({
        "pass": passed_all,
        "score": round(score, 4),
        "dims": dims,
        "weights": weights,
    }, indent=2))

    sys.exit(0 if passed_all else 1)


if __name__ == "__main__":
    main()
