"""Grader for T07a Surveillance Event Aggregation (v2 — VLM grounding heavy).

v2 changes vs v1 hardened:
  - event_grounding (NEW): per-event VLM grounding. Samples 5 events
    from incidents.csv, extracts a midpoint frame from the source clip
    via ffmpeg, asks VLM whether the predicted event_type is plausibly
    happening in that frame.
  - storytelling_axes (replaces single-score storytelling): 5 yes/no
    axes (Summary present, Detail present, counts match CSV, per-event
    narration >= ceil(N/2), factual tone), mean must be >= 0.70.
  - Capability dims (recall_any, recall_type, fp_per_clip, time_iou,
    header_structure) and report_consistent kept as-is.
"""
import argparse
import csv
import json
import math
import os
import random
import re
import subprocess
import sys
import tempfile
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT / "_lib"))

from open_vocab_judge import (  # noqa: E402
    llm_semantic_pairs,  # re-exported for symmetry with v2 template
    vlm_grounding,
    axes_yes_no_score,
    mllm_video_axes,
)


# Multimodal checklist for `incidents.mp4` content quality. 4 evenly-
# spaced frames, majority vote per axis. Catches agents that produce
# black/test-pattern mp4 even though incidents.csv looks correct.
INCIDENTS_VIDEO_AXES = [
    ("real_cctv_footage",
     "Is this frame real CCTV / surveillance footage (street / "
     "storefront / parking / interior with cameras / people / "
     "vehicles) — NOT a black screen, color bars, test pattern, or "
     "single-color placeholder?"),
    ("not_static_card",
     "Is this frame from continuous video footage rather than a "
     "static title card / single text caption / freeze frame?"),
    ("scene_has_subject",
     "Does this frame show at least one identifiable subject "
     "(person, vehicle, building, or scene) consistent with public "
     "surveillance content?"),
    ("non_synthetic",
     "Does this frame look like real-world recorded footage rather "
     "than synthetic 3D render, animation, drawn diagram, or PowerPoint slide?"),
]

CRITICAL_CHECKS = [
    "recall_any_ok",
    "recall_type_ok",
    "fp_ok",
    "time_iou_ok",
    "report_consistent_ok",
    "storytelling_axes_ok",
    "incidents_video_axes_ok",
    "vocab_consistency_ok",
]
PASS_SCORE_THRESHOLD = 0.85

VALID_EVENTS = {"burglary", "fighting", "explosion", "robbery",
                "road_accident"}
DECOY_EVENTS = {"vandalism", "shoplifting", "assault", "arson", "arrest"}
TIME_TOL = 2.0  # ±2 s boundary slack


def vocab_consistency(out: Path) -> dict:
    """Hard-fail any incidents.csv row whose event_type is not in the 5
    real UCF-Crime classes. Decoy classes (same-domain UCF labels not
    used in this fixture) and out-of-vocab strings count as violations.
    """
    rows = list(csv.DictReader((out / "incidents.csv").open()))
    bad_rows = []
    decoy_rows = []
    for i, r in enumerate(rows):
        et = (r.get("event_type") or "").strip().lower()
        if et in VALID_EVENTS:
            continue
        bad_rows.append({"row": i, "event_type": et})
        if et in DECOY_EVENTS:
            decoy_rows.append({"row": i, "event_type": et})
    return {
        "n_rows": len(rows),
        "n_bad": len(bad_rows),
        "n_decoy_used": len(decoy_rows),
        "examples": bad_rows[:5],
        "ok": len(bad_rows) == 0,
    }


# --------------------------------------------------------------------- #
# Gating + small parsing helpers (kept verbatim from v1).
# --------------------------------------------------------------------- #
def gating(out: Path):
    errs = []
    for f in ["incidents.mp4", "incidents.csv", "report.md"]:
        if not (out / f).exists():
            errs.append(f"missing {f}")
    if (out / "incidents.mp4").exists():
        try:
            d = float(subprocess.check_output(
                ["ffprobe", "-v", "error", "-show_entries", "format=duration",
                 "-of", "default=nw=1:nk=1", str(out / "incidents.mp4")]).strip())
            if d > 480:
                errs.append(f"incidents.mp4 {d:.0f}s > 480s")
        except (subprocess.CalledProcessError, ValueError, FileNotFoundError) as e:
            errs.append(f"mp4 probe: {e}")
    if (out / "incidents.csv").exists():
        try:
            rows = list(csv.DictReader((out / "incidents.csv").open()))
            for r in rows:
                for k in ("source_clip", "source_start", "source_end",
                          "event_type", "confidence", "summary"):
                    assert k in r, f"missing csv column: {k}"
        except (FileNotFoundError, csv.Error, AssertionError, KeyError) as e:
            errs.append(f"csv schema: {e}")
    if (out / "report.md").exists():
        text = (out / "report.md").read_text()
        if "Summary" not in text or "Detail" not in text:
            errs.append("report.md missing Summary/Detail sections")
    return len(errs) == 0, errs


def _match(p, g, tol=TIME_TOL):
    return (abs(p["start_sec"] - g["start_sec"]) <= tol and
            abs(p["end_sec"] - g["end_sec"]) <= tol)


def _iou(p, g):
    inter_s = max(p["start_sec"], g["start_sec"])
    inter_e = min(p["end_sec"], g["end_sec"])
    inter = max(0.0, inter_e - inter_s)
    union_s = min(p["start_sec"], g["start_sec"])
    union_e = max(p["end_sec"], g["end_sec"])
    union = max(1e-6, union_e - union_s)
    return inter / union


def _parse_time(v) -> float:
    """Accept either float seconds (e.g. 70.8) or HH:MM:SS / MM:SS / SS strings."""
    if v is None:
        raise ValueError("None")
    s = str(v).strip()
    if not s:
        raise ValueError("empty")
    try:
        return float(s)
    except ValueError:
        pass
    parts = s.split(":")
    if not 1 <= len(parts) <= 3:
        raise ValueError(f"bad time: {v}")
    parts = [float(p) for p in parts]
    while len(parts) < 3:
        parts.insert(0, 0.0)
    h, m, sec = parts
    return h * 3600 + m * 60 + sec


# --------------------------------------------------------------------- #
# Capability dims.
# --------------------------------------------------------------------- #
def event_recall(out: Path, gt: dict) -> dict:
    rows = list(csv.DictReader((out / "incidents.csv").open()))
    pred_by_clip = {}
    for r in rows:
        try:
            pred_by_clip.setdefault(r["source_clip"], []).append({
                "start_sec": _parse_time(r["source_start"]),
                "end_sec": _parse_time(r["source_end"]),
                "type": r["event_type"].lower().strip(),
            })
        except (ValueError, KeyError, TypeError):
            pass

    matched_any = matched_type = total = fp = 0
    iou_sum = 0.0
    iou_n = 0
    iou_pass = 0
    for clip, evs in gt.items():
        preds = pred_by_clip.get(clip, [])
        used = set()
        for g in evs:
            total += 1
            best_iou = 0.0
            best_i = None
            for i, p in enumerate(preds):
                if i in used:
                    continue
                if _match(p, g):
                    iou_here = _iou(p, g)
                    if iou_here > best_iou:
                        best_iou = iou_here
                        best_i = i
            if best_i is not None:
                matched_any += 1
                used.add(best_i)
                iou_sum += best_iou
                iou_n += 1
                if best_iou >= 0.50:
                    iou_pass += 1
                if preds[best_i]["type"] == g["type"]:
                    matched_type += 1
        fp += max(0, len(preds) - len(used))

    return {
        "recall_any": round(matched_any / max(1, total), 3),
        "recall_type_correct": round(matched_type / max(1, total), 3),
        "matched": matched_any,
        "type_matched": matched_type,
        "total": total,
        "false_positives": fp,
        "fp_per_clip": round(fp / max(1, len(gt)), 3),
        "time_iou_pass_ratio": round(iou_pass / max(1, total), 3),
        "mean_iou_matched": round(iou_sum / max(1, iou_n), 3),
    }


def report_consistency(out: Path) -> dict:
    """Cross-check report.md Summary counts vs CSV per-type totals."""
    md = (out / "report.md").read_text()
    rows = list(csv.DictReader((out / "incidents.csv").open()))
    csv_total = len(rows)
    csv_by_type = {}
    for r in rows:
        t = r.get("event_type", "").lower().strip()
        csv_by_type[t] = csv_by_type.get(t, 0) + 1

    m_total = re.search(r"total\s+events?\s*:?\s*(\d+)", md, re.I)
    md_total = int(m_total.group(1)) if m_total else None

    md_by_type = {}
    for t in VALID_EVENTS:
        m = re.search(rf"\b{t}\b\s*:?\s*(\d+)", md, re.I)
        if m:
            md_by_type[t] = int(m.group(1))

    issues = []
    if md_total is None:
        issues.append("report missing 'Total events: N'")
    elif md_total != csv_total:
        issues.append(f"report Total={md_total} vs csv rows={csv_total}")
    for t, n in csv_by_type.items():
        if t not in md_by_type:
            issues.append(f"report missing per-type count for {t}")
        elif md_by_type[t] != n:
            issues.append(f"{t}: report={md_by_type[t]} csv={n}")
    return {"csv_total": csv_total,
            "csv_by_type": csv_by_type,
            "md_total": md_total,
            "md_by_type": md_by_type,
            "issues": issues,
            "ok": not issues}


def header_structure(out: Path, expected_n_events: int) -> dict:
    """Sample frames at 0.5 s intervals across incidents.mp4 and count
    how many ~2 s low-luma stretches exist. Each event should be
    preceded by a 2 s black titlecard (per the spec)."""
    mp4 = out / "incidents.mp4"
    if not mp4.exists():
        return {"ok": False, "reasons": "incidents.mp4 missing"}
    try:
        d = float(subprocess.check_output(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "default=nw=1:nk=1", str(mp4)]).strip())
    except (subprocess.CalledProcessError, ValueError, FileNotFoundError) as e:
        return {"ok": False, "reasons": f"probe: {e}"}
    if d <= 0:
        return {"ok": False, "reasons": "bad duration"}
    try:
        out_stats = subprocess.check_output([
            "ffmpeg", "-loglevel", "error", "-i", str(mp4),
            "-vf", "fps=2,signalstats,metadata=mode=print:key=lavfi.signalstats.YAVG",
            "-f", "null", "-",
        ], stderr=subprocess.STDOUT, timeout=180).decode("utf-8", errors="ignore")
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, FileNotFoundError) as e:
        return {"ok": False, "reasons": f"ffmpeg: {e}"}
    yavgs = []
    for line in out_stats.splitlines():
        if "YAVG" in line:
            try:
                yavgs.append(float(line.split("=")[-1]))
            except (ValueError, IndexError):
                pass
    dark_runs = 0
    run = 0
    for y in yavgs:
        if y < 25:
            run += 1
        else:
            if run >= 3:
                dark_runs += 1
            run = 0
    if run >= 3:
        dark_runs += 1
    needed = max(1, int(expected_n_events * 0.6))
    return {
        "expected_n_events": expected_n_events,
        "needed_dark_runs": needed,
        "dark_runs_found": dark_runs,
        "duration_sec": round(d, 1),
        "ok": dark_runs >= needed,
    }


# --------------------------------------------------------------------- #
# v2 — per-event VLM grounding.
# --------------------------------------------------------------------- #
def event_grounding(out: Path, clips_dir: Path,
                    max_samples: int = 5,
                    frames_per_event: int = 3) -> dict:
    """Sample up to `max_samples` events from incidents.csv. For each,
    extract `frames_per_event` evenly-spaced frames from the source
    clip via ffmpeg and ask the VLM whether the predicted event_type
    is plausibly happening. The dim is the median over frames per
    event, then ratio of events whose median is yes/score>=0.5.

    Multi-frame sampling cuts single-frame VLM noise (e.g. fighting
    looks like nothing on a quiet frame between two punches)."""
    csv_path = out / "incidents.csv"
    if not csv_path.exists():
        return {"ok": False, "reasons": "incidents.csv missing",
                "ratio": 0.0, "informative": 0, "passed": 0}
    rows = list(csv.DictReader(csv_path.open()))
    rng = random.Random(0xFEED)
    indexed = list(enumerate(rows))
    rng.shuffle(indexed)

    samples_meta = []
    extract_errs = []
    with tempfile.TemporaryDirectory() as td:
        td_path = Path(td)
        for orig_i, r in indexed:
            if len(samples_meta) >= max_samples:
                break
            try:
                start = _parse_time(r.get("source_start"))
                end = _parse_time(r.get("source_end"))
            except (ValueError, TypeError) as e:
                extract_errs.append(f"row {orig_i}: bad time ({e})")
                continue
            if end <= start:
                extract_errs.append(f"row {orig_i}: end<=start")
                continue
            duration = end - start
            source_clip = (r.get("source_clip") or "").strip()
            event_type = (r.get("event_type") or "").strip().lower()
            if not source_clip or not event_type:
                extract_errs.append(f"row {orig_i}: missing clip/type")
                continue
            clip_path = clips_dir / source_clip
            if not clip_path.exists():
                clip_path = clips_dir / Path(source_clip).name
            if not clip_path.exists():
                extract_errs.append(f"row {orig_i}: clip {source_clip} missing")
                continue
            # Extract `frames_per_event` evenly-spaced frames inside [start, end].
            # Pick at 1/(N+1), 2/(N+1), ..., N/(N+1) of the span so we avoid
            # frame-zero edge cases.
            n = max(1, frames_per_event)
            frame_files = []
            for k in range(1, n + 1):
                t = start + duration * (k / (n + 1))
                fname = f"sample_{len(samples_meta):02d}_f{k}.jpg"
                jpg_path = td_path / fname
                try:
                    subprocess.check_call(
                        ["ffmpeg", "-y", "-loglevel", "error",
                         "-ss", f"{t:.2f}",
                         "-i", str(clip_path),
                         "-frames:v", "1", str(jpg_path)],
                        stderr=subprocess.DEVNULL,
                    )
                except (subprocess.CalledProcessError, FileNotFoundError) as e:
                    extract_errs.append(f"row {orig_i} f{k}: ffmpeg {e}")
                    continue
                if jpg_path.exists():
                    frame_files.append(fname)
            if not frame_files:
                extract_errs.append(f"row {orig_i}: no frames extracted")
                continue
            samples_meta.append({
                "files": frame_files,
                "pred_label": event_type,
                "kind": "event class",
                "source_clip": source_clip,
                "span_sec": [round(start, 2), round(end, 2)],
            })

        grounding_results = vlm_grounding(
            samples_meta, photos_dir=td_path,
            frames_per_sample=frames_per_event,
        )

    informative = [r for r in grounding_results
                   if "ok" in r and not r.get("err") and not r.get("skipped")]
    passed = sum(1 for r in informative if r.get("ok"))
    ratio = passed / len(informative) if informative else 0.0
    return {
        "informative": len(informative),
        "passed": passed,
        "ratio": round(ratio, 3),
        "samples": grounding_results,
        "extract_errors": extract_errs[:5],
        "threshold": 0.65,
        "ok": len(informative) >= 3 and ratio >= 0.65,
    }


# --------------------------------------------------------------------- #
# v2 — storytelling_axes (replaces single-score storytelling).
# --------------------------------------------------------------------- #
def _storytelling_axes_for(csv_total: int):
    needed = math.ceil(csv_total / 2) if csv_total > 0 else 1
    return [
        ("summary_section_present",
         "Does REPORT.MD contain a markdown header line starting with "
         "`## Summary` (case-insensitive)?"),
        ("detail_section_present",
         "Does REPORT.MD contain a markdown header line starting with "
         "`## Detail` (case-insensitive)?"),
        ("counts_match_csv",
         "Do the totals reported in REPORT.MD match the CSV totals "
         "EXACTLY? Specifically the overall total events and the "
         "per-event-type counts must equal the values in CSV_TOTAL "
         "and CSV_BY_TYPE without contradiction."),
        ("per_event_narration",
         f"Does the Detail section narrate at least {needed} distinct "
         "incidents in prose (clip + what happened), as opposed to a "
         "raw CSV-style row dump or table?"),
        ("factual_tone",
         "Does the prose read as a factual security-operations incident "
         "report, free of fluff, lorem-ipsum, or template placeholders?"),
    ]


def storytelling_axes_score(out: Path, csv_total: int,
                            csv_by_type: dict) -> dict:
    md = (out / "report.md").read_text()
    payload = (
        f"CSV_TOTAL: {csv_total}\n"
        f"CSV_BY_TYPE: {json.dumps(csv_by_type, ensure_ascii=False)}\n\n"
        f"REPORT.MD:\n{md[:8000]}\n"
    )
    axes = _storytelling_axes_for(csv_total)
    return axes_yes_no_score(payload, axes, threshold=0.70)


# --------------------------------------------------------------------- #
# Main.
# --------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", required=True)
    ap.add_argument("--gt", required=True)
    ap.add_argument(
        "--clips_dir",
        default=str(Path(__file__).resolve().parent / "fixtures" / "cam"),
        help="Directory containing the source CCTV clips for VLM grounding.",
    )
    args = ap.parse_args()
    out = Path(args.output)
    clips_dir = Path(args.clips_dir)
    ok, errs = gating(out)
    if not ok:
        print(json.dumps({"pass": False, "stage": "gating", "errors": errs},
                         indent=2))
        return
    gt = json.load(open(args.gt))
    er = event_recall(out, gt)
    rc = report_consistency(out)
    n_events_csv = rc.get("csv_total", 0)
    hs = header_structure(out, expected_n_events=n_events_csv)
    sa = storytelling_axes_score(out, rc.get("csv_total", 0),
                                 rc.get("csv_by_type", {}))
    # MLLM video content checklist (4 frames, majority per axis).
    iv = mllm_video_axes(out / "incidents.mp4", INCIDENTS_VIDEO_AXES,
                         n_frames=4, threshold=0.70)
    vc = vocab_consistency(out)

    checks = {
        "recall_any_ok": er["recall_any"] >= 0.80,
        "recall_type_ok": er["recall_type_correct"] >= 0.65,
        "fp_ok": er["fp_per_clip"] <= 0.25,
        "time_iou_ok": er["time_iou_pass_ratio"] >= 0.70,
        "report_consistent_ok": rc["ok"],
        "header_structure_ok": hs["ok"],
        "storytelling_axes_ok": bool(sa.get("ok")),
        "incidents_video_axes_ok": bool(iv.get("ok")),
        "vocab_consistency_ok": vc["ok"],
    }
    # v3 weights (sum = 1.0). Storytelling axes (free-form prose) +
    # multimodal incidents.mp4 axes (anti-cheat content checklist).
    score = (
        0.20 * er["recall_any"]
        + 0.15 * er["recall_type_correct"]
        + 0.10 * (1.0 if er["fp_per_clip"] <= 0.25
                  else max(0.0, 1 - er["fp_per_clip"]))
        + 0.10 * er["time_iou_pass_ratio"]
        + 0.05 * (1.0 if hs["ok"] else 0.0)
        + 0.10 * (1.0 if rc["ok"] else 0.0)
        + 0.10 * sa.get("score", 0.0)
        + 0.15 * iv.get("score", 0.0)
        + 0.05  # format gate (gating already passed at this point)
    )
    score = round(score, 3)
    critical_ok = all(checks[name] for name in CRITICAL_CHECKS)
    passed = critical_ok and score >= PASS_SCORE_THRESHOLD
    weights = {
        "recall_any": 0.20,
        "recall_type_correct": 0.15,
        "fp_per_clip": 0.10,
        "time_iou_pass_ratio": 0.10,
        "header_structure": 0.05,
        "report_consistent": 0.10,
        "storytelling_axes": 0.10,
        "incidents_video_axes": 0.15,
        "format_gate": 0.05,
    }
    thresholds = {
        "recall_any": 0.80,
        "recall_type_correct": 0.65,
        "fp_per_clip_max": 0.25,
        "tol_sec": TIME_TOL,
        "time_iou_pass_ratio": 0.70,
        "storytelling_axes": 0.70,
        "incidents_video_axes": 0.70,
    }
    print(json.dumps({
        "pass": passed,
        "checks": checks,
        "thresholds": thresholds,
        "events": er,
        "report": rc,
        "header_structure": hs,
        "storytelling_axes": sa,
        "incidents_video_axes": iv,
        "vocab_consistency": vc,
        "weights": weights,
        "score": score,
        "pass_policy": {
            "critical_checks": CRITICAL_CHECKS,
            "non_critical_checks": ["header_structure_ok"],
            "score_threshold": PASS_SCORE_THRESHOLD,
        },
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
