#!/usr/bin/env python3
"""T107a grader — Live Concert Audio Forensics & Setlist Reconstruction.

Eight evaluation dimensions:
  1. boundary_iou           — Hungarian-matched mean IoU on song segments
  2. song_title_accuracy    — closed-vocab title match for each matched song
  3. setlist_set_jaccard    — set Jaccard of predicted vs GT titles (any order)
  4. tempo_score            — 1 - clamp(MAE / 16, 0, 1) over matched songs
  5. transition_iou         — Hungarian-matched mean IoU on inter-song gaps
  6. defect_f1              — F1 on detected vs GT defects (closed vocab; +/-3s match)
  7. cross_doc_consistency  — schema, contiguity, non-overlap, vocab compliance
  8. summary_quality        — LLM judge on the concert summary (5 axes)
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
from pathlib import Path

# The release runner supplies CLAW_BENCH_LIB_DIR on PYTHONPATH. Keep a local
# fallback so the grader also works when invoked directly from this checkout.
_LIB_DIR = Path(
    os.environ.get(
        "CLAW_BENCH_LIB_DIR",
        str(Path(__file__).resolve().parents[2] / "_lib"),
    )
)
if _LIB_DIR.exists():
    sys.path.insert(0, str(_LIB_DIR))

try:
    from open_vocab_judge import axes_yes_no_score
except Exception:
    axes_yes_no_score = None


# ---------------------------------------------------------------------------
# Loading helpers
# ---------------------------------------------------------------------------
def _read_json(p: Path):
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None


def _read_csv(p: Path):
    if not p.exists():
        return []
    with p.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


# ---------------------------------------------------------------------------
# Hungarian matching by IoU
# ---------------------------------------------------------------------------
def _iou(a, b):
    s = max(a[0], b[0])
    e = min(a[1], b[1])
    inter = max(0.0, e - s)
    union = max(a[1], b[1]) - min(a[0], b[0])
    if union <= 0:
        return 0.0
    return inter / union


def _hungarian_iou(preds, gts):
    """Greedy 1-to-1 Hungarian on IoU; preds/gts are lists of [start,end].

    Returns (mean_matched_iou, matches:list[(pi,gi,iou)]).
    """
    if not preds or not gts:
        return 0.0, []
    pairs = []
    for pi, p in enumerate(preds):
        for gi, g in enumerate(gts):
            pairs.append((_iou(p, g), pi, gi))
    pairs.sort(reverse=True)
    used_p, used_g = set(), set()
    matches = []
    for v, pi, gi in pairs:
        if pi in used_p or gi in used_g:
            continue
        if v <= 0:
            continue
        used_p.add(pi); used_g.add(gi)
        matches.append((pi, gi, v))
    n_max = max(len(preds), len(gts))
    mean = sum(m[2] for m in matches) / n_max if n_max else 0.0
    return mean, matches


# ---------------------------------------------------------------------------
# Output parsers
# ---------------------------------------------------------------------------
def _parse_boundaries(out_dir: Path):
    j = _read_json(out_dir / "boundaries.json") or {}
    songs = j.get("songs", []) if isinstance(j, dict) else []
    spans = []
    for s in songs:
        try:
            ts = float(s.get("t_start"))
            te = float(s.get("t_end"))
            spans.append([ts, te])
        except Exception:
            continue
    return spans, songs


def _parse_songs_csv(out_dir: Path):
    rows = _read_csv(out_dir / "songs.csv")
    keep = []
    for r in rows:
        try:
            keep.append({
                "song_idx": int(r.get("song_idx")),
                "song_title": (r.get("song_title") or "").strip(),
                "tempo_bpm": float(r.get("tempo_bpm")),
            })
        except Exception:
            continue
    return keep


def _parse_transitions(out_dir: Path):
    j = _read_json(out_dir / "transitions.json") or {}
    trs = j.get("transitions", []) if isinstance(j, dict) else []
    spans = []
    for t in trs:
        try:
            spans.append([float(t.get("t_start")), float(t.get("t_end"))])
        except Exception:
            continue
    return spans


def _parse_defects(out_dir: Path):
    j = _read_json(out_dir / "defects.json") or {}
    ds = j.get("defects", []) if isinstance(j, dict) else []
    out = []
    for d in ds:
        try:
            out.append({
                "type": (d.get("defect_type") or "").strip(),
                "t_start": float(d.get("t_start")),
                "t_end": float(d.get("t_end")),
            })
        except Exception:
            continue
    return out


# ---------------------------------------------------------------------------
# Dimensions
# ---------------------------------------------------------------------------
def dim_boundary_iou(pred_spans, gt_boundaries):
    gt_spans = [[b["t_start"], b["t_end"]] for b in gt_boundaries]
    score, matches = _hungarian_iou(pred_spans, gt_spans)
    return round(score, 4), matches


def dim_song_title_accuracy(pred_songs_csv, pred_spans, gt_boundaries, matches, vocab):
    """For each matched (pred, gt) pair, check if pred title equals gt title."""
    if not matches:
        return 0.0
    # Map song_idx → title from CSV (1-based ordering follows boundaries.json order)
    pred_title_by_pi = {}
    for r in pred_songs_csv:
        idx = r["song_idx"] - 1
        if 0 <= idx < len(pred_spans):
            pred_title_by_pi[idx] = r["song_title"]
    correct = 0
    total = len(gt_boundaries)
    for pi, gi, _ in matches:
        pred_t = pred_title_by_pi.get(pi)
        if pred_t and pred_t in vocab and pred_t == gt_boundaries[gi]["title"]:
            correct += 1
    return round(correct / total, 4)


def dim_setlist_set_jaccard(pred_songs_csv, gt_boundaries, vocab):
    pred_set = set(r["song_title"] for r in pred_songs_csv if r["song_title"] in vocab)
    gt_set = set(b["title"] for b in gt_boundaries)
    if not pred_set and not gt_set:
        return 1.0
    inter = len(pred_set & gt_set)
    union = len(pred_set | gt_set)
    return round(inter / union if union else 0.0, 4)


def dim_tempo_score(pred_songs_csv, pred_spans, gt_tempos, matches):
    """1 - clamp(|err|/16, 0, 1) averaged over matched songs."""
    if not matches:
        return 0.0
    pred_bpm_by_pi = {}
    for r in pred_songs_csv:
        idx = r["song_idx"] - 1
        if 0 <= idx < len(pred_spans):
            pred_bpm_by_pi[idx] = r["tempo_bpm"]
    scores = []
    for pi, gi, _ in matches:
        gt_bpm = gt_tempos[gi]["tempo_bpm"]
        pred_bpm = pred_bpm_by_pi.get(pi)
        if pred_bpm is None:
            scores.append(0.0)
            continue
        # Allow octave error (BPM/2 or BPM*2) at half credit
        errs = [
            abs(pred_bpm - gt_bpm),
            abs(pred_bpm * 2 - gt_bpm),
            abs(pred_bpm / 2 - gt_bpm),
        ]
        err = min(errs)
        scores.append(max(0.0, 1.0 - err / 16.0))
    return round(sum(scores) / len(gt_tempos), 4)


def dim_transition_iou(pred_trans, gt_trans):
    gt_spans = [[t["t_start"], t["t_end"]] for t in gt_trans]
    score, _ = _hungarian_iou(pred_trans, gt_spans)
    return round(score, 4)


def dim_defect_f1(pred_defects, gt_defects, vocab, tol=3.0):
    """Defect detected if same type and either start time within ±tol of GT start
    OR pred interval overlaps GT timestamp."""
    if not pred_defects and not gt_defects:
        return 1.0
    if not pred_defects:
        return 0.0
    if not gt_defects:
        return 0.0
    # Filter preds to vocab
    pred_v = [p for p in pred_defects if p["type"] in vocab]
    matched_g = set()
    tp = 0
    for p in pred_v:
        for gi, g in enumerate(gt_defects):
            if gi in matched_g:
                continue
            if p["type"] != g["defect_type"]:
                continue
            if (abs(p["t_start"] - g["t_start"]) <= tol
                    or (p["t_start"] <= g["t_start"] <= p["t_end"])):
                tp += 1
                matched_g.add(gi)
                break
    fp = len(pred_v) - tp
    fn = len(gt_defects) - len(matched_g)
    if tp == 0:
        return 0.0
    prec = tp / (tp + fp)
    rec = tp / (tp + fn)
    return round(2 * prec * rec / (prec + rec) if (prec + rec) else 0.0, 4)


def dim_cross_doc_consistency(pred_spans, pred_trans, pred_defects, songs_csv, total_dur, vocab):
    issues = []
    # Boundaries: contiguous, non-overlapping, in-bounds
    if not pred_spans:
        issues.append("no song spans")
    else:
        prev_end = -1
        for i, (a, b) in enumerate(sorted(pred_spans)):
            if a < 0 or b > total_dur + 1.0:
                issues.append(f"span {i} OOB")
            if a >= b:
                issues.append(f"span {i} non-positive")
            if a < prev_end - 0.5:
                issues.append(f"span {i} overlaps previous")
            prev_end = b
    # Songs CSV row count == boundaries count
    if len(songs_csv) != len(pred_spans):
        issues.append(f"songs.csv rows ({len(songs_csv)}) != boundaries ({len(pred_spans)})")
    # All song titles in vocab
    for r in songs_csv:
        if r["song_title"] not in vocab:
            issues.append(f"title {r['song_title']!r} not in vocab")
    return 1.0 if not issues else 0.0


def dim_summary_quality(out_dir: Path, songs_csv, defects, gt_boundaries):
    p = out_dir / "concert_summary.md"
    if not p.exists():
        return 0.0
    txt = p.read_text(encoding="utf-8", errors="ignore").strip()
    n_words = len(txt.split())
    if n_words == 0 or n_words > 250:
        return 0.0
    titles = [r["song_title"] for r in songs_csv]
    found = [t for t in titles if t and t in txt]
    has_two_titles = len(set(found)) >= 2
    has_header = bool(re.search(r"^##\s", txt, flags=re.M))
    mentions_count = (str(len(songs_csv)) in txt) or any(
        w in txt.lower() for w in ("songs", "tracks", "pieces"))
    mentions_defect = (
        bool(defects) and any(d["type"] in txt for d in defects)
    ) or ("no defect" in txt.lower() or "no audio defect" in txt.lower())

    # Prefer LLM judge if available.
    if axes_yes_no_score is not None and os.environ.get("CLAW_BENCH_SKIP_VLM") != "1":
        try:
            payload = (
                "CONTEXT:\n"
                "Concert summary memo for a live concert recording. "
                f"The agent's CSV reports {len(songs_csv)} songs and "
                f"{len(defects)} defects; titles include "
                + ", ".join(repr(t) for t in titles[:4])
                + ".\n\n"
                f"CONCERT_SUMMARY.MD:\n{txt[:4000]}\n"
            )
            axes = [
                ("song_count", "Does the memo mention a clear count of songs identified?"),
                ("specific_titles", "Does the memo reference at least two specific song titles by name?"),
                ("defect_status", "Does the memo mention whether an audio defect was found, and what type if present?"),
                ("structured_prose", "Does the memo have at least one Markdown ## header and coherent prose, not a data dump?"),
                ("no_fabrication", "Does the memo stay under 250 words and avoid fabrication of unrelated facts?"),
            ]
            judged = axes_yes_no_score(payload, axes, threshold=0.55)
            if not judged.get("skipped"):
                return round(float(judged.get("score", 0.0)), 4)
        except Exception:
            pass

    # Heuristic fallback
    axes = [has_two_titles, has_header, mentions_count, mentions_defect, n_words <= 250]
    return round(sum(1 for a in axes if a) / len(axes), 4)


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------
THRESHOLDS = {
    "boundary_iou":           0.65,
    "song_title_accuracy":    0.50,
    "setlist_set_jaccard":    0.55,
    "tempo_score":            0.55,
    "transition_iou":         0.45,
    "defect_f1":              0.45,
    "cross_doc_consistency":  1.0,
    "summary_quality":        0.55,
}

WEIGHTS = {
    "boundary_iou":           0.18,
    "song_title_accuracy":    0.16,
    "setlist_set_jaccard":    0.12,
    "tempo_score":            0.10,
    "transition_iou":         0.10,
    "defect_f1":              0.12,
    "cross_doc_consistency":  0.12,
    "summary_quality":        0.10,
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", required=True)
    ap.add_argument("--gt", default=None)
    ap.add_argument("--fixtures", default=None)
    args = ap.parse_args()

    out_dir = Path(args.output)
    here = Path(__file__).resolve().parent
    gt_path = Path(args.gt) if args.gt else here / "reference" / "gt.json"
    fix_dir = Path(args.fixtures) if args.fixtures else here / "fixtures"

    gt = _read_json(gt_path) or {}
    setlist_vocab = (_read_json(fix_dir / "setlist_vocab.json") or {}).get("songs", [])
    defect_vocab = (_read_json(fix_dir / "defect_vocab.json") or {}).get("defect_types", [])

    pred_spans, pred_song_objs = _parse_boundaries(out_dir)
    songs_csv = _parse_songs_csv(out_dir)
    pred_trans = _parse_transitions(out_dir)
    pred_defs = _parse_defects(out_dir)

    # ---- dims ------------------------------------------------------------
    boundary_score, matches = dim_boundary_iou(pred_spans, gt["boundaries"])
    title_acc = dim_song_title_accuracy(songs_csv, pred_spans, gt["boundaries"], matches, set(setlist_vocab))
    set_jac = dim_setlist_set_jaccard(songs_csv, gt["boundaries"], set(setlist_vocab))
    tempo_s = dim_tempo_score(songs_csv, pred_spans, gt["tempos"], matches)
    trans_iou = dim_transition_iou(pred_trans, gt["transitions"])
    defect_f1 = dim_defect_f1(pred_defs, gt["defects"], set(defect_vocab))
    cdc = dim_cross_doc_consistency(pred_spans, pred_trans, pred_defs, songs_csv, gt["_total_duration_s"], set(setlist_vocab))
    summary_q = dim_summary_quality(out_dir, songs_csv, pred_defs, gt["boundaries"])

    dims = {
        "boundary_iou":          boundary_score,
        "song_title_accuracy":   title_acc,
        "setlist_set_jaccard":   set_jac,
        "tempo_score":           tempo_s,
        "transition_iou":        trans_iou,
        "defect_f1":             defect_f1,
        "cross_doc_consistency": cdc,
        "summary_quality":       summary_q,
    }
    passed = {k: (v >= THRESHOLDS[k]) for k, v in dims.items()}
    overall_pass = all(passed.values())
    score = round(
        sum(WEIGHTS[k] * min(1.0, max(0.0, dims[k] / max(THRESHOLDS[k], 1e-9)))
            for k in dims),
        4,
    )
    dims_report = {
        k: {
            "value": dims[k],
            "threshold": THRESHOLDS[k],
            "direction": "ge",
            "ok": passed[k],
            "weight": WEIGHTS[k],
        }
        for k in dims
    }

    result = {
        "task": "T107a_concert_setlist_audio_forensics",
        "dims": dims_report,
        "thresholds": THRESHOLDS,
        "weights": WEIGHTS,
        "passed": passed,
        "pass": overall_pass,
        "score": score,
    }
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
