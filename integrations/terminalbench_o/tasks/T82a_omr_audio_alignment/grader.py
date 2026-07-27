"""Grader for T82a OMR + Score-Audio Alignment (v3 — deterministic + summary axes).

v3 redesign:
  - Closed-vocab dims (deviation_type, decoy_type, voice, pitch_midi)
    remain strict normalised match — no LLM, no VLM.
  - musicology_review.md is genuinely free-form prose; replaced single
    VLM judge with 5-axis yes/no `musicology_quality_axes`.
  - Reweighted: deterministic dims dominate (~0.78); only the prose
    review uses LLM axes (0.15).
  - deviation_categories.json + decoy_categories.json now include 4-6
    decoy strings each (real musicology vocab not used in this piece)
    so the agent can't shortcut by dumping the closed vocab.

Capability dims  (~0.62): pitch_recall + alignment_mae_sec +
                          deviation_detection_f1 + decoy_rejection
Format / sanity  (~0.23): cross_doc_consistency
Quality dims     (~0.15): musicology_quality_axes
"""
import argparse
import csv
import json
import os
import re
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT / "_lib"))

from open_vocab_judge import axes_yes_no_score  # noqa: E402


REQUIRED_FILES = ["score_notes.csv", "alignment.csv", "deviations.json",
                  "decoy_rejections.json", "musicology_review.md"]
NOTE_FIELDS = {"note_id", "page_no", "system_no", "voice", "pitch_midi",
               "onset_beat", "dur_beat", "slur_id_or_null"}
ALIGN_FIELDS = {"note_id", "audio_onset_sec", "audio_offset_sec",
                "alignment_confidence"}
VALID_VOICES = {"melody", "bass", "middle", "1", "2", "3", "4"}
REVIEW_WORD_CAP = 500
NOTE_ID_RE = re.compile(r"\bN\d{3,5}\b")
RANGE_RE = re.compile(r"^N\d{3,5}(-N\d{3,5})?$")
ONSET_TOL_BEAT = 0.25
PAGE_TOL = 1
TIME_TOL_SEC = 2.0
ALIGN_HIT_THR = 0.15

REVIEW_AXES = [
    ("note_count_matches",
     "Does the review state the total note count and does that integer "
     "match GROUND_DATA.n_score_notes verbatim?"),
    ("nxxxx_in_context",
     "Does the review cite at least 3 specific Nxxxx note_ids in "
     "context (each used in a sentence about a phrase, deviation, or "
     "tempo observation, not just listed)?"),
    ("deviation_discussed",
     "Does the review discuss at least one specific performance "
     "deviation (timing / dynamics / ornament / wrong-note) using "
     "GROUND_DATA.deviation_vocab terminology?"),
    ("musicological_prose",
     "Does the review read as coherent musicological prose with at "
     "least one sentence on overall interpretation, tempo, or stylistic "
     "judgement?"),
    ("no_fabrication",
     "Is the review free of fabricated Nxxxx ids absent from "
     "score_notes.csv and free of contradictions with the supplied "
     "deviation counts/types?"),
]


def gating(out: Path):
    errs = []
    for f in REQUIRED_FILES:
        p = out / f
        if not p.exists():
            errs.append(f"missing: {f}")
        elif p.stat().st_size == 0:
            errs.append(f"empty: {f}")
    if errs:
        return False, errs
    with (out / "score_notes.csv").open(newline="") as fh:
        cols = set((csv.DictReader(fh).fieldnames or []))
        if not NOTE_FIELDS.issubset(cols):
            return False, [f"score_notes.csv missing fields: {sorted(NOTE_FIELDS - cols)}"]
    with (out / "alignment.csv").open(newline="") as fh:
        cols = set((csv.DictReader(fh).fieldnames or []))
        if not ALIGN_FIELDS.issubset(cols):
            return False, [f"alignment.csv missing fields: {sorted(ALIGN_FIELDS - cols)}"]
    try:
        json.loads((out / "deviations.json").read_text())
        json.loads((out / "decoy_rejections.json").read_text())
    except Exception as e:
        return False, [f"json parse: {e}"]
    return True, []


def _f(x):
    try:
        return float(x)
    except Exception:
        return None


def _i(x):
    try:
        return int(float(x))
    except Exception:
        return None


def parse_range(rs):
    rs = (rs or "").strip()
    if not RANGE_RE.match(rs):
        return None
    if "-" in rs:
        a, b = rs.split("-")
        ai, bi = _i(a[1:]), _i(b[1:])
        if ai is None or bi is None or bi < ai:
            return None
        return (ai, bi)
    n = _i(rs[1:])
    return None if n is None else (n, n)


def overlap_frac(a, b):
    inter = max(0, min(a[1], b[1]) - max(a[0], b[0]) + 1)
    union = max(a[1], b[1]) - min(a[0], b[0]) + 1
    return inter / union if union > 0 else 0.0


def pitch_recall_eval(pred_notes, gt_notes):
    by_pitch = {}
    for r in pred_notes:
        p = _i(r.get("pitch_midi")); ob = _f(r.get("onset_beat"))
        nid = (r.get("note_id") or "").strip()
        if p is None or ob is None or not nid:
            continue
        by_pitch.setdefault(p, []).append((ob, nid, r))
    hits, used = [], set()
    for g in gt_notes:
        cands = by_pitch.get(g["pitch_midi"], [])
        best = None; best_d = ONSET_TOL_BEAT + 1e-6
        for ob, nid, r in cands:
            if nid in used:
                continue
            d = abs(ob - g["onset_beat"])
            if d <= ONSET_TOL_BEAT and d < best_d:
                best_d = d; best = (nid, r, g)
        if best is not None:
            used.add(best[0]); hits.append(best)
    return len(hits) / max(1, len(gt_notes)), hits


def alignment_mae(hits, align_by_id, gt_by_id):
    diffs = []
    for nid, _, g in hits:
        a = align_by_id.get(nid); gt = gt_by_id.get(g["note_id"])
        if a is None or gt is None:
            continue
        ao = _f(a.get("audio_onset_sec")); go = gt.get("audio_onset_sec_gt")
        if ao is None or go is None:
            continue
        diffs.append(abs(ao - go))
    if not diffs:
        return None, 0
    return sum(diffs) / len(diffs), len(diffs)


def _norm(s):
    return str(s or "").strip().lower().replace("-", "_").replace(" ", "_")


def deviation_f1(pred_devs, gt_devs, dev_vocab):
    """Closed-vocab strict match on deviation_type + range overlap > 0.5."""
    norm_vocab = {_norm(v) for v in dev_vocab}
    gt = [(_norm(d["deviation_type"]), parse_range(d.get("note_id_range")))
          for d in gt_devs]
    gt = [(t, r) for t, r in gt if r is not None]
    pr = []
    for d in pred_devs:
        t = _norm(d.get("deviation_type"))
        if t not in norm_vocab:
            continue
        r = parse_range(d.get("note_id_range"))
        if r is None:
            continue
        pr.append((t, r))
    matched = set(); tp = 0
    for pt, prng in pr:
        for gi, (gt_t, gt_rng) in enumerate(gt):
            if gi in matched or pt != gt_t:
                continue
            if overlap_frac(prng, gt_rng) > 0.5:
                matched.add(gi); tp += 1; break
    fp = len(pr) - tp; fn = len(gt) - tp
    if tp == 0:
        return 0.0, tp, fp, fn
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    return f1, tp, fp, fn


def decoy_eval(pred_decoys, gt_decoys, dec_vocab):
    """Closed-vocab strict match on decoy_type + loc_type + page/time tolerance."""
    norm_vocab = {_norm(v) for v in dec_vocab}
    matched = set(); hits = 0
    for d in pred_decoys:
        t = _norm(d.get("decoy_type"))
        loc = (d.get("loc_type") or "").strip().lower()
        pt = d.get("page_or_t")
        if t not in norm_vocab or loc not in {"score", "audio"}:
            continue
        for gi, g in enumerate(gt_decoys):
            if gi in matched:
                continue
            if _norm(g["decoy_type"]) != t or g["loc_type"] != loc:
                continue
            if loc == "score":
                pp, gp = _i(pt), _i(g["page_or_t"])
                if pp is not None and gp is not None and abs(pp - gp) <= PAGE_TOL:
                    matched.add(gi); hits += 1; break
            else:
                pf, gf = _f(pt), _f(g["page_or_t"])
                if pf is not None and gf is not None and abs(pf - gf) <= TIME_TOL_SEC:
                    matched.add(gi); hits += 1; break
    return hits / max(1, len(gt_decoys)), hits, len(gt_decoys)


def cross_doc_check(pred_notes, align_rows, devs, decoys, review,
                    dev_vocab, dec_vocab):
    norm_dev = {_norm(v) for v in dev_vocab}
    norm_dec = {_norm(v) for v in dec_vocab}
    errs = []; note_ids = []; seen = set()
    for i, r in enumerate(pred_notes):
        nid = (r.get("note_id") or "").strip()
        if not nid:
            errs.append(f"note row {i}: empty note_id"); continue
        if nid in seen:
            errs.append(f"note row {i}: duplicate note_id {nid}")
        seen.add(nid); note_ids.append(nid)
        p = _i(r.get("pitch_midi"))
        if p is None or p < 21 or p > 108:
            errs.append(f"note row {i}: pitch_midi {r.get('pitch_midi')} not in [21,108]")
        v = (r.get("voice") or "").strip()
        if v not in VALID_VOICES:
            errs.append(f"note row {i}: voice '{v}' not in vocab")
        if _f(r.get("onset_beat")) is None:
            errs.append(f"note row {i}: onset_beat invalid")
    note_id_set = set(note_ids)
    if len(note_ids) < 200:
        errs.append(f"score_notes.csv has {len(note_ids)} rows (need >= 200)")
    align_ids = set()
    for i, a in enumerate(align_rows):
        nid = (a.get("note_id") or "").strip()
        if not nid:
            errs.append(f"align row {i}: empty note_id"); continue
        if nid in align_ids:
            errs.append(f"align row {i}: duplicate note_id {nid}")
        align_ids.add(nid)
        ao, af, ac = _f(a.get("audio_onset_sec")), _f(a.get("audio_offset_sec")), _f(a.get("alignment_confidence"))
        if ao is None or af is None or af < ao:
            errs.append(f"align row {i}: bad onset/offset {ao}/{af}")
        if ac is None or not 0.0 <= ac <= 1.0:
            errs.append(f"align row {i}: alignment_confidence {ac} not in [0,1]")
    if note_id_set != align_ids:
        miss = list(note_id_set - align_ids)[:3]; extra = list(align_ids - note_id_set)[:3]
        errs.append(f"alignment.csv note_id != score_notes.csv (missing {len(note_id_set - align_ids)} e.g. {miss}; extra {len(align_ids - note_id_set)} e.g. {extra})")
    if not isinstance(devs, list) or len(devs) < 4:
        errs.append(f"deviations.json needs >= 4 entries (got {len(devs) if isinstance(devs, list) else 'non-list'})")
    for i, d in enumerate(devs if isinstance(devs, list) else []):
        if _norm(d.get("deviation_type")) not in norm_dev:
            errs.append(f"deviation {i}: type '{d.get('deviation_type')}' not in vocab")
        if parse_range(d.get("note_id_range")) is None:
            errs.append(f"deviation {i}: note_id_range '{d.get('note_id_range')}' invalid")
    if not isinstance(decoys, list) or len(decoys) < 4:
        errs.append(f"decoy_rejections.json needs >= 4 entries (got {len(decoys) if isinstance(decoys, list) else 'non-list'})")
    for i, d in enumerate(decoys if isinstance(decoys, list) else []):
        if _norm(d.get("decoy_type")) not in norm_dec:
            errs.append(f"decoy {i}: type '{d.get('decoy_type')}' not in vocab")
        if d.get("loc_type") not in {"score", "audio"}:
            errs.append(f"decoy {i}: loc_type '{d.get('loc_type')}' invalid")
    n_words = len(re.findall(r"\S+", review))
    if n_words > REVIEW_WORD_CAP:
        errs.append(f"review word count {n_words} > {REVIEW_WORD_CAP}")
    if n_words < 50:
        errs.append(f"review too short: {n_words} words")
    cited = NOTE_ID_RE.findall(review)
    cited_in_csv = [c for c in cited if c in note_id_set]
    if len(set(cited_in_csv)) < 3:
        errs.append(f"review cites {len(set(cited_in_csv))} valid Nxxxx ids (need >= 3 in score_notes.csv); raw cites={cited[:5]}")
    nums = re.findall(r"\b\d+\b", review)
    n_csv = len(note_ids)
    if str(n_csv) not in nums:
        errs.append(f"review must mention total note count {n_csv} verbatim")
    return errs


def evaluate(out: Path, gt: dict):
    with (out / "score_notes.csv").open(newline="") as fh:
        pred_notes = list(csv.DictReader(fh))
    with (out / "alignment.csv").open(newline="") as fh:
        align_rows = list(csv.DictReader(fh))
    devs = json.loads((out / "deviations.json").read_text())
    decoys = json.loads((out / "decoy_rejections.json").read_text())
    review = (out / "musicology_review.md").read_text()
    dev_vocab = set(gt.get("deviation_vocab", []))
    dec_vocab = set(gt.get("decoy_vocab", []))
    cd_errs = cross_doc_check(pred_notes, align_rows, devs, decoys, review,
                               dev_vocab, dec_vocab)
    gt_aligned = [n for n in gt["notes"] if n.get("audio_onset_sec_gt") is not None]
    pr_ratio, hits = pitch_recall_eval(pred_notes, gt_aligned)
    align_by_id = {(a.get("note_id") or "").strip(): a for a in align_rows}
    gt_by_id = {n["note_id"]: n for n in gt["notes"]}
    mae, n_mae = alignment_mae(hits, align_by_id, gt_by_id)
    f1, tp, fp, fn = deviation_f1(devs if isinstance(devs, list) else [],
                                    gt["deviations"], dev_vocab)
    dr_ratio, dr_hit, dr_total = decoy_eval(decoys if isinstance(decoys, list) else [],
                                             gt["decoys"], dec_vocab)
    return {
        "pitch_recall": {"ratio": round(pr_ratio, 3), "hits": len(hits),
                         "n_gt": len(gt_aligned), "ok": pr_ratio >= 0.75},
        "alignment_mae_sec": {"mae": None if mae is None else round(mae, 4),
                              "n_eval": n_mae,
                              "ok": mae is not None and mae <= ALIGN_HIT_THR},
        "deviation_detection_f1": {"f1": round(f1, 3), "tp": tp, "fp": fp, "fn": fn,
                                   "ok": f1 >= 0.50},
        "decoy_rejection": {"ratio": round(dr_ratio, 3), "hits": dr_hit,
                            "n_gt": dr_total, "ok": dr_ratio >= 0.75},
        "cross_doc_consistency": {"errors": cd_errs[:10], "n_errors": len(cd_errs),
                                  "ok": len(cd_errs) == 0},
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", required=True)
    ap.add_argument("--gt", required=True)
    ap.add_argument("--fixtures", default="fixtures")
    args = ap.parse_args()
    out = Path(args.output)
    ok, errs = gating(out)
    if not ok:
        print(json.dumps({"pass": False, "stage": "gating", "errors": errs}, indent=2))
        return
    gt = json.load(open(args.gt))
    res = evaluate(out, gt)

    # Build ground for review (free-form prose dim).
    with (out / "score_notes.csv").open(newline="") as fh:
        n_notes_pred = sum(1 for _ in csv.DictReader(fh))
    try:
        devs_for_summary = json.loads((out / "deviations.json").read_text())
        if not isinstance(devs_for_summary, list):
            devs_for_summary = []
    except Exception:
        devs_for_summary = []
    ground_for_review = {
        "n_score_notes": n_notes_pred,
        "n_deviations_pred": len(devs_for_summary),
        "n_deviations_gt": len(gt.get("deviations", [])),
        "deviation_vocab": list(gt.get("deviation_vocab", [])),
        "pitch_recall_hits": res["pitch_recall"].get("hits"),
        "alignment_mae_sec": res["alignment_mae_sec"].get("mae"),
        "deviation_f1": res["deviation_detection_f1"].get("f1"),
    }
    review_text = (out / "musicology_review.md").read_text()
    payload = (
        f"GROUND_DATA:\n{json.dumps(ground_for_review, ensure_ascii=False)}\n\n"
        f"MUSICOLOGY_REVIEW.MD:\n{review_text[:6000]}\n"
    )
    sq = axes_yes_no_score(payload, REVIEW_AXES, threshold=0.70)
    res["musicology_quality_axes"] = {
        "axes": sq.get("axes", []),
        "score": sq.get("score", 0.0),
        "ok": bool(sq.get("ok", False)),
        "threshold": 0.70,
    }

    all_ok = all(v["ok"] for v in res.values())
    weights = {
        "pitch_recall": 0.18,
        "alignment_mae_sec": 0.16,
        "deviation_detection_f1": 0.14,
        "decoy_rejection": 0.14,
        "cross_doc_consistency": 0.23,
        "musicology_quality_axes": 0.15,
    }

    def dim_score(k):
        v = res[k]
        if k == "cross_doc_consistency":
            return 1.0 if v["ok"] else 0.0
        if k == "musicology_quality_axes":
            return float(v.get("score", 0.0))
        if k == "alignment_mae_sec":
            mae = v.get("mae")
            if mae is None:
                return 0.0
            return max(0.0, min(1.0, 1.0 - mae / ALIGN_HIT_THR * 0.5))
        return max(0.0, v.get("ratio", v.get("f1", 0.0)))

    score = round(sum(weights[k] * dim_score(k) for k in weights), 3)
    print(json.dumps({
        "pass": all_ok,
        "checks": {k: v["ok"] for k, v in res.items()},
        **res,
        "score": score,
        "weights": weights,
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
