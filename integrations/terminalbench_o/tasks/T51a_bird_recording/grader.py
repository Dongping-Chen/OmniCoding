"""Grader for T51a Bird Field Recording Species Timeline (v3).

v3 pattern:
  - Closed-vocab `sci_name` is CLOSED_WITH_DECOYS (candidate + decoy
    sci-name lists in species_vocab.json). decoy_rejection /
    species_precision / cross-doc-gating already deterministically
    enforce GT match.
  - summary.md is genuinely free-form prose; replaced single
    `judge_memo_grounded` continuous-score with `axes_yes_no_score`
    over 5 yes/no axes.
  - Reweighted: deterministic dims ~0.85, axes-LLM 0.15.

6 dimensions (weights sum 1.0):
  species_recall          >= 0.70  (weight 0.20)
  species_precision       >= 0.75  (weight 0.20)
  time_iou                >= 0.40  (weight 0.15)
  decoy_rejection         <= 0.10  (weight 0.10)
  cooccurrence_f1         >= 0.55  (weight 0.20)  cross-doc gating bundled in
  summary_quality_axes    >= 0.60  (weight 0.15)  axes yes/no on summary.md

Gating fails on missing files OR unparseable timeline.json. Other dims
that throw report ok=false (NEVER silently passed).
"""
import argparse, csv, json, os, re, sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT / "_lib"))

from open_vocab_judge import axes_yes_no_score  # noqa: E402

REQUIRED_FILES = ["timeline.json", "cooccurrence.csv", "summary.md"]
COOCC_FIELDS = {"species_a", "species_b", "n_overlaps", "total_overlap_sec"}
SUMMARY_WORD_CAP = 320
MIN_SEGMENT_SEC = 0.5
MERGE_GAP_SEC = 1.0
IOU_EDGE_TOL = 2.0


def gating(out: Path):
    errs = []
    for f in REQUIRED_FILES:
        p = out / f
        if not p.exists(): errs.append(f"missing: {f}")
        elif p.stat().st_size == 0: errs.append(f"empty: {f}")
    if errs: return False, errs
    try:
        json.loads((out / "timeline.json").read_text())
    except Exception as e:
        return False, [f"timeline.json not valid JSON: {e}"]
    with (out / "cooccurrence.csv").open(newline="") as fh:
        cols = set((csv.DictReader(fh).fieldnames or []))
        if not COOCC_FIELDS.issubset(cols):
            return False, [f"cooccurrence.csv missing fields: {sorted(COOCC_FIELDS - cols)}"]
    return True, []


def load_timeline(out: Path):
    raw = json.loads((out / "timeline.json").read_text())
    if not isinstance(raw, dict):
        raise ValueError("timeline.json must be a dict keyed by sci_name")
    res = {}
    for sci, segs in raw.items():
        if not isinstance(segs, list):
            raise ValueError(f"timeline['{sci}'] must be a list")
        out_segs = []
        for seg in segs:
            if not isinstance(seg, dict):
                raise ValueError(f"timeline['{sci}'] item not a dict")
            try:
                s, e = float(seg["start"]), float(seg["end"])
            except (KeyError, TypeError, ValueError):
                raise ValueError(f"timeline['{sci}'] segment missing/invalid start/end")
            out_segs.append((s, e))
        res[sci] = out_segs
    return res


def merge_segs(segs):
    if not segs: return []
    s = sorted(segs)
    out = [list(s[0])]
    for a, b in s[1:]:
        if a <= out[-1][1]: out[-1][1] = max(out[-1][1], b)
        else: out.append([a, b])
    return [(a, b) for a, b in out]


def total_dur(segs):
    return sum(e - s for s, e in segs)


def iou_with_tol(pred_segs, gt_segs, tol):
    pred_m = merge_segs(pred_segs)
    gt_m = merge_segs(gt_segs)
    if not pred_m or not gt_m:
        union = total_dur(merge_segs(pred_m + gt_m))
        return 0.0 if union > 0 else 0.0
    expanded = merge_segs([(s - tol, e + tol) for s, e in gt_m])
    inter = []
    for ps, pe in pred_m:
        for gs, ge in expanded:
            s, e = max(ps, gs), min(pe, ge)
            if e > s: inter.append((s, e))
            if gs > pe: break
    inter_sec = total_dur(merge_segs(inter))
    union = total_dur(merge_segs(pred_m + gt_m))
    return min(1.0, inter_sec / union) if union > 0 else 0.0


def cross_doc_check(pred, cooc_rows, summary_text, gt):
    errs = []
    cand = set(gt["candidate_sci_names"])
    full_vocab = cand | set(gt["decoy_sci_names"])
    duration = float(gt.get("duration_sec", 600.0))

    bad = [k for k in pred if k not in full_vocab]
    if bad:
        errs.append(f"timeline has out-of-vocab sci_names: {sorted(bad)[:5]}")

    for sci, segs in pred.items():
        if not segs:
            errs.append(f"timeline['{sci}'] is empty list (omit instead)")
            continue
        for s, e in segs:
            if not (0.0 <= s < e <= duration + 0.001):
                errs.append(f"{sci}: bad segment [{s}, {e}]")
            elif (e - s) < MIN_SEGMENT_SEC:
                errs.append(f"{sci}: segment shorter than {MIN_SEGMENT_SEC}s: [{s}, {e}]")
        sorted_segs = sorted(segs)
        for (s1, e1), (s2, e2) in zip(sorted_segs, sorted_segs[1:]):
            if 0 <= s2 - e1 < MERGE_GAP_SEC:
                errs.append(f"{sci}: unmerged segments around {e1:.2f}s (gap {s2-e1:.2f}s < {MERGE_GAP_SEC}s)")
                break

    for r in cooc_rows:
        a = (r.get("species_a") or "").strip()
        b = (r.get("species_b") or "").strip()
        if a not in cand: errs.append(f"cooccurrence species_a='{a}' not in candidate vocab")
        if b not in cand: errs.append(f"cooccurrence species_b='{b}' not in candidate vocab")
        if a >= b: errs.append(f"cooccurrence pair ({a}, {b}) not sorted (a < b)")
        try:
            n = int(r.get("n_overlaps", "0"))
            if n < 1: errs.append(f"cooccurrence ({a},{b}) n_overlaps={n} < 1")
        except (ValueError, TypeError):
            errs.append(f"cooccurrence ({a},{b}) n_overlaps not int")
        try:
            t = float(r.get("total_overlap_sec", "0"))
            if t < 0: errs.append(f"cooccurrence ({a},{b}) total_overlap_sec={t} < 0")
        except (ValueError, TypeError):
            errs.append(f"cooccurrence ({a},{b}) total_overlap_sec not float")

    n_words = len(re.findall(r"\S+", summary_text))
    if n_words > SUMMARY_WORD_CAP:
        errs.append(f"summary.md word count {n_words} > {SUMMARY_WORD_CAP}")
    return errs


def evaluate(out: Path, gt: dict):
    pred = load_timeline(out)
    with (out / "cooccurrence.csv").open(newline="") as fh:
        cooc_rows = list(csv.DictReader(fh))
    summary_text = (out / "summary.md").read_text()

    cand = set(gt["candidate_sci_names"])
    decoy = set(gt["decoy_sci_names"])
    truly = set(gt["truly_present_sci"])
    gt_timelines = {k: [(s["start"], s["end"]) for s in v]
                    for k, v in gt["species_timelines"].items()}

    cd_errs = cross_doc_check(pred, cooc_rows, summary_text, gt)

    pred_cand = {sci for sci, segs in pred.items() if sci in cand and segs}
    tp = pred_cand & truly
    rec = len(tp) / max(1, len(truly))
    prec = (len(tp) / max(1, len(pred_cand))) if pred_cand else 0.0

    ious = []
    for sp in sorted(truly):
        clean = [(s, e) for s, e in pred.get(sp, [])
                 if e > s and (e - s) >= MIN_SEGMENT_SEC]
        ious.append(iou_with_tol(clean, gt_timelines[sp], IOU_EDGE_TOL))
    iou_macro = (sum(ious) / len(ious)) if ious else 0.0

    pred_decoys = {sci for sci, segs in pred.items() if sci in decoy and segs}
    decoy_err = len(pred_decoys) / max(1, len(decoy))

    gt_pairs = {tuple(sorted([p["species_a"], p["species_b"]]))
                for p in gt["cooccurrence_pairs"]}
    pred_pairs = set()
    for r in cooc_rows:
        a = (r.get("species_a") or "").strip()
        b = (r.get("species_b") or "").strip()
        if a in cand and b in cand and a != b:
            pred_pairs.add(tuple(sorted([a, b])))
    if not pred_pairs and not gt_pairs:
        f1 = 1.0
    elif not pred_pairs or not gt_pairs:
        f1 = 0.0
    else:
        ptp = len(pred_pairs & gt_pairs)
        pp = ptp / len(pred_pairs)
        rr = ptp / len(gt_pairs)
        f1 = (2 * pp * rr / (pp + rr)) if (pp + rr) > 0 else 0.0

    return {
        "species_recall": {"ratio": round(rec, 3), "tp": len(tp),
                           "fp": len(pred_cand - truly), "fn": len(truly - pred_cand),
                           "ok": rec >= 0.70},
        "species_precision": {"ratio": round(prec, 3), "tp": len(tp),
                              "n_pred": len(pred_cand), "ok": prec >= 0.75},
        "time_iou": {"ratio": round(iou_macro, 3), "n_species": len(ious),
                     "ok": iou_macro >= 0.40},
        "decoy_rejection": {"ratio": round(max(0.0, 1.0 - decoy_err), 3),
                            "err_frac": round(decoy_err, 3),
                            "n_decoys_emitted": len(pred_decoys),
                            "ok": decoy_err <= 0.10},
        "cooccurrence_f1": {"ratio": round(f1, 3),
                            "n_gt_pairs": len(gt_pairs),
                            "n_pred_pairs": len(pred_pairs),
                            "cross_doc_errors": cd_errs[:8],
                            "n_cross_doc_errors": len(cd_errs),
                            "ok": f1 >= 0.55 and len(cd_errs) == 0},
    }


SUMMARY_AXES = [
    ("detected_count_present",
     "Does SUMMARY.MD report a detected-species count in `n_target / "
     "n_candidate` style consistent with GROUND_DATA's "
     "n_pred_candidate_species (within +/-1)?"),
    ("top_species_listed",
     "Does SUMMARY.MD list the top 3 species by total vocalisation "
     "duration, using sci_name (binomial Latin) form — not 'Bird A'?"),
    ("top_pairs_listed",
     "Does SUMMARY.MD list the top 2 co-occurring species pairs by "
     "total_overlap_sec?"),
    ("ecological_remark",
     "Does SUMMARY.MD include at least one ecological remark on dawn "
     "chorus, habitat, seasonality, or comparable field-guide context?"),
    ("no_fabrication",
     "Is SUMMARY.MD free of fabricated species (no decoy sci_name "
     "presented as detected; no species absent from agent's timeline)?"),
]


def summary_quality(out: Path, gt: dict, eval_res: dict) -> dict:
    prec = eval_res.get("species_precision", {})
    cooc = eval_res.get("cooccurrence_f1", {})
    ground = {
        "n_candidate_species": len(gt.get("candidate_sci_names", [])),
        "n_truly_present_species": len(gt.get("truly_present_sci", [])),
        "n_pred_candidate_species": prec.get("n_pred"),
        "n_pred_pairs": cooc.get("n_pred_pairs"),
        "n_gt_pairs": cooc.get("n_gt_pairs"),
        "decoy_sci_names": gt.get("decoy_sci_names", []),
    }
    summary = (out / "summary.md").read_text()
    payload = (
        f"GROUND_DATA:\n{json.dumps(ground, ensure_ascii=False)}\n\n"
        f"SUMMARY.MD:\n{summary[:6000]}\n"
    )
    return axes_yes_no_score(payload, SUMMARY_AXES, threshold=0.60)


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

    # v3: replace single-LLM continuous judge with axes-yes/no.
    sq = summary_quality(out, gt, res)
    res["summary_quality_axes"] = {
        "axes": sq.get("axes", []),
        "score": sq.get("score", 0.0),
        "ok": bool(sq.get("ok", False)),
        "threshold": 0.60,
    }

    all_ok = all(v["ok"] for v in res.values())
    # v3 weights (sum = 1.000): deterministic dims dominate.
    weights = {"species_recall": 0.20, "species_precision": 0.20,
               "time_iou": 0.15, "decoy_rejection": 0.10,
               "cooccurrence_f1": 0.20, "summary_quality_axes": 0.15}
    score = 0.0
    for k, w in weights.items():
        if k == "cooccurrence_f1":
            score += w * (res[k]["ratio"]
                          if len(res[k]["cross_doc_errors"]) == 0 else 0.0)
        elif k == "summary_quality_axes":
            score += w * res[k].get("score", 0.0)
        else:
            score += w * res[k].get("ratio", 0.0)
    score = round(score, 3)
    print(json.dumps({"pass": all_ok, "checks": {k: v["ok"] for k, v in res.items()},
                      **res, "score": score, "weights": weights},
                     indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
