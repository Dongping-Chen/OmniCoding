"""Grader for T83a Egocentric Cooking Procedure Verification (v3 — deterministic + summary axes).

v3 redesign:
  - Closed-vocab fields (verb, noun, hazard_type, decoy_type,
    compliance_status) remain strict normalised match — no LLM, no VLM.
  - supervisor_note.md is genuinely free-form prose; replaced single
    VLM judge with 5-axis yes/no `supervisor_quality_axes`.
  - Reweighted: deterministic dims dominate (~0.85); only the prose
    note uses LLM axes (0.15).
  - hazard_categories.json + decoy_categories.json now include 4-5
    decoy strings each (real plausible categories not in this clip).
    Active subset is checked via `active_categories` field.

Capability dims  (~0.62): action_recall + step_compliance_f1 +
                          hazard_detection + decoy_rejection
Format / sanity  (~0.23): cross_doc_consistency
Quality dims     (~0.15): supervisor_quality_axes
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


REQUIRED = ["action_timeline.csv", "compliance_report.csv",
            "hazards.json", "decoy_rejections.json", "supervisor_note.md"]
ACTION_FIELDS = ["event_id", "t_start", "t_end", "verb", "noun", "confidence"]
COMPLIANCE_FIELDS = ["recipe_step_idx", "expected_verb", "expected_noun",
                     "actual_t_start", "actual_t_end", "compliance_status"]
COMPLIANCE_CLASSES = ["on_time", "late", "out_of_order", "skipped", "unexpected"]


SUPERVISOR_AXES = [
    ("compliance_rate_present",
     "Does supervisor_note.md state the on-time compliance rate within "
     "±5% of GROUND_DATA.on_time_compliance_pct (the integer / percentage "
     "must appear verbatim in the prose)?"),
    ("verb_noun_citations",
     "Does supervisor_note.md cite at least 3 specific (verb, noun) "
     "action pairs that appear in GROUND_DATA.sample_action_pairs, "
     "written in `(verb, noun)` form?"),
    ("hazard_or_compliance_issue",
     "Does supervisor_note.md reference at least one hazard or "
     "compliance issue (skipped / out_of_order / late) consistent with "
     "GROUND_DATA?"),
    ("supervisor_prose_with_guidance",
     "Does supervisor_note.md read as coherent kitchen-supervisor "
     "English with at least one sentence on training / improvement "
     "guidance grounded in the observed actions?"),
    ("no_fabrication",
     "Is supervisor_note.md free of fabricated (verb, noun) pairs not "
     "in the supplied vocab and free of hazard / compliance counts that "
     "contradict GROUND_DATA?"),
]


def _norm(s):
    return str(s or "").strip().lower().replace("-", "_").replace(" ", "_")


def _load_categories(path: Path):
    """Load a category JSON. If `active_categories` is present, return
    that subset (the rest are decoy strings); else fall back to
    `categories`."""
    raw = json.load(open(path))
    if isinstance(raw, dict):
        if "active_categories" in raw and isinstance(raw["active_categories"], list):
            return list(raw["active_categories"])
        return list(raw.get("categories") or [])
    return []


def gating(out: Path):
    errs = []
    for f in REQUIRED:
        p = out / f
        if not p.exists():
            errs.append(f"missing: {f}")
        elif p.stat().st_size == 0:
            errs.append(f"empty: {f}")
    if errs:
        return False, errs
    with open(out / "action_timeline.csv") as f:
        r = csv.DictReader(f)
        if not r.fieldnames or set(r.fieldnames) < set(ACTION_FIELDS):
            return False, [f"action_timeline.csv missing fields, got {r.fieldnames}"]
        rows = list(r)
    if len(rows) < 80:
        errs.append(f"action_timeline.csv has {len(rows)} rows (<80)")
    with open(out / "compliance_report.csv") as f:
        r = csv.DictReader(f)
        if not r.fieldnames or set(r.fieldnames) < set(COMPLIANCE_FIELDS):
            errs.append(f"compliance_report.csv missing fields, got {r.fieldnames}")
    try:
        h = json.load(open(out / "hazards.json"))
        if not isinstance(h, list) or len(h) < 3:
            errs.append("hazards.json must be a list of >=3 entries")
    except Exception as e:
        errs.append(f"hazards.json parse: {e}")
    try:
        d = json.load(open(out / "decoy_rejections.json"))
        if not isinstance(d, list) or len(d) < 3:
            errs.append("decoy_rejections.json must be a list of >=3 entries")
    except Exception as e:
        errs.append(f"decoy_rejections.json parse: {e}")
    return len(errs) == 0, errs


def load_pred(out: Path):
    actions = []
    with open(out / "action_timeline.csv") as f:
        for r in csv.DictReader(f):
            try:
                actions.append({
                    "event_id": (r.get("event_id") or "").strip(),
                    "t_start": float(r["t_start"]),
                    "t_end":   float(r["t_end"]),
                    "verb":    (r["verb"] or "").strip(),
                    "noun":    (r["noun"] or "").strip(),
                    "confidence": float(r["confidence"]) if r.get("confidence") else 0.0,
                })
            except (ValueError, KeyError):
                pass
    compliance = []
    with open(out / "compliance_report.csv") as f:
        for r in csv.DictReader(f):
            try:
                ts = r.get("actual_t_start") or ""
                te = r.get("actual_t_end") or ""
                compliance.append({
                    "step_idx": int(r["recipe_step_idx"]),
                    "verb": (r["expected_verb"] or "").strip(),
                    "noun": (r["expected_noun"] or "").strip(),
                    "t_start": float(ts) if ts.strip() else None,
                    "t_end":   float(te) if te.strip() else None,
                    "status":  (r["compliance_status"] or "").strip(),
                })
            except (ValueError, KeyError):
                pass
    hazards = json.load(open(out / "hazards.json"))
    decoys = json.load(open(out / "decoy_rejections.json"))
    return actions, compliance, hazards, decoys


def iou(a0, a1, b0, b1):
    inter = max(0.0, min(a1, b1) - max(a0, b0))
    union = max(a1, b1) - min(a0, b0)
    return inter / union if union > 0 else 0.0


def hungarian(preds, gts, key_eq, iou_thr=0.3):
    """Match preds to gts requiring key_eq(p,g) AND IoU>iou_thr."""
    try:
        import numpy as np
        from scipy.optimize import linear_sum_assignment
    except ImportError:
        return None
    if not preds or not gts:
        return []
    cost = np.full((len(gts), len(preds)), 1.0, dtype=float)
    for i, g in enumerate(gts):
        for j, p in enumerate(preds):
            if not key_eq(p, g):
                continue
            ov = iou(g["t_start"], g["t_end"], p["t_start"], p["t_end"])
            if ov > iou_thr:
                cost[i, j] = -ov
    rs, cs = linear_sum_assignment(cost)
    return [(int(r), int(c), float(-cost[r, c])) for r, c in zip(rs, cs) if cost[r, c] < 0]


def macro_f1(pred_labels, gt_labels, classes):
    f1s = []
    for cls in classes:
        tp = sum(1 for p, g in zip(pred_labels, gt_labels) if p == cls and g == cls)
        fp = sum(1 for p, g in zip(pred_labels, gt_labels) if p == cls and g != cls)
        fn = sum(1 for p, g in zip(pred_labels, gt_labels) if p != cls and g == cls)
        if tp == 0 and fp == 0 and fn == 0:
            continue
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1s.append(2 * prec * rec / (prec + rec) if prec + rec else 0.0)
    return sum(f1s) / len(f1s) if f1s else 0.0


def evaluate(out: Path, gt: dict, fix: Path) -> dict:
    pred_acts, pred_comp, pred_haz, pred_dec = load_pred(out)

    # ---- action_recall (Hungarian on IoU>0.3 + verb/noun strict) -----------
    gt_acts = [{"t_start": a["t_start"], "t_end": a["t_end"],
                "verb": a["verb"], "noun": a["noun"]} for a in gt["actions"]]
    n_gt_acts = len(gt_acts)
    hits = hungarian(pred_acts, gt_acts,
                     key_eq=lambda p, g: _norm(p["verb"]) == _norm(g["verb"])
                                          and _norm(p["noun"]) == _norm(g["noun"]))
    if hits is None:
        action_recall = {"ratio": 0.0, "ok": False, "error": "scipy/numpy missing"}
    else:
        ratio = len(hits) / max(1, n_gt_acts)
        action_recall = {"hits": len(hits), "n_gt": n_gt_acts,
                         "ratio": round(ratio, 3), "ok": ratio >= 0.55}

    # ---- step_compliance_f1 (macro F1 over 5 classes) ----------------------
    gt_comp = {c["recipe_step_idx"]: c for c in gt["recipe_compliance_gt"]}
    pred_by_idx = {c["step_idx"]: c for c in pred_comp}
    g_lbls, p_lbls = [], []
    for s_idx in sorted(gt_comp):
        g_lbls.append(_norm(gt_comp[s_idx]["compliance_status"]))
        p_lbls.append(_norm((pred_by_idx.get(s_idx) or {"status": "MISSING"})["status"]))
    f1 = macro_f1(p_lbls, g_lbls, COMPLIANCE_CLASSES)
    step_compliance_f1 = {"macro_f1": round(f1, 3), "ratio": round(f1, 3),
                          "n_steps": len(gt_comp), "ok": f1 >= 0.65}

    # ---- hazard_detection --------------------------------------------------
    gt_haz = gt["hazards"]
    haz_hits = hungarian(pred_haz, gt_haz,
                         key_eq=lambda p, g: _norm(p.get("hazard_type") or "")
                                              == _norm(g["hazard_type"]))
    if haz_hits is None:
        hazard_detection = {"ratio": 0.0, "ok": False, "error": "scipy missing"}
    else:
        h_ratio = len(haz_hits) / max(1, len(gt_haz))
        hazard_detection = {"hits": len(haz_hits), "n_gt": len(gt_haz),
                            "ratio": round(h_ratio, 3), "ok": h_ratio >= 0.50}

    # ---- decoy_rejection ---------------------------------------------------
    gt_dec = gt["decoys"]
    dec_hits = hungarian(pred_dec, gt_dec,
                         key_eq=lambda p, g: _norm(p.get("decoy_type") or "")
                                              == _norm(g["decoy_type"]))
    if dec_hits is None:
        decoy_rejection = {"ratio": 0.0, "ok": False, "error": "scipy missing"}
    else:
        d_ratio = len(dec_hits) / max(1, len(gt_dec))
        decoy_rejection = {"hits": len(dec_hits), "n_gt": len(gt_dec),
                           "ratio": round(d_ratio, 3), "ok": d_ratio >= 0.75}

    # ---- cross_doc_consistency ---------------------------------------------
    cd_errs = []
    verb_vocab = {_norm(v) for v in json.load(open(fix / "verb_vocab.json"))["verbs"]}
    noun_vocab = {_norm(v) for v in json.load(open(fix / "noun_vocab.json"))["nouns"]}
    for a in pred_acts:
        if _norm(a["verb"]) not in verb_vocab:
            cd_errs.append(f"verb '{a['verb']}' OOV"); break
        if _norm(a["noun"]) not in noun_vocab:
            cd_errs.append(f"noun '{a['noun']}' OOV"); break
    if not cd_errs:
        for i in range(1, len(pred_acts)):
            if pred_acts[i]["t_start"] < pred_acts[i-1]["t_start"]:
                cd_errs.append(f"non-monotonic at row {i+1}: "
                               f"{pred_acts[i-1]['t_start']} -> {pred_acts[i]['t_start']}")
                break
    for a in pred_acts:
        if a["t_end"] <= a["t_start"]:
            cd_errs.append(f"non-positive duration: {a['t_start']}-{a['t_end']}")
            break
        if not (0.10 <= a["t_end"] - a["t_start"] <= 30.0):
            cd_errs.append(f"duration out of [0.10,30] at {a['t_start']}: "
                           f"{a['t_end']-a['t_start']:.2f}s")
            break
    n_recipe = gt["n_recipe_steps_gt"]
    if len(pred_comp) != n_recipe:
        cd_errs.append(f"compliance_report.csv has {len(pred_comp)} rows, expected {n_recipe}")
    bad_status = [c for c in pred_comp if _norm(c["status"]) not in COMPLIANCE_CLASSES]
    if bad_status:
        cd_errs.append(f"compliance_status enum violation: {bad_status[0]['status']!r}")
    # Active enums (exclude decoy strings) — agents must use only the active subset.
    haz_enum = {_norm(v) for v in _load_categories(fix / "hazard_categories.json")}
    for h in pred_haz:
        if _norm(h.get("hazard_type") or "") not in haz_enum:
            cd_errs.append(f"hazard_type '{h.get('hazard_type')}' not in active enum"); break
        if not (h.get("t_end", 0) > h.get("t_start", 0)):
            cd_errs.append(f"bad hazard interval: {h}"); break
    dec_enum = {_norm(v) for v in _load_categories(fix / "decoy_categories.json")}
    for d in pred_dec:
        if _norm(d.get("decoy_type") or "") not in dec_enum:
            cd_errs.append(f"decoy_type '{d.get('decoy_type')}' not in active enum"); break
        if not (d.get("t_end", 0) > d.get("t_start", 0)):
            cd_errs.append(f"bad decoy interval: {d}"); break
    note = (out / "supervisor_note.md").read_text(encoding="utf-8", errors="ignore")
    word_count = len(re.findall(r"\S+", note))
    if word_count > 400:
        cd_errs.append(f"supervisor_note too long: {word_count} words (>400)")
    pred_pairs = {(_norm(a["verb"]), _norm(a["noun"])) for a in pred_acts}
    note_low = note.lower()
    cite_hits = 0
    for v, n in pred_pairs:
        if v and n and re.search(rf"\(\s*{re.escape(v)}\s*,\s*{re.escape(n)}\s*\)",
                                 note_low):
            cite_hits += 1
    if cite_hits < 3:
        cd_errs.append(f"supervisor_note has only {cite_hits} (verb, noun) citations from "
                       f"action_timeline.csv (need ≥ 3)")
    on_time_pred = sum(1 for c in pred_comp if _norm(c["status"]) == "on_time")
    rate_pct = on_time_pred / max(1, len(pred_comp)) * 100.0
    nums = [float(x) for x in re.findall(r"\b\d+(?:\.\d+)?\b", note)]
    rate_match = any(abs(num - rate_pct) <= 5.0 for num in nums) or \
                 any(abs(num - rate_pct/100.0) <= 0.05 for num in nums)
    if not rate_match:
        cd_errs.append(f"supervisor_note lacks compliance-rate number "
                       f"matching {rate_pct:.1f}% (or {rate_pct/100:.2f}) ±5%")
    cross_doc_consistency = {"errors": cd_errs[:8],
                             "n_errors": len(cd_errs),
                             "ok": len(cd_errs) == 0}

    return {
        "action_recall": action_recall,
        "step_compliance_f1": step_compliance_f1,
        "hazard_detection": hazard_detection,
        "decoy_rejection": decoy_rejection,
        "cross_doc_consistency": cross_doc_consistency,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", required=True)
    ap.add_argument("--gt", required=True)
    ap.add_argument("--fixtures", default="fixtures")
    args = ap.parse_args()
    out, fix = Path(args.output), Path(args.fixtures)
    ok, errs = gating(out)
    if not ok:
        print(json.dumps({"pass": False, "stage": "gating", "errors": errs},
                         indent=2)); return
    gt = json.load(open(args.gt))
    res = evaluate(out, gt, fix)

    pred_acts_summary, pred_comp_summary, pred_haz_summary, _ = load_pred(out)
    on_time_pred = sum(1 for c in pred_comp_summary if _norm(c["status"]) == "on_time")
    n_comp = max(1, len(pred_comp_summary))
    rate_pct = round(on_time_pred / n_comp * 100.0, 1)
    sample_pairs = list({(a["verb"], a["noun"]) for a in pred_acts_summary})[:8]
    ground_for_note = {
        "on_time_compliance_pct": rate_pct,
        "n_compliance_steps": len(pred_comp_summary),
        "n_actions": len(pred_acts_summary),
        "n_hazards_pred": len(pred_haz_summary) if isinstance(pred_haz_summary, list) else 0,
        "sample_action_pairs": sample_pairs,
        "action_recall_ratio": res["action_recall"].get("ratio"),
        "step_compliance_f1": res["step_compliance_f1"].get("macro_f1"),
    }
    note_text = (out / "supervisor_note.md").read_text(encoding="utf-8", errors="ignore")
    payload = (
        f"GROUND_DATA:\n{json.dumps(ground_for_note, ensure_ascii=False)}\n\n"
        f"SUPERVISOR_NOTE.MD:\n{note_text[:6000]}\n"
    )
    sq = axes_yes_no_score(payload, SUPERVISOR_AXES, threshold=0.70)
    res["supervisor_quality_axes"] = {
        "axes": sq.get("axes", []),
        "score": sq.get("score", 0.0),
        "ok": bool(sq.get("ok", False)),
        "threshold": 0.70,
    }

    weights = {
        "action_recall": 0.22,
        "step_compliance_f1": 0.18,
        "hazard_detection": 0.13,
        "decoy_rejection": 0.13,
        "cross_doc_consistency": 0.19,
        "supervisor_quality_axes": 0.15,
    }
    score = 0.0
    for k, w in weights.items():
        if k == "cross_doc_consistency":
            score += w * (1.0 if res[k]["ok"] else 0.0)
        elif k == "supervisor_quality_axes":
            score += w * float(res[k].get("score", 0.0))
        else:
            score += w * max(0.0, res[k].get("ratio", 0.0))
    score = round(max(0.0, min(score, 1.0)), 3)
    all_ok = all(v["ok"] for v in res.values())
    print(json.dumps({"pass": all_ok,
                      "checks": {k: v["ok"] for k, v in res.items()},
                      **res,
                      "score": score,
                      "weights": weights},
                     indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
