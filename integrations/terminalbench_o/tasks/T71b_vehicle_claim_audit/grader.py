"""Grader for T71a Vehicle Insurance Claim Audit (v3).

v3 redesign:
  - Closed-vocab fields (part_name / damage_type / recommendation /
    direction_consistent / overcharge_flag) graded by strict
    deterministic match. NO LLM / VLM on these.
  - Free-form `consistency_check.md` graded by 5-axis yes/no
    (`audit_quality_axes`), replacing the single-shot
    `judge_memo_grounded` call.
  - Closed-vocab fixture `damage_taxonomy.json["parts"]` augmented
    with realistic decoys to defeat blanket-include shortcuts.

Gates (all must pass):
  - part_localization_f1     >= 0.55
  - direction_consistency_acc>= 0.75
  - pre_existing_recall      >= 0.55
  - overcharge_detection_f1  >= 0.55
  - cross_doc_consistency    == 1.0
  - audit_quality_axes       >= 0.70
"""
import argparse
import csv
import json
import os
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT / "_lib"))
from open_vocab_judge import axes_yes_no_score


AUDIT_AXES = [
    ("case_ids_present",
     "Does the memo explicitly mention every case_id listed in "
     "GROUND_DATA.case_ids (e.g. case_01, case_02, ...)?"),
    ("direction_verdicts",
     "For each case, does the memo give a clear "
     "consistent/inconsistent verdict on the stated impact direction, "
     "and do these verdicts match GROUND_DATA.direction_consistent for "
     "at least most cases (>=4 of 6)?"),
    ("pre_existing_callout",
     "Does the memo call out at least one pre-existing (old) damage "
     "finding, referencing a body part from the closed parts vocabulary?"),
    ("overcharge_callout",
     "Does the memo call out at least one OEM-mismatch or overcharge "
     "finding, citing case_id and line_no consistent with GROUND_DATA?"),
    ("auditor_tone_no_oov",
     "Does the memo read as coherent claims-auditor English using "
     "concrete part names / case_ids / line numbers, with "
     "recommendations only from the closed vocabulary "
     "(approve / investigate / deny) and no fabricated case_ids?"),
]


def gating(out: Path):
    errs = []
    for f in ["damage_findings.csv", "consistency_check.md",
              "cost_audit.csv", "fraud_assessment.json"]:
        p = out / f
        if not p.exists():
            errs.append(f"missing {f}")
        elif p.stat().st_size == 0:
            errs.append(f"empty {f}")
    return len(errs) == 0, errs


def _f(s, default=0.0):
    try: return float(str(s).strip())
    except Exception: return default


def _b(s):
    return str(s).strip().lower() in ("true", "1", "yes", "y", "t")


def iou_xywh(a, b):
    ax, ay, aw, ah = a; bx, by, bw, bh = b
    ix1, iy1 = max(ax, bx), max(ay, by)
    ix2, iy2 = min(ax + aw, bx + bw), min(ay + ah, by + bh)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    ua = aw * ah + bw * bh - inter
    return inter / ua if ua > 0 else 0.0


def evaluate(out: Path, gt):
    findings = list(csv.DictReader((out / "damage_findings.csv").open()))
    cost_rows = list(csv.DictReader((out / "cost_audit.csv").open()))
    fraud = json.load((out / "fraud_assessment.json").open())
    consistency_md = (out / "consistency_check.md").read_text(
        encoding="utf-8", errors="replace")

    parts_vocab = set(gt["vocab"]["parts"])
    types_vocab = set(gt["vocab"]["damage_types"])
    rec_vocab = set(gt["vocab"]["recommendations"])

    # Pre-build OEM PN -> ref price map and operation -> std_hours map
    # by reading fixtures via the GT (cost_audit_truth has the matched
    # ref values per line).
    gt_audit = {}        # (case_id, line_no) -> truth dict
    for c in gt["cases"]:
        for a in c.get("audit_truth", []):
            gt_audit[(a["case_id"], int(a["line_no"]))] = a

    # ---- 1. part_localization_f1 ----
    gt_dmgs = []  # (case_id, idx) -> dict
    for c in gt["cases"]:
        for di, d in enumerate(c["true_damages"]):
            gt_dmgs.append({
                "key": (c["case_id"], di),
                "case_id": c["case_id"],
                "part": d["part"],
                "damage_type": d["damage_type"],
                "age": d["age"],
                "bbox": d["bbox_xywh"],
            })
    matched_gt = set()  # GT keys hit by some pred
    matched_old_gt = set()  # GT keys (age=old) where pred age matches
    tp = 0
    pred_used_count = 0
    for f in findings:
        cid = (f.get("case_id") or "").strip()
        part = (f.get("part_name") or "").strip()
        dtype = (f.get("damage_type") or "").strip()
        age = (f.get("age_estimate") or "").strip().lower()
        try:
            bb = [_f(f["bbox_x"]), _f(f["bbox_y"]),
                  _f(f["bbox_w"]), _f(f["bbox_h"])]
        except Exception:
            continue
        if not part or not dtype:
            continue
        pred_used_count += 1
        # Find best matching GT in same case with same part
        best = None
        for g in gt_dmgs:
            if g["case_id"] != cid:
                continue
            if g["key"] in matched_gt:
                continue
            if g["part"] != part:
                continue
            iou = iou_xywh(bb, g["bbox"])
            if iou >= 0.4:
                if best is None or iou > best[0]:
                    best = (iou, g)
        if best is not None:
            tp += 1
            matched_gt.add(best[1]["key"])
            if best[1]["age"] == "old" and age == "old":
                matched_old_gt.add(best[1]["key"])
    n_gt = len(gt_dmgs)
    n_pred = pred_used_count
    prec = tp / max(1, n_pred)
    rec = tp / max(1, n_gt)
    pf1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0

    # ---- 2. pre_existing_recall ----
    n_old_gt = sum(1 for g in gt_dmgs if g["age"] == "old")
    old_rec = (len(matched_old_gt) / n_old_gt) if n_old_gt else 1.0

    # ---- 3. direction_consistency_acc ----
    dir_hits = 0
    for c in gt["cases"]:
        cid = c["case_id"]
        gt_consistent = bool(c["direction_consistent"])
        case_pred = fraud.get(cid) or {}
        pred_consistent = case_pred.get("direction_consistent")
        if isinstance(pred_consistent, str):
            pred_consistent = pred_consistent.strip().lower() in (
                "true", "yes", "1")
        if pred_consistent == gt_consistent:
            dir_hits += 1
    dir_acc = dir_hits / max(1, len(gt["cases"]))

    # ---- 4. overcharge_detection_f1 ----
    # Build pred map: (case_id, line_no) -> overcharge bool
    pred_oc = {}
    for r in cost_rows:
        cid = (r.get("case_id") or "").strip()
        try:
            ln = int(str(r.get("line_no") or "").strip())
        except Exception:
            continue
        pred_oc[(cid, ln)] = _b(r.get("overcharge_flag"))
    tp_o = fp_o = fn_o = 0
    for k, t in gt_audit.items():
        gt_flag = bool(t["overcharge_flag"])
        p_flag = pred_oc.get(k, None)
        if p_flag is None:
            if gt_flag:
                fn_o += 1
            # missing line is also caught by cross_doc_consistency
            continue
        if gt_flag and p_flag:
            tp_o += 1
        elif p_flag and not gt_flag:
            fp_o += 1
        elif gt_flag and not p_flag:
            fn_o += 1
    o_prec = tp_o / max(1, tp_o + fp_o)
    o_rec = tp_o / max(1, tp_o + fn_o)
    o_f1 = ((2 * o_prec * o_rec / (o_prec + o_rec))
            if (o_prec + o_rec) > 0 else 0.0)

    # ---- 5. cross_doc_consistency ----
    cdc_errs = []
    # 5a. vocab compliance for findings
    for f in findings:
        p = (f.get("part_name") or "").strip()
        d = (f.get("damage_type") or "").strip()
        if p and p not in parts_vocab:
            cdc_errs.append(f"oov part_name: {p!r}")
        if d and d not in types_vocab:
            cdc_errs.append(f"oov damage_type: {d!r}")
    # 5b. recommendations vocab + case set
    pred_cases = set(fraud.keys())
    gt_cases = {c["case_id"] for c in gt["cases"]}
    if pred_cases != gt_cases:
        cdc_errs.append(
            f"fraud_assessment cases mismatch: extra={sorted(pred_cases-gt_cases)} "
            f"missing={sorted(gt_cases-pred_cases)}")
    for cid, v in fraud.items():
        rec_str = (v.get("recommendation") or "").strip()
        if rec_str and rec_str not in rec_vocab:
            cdc_errs.append(f"{cid} oov recommendation: {rec_str!r}")
    # 5c. cost_audit case set covers all cases that have quote items
    cost_cases = {(r.get("case_id") or "").strip() for r in cost_rows}
    if cost_cases != gt_cases:
        cdc_errs.append(
            f"cost_audit cases mismatch: extra={sorted(cost_cases-gt_cases)} "
            f"missing={sorted(gt_cases-cost_cases)}")
    # 5d. cost_audit line set per case matches GT
    pred_lines = set(pred_oc.keys())
    gt_lines = set(gt_audit.keys())
    if pred_lines != gt_lines:
        diff_extra = sorted(pred_lines - gt_lines)[:3]
        diff_missing = sorted(gt_lines - pred_lines)[:3]
        cdc_errs.append(
            f"cost_audit line set mismatch: extra={diff_extra} "
            f"missing={diff_missing}")
    # 5e. damage_findings cases ⊆ GT cases
    finding_cases = {(f.get("case_id") or "").strip() for f in findings}
    extra_cids = finding_cases - gt_cases
    if extra_cids:
        cdc_errs.append(f"findings have unknown case_ids: {sorted(extra_cids)}")
    # 5f. cost_audit oem_match consistency
    for r in cost_rows:
        cid = (r.get("case_id") or "").strip()
        try:
            ln = int(str(r.get("line_no") or "").strip())
        except Exception:
            continue
        truth = gt_audit.get((cid, ln))
        if truth is None:
            continue
        m = _b(r.get("oem_match"))
        # If pred says oem_match=true but truth says false → bad
        if m and not truth["oem_match"]:
            cdc_errs.append(f"{cid} L{ln}: claims OEM match for fake PN")
        # If pred says oem_match=true, ref_price columns should not be
        # blank
        if m:
            lo_blank = (r.get("ref_price_low") or "").strip() == ""
            hi_blank = (r.get("ref_price_high") or "").strip() == ""
            if lo_blank or hi_blank:
                cdc_errs.append(
                    f"{cid} L{ln}: oem_match=true but ref_price blank")
    # 5g. consistency_check.md mentions every case
    for cid in gt_cases:
        if cid not in consistency_md:
            cdc_errs.append(f"consistency_check.md missing {cid}")
    cdc_ok = len(cdc_errs) == 0

    return {
        "part_localization_f1": {
            "tp": tp, "fp": n_pred - tp, "fn": n_gt - tp,
            "precision": round(prec, 3),
            "recall": round(rec, 3),
            "f1": round(pf1, 3),
            "ok": pf1 >= 0.55,
        },
        "direction_consistency_acc": {
            "hits": dir_hits, "n_cases": len(gt["cases"]),
            "ratio": round(dir_acc, 3),
            "ok": dir_acc >= 0.75,
        },
        "pre_existing_recall": {
            "matched": len(matched_old_gt), "n_old_gt": n_old_gt,
            "ratio": round(old_rec, 3),
            "ok": old_rec >= 0.55,
        },
        "overcharge_detection_f1": {
            "tp": tp_o, "fp": fp_o, "fn": fn_o,
            "precision": round(o_prec, 3),
            "recall": round(o_rec, 3),
            "f1": round(o_f1, 3),
            "ok": o_f1 >= 0.55,
        },
        "cross_doc_consistency": {
            "errors": cdc_errs[:8],
            "ok": cdc_ok,
        },
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", required=True)
    ap.add_argument("--gt", required=True)
    args = ap.parse_args()
    out = Path(args.output)
    ok, errs = gating(out)
    if not ok:
        print(json.dumps({"pass": False, "stage": "gating",
                          "errors": errs}, indent=2))
        return
    gt = json.load(open(args.gt))
    res = evaluate(out, gt)
    ground = {
        "case_ids": [c["case_id"] for c in gt["cases"]],
        "direction_consistent": {c["case_id"]: bool(c["direction_consistent"])
                                  for c in gt["cases"]},
        "vocab_parts": gt["vocab"]["parts"][:25],
        "vocab_damage_types": gt["vocab"]["damage_types"],
        "vocab_recommendations": gt["vocab"]["recommendations"],
        "n_old_damages_total": sum(
            1 for c in gt["cases"] for d in c["true_damages"]
            if d.get("age") == "old"),
    }
    # ---- audit_quality_axes (5 yes/no axes on free-form prose) ----
    consistency_md = (out / "consistency_check.md").read_text(
        encoding="utf-8", errors="replace")
    payload = (
        f"GROUND_DATA:\n{json.dumps(ground, ensure_ascii=False)}\n\n"
        f"CONSISTENCY_CHECK.MD:\n{consistency_md[:6000]}\n"
    )
    aq = axes_yes_no_score(payload, AUDIT_AXES, threshold=0.70)
    res["audit_quality_axes"] = {
        "axes": aq.get("axes", []),
        "score": aq.get("score", 0.0),
        "ok": bool(aq.get("ok", False)) or bool(aq.get("skipped", False)),
        "threshold": 0.70,
    }
    all_ok = all(v["ok"] for v in res.values())
    # v3 reweight: deterministic dims dominate (~0.80), LLM axes <=0.20.
    score = (
        0.20 * res["part_localization_f1"]["f1"]
        + 0.15 * res["direction_consistency_acc"]["ratio"]
        + 0.15 * res["pre_existing_recall"]["ratio"]
        + 0.15 * res["overcharge_detection_f1"]["f1"]
        + 0.15 * (1 if res["cross_doc_consistency"]["ok"] else 0)
        + 0.20 * res["audit_quality_axes"]["score"]
    )
    print(json.dumps({
        "pass": all_ok,
        "checks": {k: v["ok"] for k, v in res.items()},
        **res,
        "score_so_far": round(score, 3),
        "weights": {
            "part_localization_f1": 0.20,
            "direction_consistency_acc": 0.15,
            "pre_existing_recall": 0.15,
            "overcharge_detection_f1": 0.15,
            "cross_doc_consistency": 0.15,
            "audit_quality_axes": 0.20,
        },
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
