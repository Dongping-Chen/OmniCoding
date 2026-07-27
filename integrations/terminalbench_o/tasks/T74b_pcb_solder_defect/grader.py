"""Grader for T74a PCB Solder Defect Triage (v3).

v3 redesign:
  - Closed-vocab fields (defect_type / severity / ipc_class /
    ship_decision / action) graded by deterministic strict match.
  - Free-form `process_correlation.md` graded by 5-axis yes/no
    (`process_quality_axes`), replacing single-shot VLM judge.
  - `defect_taxonomy.json` augmented with decoy entries
    (CLOSED_WITH_DECOYS).

Pass criteria (all 6 gates):
  defect_localization_f1     >= 0.55  (weight 0.25)
  refdes_grounding           >= 0.75  (weight 0.15)
  ipc_classification_kappa   >= 0.45  (weight 0.10)
  process_correlation_recall >= 0.65  (weight 0.10)
  cross_doc_consistency      == 1.0   (weight 0.20)
  process_quality_axes       >= 0.70  (weight 0.20)
"""
import argparse
import csv
import json
import os
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT / "_lib"))
from open_vocab_judge import axes_yes_no_score


PROCESS_AXES = [
    ("worst_factor_ided",
     "Does the memo identify the worst-performing factors (a specific "
     "reflow_profile and a specific operator_shift) consistent with "
     "GROUND_DATA.bad_factors (factors strongly correlated with "
     "defects, e.g. paste profile X, shift Y) and correlate them with "
     "elevated defect rate?"),
    ("two_defect_types_cited",
     "Does the memo reference >=2 distinct defect_types from the "
     "closed defect_vocab (e.g. cold_joint, solder_bridge, "
     "insufficient_solder, lifted_pad)?"),
    ("corrective_action",
     "Does the memo include >=1 concrete corrective-action "
     "recommendation tied to the worst factor (process tweak, paste "
     "deposit change, operator retraining, etc.)?"),
    ("engineer_tone",
     "Does the memo read as coherent process-engineer English citing "
     "concrete board_ids / refdes / IPC class numbers consistent with "
     "GROUND_DATA?"),
    ("self_consistent_no_oov",
     "Is the memo self-consistent with GROUND_DATA's bad_factors and "
     "defect counts, and free of OOV defect_type vocabulary?"),
]

SILK_SLACK = 20
IOU_THRESHOLD = 0.30
ACTIONS = {"rework", "replace", "scrap"}
SHIPS = {"ship", "rework", "scrap"}
DELIVERABLES = ["defects.csv", "rework_plan.csv",
                "process_correlation.md", "rma_risk.json",
                "defect_correlations.json"]


def gating(out: Path):
    errs = []
    for f in DELIVERABLES:
        p = out / f
        if not p.exists():
            errs.append(f"missing {f}")
        elif p.stat().st_size == 0:
            errs.append(f"empty {f}")
    return len(errs) == 0, errs


def parse_bbox(s):
    parts = [p.strip() for p in str(s).replace("[", "").replace("]", "").split(",")]
    if len(parts) != 4:
        return None
    try:
        return [int(float(x)) for x in parts]
    except ValueError:
        return None


def iou(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    inter = (ix2 - ix1) * (iy2 - iy1)
    aa = max(1, (a[2] - a[0]) * (a[3] - a[1]))
    bb = max(1, (b[2] - b[0]) * (b[3] - b[1]))
    return inter / (aa + bb - inter)


def cohens_kappa(pairs):
    # Require >=5 pairs for kappa to be meaningful; below that we
    # treat the agent as having insufficient evidence -> 0.
    if len(pairs) < 5:
        return 0.0
    labels = sorted({l for p in pairs for l in p})
    n = len(pairs)
    obs = sum(1 for g, p in pairs if g == p) / n
    if len(labels) < 2:
        return obs
    gd = Counter(g for g, _ in pairs)
    pd = Counter(p for _, p in pairs)
    exp = sum((gd[l] / n) * (pd[l] / n) for l in labels)
    if exp >= 1.0:
        return 1.0 if obs >= 1.0 else 0.0
    return (obs - exp) / (1 - exp)


def load_predictions(out: Path):
    rows = list(csv.DictReader((out / "defects.csv").open()))
    by_board = defaultdict(list)
    parsed = []
    for r in rows:
        bbox = parse_bbox(r.get("bbox", ""))
        try:
            ipc = int(float(r.get("ipc_class", "0")))
        except ValueError:
            ipc = 0
        rec = {
            "finding_id":  (r.get("finding_id") or "").strip(),
            "board_id":    (r.get("board_id") or "").strip(),
            "refdes":      (r.get("refdes") or "").strip(),
            "defect_type": (r.get("defect_type") or "").strip(),
            "severity":    (r.get("severity") or "").strip(),
            "ipc_class":   ipc,
            "bbox":        bbox,
        }
        parsed.append(rec)
        if bbox is not None:
            by_board[rec["board_id"]].append(rec)
    return rows, parsed, by_board


def match_pairs(pred_list, gt_list):
    cands = []
    for ip, p in enumerate(pred_list):
        for ig, g in enumerate(gt_list):
            if p["defect_type"] != g["defect_type"]:
                continue
            s = iou(p["bbox"], g["bbox"])
            if s > IOU_THRESHOLD:
                cands.append((s, ip, ig))
    cands.sort(key=lambda t: -t[0])
    used_p, used_g, pairs = set(), set(), []
    for s, ip, ig in cands:
        if ip in used_p or ig in used_g:
            continue
        used_p.add(ip); used_g.add(ig)
        pairs.append((gt_list[ig], pred_list[ip]))
    return pairs


def silk_contains(layout, refdes, cx, cy):
    box = layout.get(refdes)
    if box is None:
        return False
    x0, y0, x1, y1 = box
    return (x0 - SILK_SLACK <= cx <= x1 + SILK_SLACK
            and y0 - SILK_SLACK <= cy <= y1 + SILK_SLACK)


def near_in(text, needle, key_terms, window=200):
    tl = text.lower()
    for m in re.finditer(re.escape(needle.lower()), tl):
        ctx = tl[max(0, m.start() - window): min(len(tl), m.end() + window)]
        if any(kt in ctx for kt in key_terms):
            return True
    return False


def evaluate(out: Path, gt):
    pred_rows, pred_list_full, pred_by_board = load_predictions(out)
    rework_rows = list(csv.DictReader((out / "rework_plan.csv").open()))
    rma = json.load((out / "rma_risk.json").open())
    proc_md = (out / "process_correlation.md").read_text(errors="ignore")

    layout = gt["refdes_layout"]
    defect_vocab = set(gt["defect_vocab"])
    sev_vocab = set(gt["severity_vocab"])

    # 1) localization F1 + 2) refdes_grounding + 3) IPC kappa
    total_gt = total_pred = grounded = 0
    ipc_pairs = []
    for b in gt["boards"]:
        gtl = b["true_defects"]
        prl = pred_by_board.get(b["board_id"], [])
        total_gt += len(gtl)
        total_pred += len(prl)
        for gd, pdr in match_pairs(prl, gtl):
            ipc_pairs.append((gd["ipc_class"], pdr["ipc_class"]))
            cx = (pdr["bbox"][0] + pdr["bbox"][2]) // 2
            cy = (pdr["bbox"][1] + pdr["bbox"][3]) // 2
            if silk_contains(layout, pdr["refdes"], cx, cy):
                grounded += 1
    tp = len(ipc_pairs)
    prec = tp / max(1, total_pred)
    rec = tp / max(1, total_gt)
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    grounding = grounded / max(1, total_gt)
    kappa = cohens_kappa(ipc_pairs)

    # 4) process correlation recall — deterministic recall over the
    # GT (reflow_profile, operator_shift) factor pairs declared in
    # `defect_correlations.json`. We do NOT scan free-form prose for
    # the literal "P3" / "night" substrings (that would let an agent
    # pass by writing the answer anywhere in the memo).
    try:
        dc_raw = json.load((out / "defect_correlations.json").open())
    except Exception:
        dc_raw = None
    truth = gt.get("process_correlation_truth", {}) or {}
    gt_profiles = {str(p).strip().lower()
                   for p in truth.get("high_correlation_profiles", [])}
    gt_shifts = {str(s).strip().lower()
                 for s in truth.get("high_correlation_shifts", [])}
    n_factors = len(gt_profiles) + len(gt_shifts)

    def _norm_list(x):
        if x is None:
            return []
        if isinstance(x, str):
            return [x]
        if isinstance(x, list):
            return [str(v) for v in x]
        return []

    agent_profiles, agent_shifts = set(), set()
    if isinstance(dc_raw, dict):
        for v in _norm_list(dc_raw.get("high_correlation_profiles")):
            agent_profiles.add(v.strip().lower())
        for v in _norm_list(dc_raw.get("high_correlation_shifts")):
            agent_shifts.add(v.strip().lower())
        # Also accept nested forms like
        #   {"factors": [{"profile": "P3", "shift": "night"}, ...]}
        for f in dc_raw.get("factors", []) or []:
            if isinstance(f, dict):
                p = f.get("reflow_profile") or f.get("profile")
                s = f.get("operator_shift") or f.get("shift")
                if isinstance(p, str):
                    agent_profiles.add(p.strip().lower())
                if isinstance(s, str):
                    agent_shifts.add(s.strip().lower())

    p_hits = len(agent_profiles & gt_profiles)
    s_hits = len(agent_shifts & gt_shifts)
    hits = p_hits + s_hits
    proc_recall = hits / max(1, n_factors)

    # 5) cross-doc consistency
    errs = []
    fids = []
    pairs_def = set()  # (board_id, refdes)
    for r in pred_rows:
        fid = (r.get("finding_id") or "").strip()
        bid = (r.get("board_id") or "").strip()
        rd = (r.get("refdes") or "").strip()
        dt = (r.get("defect_type") or "").strip()
        sv = (r.get("severity") or "").strip()
        try:
            ipc = int(float(r.get("ipc_class", "")))
        except ValueError:
            ipc = -1
        fids.append(fid)
        if dt and dt not in defect_vocab:
            errs.append(f"defect_type out of vocab: {dt}")
        if sv and sv not in sev_vocab:
            errs.append(f"severity out of vocab: {sv}")
        if ipc not in (1, 2, 3):
            errs.append(f"ipc_class out of {{1,2,3}}: row {fid}")
        if not bid:
            errs.append(f"missing board_id on {fid or '?'}")
        if not rd:
            errs.append(f"missing refdes on {fid or '?'}")
        else:
            pairs_def.add((bid, rd))
    if len(fids) != len(set(fids)):
        errs.append("duplicate finding_id in defects.csv")

    # refdes ∈ BOM
    bom_cache = {}
    for bid, rd in pairs_def:
        if bid not in bom_cache:
            p = Path("fixtures/boards") / f"{bid}_bom.csv"
            bom_cache[bid] = ({row["refdes"].strip()
                               for row in csv.DictReader(p.open())}
                              if p.exists() else None)
        bom = bom_cache[bid]
        if bom is None:
            errs.append(f"BOM missing for {bid}")
        elif rd not in bom:
            errs.append(f"refdes {rd} not in BOM of {bid}")

    # rework_plan three-way
    pairs_rew = set()
    for r in rework_rows:
        bid = (r.get("board_id") or "").strip()
        rd = (r.get("refdes") or "").strip()
        act = (r.get("action") or "").strip()
        if act and act not in ACTIONS:
            errs.append(f"rework action out of vocab: {act}")
        if bid and rd:
            pairs_rew.add((bid, rd))
    miss = pairs_def - pairs_rew
    extra = pairs_rew - pairs_def
    if miss:
        errs.append(f"{len(miss)} (board,refdes) missing from rework_plan.csv")
    if extra:
        errs.append(f"{len(extra)} extra (board,refdes) in rework_plan.csv")

    # rma_risk
    if not isinstance(rma, dict):
        errs.append("rma_risk.json is not an object"); rma = {}
    for bid in {b for b, _ in pairs_def}:
        if bid not in rma:
            errs.append(f"rma_risk missing entry for {bid}")
    for bid, entry in rma.items():
        if not isinstance(entry, dict):
            errs.append(f"rma_risk[{bid}] not an object"); continue
        if entry.get("ship_decision", "") not in SHIPS:
            errs.append(f"ship_decision out of vocab: {bid}")
        try:
            p = float(entry.get("rma_fail_probability", -1))
            if not (0.0 <= p <= 1.0):
                errs.append(f"rma_fail_probability out of [0,1]: {bid}")
        except (TypeError, ValueError):
            errs.append(f"rma_fail_probability not numeric: {bid}")
        crit = entry.get("top_3_critical_defects", None)
        if not isinstance(crit, list):
            errs.append(f"top_3_critical_defects not a list: {bid}")
        else:
            for fid in crit:
                if fid and fid not in fids:
                    errs.append(f"rma_risk[{bid}] -> unknown finding {fid}")

    consistency_ok = len(errs) == 0

    return {
        "defect_localization_f1":     {"tp": tp, "n_pred": total_pred, "n_gt": total_gt, "precision": round(prec, 3), "recall": round(rec, 3), "value": round(f1, 3), "ok": f1 >= 0.55},
        "refdes_grounding":           {"grounded_tp": grounded, "n_gt": total_gt, "value": round(grounding, 3), "ok": grounding >= 0.75},
        "ipc_classification_kappa":   {"n_pairs": len(ipc_pairs), "value": round(kappa, 3), "ok": kappa >= 0.45},
        "process_correlation_recall": {"hits": hits, "n_factors": n_factors,
                                       "agent_profiles": sorted(agent_profiles),
                                       "agent_shifts": sorted(agent_shifts),
                                       "gt_profiles": sorted(gt_profiles),
                                       "gt_shifts": sorted(gt_shifts),
                                       "value": round(proc_recall, 3),
                                       "ok": proc_recall >= 0.65},
        "cross_doc_consistency":      {"errors": errs[:8], "n_errors": len(errs), "value": 1.0 if consistency_ok else 0.0, "ok": consistency_ok},
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", required=True)
    ap.add_argument("--gt", required=True)
    a = ap.parse_args()
    out = Path(a.output)
    ok, errs = gating(out)
    if not ok:
        print(json.dumps({"pass": False, "stage": "gating", "errors": errs}, indent=2))
        return
    gt = json.load(open(a.gt))
    res = evaluate(out, gt)
    # Build ground dict for memo judge. We deliberately do NOT leak
    # the literal worst reflow_profile / operator_shift values
    # ("P3" / "night shift") into the prompt — the judge only sees an
    # abstract description so it cannot string-match the answer.
    ground = {
        "n_boards": len(gt["boards"]),
        "defect_vocab": gt.get("defect_vocab", []),
        "ipc_classes": [1, 2, 3],
        "bad_factors": (
            "factors strongly correlated with defects "
            "(e.g. paste profile X, shift Y) — withheld from this "
            "prompt; judge only on whether the memo names a single "
            "specific reflow_profile and a single specific "
            "operator_shift and ties them to elevated defect rate."
        ),
        "total_defects": sum(len(b.get("true_defects") or []) for b in gt["boards"]),
        "sample_board_ids": [b["board_id"] for b in gt["boards"][:5]],
    }
    proc_md = (out / "process_correlation.md").read_text(errors="ignore")
    payload = (
        f"GROUND_DATA:\n{json.dumps(ground, ensure_ascii=False)}\n\n"
        f"PROCESS_CORRELATION.MD:\n{proc_md[:6000]}\n"
    )
    pq = axes_yes_no_score(payload, PROCESS_AXES, threshold=0.70)
    res["process_quality_axes"] = {
        "value": round(float(pq.get("score", 0.0)), 3),
        "axes": pq.get("axes", []),
        "skipped": bool(pq.get("skipped", False)),
        "ok": bool(pq.get("ok", False)) or bool(pq.get("skipped", False)),
        "threshold": 0.70,
    }
    all_ok = all(v["ok"] for v in res.values())
    w = {"defect_localization_f1": 0.25, "refdes_grounding": 0.15,
         "ipc_classification_kappa": 0.10, "process_correlation_recall": 0.10,
         "cross_doc_consistency": 0.20, "process_quality_axes": 0.20}
    score = round(sum(w[k] * res[k]["value"] for k in w), 3)
    print(json.dumps({"pass": all_ok,
                      "checks": {k: v["ok"] for k, v in res.items()},
                      "score": score, "weights": w, "dims": res},
                     indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
