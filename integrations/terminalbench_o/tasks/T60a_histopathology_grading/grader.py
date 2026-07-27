"""Grader for T60a Histopathology Tumor Grading (v3 — deterministic +
summary-only-LLM).

v3 redesign:
  - Capability dims (mask IoU, cellularity MAPE, classification kappa,
    case recall, cross_doc) all deterministic. classification is
    closed-vocab {macro, micro, itc, none} with 6 advertised decoys
    (suspicious, atypia, dysplasia, ...) that fail cross_doc and
    misclassify under kappa.
  - report_quality switches from a single VLM rubric judge to
    `axes_yes_no_score` over 5 yes/no axes.
  - Reweighted: deterministic dims ~0.82; LLM-axes 0.18.

6 dimensions (weights sum 1.0):
  mask_iou_macro          >= 0.45  (weight 0.16)
  cellularity_mape        <= 0.25  (weight 0.12)
  classification_kappa    >= 0.50  (weight 0.21)
  case_level_recall       >= 0.70  (weight 0.16)
  cross_doc_consistency   == 1.0   (weight 0.17)
  report_quality          >= 0.70  (weight 0.18)  axes yes/no on pathology_report.md
"""
import argparse, csv, json, os, re, sys
from pathlib import Path
import numpy as np
from PIL import Image

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT / "_lib"))

from open_vocab_judge import axes_yes_no_score  # noqa: E402


REPORT_AXES = [
    ("six_section_headings",
     "Does the pathology_report include all six required section "
     "headings (Specimen, Gross Description, Microscopic Findings, "
     "Diagnosis, Prognosis, Recommendation) and stay within ~350 "
     "words?"),
    ("three_or_more_valid_tile_ids",
     "Does the pathology_report cite >=3 distinct tile_NN identifiers "
     "that appear in GROUND_DATA `valid_tile_ids`, each linked to its "
     "classification or cellularity_pct from GROUND_DATA?"),
    ("ajcc_and_histopath_language",
     "Does the pathology_report reference AJCC pN classification "
     "(macro / micro / itc) and use concrete histopathology language "
     "(mitotic figures, reactive germinal centers, viable tumor, "
     "etc.) — not generic prose?"),
    ("decoys_or_neoadjuvant_addressed",
     "Does the pathology_report mention reactive-hyperplasia decoys "
     "or the prior-neoadjuvant convention where the supplied cases "
     "warrant it, and include a concrete recommendation (re-stage / "
     "IHC keratin / MDT review)?"),
    ("self_consistent_no_fabrication",
     "Is the pathology_report self-consistent with GROUND_DATA "
     "case_summary (per-case max classification, total_metastatic_area_"
     "mm2) and free of fabricated tile_ids not in `valid_tile_ids`?"),
]

VOCAB_CLASS = {"macro", "micro", "itc", "none"}
SEV_RANK = {"none": 0, "itc": 1, "micro": 2, "macro": 3}
SEV_NAME = {v: k for k, v in SEV_RANK.items()}
REQUIRED_FILES = ["findings.json", "case_summary.csv", "pathology_report.md"]
CASE_COLS = {"case_id", "n_tiles", "max_classification",
             "total_metastatic_area_mm2", "recommendation"}
SECTIONS = ["Specimen", "Gross Description", "Microscopic Findings",
            "Diagnosis", "Prognosis", "Recommendation"]


def gating(out: Path):
    errs = []
    for f in REQUIRED_FILES:
        p = out / f
        if not p.exists(): errs.append(f"missing: {f}")
        elif p.stat().st_size == 0: errs.append(f"empty: {f}")
    if not (out / "regions").is_dir():
        errs.append("missing: regions/ (mask directory)")
    if errs: return False, errs
    try:
        findings = json.loads((out / "findings.json").read_text())
    except Exception as e:
        return False, [f"findings.json invalid JSON: {e}"]
    if not isinstance(findings, list) or not findings:
        return False, ["findings.json must be a non-empty JSON array"]
    with (out / "case_summary.csv").open(newline="") as fh:
        cols = set((csv.DictReader(fh).fieldnames or []))
    miss = CASE_COLS - cols
    if miss:
        return False, [f"case_summary.csv missing fields: {sorted(miss)}"]
    return True, []


def load_mask(p: Path):
    return np.asarray(Image.open(p).convert("L"), dtype=np.uint8) > 127


def iou(a: np.ndarray, g: np.ndarray):
    if a.shape != g.shape: return 0.0
    inter = int(np.logical_and(a, g).sum())
    union = int(np.logical_or(a, g).sum())
    return 1.0 if union == 0 else inter / union


def cohen_kappa(yt, yp, labels):
    n = len(yt)
    if n == 0: return 0.0
    cm = {(t, p): 0 for t in labels for p in labels}
    for t, p in zip(yt, yp):
        if t in labels and p in labels: cm[(t, p)] += 1
    po = sum(cm[(l, l)] for l in labels) / n
    pe = sum((sum(cm[(l, q)] for q in labels) / n) *
             (sum(cm[(t, l)] for t in labels) / n) for l in labels)
    if pe >= 1.0: return 1.0 if po >= 1.0 else 0.0
    return (po - pe) / (1.0 - pe)


def cross_doc_check(findings, case_rows, report, regions, gt):
    errs = []
    gt_tids = {t["tile_id"] for t in gt["tiles"]}
    gt_cids = {c["case_id"] for c in gt["case_summaries"]}
    seen, by_id = set(), {}
    for f in findings:
        if not isinstance(f, dict):
            errs.append("findings item not dict"); continue
        tid = (f.get("tile_id") or "").strip()
        if not tid: errs.append("finding missing tile_id"); continue
        if tid in seen: errs.append(f"duplicate tile_id {tid}")
        seen.add(tid); by_id[tid] = f
        cls = (f.get("classification") or "").strip().lower()
        if cls not in VOCAB_CLASS:
            errs.append(f"{tid} classification='{f.get('classification')}' not in vocab")
        cell = f.get("cellularity_pct")
        if not isinstance(cell, int) or isinstance(cell, bool) or not (0 <= cell <= 100):
            errs.append(f"{tid} cellularity_pct={cell!r} not int [0,100]")
        tp = f.get("tumor_present")
        if not isinstance(tp, bool):
            errs.append(f"{tid} tumor_present must be JSON bool")
        elif tp and cls == "none":
            errs.append(f"{tid} tumor_present=true but classification=none")
        elif (not tp) and cls != "none":
            errs.append(f"{tid} tumor_present=false but classification={cls}")
        cf = f.get("confidence")
        if not isinstance(cf, (int, float)) or isinstance(cf, bool) or not (0.0 <= float(cf) <= 1.0):
            errs.append(f"{tid} confidence={cf!r} not in [0,1]")
    if seen != gt_tids:
        miss = sorted(gt_tids - seen)[:5]; extra = sorted(seen - gt_tids)[:5]
        if miss: errs.append(f"findings missing tile_ids: {miss}")
        if extra: errs.append(f"findings extra tile_ids: {extra}")
    # mask file + cellularity-vs-mask consistency
    for f in findings:
        tid = (f.get("tile_id") or "").strip()
        if not tid: continue
        mp = regions / f"{tid}.png"
        if not mp.exists():
            errs.append(f"{tid} mask file missing: regions/{tid}.png"); continue
        try:
            m = load_mask(mp)
        except Exception as e:
            errs.append(f"{tid} mask read error: {e}"); continue
        pct = 100.0 * float(m.sum()) / m.size
        cell = f.get("cellularity_pct")
        if isinstance(cell, int) and not isinstance(cell, bool) and abs(pct - cell) > 10.0:
            errs.append(f"{tid} mask area {pct:.1f}% vs cellularity_pct {cell}% differ > 10pp")
    # case_summary
    cseen, c_by_id = set(), {}
    for r in case_rows:
        cid = (r.get("case_id") or "").strip()
        if not cid: errs.append("case_summary row missing case_id"); continue
        if cid in cseen: errs.append(f"duplicate case_id in case_summary: {cid}")
        cseen.add(cid); c_by_id[cid] = r
        mc = (r.get("max_classification") or "").strip().lower()
        if mc not in VOCAB_CLASS:
            errs.append(f"{cid} max_classification='{r.get('max_classification')}' not in vocab")
        try: int(r.get("n_tiles"))
        except (ValueError, TypeError): errs.append(f"{cid} n_tiles not integer")
        try: float(r.get("total_metastatic_area_mm2"))
        except (ValueError, TypeError): errs.append(f"{cid} total_metastatic_area_mm2 not float")
        if not (r.get("recommendation") or "").strip():
            errs.append(f"{cid} recommendation empty")
    if cseen != gt_cids:
        miss = sorted(gt_cids - cseen); extra = sorted(cseen - gt_cids)
        if miss: errs.append(f"case_summary missing case_ids: {miss}")
        if extra: errs.append(f"case_summary extra case_ids: {extra}")
    # case max monotonicity vs findings
    tile_to_case = {t["tile_id"]: t["case_id"] for t in gt["tiles"]}
    case_pred_max = {}
    for f in findings:
        tid = (f.get("tile_id") or "").strip()
        cid = tile_to_case.get(tid)
        cls = (f.get("classification") or "").strip().lower()
        if cid and cls in VOCAB_CLASS:
            case_pred_max[cid] = max(case_pred_max.get(cid, 0), SEV_RANK[cls])
    for cid, row in c_by_id.items():
        mc = (row.get("max_classification") or "").strip().lower()
        if mc not in VOCAB_CLASS: continue
        exp = case_pred_max.get(cid, 0)
        if SEV_RANK[mc] != exp:
            errs.append(f"{cid} case_summary max={mc} but per-tile findings max={SEV_NAME[exp]}")
    # report sections + word count + tile_id mentions
    for sec in SECTIONS:
        if not re.search(rf"(?im)^\s*#{{1,3}}\s*{re.escape(sec)}\s*$", report):
            errs.append(f"report missing section: {sec}")
    nw = len(re.findall(r"\S+", report))
    if nw > 350: errs.append(f"pathology_report word count {nw} > 350")
    mtids = set(re.findall(r"tile_\d{2}", report))
    bad = mtids - gt_tids
    if bad: errs.append(f"report mentions unknown tile_ids: {sorted(bad)[:5]}")
    if len(mtids & gt_tids) < 3:
        errs.append(f"report mentions only {len(mtids & gt_tids)} valid tile_ids (need >= 3)")
    return errs, by_id, c_by_id


def evaluate(out: Path, gt: dict, gt_mask_dir: Path):
    findings = json.loads((out / "findings.json").read_text())
    with (out / "case_summary.csv").open(newline="") as fh:
        case_rows = list(csv.DictReader(fh))
    report = (out / "pathology_report.md").read_text(encoding="utf-8")
    regions = out / "regions"

    cd_errs, by_id, _ = cross_doc_check(findings, case_rows, report, regions, gt)
    gt_tiles = gt["tiles"]
    gt_cases = gt["case_summaries"]

    # mask_iou_macro
    ious = []
    for t in gt_tiles:
        gp = gt_mask_dir / f"{t['tile_id']}.png"
        ap = regions / f"{t['tile_id']}.png"
        if not gp.exists(): continue
        if not ap.exists():
            ious.append(0.0); continue
        try:
            ious.append(iou(load_mask(ap), load_mask(gp)))
        except Exception:
            ious.append(0.0)
    mean_iou = sum(ious) / len(ious) if ious else 0.0

    # cellularity_mape on tumor tiles
    abs_errs = []
    for t in gt_tiles:
        if t["true_classification"] == "none": continue
        f = by_id.get(t["tile_id"])
        if not f: abs_errs.append(1.0); continue
        ag = f.get("cellularity_pct")
        if not isinstance(ag, int) or isinstance(ag, bool):
            abs_errs.append(1.0); continue
        gtv = t["expected_cellularity_pct"]
        denom = max(5, gtv)
        abs_errs.append(abs(ag - gtv) / denom)
    mape = sum(abs_errs) / max(1, len(abs_errs))

    # classification_kappa
    yt, yp = [], []
    for t in gt_tiles:
        f = by_id.get(t["tile_id"])
        cls = (f.get("classification") or "").strip().lower() if f else ""
        if cls not in VOCAB_CLASS: cls = "none"
        yt.append(t["true_classification"]); yp.append(cls)
    kappa = cohen_kappa(yt, yp, list(VOCAB_CLASS))

    # case_level_recall (positive = max != none)
    case_pred_max = {c["case_id"]: 0 for c in gt_cases}
    for c in gt_cases:
        for tid in c["tile_ids"]:
            f = by_id.get(tid)
            cls = (f.get("classification") or "").strip().lower() if f else ""
            if cls in VOCAB_CLASS:
                case_pred_max[c["case_id"]] = max(case_pred_max[c["case_id"]], SEV_RANK[cls])
    pos_total = pos_correct = 0
    for c in gt_cases:
        if c["max_classification"] != "none":
            pos_total += 1
            if case_pred_max[c["case_id"]] >= 1: pos_correct += 1
    case_recall = pos_correct / max(1, pos_total)

    return {
        "mask_iou_macro": {"ratio": round(mean_iou, 3), "n_tiles": len(ious),
                           "ok": mean_iou >= 0.45},
        "cellularity_mape": {"value": round(mape, 3), "n_tiles": len(abs_errs),
                             "ratio": round(max(0.0, 1.0 - mape), 3),
                             "ok": mape <= 0.25},
        "classification_kappa": {"ratio": round(max(0.0, kappa), 3),
                                 "kappa": round(kappa, 3),
                                 "ok": kappa >= 0.50},
        "case_level_recall": {"ratio": round(case_recall, 3),
                              "tp": pos_correct, "n": pos_total,
                              "ok": case_recall >= 0.70},
        "cross_doc_consistency": {"errors": cd_errs[:10],
                                  "n_errors": len(cd_errs),
                                  "ok": len(cd_errs) == 0,
                                  "ratio": 1.0 if len(cd_errs) == 0 else 0.0},
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", required=True)
    ap.add_argument("--gt", required=True)
    ap.add_argument("--fixtures", default="fixtures")
    args = ap.parse_args()
    out = Path(args.output)
    gt_mask_dir = Path(args.gt).parent / "masks"
    ok, errs = gating(out)
    if not ok:
        print(json.dumps({"pass": False, "stage": "gating", "errors": errs}, indent=2))
        return
    gt = json.load(open(args.gt))
    res = evaluate(out, gt, gt_mask_dir)
    # ground for report quality judge
    try:
        findings = json.loads((out / "findings.json").read_text())
        sample_findings = []
        for f in findings[:8]:
            if isinstance(f, dict):
                sample_findings.append({
                    "tile_id": f.get("tile_id"),
                    "classification": f.get("classification"),
                    "cellularity_pct": f.get("cellularity_pct"),
                })
        with (out / "case_summary.csv").open(newline="") as fh:
            cs_rows = list(csv.DictReader(fh))
        case_summary = [{"case_id": r.get("case_id"),
                          "max_classification": (r.get("max_classification") or "").strip(),
                          "total_metastatic_area_mm2":
                              r.get("total_metastatic_area_mm2")}
                         for r in cs_rows]
    except Exception:
        sample_findings = []
        case_summary = []
    ground_for_memo = {
        "n_tiles": len(gt.get("tiles", [])),
        "n_cases": len(gt.get("case_summaries", [])),
        "vocab_classification": ["macro", "micro", "itc", "none"],
        "sample_findings": sample_findings,
        "case_summary": case_summary,
        "valid_tile_ids": sorted({t.get("tile_id") for t in gt.get("tiles", [])
                                    if t.get("tile_id")}),
    }
    report_text = (out / "pathology_report.md").read_text(errors="ignore")
    payload = (
        f"GROUND_DATA:\n{json.dumps(ground_for_memo, ensure_ascii=False)}\n\n"
        f"PATHOLOGY_REPORT.MD:\n{report_text[:6000]}\n"
    )
    rq = axes_yes_no_score(payload, REPORT_AXES, threshold=0.70)
    rq_score = float(rq.get("score", 0.0))
    res["report_quality"] = {"axes": rq.get("axes", []),
                              "score": round(rq_score, 3),
                              "ratio": round(rq_score, 3),
                              "threshold": 0.70,
                              "ok": bool(rq.get("ok", False))}
    all_ok = all(v["ok"] for v in res.values())
    # v3 reweight: deterministic dims dominate (~0.82); LLM axes 0.18.
    weights = {"mask_iou_macro": 0.16, "cellularity_mape": 0.12,
               "classification_kappa": 0.21, "case_level_recall": 0.16,
               "cross_doc_consistency": 0.17, "report_quality": 0.18}
    score = round(sum(weights[k] *
                      ((1.0 if res[k]["ok"] else 0.0) if k == "cross_doc_consistency"
                       else max(0.0, res[k].get("ratio", 0.0)))
                      for k in weights), 3)
    print(json.dumps({"pass": all_ok,
                      "checks": {k: v["ok"] for k, v in res.items()},
                      **res, "score": score, "weights": weights},
                     indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
