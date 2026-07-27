"""Grader for T28a Cross-Vendor Catalog Reconciliation (v3 deterministic).

v3 redesign:
  - Match pairs / anomalies remain strict deterministic set comparisons.
  - Replaced single-LLM `report_quality` with `report_axes`
    (4 yes/no axes).
  - Reweighted: deterministic dims ~0.85, axes ~0.15.
"""
import argparse
import csv
import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT / "_lib"))
from open_vocab_judge import axes_yes_no_score

try:
    from PIL import Image
except Exception:
    Image = None


REPORT_AXES = [
    ("counts_consistent",
     "Does the report state the total number of matched SKU pairs and "
     "the number of price anomalies, BOTH within +/-1 of the supplied "
     "GROUND_DATA n_matches and n_price_anomalies?"),
    ("cites_specific_skus",
     "Does the report cite >=2 specific A- or B- SKUs that appear in "
     "the supplied sample_pairs list (no fabricated SKUs)?"),
    ("unmatched_split_discussed",
     "Does the report discuss the unmatched-A vs unmatched-B counts "
     "consistent with the supplied n_unmatched_a / n_unmatched_b?"),
    ("analyst_prose_with_action",
     "Does the report read as coherent procurement-analyst prose under "
     "320 words AND include >=1 sentence on follow-up vendor action or "
     "pricing concerns?"),
]


def report_axes_score(out: Path, ground: dict) -> dict:
    rep = (out / "report.md").read_text()
    payload = (
        f"GROUND_DATA:\n{json.dumps(ground, ensure_ascii=False)}\n\n"
        f"REPORT.MD:\n{rep[:6000]}\n"
    )
    return axes_yes_no_score(payload, REPORT_AXES, threshold=0.70)


def gating(out: Path):
    errs = []
    for f in ["matches.json", "unmatched.csv",
                "price_diff_chart.png", "report.md"]:
        if not (out / f).exists():
            errs.append(f"missing {f}")
    return len(errs) == 0, errs


def _matches_of(obj):
    if isinstance(obj, list):
        return obj
    if isinstance(obj, dict):
        for k in ("matches", "items", "results", "pairs"):
            if isinstance(obj.get(k), list):
                return obj[k]
    return []


def evaluate(out: Path, gt):
    pred_obj = json.load((out / "matches.json").open())
    pred_matches = _matches_of(pred_obj)
    unm_rows = list(csv.DictReader((out / "unmatched.csv").open()))
    report = (out / "report.md").read_text()

    gt_pairs = {(m["a_sku"], m["b_sku"]) for m in gt["matches"]}
    gt_anomalies = {tuple(p) for p in gt["anomaly_pairs"]}

    pred_pairs = set()
    pred_anomalies = set()
    for m in pred_matches:
        a = (m.get("a_sku") or "").strip()
        b = (m.get("b_sku") or "").strip()
        if not a or not b:
            continue
        pred_pairs.add((a, b))
        if m.get("price_anomaly") is True:
            pred_anomalies.add((a, b))

    tp = len(pred_pairs & gt_pairs)
    rec = tp / max(1, len(gt_pairs))
    prec = tp / max(1, len(pred_pairs))

    pa_tp = len(pred_anomalies & gt_anomalies)
    pa_rec = pa_tp / max(1, len(gt_anomalies))

    cs_ok = True; cs_err = []
    if Image is not None:
        try:
            w, h = Image.open(out / "price_diff_chart.png").size
            if w < 1280 or h < 720:
                cs_ok = False; cs_err.append(f"{w}x{h}")
        except Exception as e:
            cs_ok = False; cs_err.append(str(e))
    else:
        cs_ok = False; cs_err.append("PIL missing")

    cdc_ok = True; cdc_err = []
    for m in pred_matches:
        a = (m.get("a_sku") or "")
        b = (m.get("b_sku") or "")
        if not a.startswith("A-") or not b.startswith("B-"):
            cdc_ok = False
            cdc_err.append(f"bad sku pair {a}/{b}")
            break
    for r in unm_rows:
        v = (r.get("vendor") or "").strip()
        sk = (r.get("sku") or "").strip()
        if v not in {"A", "B", "a", "b"}:
            cdc_ok = False
            cdc_err.append(f"bad vendor: {v}")
            break
        if v.upper() == "A" and not sk.startswith("A-"):
            cdc_ok = False
            cdc_err.append(f"A vendor sku must start A-: {sk}")
            break
        if v.upper() == "B" and not sk.startswith("B-"):
            cdc_ok = False
            cdc_err.append(f"B vendor sku must start B-: {sk}")
            break
    matched_a = {a for (a, _) in pred_pairs}
    matched_b = {b for (_, b) in pred_pairs}
    unm_skus = {(r.get("sku") or "").strip() for r in unm_rows}
    if (matched_a | matched_b) & unm_skus:
        cdc_ok = False
        cdc_err.append("sku appears in both matches and unmatched")
    if len(report.split()) > 320:
        cdc_ok = False
        cdc_err.append(f"report {len(report.split())} words")

    return {
        "match_recall": {
            "tp": tp, "total": len(gt_pairs),
            "ratio": round(rec, 3),
            "ok": rec >= 0.75,
        },
        "match_precision": {
            "tp": tp, "predicted": len(pred_pairs),
            "ratio": round(prec, 3),
            "ok": prec >= 0.80,
        },
        "price_anomaly_recall": {
            "tp": pa_tp, "total": len(gt_anomalies),
            "ratio": round(pa_rec, 3),
            "ok": pa_rec >= 0.70,
        },
        "chart_size": {
            "errors": cs_err,
            "ok": cs_ok,
        },
        "cross_doc_consistency": {
            "errors": cdc_err[:5],
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
    pred_obj = json.load((out / "matches.json").open())
    pred_matches = _matches_of(pred_obj)
    unm_rows = list(csv.DictReader((out / "unmatched.csv").open()))
    n_anomalies = sum(1 for m in pred_matches if m.get("price_anomaly") is True)
    sample_pairs = []
    for m in pred_matches[:6]:
        a = (m.get("a_sku") or "").strip()
        b = (m.get("b_sku") or "").strip()
        if a and b:
            sample_pairs.append(f"{a}<->{b}")
    n_unm_a = sum(1 for r in unm_rows if (r.get("vendor") or "").upper() == "A")
    n_unm_b = sum(1 for r in unm_rows if (r.get("vendor") or "").upper() == "B")
    ground_for_memo = {
        "n_matches": len(pred_matches),
        "n_price_anomalies": n_anomalies,
        "n_unmatched_a": n_unm_a,
        "n_unmatched_b": n_unm_b,
        "sample_pairs": sample_pairs,
    }
    sa = report_axes_score(out, ground_for_memo)
    res["report_axes"] = {
        "axes": sa.get("axes", []),
        "score": sa.get("score", 0.0),
        "ok": bool(sa.get("ok", False)),
        "threshold": 0.70,
    }
    all_ok = all(v["ok"] for v in res.values())
    score = (
        0.18 * res["match_recall"]["ratio"]
        + 0.18 * res["match_precision"]["ratio"]
        + 0.22 * res["price_anomaly_recall"]["ratio"]
        + 0.07 * (1 if res["chart_size"]["ok"] else 0)
        + 0.20 * (1 if res["cross_doc_consistency"]["ok"] else 0)
        + 0.15 * res["report_axes"]["score"]
    )
    print(json.dumps({
        "pass": all_ok,
        "checks": {k: v["ok"] for k, v in res.items()},
        **res,
        "score_so_far": round(score, 3),
        "weights": {
            "match_recall": 0.18,
            "match_precision": 0.18,
            "price_anomaly_recall": 0.22,
            "chart_size": 0.07,
            "cross_doc_consistency": 0.20,
            "report_axes": 0.15,
        },
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
