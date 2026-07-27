"""Grader for T25a Travel Reimbursement (v3 deterministic).

v3 redesign:
  - decision is closed-vocab (approve/reject) strict normalised match.
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
    ("approved_total_match",
     "Does the report state the approved-total KRW figure within +/-1 "
     "of the supplied GROUND_DATA approved_total_krw?"),
    ("cites_specific_files",
     "Does the report cite >=2 specific receipt filenames (e.g. "
     "r001.png) that appear in the supplied sample_files list?"),
    ("decision_split",
     "Does the report discuss the approve / reject split with counts "
     "consistent with the supplied n_approve / n_reject figures?"),
    ("controller_prose_with_action",
     "Does the report read as coherent expense-controller prose under "
     "280 words AND include >=1 sentence on policy reasoning or follow-"
     "up action?"),
]


def report_axes_score(out: Path, ground: dict) -> dict:
    rep = (out / "report.md").read_text()
    payload = (
        f"GROUND_DATA:\n{json.dumps(ground, ensure_ascii=False)}\n\n"
        f"REPORT.MD:\n{rep[:6000]}\n"
    )
    return axes_yes_no_score(payload, REPORT_AXES, threshold=0.70)


def _norm(s):
    return str(s or "").strip().lower().replace("-", "_").replace(" ", "_")


def gating(out: Path):
    errs = []
    for f in ["reimbursement.json", "summary.csv",
                "expense_chart.png", "report.md"]:
        if not (out / f).exists():
            errs.append(f"missing {f}")
    if not (out / "thumbnails").is_dir():
        errs.append("missing thumbnails/")
    return len(errs) == 0, errs


def evaluate(out: Path, gt):
    data = json.load((out / "reimbursement.json").open())
    items = data.get("receipts", [])
    by_file = {it.get("file", ""): it for it in items}

    rows = list(csv.DictReader((out / "summary.csv").open()))
    report = (out / "report.md").read_text()

    gt_recs = {r["file"]: r for r in gt["receipts"]}

    te_total = 0; te_ok = 0
    for fpath, gr in gt_recs.items():
        it = by_file.get(fpath)
        if not it:
            te_total += 1; continue
        try:
            t = int(it.get("total_krw"))
        except Exception:
            te_total += 1; continue
        te_total += 1
        if abs(t - gr["total_krw"]) <= 1:
            te_ok += 1
    te_ratio = te_ok / max(1, te_total)

    dec_total = 0; dec_ok = 0
    for fpath, gr in gt_recs.items():
        it = by_file.get(fpath)
        if not it:
            dec_total += 1; continue
        dec_total += 1
        if _norm(it.get("decision")) == _norm(gr["decision"]):
            dec_ok += 1
    dec_ratio = dec_ok / max(1, dec_total)

    tpa_ok = True; tpa_err = []
    approved_files = [f for f, r in gt_recs.items()
                       if r["decision"] == "approve"]
    for fpath in approved_files:
        stem = Path(fpath).stem
        candidates = [out / "thumbnails" / f"{stem}.png",
                       out / "thumbnails" / f"{stem}.jpg"]
        thumb = next((c for c in candidates if c.exists()), None)
        if not thumb:
            tpa_ok = False; tpa_err.append(f"no thumb for {fpath}")
            continue
        if Image is not None:
            try:
                w, h = Image.open(thumb).size
                if w < 400 or h < 400:
                    tpa_ok = False
                    tpa_err.append(f"{stem} {w}x{h}")
            except Exception as e:
                tpa_ok = False; tpa_err.append(f"{stem}: {e}")

    sa_ok = True; sa_err = []
    declared = data.get("approved_total_krw")
    actual = sum(int(it.get("total_krw") or 0) for it in items
                  if _norm(it.get("decision")) == "approve")
    if declared is None or int(declared) != actual:
        sa_ok = False
        sa_err.append(f"declared {declared} vs computed {actual}")

    # ---- decision_balance_ok ----
    # Anti-degenerate: a binary {approve, reject} decision space at
    # ~50% chance accuracy permits all-approve / all-reject baselines.
    # Require each class >= 15% of total over the agent's reimbursement.json.
    n_total_dec = max(1, len(items))
    n_app = sum(1 for it in items if _norm(it.get("decision")) == "approve")
    n_rej = sum(1 for it in items if _norm(it.get("decision")) == "reject")
    min_frac_dec = min(n_app, n_rej) / n_total_dec
    db_ok = min_frac_dec >= 0.15
    db_err = []
    if not db_ok:
        db_err.append(
            f"degenerate decision split: approve={n_app}/{n_total_dec}, "
            f"reject={n_rej}/{n_total_dec}; min_frac={min_frac_dec:.2f} < 0.15")

    cdc_ok = True; cdc_err = []
    if len(rows) != len(gt_recs):
        cdc_ok = False; cdc_err.append(f"summary rows {len(rows)}")
    csv_files = {r.get("file", "") for r in rows}
    if csv_files != set(gt_recs.keys()):
        cdc_ok = False
        miss = sorted(set(gt_recs.keys()) - csv_files)[:3]
        cdc_err.append(f"csv missing {miss}")
    if set(by_file.keys()) != set(gt_recs.keys()):
        cdc_ok = False
        diff = sorted(set(gt_recs.keys()) ^ set(by_file.keys()))[:3]
        cdc_err.append(f"json file diff {diff}")
    if Image is not None:
        try:
            w, h = Image.open(out / "expense_chart.png").size
            if w < 800 or h < 500:
                cdc_ok = False; cdc_err.append(f"chart {w}x{h}")
        except Exception as e:
            cdc_ok = False; cdc_err.append(f"chart: {e}")
    if len(report.split()) > 280:
        cdc_ok = False
        cdc_err.append(f"report {len(report.split())} words > 280")

    return {
        "total_extraction_acc": {
            "matched": te_ok, "total": te_total,
            "ratio": round(te_ratio, 3),
            "ok": te_ratio >= 0.85,
        },
        "decision_acc": {
            "matched": dec_ok, "total": dec_total,
            "ratio": round(dec_ratio, 3),
            "ok": dec_ratio >= 0.85,
        },
        "thumbnail_per_approved": {
            "errors": tpa_err[:5],
            "ok": tpa_ok,
        },
        "summary_aggregate_match": {
            "errors": sa_err[:5],
            "ok": sa_ok,
        },
        "decision_balance_ok": {
            "n_approve": n_app, "n_reject": n_rej, "n_total": n_total_dec,
            "min_frac": round(min_frac_dec, 3),
            "errors": db_err,
            "ok": db_ok,
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
    data = json.load((out / "reimbursement.json").open())
    items = data.get("receipts", [])
    approved = [it for it in items if _norm(it.get("decision")) == "approve"]
    rejected = [it for it in items if _norm(it.get("decision")) == "reject"]
    other = [it for it in items if _norm(it.get("decision")) not in ("approve", "reject")]
    approved_total = sum(int(it.get("total_krw") or 0) for it in approved)
    sample_files = [it.get("file") for it in items[:8] if it.get("file")]
    ground_for_memo = {
        "approved_total_krw": approved_total,
        "n_approve": len(approved),
        "n_reject": len(rejected),
        "n_other": len(other),
        "n_total": len(items),
        "sample_files": sample_files,
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
        0.27 * res["total_extraction_acc"]["ratio"]
        + 0.22 * res["decision_acc"]["ratio"]
        + 0.07 * (1 if res["thumbnail_per_approved"]["ok"] else 0)
        + 0.12 * (1 if res["summary_aggregate_match"]["ok"] else 0)
        + 0.05 * (1 if res["decision_balance_ok"]["ok"] else 0)
        + 0.12 * (1 if res["cross_doc_consistency"]["ok"] else 0)
        + 0.15 * res["report_axes"]["score"]
    )
    print(json.dumps({
        "pass": all_ok,
        "checks": {k: v["ok"] for k, v in res.items()},
        **res,
        "score_so_far": round(score, 3),
        "weights": {
            "total_extraction_acc": 0.27,
            "decision_acc": 0.22,
            "thumbnail_per_approved": 0.07,
            "summary_aggregate_match": 0.12,
            "decision_balance_ok": 0.05,
            "cross_doc_consistency": 0.12,
            "report_axes": 0.15,
        },
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
