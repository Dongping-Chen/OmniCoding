"""Grader for T15a News Photo Editorial Ledger (v3 deterministic).

v3 redesign:
  - Replaced single-LLM `summary_quality` with 5-axis yes/no (`summary_axes`).
  - Reweighted: deterministic dims ~0.75, axes ~0.20.
"""
import argparse
import csv
import json
import sys
from pathlib import Path

try:
    from PIL import Image
except Exception:
    Image = None

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT / "_lib"))
from open_vocab_judge import axes_yes_no_score


SUMMARY_AXES = [
    ("hero_rationale_present",
     "Does the memo include a `## Hero rationale` section that names the "
     "chosen hero filename (matching GROUND_DATA hero) and gives a concrete "
     "editorial reason (subject framing, lighting, decisive moment) — not "
     "just generic adjectives?"),
    ("rejections_breakdown",
     "Does the memo include a `## Rejections by reason` section that lists "
     "at least two specific rejection reasons (banned / embargoed / duplicate / "
     "low_res) with file count or example filename matching the rejected list?"),
    ("alt_slate_cited",
     "Does the memo reference the alt slate by citing >=2 alt filenames so "
     "the editor can trace the pack composition?"),
    ("species_and_quality_bar",
     "Does the memo read as coherent picture-desk English AND mention species "
     "relevance (lion / tiger / leopard) and quality bar (>=1280x720)?"),
    ("no_forbidden_files",
     "Is the memo free of praising or citing banned / embargoed files as hero "
     "or alt, and are all cited filenames drawn from the supplied ledger?"),
]

CRITICAL_CHECKS = [
    "banned_excluded",
    "dedup_recall",
    "crop_size_match",
    "candidate_set_match",
    "cross_doc_consistency",
]
PASS_SCORE_THRESHOLD = 0.80


def summary_axes_score(out: Path, ground: dict) -> dict:
    summary = (out / "summary.md").read_text()
    payload = (
        f"GROUND_DATA:\n{json.dumps(ground, ensure_ascii=False)}\n\n"
        f"SUMMARY.MD:\n{summary[:6000]}\n"
    )
    return axes_yes_no_score(payload, SUMMARY_AXES, threshold=0.70)


def gating(out: Path):
    errs = []
    if not (out / "ledger.json").exists():
        errs.append("missing ledger.json")
    if not (out / "provenance.csv").exists():
        errs.append("missing provenance.csv")
    if not (out / "summary.md").exists():
        errs.append("missing summary.md")
    if not (out / "crops").is_dir():
        errs.append("missing crops/")
    return len(errs) == 0, errs


def evaluate(out: Path, gt, sources_csv: Path):
    ledger = json.load((out / "ledger.json").open())
    rows = list(csv.DictReader((out / "provenance.csv").open()))
    summary = (out / "summary.md").read_text()

    hero = ledger.get("hero")
    alts = ledger.get("alts") or []
    crops = ledger.get("crops") or []
    rejected = ledger.get("rejected") or []

    banned = set(gt["banned"])
    embargoed = set(gt["embargoed"])
    forbidden = banned | embargoed
    dup_groups = [set(g) for g in gt["duplicate_groups"]]
    oracle_top = set(gt["oracle_top"])
    valid_ratios = gt["valid_ratios"]

    used = {hero} if hero else set()
    used |= set(alts)
    used |= {c.get("file") for c in crops if c.get("file")}

    # ---- banned_excluded ----
    bad_used = used & forbidden
    be_ok = len(bad_used) == 0

    # ---- dedup_recall ----
    dr_total = 0; dr_ok = 0
    for grp in dup_groups:
        dr_total += 1
        if len(grp & used) <= 1:
            dr_ok += 1
    dr_ratio = dr_ok / max(1, dr_total) if dr_total else 1.0

    # ---- crop_size_match ----
    cs_ok = True; cs_err = []
    if Image is None:
        cs_ok = False; cs_err.append("PIL missing")
    # group crops by file
    by_file = {}
    for c in crops:
        by_file.setdefault(c.get("file"), []).append(c)
    for fname, items in by_file.items():
        ratios = [(c.get("ratio") or "").strip() for c in items]
        if not (set(ratios) >= set(valid_ratios.keys())):
            cs_ok = False; cs_err.append(f"{fname} ratios {ratios}")
        for c in items:
            r = (c.get("ratio") or "").strip()
            if r not in valid_ratios:
                cs_ok = False; cs_err.append(f"{fname} bad ratio {r}")
                continue
            outpath = out / (c.get("out_path") or "")
            if not outpath.exists():
                cs_ok = False; cs_err.append(f"{fname} {r} missing")
                continue
            try:
                w, h = Image.open(outpath).size
                ew, eh = valid_ratios[r]
                if (w, h) != (ew, eh):
                    cs_ok = False
                    cs_err.append(f"{fname} {r} {w}x{h}")
            except Exception as e:
                cs_ok = False; cs_err.append(f"{fname}: {e}")

    # ---- candidate_set_match ----
    pick = [hero] + alts
    pick = [p for p in pick if p]
    csm_ok_count = sum(1 for p in pick if p in oracle_top)
    csm_ratio = csm_ok_count / max(1, len(pick))

    # ---- cross_doc_consistency ----
    cdc_ok = True; cdc_err = []
    if not hero:
        cdc_ok = False; cdc_err.append("no hero")
    if len(alts) != 3:
        cdc_ok = False; cdc_err.append(f"alts {len(alts)} ≠ 3")
    # provenance covers used files + rejected
    prov_files = {(r.get("file") or "").strip() for r in rows}
    expected_files = (used | {r.get("file") for r in rejected if r.get("file")})
    missing_prov = expected_files - prov_files
    if missing_prov:
        cdc_ok = False
        cdc_err.append(f"provenance missing {sorted(missing_prov)[:3]}")
    # summary mentions hero filename
    if hero and hero not in summary:
        cdc_ok = False
        cdc_err.append(f"summary missing hero {hero}")
    # summary mentions rejection reasons (≥ 1)
    reasons = {(r.get("reason") or "").strip() for r in rejected
                if r.get("reason")}
    if not any(r in summary for r in reasons):
        cdc_ok = False
        cdc_err.append("summary missing rejection reason")

    return {
        "banned_excluded": {
            "bad_used": sorted(bad_used),
            "ok": be_ok,
        },
        "dedup_recall": {
            "matched": dr_ok, "total": dr_total,
            "ratio": round(dr_ratio, 3),
            "ok": dr_ratio >= 0.85,
        },
        "crop_size_match": {
            "errors": cs_err[:5],
            "ok": cs_ok,
        },
        "candidate_set_match": {
            "matched": csm_ok_count, "total": len(pick),
            "ratio": round(csm_ratio, 3),
            "ok": csm_ratio >= 0.80,
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
    ap.add_argument("--sources", required=True)
    args = ap.parse_args()
    out = Path(args.output)
    ok, errs = gating(out)
    if not ok:
        print(json.dumps({"pass": False, "stage": "gating", "errors": errs},
                         indent=2))
        return
    gt = json.load(open(args.gt))
    res = evaluate(out, gt, Path(args.sources))
    # ground for summary axes
    try:
        ledger = json.load((out / "ledger.json").open())
        rejected = ledger.get("rejected") or []
        from collections import Counter as _Counter
        reason_counts = _Counter((r.get("reason") or "").strip() for r in rejected)
        ground_for_memo = {
            "hero": ledger.get("hero"),
            "alts": ledger.get("alts") or [],
            "rejection_reason_counts": dict(reason_counts),
            "n_rejected": len(rejected),
            "n_crops": len(ledger.get("crops") or []),
            "banned_files": gt.get("banned", [])[:8],
            "embargoed_files": gt.get("embargoed", [])[:8],
        }
    except Exception:
        ground_for_memo = {}
    sq = summary_axes_score(out, ground_for_memo)
    res["summary_axes"] = {
        "axes": sq.get("axes", []),
        "score": sq.get("score", 0.0),
        "ok": bool(sq.get("ok", False)),
        "threshold": 0.70,
    }
    # v3: deterministic dims dominate (~0.75), only summary uses LLM judge (~0.20).
    score = (
        0.20 * (1 if res["banned_excluded"]["ok"] else 0)
        + 0.20 * res["dedup_recall"]["ratio"]
        + 0.15 * (1 if res["crop_size_match"]["ok"] else 0)
        + 0.20 * res["candidate_set_match"]["ratio"]
        + 0.05 * (1 if res["cross_doc_consistency"]["ok"] else 0)
        + 0.20 * res["summary_axes"]["score"]
    )
    score = round(score, 3)
    critical_ok = all(res[name]["ok"] for name in CRITICAL_CHECKS)
    passed = critical_ok and score >= PASS_SCORE_THRESHOLD
    print(json.dumps({
        "pass": passed,
        "checks": {k: v["ok"] for k, v in res.items()},
        **res,
        "score": score,
        "weights": {
            "banned_excluded": 0.20,
            "dedup_recall": 0.20,
            "crop_size_match": 0.15,
            "candidate_set_match": 0.20,
            "cross_doc_consistency": 0.05,
            "summary_axes": 0.20,
        },
        "pass_policy": {
            "critical_checks": CRITICAL_CHECKS,
            "non_critical_checks": ["summary_axes"],
            "score_threshold": PASS_SCORE_THRESHOLD,
        },
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
