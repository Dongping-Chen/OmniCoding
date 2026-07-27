"""Grader for T87a Multi-Epoch Astronomical Transient Triage (v3 — deterministic + memo axes).

v3 redesign:
  - Closed-vocab fields (variable_type, decoy_type, band) remain
    strict normalised match — no LLM, no VLM. Active subsets enforced
    via VARS / DCATS sets (decoy strings in fixtures must NOT appear).
  - triage_report.md is genuinely free-form prose; replaced single
    VLM judge with 5-axis yes/no `memo_quality_axes`.
  - Reweighted: deterministic dims dominate (~0.85); only the prose
    memo uses LLM axes (0.15).
"""
import argparse
import csv
import json
import math
import os
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT / "_lib"))

from open_vocab_judge import axes_yes_no_score, mllm_image_axes  # noqa: E402
from grader_format import resolve_required, check_columns  # noqa: E402


# Multimodal checklist for `light_curve_plot.png` (phase-folded /
# multi-epoch light curve panels split by variable_type, >=5 panels for
# the 5 variable classes). Anti-cheat: confirms the multipanel figure
# is actually rendered with traces, not blank or placeholder.
LIGHT_CURVE_AXES = [
    ("real_image_not_placeholder",
     "Is this image a real rendered figure with plotted curves and "
     "axes, NOT a single solid colour, blank canvas, all-black image, "
     "or single-text caption?"),
    ("multiple_panel_layout_visible",
     "Does the figure visibly contain multiple subplot panels "
     "(at least 5, one per variable type) arranged in a grid or row?"),
    ("light_curve_scatter_or_line_traces_visible",
     "Do the panels show scatter or line-trace data (points / curves "
     "over time or phase) consistent with photometric light curves "
     "rather than empty axes?"),
    ("axis_labels_or_panel_titles_visible",
     "Are axis labels, tick marks, panel titles, or legends visible "
     "(magnitude / phase / epoch / band annotations) on the panels?"),
]


REQUIRED = ["source_catalog.csv", "light_curves.csv",
            "decoy_rejections.json", "light_curve_plot.png", "triage_report.md"]
CAT_FIELDS = {"source_id", "ra_deg", "dec_deg", "gaia_id_or_null",
              "magnitude_B", "magnitude_V", "magnitude_I",
              "variable_type", "confidence"}
LC_FIELDS = {"source_id", "epoch", "band", "magnitude", "magerr"}
DCATS = {"cosmic_ray", "hot_pixel", "satellite_trail", "diffraction_spike"}
VARS = {"cepheid", "rr_lyrae", "eclipsing_binary", "cataclysmic", "supernova_candidate"}
BANDS = {"B", "V", "I"}
MEMO_WORD_CAP = 500
IMG_MIN_W, IMG_MIN_H = 1280, 800
MATCH_ARCSEC = 0.5
DECOY_RADIUS_PX = 5


MEMO_AXES = [
    ("top_candidates_per_type",
     "Does triage_report.md cover all 5 variable types (cepheid, "
     "rr_lyrae, eclipsing_binary, cataclysmic, supernova_candidate) "
     "with top-3 candidates by source_id including RA/Dec and B/V/I "
     "magnitudes, and does at least one Cepheid candidate quote a "
     "period_days value?"),
    ("delta_v_quantified",
     "Does triage_report.md quantify cross-epoch flux with at least "
     "one explicit ΔV (delta-V) magnitude between epochs?"),
    ("decoy_physics_cited",
     "Does triage_report.md cite concrete imaging physics for decoy "
     "rejection (cosmic-ray single-pixel spike width, satellite-trail "
     "Hough-line theta, diffraction-spike orientation, hot-pixel "
     "persistence)?"),
    ("source_id_grounded",
     "Does triage_report.md cite at least 3 distinct source_id values "
     "(S### form) that match rows in GROUND_DATA / source_catalog.csv?"),
    ("triage_prose_no_fabrication",
     "Does triage_report.md read as coherent astro-triage prose with a "
     "follow-up cadence / spectroscopy sentence AND avoid fabricated "
     "source_ids absent from the supplied catalog?"),
]


def _norm(s):
    return str(s or "").strip().lower().replace("-", "_").replace(" ", "_")


def img_size(p: Path):
    try:
        from PIL import Image
        with Image.open(p) as im:
            return im.size
    except (ImportError, FileNotFoundError, OSError):
        return None


def gating(out: Path):
    errs, resolved = resolve_required(out, REQUIRED)
    if errs: return False, errs, resolved
    miss = check_columns(resolved["source_catalog.csv"], CAT_FIELDS)
    if miss: return False, [f"source_catalog.csv missing cols: {miss}"], resolved
    miss = check_columns(resolved["light_curves.csv"], LC_FIELDS)
    if miss: return False, [f"light_curves.csv missing cols: {miss}"], resolved
    try:
        d = json.load(resolved["decoy_rejections.json"].open())
        assert isinstance(d, list)
    except (FileNotFoundError, json.JSONDecodeError, ValueError, AssertionError) as e:
        return False, [f"decoy_rejections.json invalid: {e}"], resolved
    return True, [], resolved


def angular_sep_arcsec(ra1, dec1, ra2, dec2):
    dra = (ra1 - ra2) * math.cos(math.radians(0.5 * (dec1 + dec2)))
    ddec = dec1 - dec2
    return 3600.0 * math.sqrt(dra * dra + ddec * ddec)


def hungarian_match(cat_rows, gt_sources, thr=MATCH_ARCSEC):
    if not cat_rows or not gt_sources:
        return [], 0
    BIG = 9.99e6
    n_gt, n_pr = len(gt_sources), len(cat_rows)
    cost = np.full((n_gt, n_pr), BIG, dtype=np.float64)
    for i, g in enumerate(gt_sources):
        for j, r in enumerate(cat_rows):
            cost[i, j] = angular_sep_arcsec(r["ra"], r["dec"], g["ra_deg"], g["dec_deg"])
    try:
        from scipy.optimize import linear_sum_assignment
        rows, cols = linear_sum_assignment(cost)
    except ImportError:
        return [], 0
    pairs, hits = [], 0
    for i, j in zip(rows, cols):
        if cost[i, j] <= thr:
            pairs.append((int(i), int(j))); hits += 1
    return pairs, hits


def magnitude_mae(pairs, cat_rows, gt_sources, lc_rows):
    if not pairs:
        return float("inf"), 0
    lc_idx = defaultdict(dict)
    for r in lc_rows:
        lc_idx[r["source_id"]][(int(r["epoch"]), r["band"])] = float(r["magnitude"])
    diffs = []
    for gi, pi in pairs:
        sid = cat_rows[pi]["source_id"]; gt_m = gt_sources[gi]["magnitudes_BVI"]
        fb = {"B": cat_rows[pi].get("mB"), "V": cat_rows[pi].get("mV"),
              "I": cat_rows[pi].get("mI")}
        for ep in (1, 2, 3):
            for k, b in enumerate(("B", "V", "I")):
                pred = lc_idx[sid].get((ep, b), fb[b])
                diffs.append(abs(float(pred) - float(gt_m[ep - 1][k])) if pred is not None else 2.0)
    return (float(sum(diffs) / len(diffs)) if diffs else float("inf")), len(diffs)


def variable_macro_f1(pairs, cat_rows, gt_sources):
    if not pairs:
        return 0.0
    from sklearn.metrics import f1_score
    y_t = [_norm(gt_sources[gi]["variable_type"]) for gi, _ in pairs]
    y_p = [_norm(cat_rows[pi]["variable_type"]) for _, pi in pairs]
    return float(f1_score(y_t, y_p, labels=sorted(VARS), average="macro", zero_division=0))


def decoy_rej(decoys, gt_decoys):
    used, hits = set(), 0
    for g in gt_decoys:
        for i, d in enumerate(decoys):
            if i in used:
                continue
            if d["epoch"] != g["epoch"] or d["band"] != g["band"]:
                continue
            if _norm(d["decoy_type"]) != _norm(g["decoy_type"]):
                continue
            if abs(d["x"] - g["x"]) + abs(d["y"] - g["y"]) <= DECOY_RADIUS_PX:
                used.add(i); hits += 1; break
    n = len(gt_decoys)
    return (hits / n if n else 0.0), hits, n


def cross_doc(cat_rows, lc_rows, decoys, memo, png, gt):
    errs = []
    ids = [r["source_id"] for r in cat_rows]
    bad_v = sorted({r["variable_type"] for r in cat_rows if _norm(r["variable_type"]) not in VARS})
    if bad_v:
        errs.append(f"variable_type out of active vocab: {bad_v[:5]}")
    bad_d = sorted({d.get("decoy_type") for d in decoys if _norm(d.get("decoy_type") or "") not in DCATS})
    if bad_d:
        errs.append(f"decoy_type out of active vocab: {bad_d[:5]}")
    bad_band = sorted({d.get("band") for d in decoys if d.get("band") not in BANDS})
    if bad_band:
        errs.append(f"decoy band out of {{B,V,I}}: {bad_band[:5]}")
    if [d for d in decoys if d.get("epoch") not in (1, 2, 3)]:
        errs.append("decoy epoch out of {1,2,3}")
    if len(cat_rows) < 30:
        errs.append(f"source_catalog.csv has {len(cat_rows)} rows (< 30)")
    if len(set(ids)) != len(ids):
        errs.append("duplicate source_id in source_catalog.csv")
    if len(lc_rows) < 9 * len(cat_rows):
        errs.append(f"light_curves.csv has {len(lc_rows)} rows (< 9*{len(cat_rows)})")
    bad_lc = sorted({r["source_id"] for r in lc_rows if r["source_id"] not in set(ids)})
    if bad_lc:
        errs.append(f"light_curves source_id not in catalog: {bad_lc[:5]}")
    crval1 = _f(gt["wcs_reference"]["crval1"]); crval2 = _f(gt["wcs_reference"]["crval2"])
    oob = [r["source_id"] for r in cat_rows
           if abs(r["ra"] - crval1) > 0.06 or abs(r["dec"] - crval2) > 0.06]
    if oob:
        errs.append(f"sources out of WCS scene bounds: {len(oob)} (e.g. {oob[:3]})")
    if len(decoys) < 6:
        errs.append(f"decoy_rejections has {len(decoys)} entries (< 6)")
    n_words = len(re.findall(r"\S+", memo))
    if n_words > MEMO_WORD_CAP:
        errs.append(f"triage_report.md {n_words} words (> {MEMO_WORD_CAP})")
    found_ids = set(re.findall(r"\bS\d{3}\b", memo))
    cited = found_ids & set(ids)
    if len(cited) < 3:
        errs.append(f"memo cites {len(cited)} catalog source_id (need >= 3)")
    bad_cite = found_ids - set(ids)
    if bad_cite:
        errs.append(f"memo cites source_id not in catalog: {sorted(bad_cite)[:5]}")
    sz = img_size(png)
    if sz is None:
        errs.append("light_curve_plot.png unreadable")
    elif sz[0] < IMG_MIN_W or sz[1] < IMG_MIN_H:
        errs.append(f"light_curve_plot.png {sz[0]}x{sz[1]} < {IMG_MIN_W}x{IMG_MIN_H}")
    return errs


def _f(v, d=0.0):
    try:
        return float(v)
    except (ValueError, TypeError):
        return d


def parse_outputs(out: Path, resolved: dict[str, Path]):
    cat_rows, lc_rows, decoys = [], [], []
    with resolved["source_catalog.csv"].open(newline="") as fh:
        for r in csv.DictReader(fh):
            try:
                cat_rows.append({"source_id": (r.get("source_id") or "").strip(),
                    "ra": _f(r.get("ra_deg")), "dec": _f(r.get("dec_deg")),
                    "gaia": (r.get("gaia_id_or_null") or "").strip(),
                    "mB": _f(r.get("magnitude_B")), "mV": _f(r.get("magnitude_V")),
                    "mI": _f(r.get("magnitude_I")),
                    "variable_type": (r.get("variable_type") or "").strip(),
                    "confidence": _f(r.get("confidence"))})
            except (KeyError, ValueError, TypeError):
                pass
    with resolved["light_curves.csv"].open(newline="") as fh:
        for r in csv.DictReader(fh):
            try:
                lc_rows.append({"source_id": (r.get("source_id") or "").strip(),
                    "epoch": int(_f(r.get("epoch"))), "band": (r.get("band") or "").strip(),
                    "magnitude": _f(r.get("magnitude")), "magerr": _f(r.get("magerr"))})
            except (KeyError, ValueError, TypeError):
                pass
    raw = json.load(resolved["decoy_rejections.json"].open())
    if isinstance(raw, list):
        for d in raw:
            try:
                decoys.append({"epoch": int(d.get("epoch")),
                    "band": (d.get("band") or "").strip(),
                    "x": int(_f(d.get("x"))), "y": int(_f(d.get("y"))),
                    "decoy_type": (d.get("decoy_type") or "").strip()})
            except (KeyError, ValueError, TypeError):
                pass
    memo = resolved["triage_report.md"].read_text(encoding="utf-8", errors="replace")
    return cat_rows, lc_rows, decoys, memo


def evaluate(out: Path, gt: dict, resolved: dict[str, Path]):
    cat_rows, lc_rows, decoys, memo = parse_outputs(out, resolved)
    gt_sources = gt["sources"]
    pairs, hits = hungarian_match(cat_rows, gt_sources, MATCH_ARCSEC)
    n_gt = len(gt_sources); recall = hits / n_gt if n_gt else 0.0
    mae, n_pairs = magnitude_mae(pairs, cat_rows, gt_sources, lc_rows)
    f1 = variable_macro_f1(pairs, cat_rows, gt_sources)
    rej, dh, dt = decoy_rej(decoys, gt["decoys"])
    cd = cross_doc(cat_rows, lc_rows, decoys, memo, resolved["light_curve_plot.png"], gt)
    mae_val = mae if math.isfinite(mae) else 99.0
    return {
        "source_position_recall": {"ratio": round(recall, 3), "n_gt": n_gt, "n_hit": hits,
                                    "ok": recall >= 0.60},
        "magnitude_mae": {"mae": round(mae_val, 3), "n_pairs": n_pairs,
                          "ratio": max(0.0, 1.0 - mae_val / 0.30),
                          "ok": mae_val <= 0.30 and n_pairs > 0},
        "variable_classification_macro_f1": {"ratio": round(float(f1), 3), "n_classes": 5,
                                              "ok": f1 >= 0.50},
        "decoy_rejection": {"ratio": round(rej, 3), "hits": dh, "gt": dt, "ok": rej >= 0.62},
        "cross_doc_consistency": {"errors": cd[:8], "n_errors": len(cd), "ok": len(cd) == 0},
    }, cat_rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", required=True)
    ap.add_argument("--gt", required=True)
    ap.add_argument("--fixtures", default="fixtures")
    a = ap.parse_args()
    out = Path(a.output)
    ok, errs, resolved = gating(out)
    if not ok:
        print(json.dumps({"pass": False, "stage": "gating", "errors": errs}, indent=2)); return
    gt = json.load(open(a.gt))
    res, cat_rows = evaluate(out, gt, resolved)
    var_counter = defaultdict(int)
    for r in cat_rows:
        var_counter[r["variable_type"]] += 1
    ground = {"n_sources_total": len(cat_rows),
              "variable_type_counts": dict(var_counter),
              "n_decoys_hit": res["decoy_rejection"].get("hits"),
              "n_decoys_gt":  res["decoy_rejection"].get("gt"),
              "magnitude_mae": res["magnitude_mae"].get("mae"),
              "macro_f1":      res["variable_classification_macro_f1"].get("ratio"),
              "position_recall": res["source_position_recall"].get("ratio"),
              "valid_source_ids_sample": [r["source_id"] for r in cat_rows[:20]]}
    memo_text = resolved["triage_report.md"].read_text(encoding="utf-8", errors="replace")
    payload = (
        f"GROUND_DATA:\n{json.dumps(ground, ensure_ascii=False)}\n\n"
        f"TRIAGE_REPORT.MD:\n{memo_text[:6000]}\n"
    )
    sq = axes_yes_no_score(payload, MEMO_AXES, threshold=0.70)
    res["memo_quality_axes"] = {
        "axes": sq.get("axes", []),
        "score": sq.get("score", 0.0),
        "ok": bool(sq.get("ok", False)),
        "threshold": 0.70,
    }
    # MLLM image content checklist on rendered light_curve_plot figure.
    la = mllm_image_axes(resolved["light_curve_plot.png"], LIGHT_CURVE_AXES,
                         threshold=0.70)
    res["light_curve_image_axes"] = {
        "axes": la.get("axes", []),
        "score": round(float(la.get("score", 0.0)), 3),
        "ok": bool(la.get("ok", False)),
        "threshold": 0.70,
    }

    W = {"source_position_recall": 0.18, "magnitude_mae": 0.13,
         "variable_classification_macro_f1": 0.18, "decoy_rejection": 0.13,
         "cross_doc_consistency": 0.13, "memo_quality_axes": 0.15,
         "light_curve_image_axes": 0.10}
    score = 0.0
    for k, w in W.items():
        if k == "cross_doc_consistency":
            score += w * (1.0 if res[k]["ok"] else 0.0)
        elif k == "memo_quality_axes":
            score += w * float(res[k].get("score", 0.0))
        elif k == "light_curve_image_axes":
            score += w * float(res[k].get("score", 0.0))
        else:
            score += w * max(0.0, res[k].get("ratio", 0.0))
    score = round(max(0.0, min(score, 1.0)), 3)
    all_ok = all(v["ok"] for v in res.values())
    print(json.dumps({"pass": all_ok,
                      "checks": {k: v["ok"] for k, v in res.items()},
                      **res, "score": score, "weights": W}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
