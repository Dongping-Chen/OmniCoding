"""Grader for T79a Wildfire Burn-Scar Multi-Temporal Sentinel-2 (v3 — deterministic + memo axes).

v3 redesign:
  - Closed-vocab fields (severity, classification) remain strict
    normalised match — no LLM, no VLM. SEV / CLS are the active subsets.
  - response_memo.md is genuinely free-form prose; replaced single
    VLM judge with 5-axis yes/no `response_memo_quality_axes`.
  - Reweighted: deterministic dims dominate (~0.85); only the prose
    memo uses LLM axes (0.15).

6 dims (sum 1.0): burn_area_mae_km2 le 15.0 (0.18), new_vs_old_classification ge 0.75 (0.18),
decoy_rejection ge 0.80 (0.18), temporal_consistency ge 0.85 (0.13), cross_doc_consistency = 1.0 (0.23),
response_memo_quality_axes >= 0.70 (0.10) 5-axis yes/no on response_memo.md.
"""
import argparse, csv, json, os, re, sys
from pathlib import Path
import numpy as np, pyproj, rasterio
from shapely.geometry import shape, mapping
from shapely.ops import transform as sh_transform, unary_union
from rasterio.features import geometry_mask
from scipy.optimize import curve_fit

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT / "_lib"))

from open_vocab_judge import axes_yes_no_score, mllm_image_axes  # noqa: E402


# Multimodal checklist for `comparison.png` (2x3 grid of Sentinel-2 RGB
# composites and dNBR false-colour rasters with red burn outlines).
# Anti-cheat: confirms the figure is actually rendered, not a blank or
# placeholder image.
COMPARISON_AXES = [
    ("real_image_not_placeholder",
     "Is this image a real rendered figure (raster panels, plots) and "
     "NOT a single solid colour, blank canvas, all-black image, or "
     "single-text caption?"),
    ("two_by_three_grid_layout_visible",
     "Does the figure visibly contain a 2-row by 3-column grid of "
     "panels (6 panels arranged in two rows)?"),
    ("satellite_imagery_or_raster_visible",
     "Do the panels show satellite imagery / raster maps (RGB scenes "
     "or false-colour rasters) rather than empty axes or blank "
     "placeholders?"),
    ("red_burn_outlines_or_annotations_visible",
     "Are red burn-perimeter outlines, polygons, or coloured "
     "annotations overlaid on the panels (highlighting primary "
     "burns)?"),
]


RESPONSE_MEMO_AXES = [
    ("burn_area_total_cited",
     "Does response_memo.md state the total burn area in km^2 within "
     "±5% of GROUND_DATA.agent_total_km2 and clearly distinguish new vs "
     "old burn classifications?"),
    ("burn_id_citations_grounded",
     "Does response_memo.md cite at least 2 specific burn_id values with "
     "area or severity consistent with the supplied perimeters (no "
     "fabricated burn_ids)?"),
    ("decoy_rejection_mentioned",
     "Does response_memo.md mention decoy rejection (cloud/water/shadow/"
     "other false positives) consistent with GROUND_DATA.n_decoys_total "
     "and n_decoys_hits?"),
    ("response_priorities_present",
     "Does response_memo.md read as coherent incident-command English "
     "with at least one sentence on response priorities or follow-up "
     "monitoring?"),
    ("no_fabrication",
     "Is response_memo.md free of fabricated burn_ids absent from the "
     "supplied perimeters and free of area numbers contradicting "
     "GROUND_DATA totals?"),
]

SEV = {"unburned", "low", "moderate", "high"}
CLS = {"new", "recent_recovering", "old_recovered", "not_burn"}
DELIVERABLES = ["burn_perimeters.geojson", "burn_timeseries.csv", "comparison.png", "response_memo.md"]
CSV_COLS = {"burn_id", "date", "mean_nbr", "mean_ndvi", "area_km2", "classification"}
WORD_CAP, WORD_BUFFER, AREA_TOL = 600, 60, 0.05
TO_UTM = pyproj.Transformer.from_crs("EPSG:4326", "EPSG:32610", always_xy=True).transform


def gating(out: Path):
    errs = []
    for f in DELIVERABLES:
        p = out / f
        if not p.exists(): errs.append(f"missing: {f}")
        elif p.stat().st_size == 0: errs.append(f"empty: {f}")
    if errs: return False, errs
    try:
        gj = json.loads((out / "burn_perimeters.geojson").read_text())
    except json.JSONDecodeError as e:
        return False, [f"burn_perimeters.geojson: invalid JSON ({e})"]
    if gj.get("type") != "FeatureCollection":
        return False, ["burn_perimeters.geojson: type != FeatureCollection"]
    if not gj.get("features"):
        return False, ["burn_perimeters.geojson: empty features"]
    with (out / "burn_timeseries.csv").open(newline="") as fh:
        cols = set((csv.DictReader(fh).fieldnames or []))
    miss = CSV_COLS - cols
    if miss:
        return False, [f"burn_timeseries.csv missing columns: {sorted(miss)}"]
    return True, []


def _coords(g):
    if g.geom_type == "Polygon":
        yield from g.exterior.coords
        for r in g.interiors: yield from r.coords
    elif g.geom_type == "MultiPolygon":
        for p in g.geoms: yield from _coords(p)


def iou(a, b):
    if a.is_empty or b.is_empty: return 0.0
    u = a.union(b).area
    return a.intersection(b).area / u if u else 0.0


def greedy_match(cost, sim_threshold):
    pairs = []
    used_r, used_c = set(), set()
    flat = sorted([(c, i, j) for i, row in enumerate(cost) for j, c in enumerate(row)])
    for c, i, j in flat:
        if i in used_r or j in used_c or (1.0 - c) < sim_threshold: continue
        used_r.add(i); used_c.add(j); pairs.append((i, j, 1.0 - c))
    return pairs


def sample_nbr_series(geom_utm, dates, fixtures: Path):
    series = {}
    for d in dates:
        tif = fixtures / "tiles" / f"{d}.tif"
        if not tif.exists(): return None
        with rasterio.open(tif) as src:
            mask = geometry_mask([mapping(geom_utm)], out_shape=(src.height, src.width),
                                 transform=src.transform, invert=True)
            if mask.sum() < 5: return None
            descs = list(src.descriptions or [])
            try:
                b08 = descs.index("B08") + 1; b12 = descs.index("B12") + 1
            except ValueError:
                return None
            nir = src.read(b08).astype(np.float32) / 10000.0
            sw = src.read(b12).astype(np.float32) / 10000.0
            sn, ss = nir[mask], sw[mask]
            valid = (sn > 0) & (ss > 0)
            if valid.sum() < 5: return None
            nbr = (sn[valid] - ss[valid]) / (sn[valid] + ss[valid] + 1e-8)
            series[d] = float(np.nanmean(nbr))
    return series


def temporal_ok(series, fire_idx):
    dates = sorted(series.keys())
    if len(dates) < 4 or fire_idx >= len(dates): return False
    pre = max(series[d] for d in dates[:fire_idx])
    post_min = min(series[d] for d in dates[fire_idx:])
    if pre - post_min < 0.27: return False
    post_dates = dates[fire_idx:]
    if len(post_dates) < 3: return False
    t = np.arange(len(post_dates), dtype=np.float64)
    y = np.array([series[d] for d in post_dates], dtype=np.float64)
    if np.std(y) < 1e-4: return True
    try:
        def lg(t, a, b, c): return a * (1.0 - np.exp(-b * t)) + c
        popt, _ = curve_fit(lg, t, y, p0=[0.3, 0.5, y[0]], maxfev=4000,
                            bounds=([-2, 0.01, -1.0], [2, 5.0, 1.0]))
        pred = lg(t, *popt)
        ss_res = float(np.sum((y - pred) ** 2))
        ss_tot = float(np.sum((y - np.mean(y)) ** 2))
        return (1.0 - ss_res / ss_tot if ss_tot > 1e-9 else 1.0) >= 0.7
    except Exception:
        return False


def evaluate(out: Path, gt: dict, fixtures: Path):
    cd_errs = []
    gj = json.loads((out / "burn_perimeters.geojson").read_text())
    feats4326 = [{"geom": shape(f["geometry"]), "props": f.get("properties", {}) or {}} for f in gj["features"]]
    bad_crs = any(not (-180 <= x <= 180 and -90 <= y <= 90) for f in feats4326 for x, y in _coords(f["geom"]))
    if bad_crs:
        cd_errs.append("burn_perimeters.geojson: coordinates not in EPSG:4326 lon/lat range")

    burn_ids_geo = []
    for f in feats4326:
        bid = f["props"].get("burn_id")
        if not bid: cd_errs.append("burn_perimeters.geojson: missing burn_id")
        else: burn_ids_geo.append(bid)
        sev = f["props"].get("severity")
        if sev not in SEV: cd_errs.append(f"burn_perimeters.geojson: severity '{sev}' out of vocab")
        ctr = f["props"].get("centroid")
        if not (isinstance(ctr, (list, tuple)) and len(ctr) == 2
                and -180 <= ctr[0] <= 180 and -90 <= ctr[1] <= 90):
            cd_errs.append(f"burn_perimeters.geojson: bad centroid for burn_id={f['props'].get('burn_id')}")
    if len(set(burn_ids_geo)) != len(burn_ids_geo):
        cd_errs.append("burn_perimeters.geojson: duplicate burn_id")

    csv_rows = list(csv.DictReader((out / "burn_timeseries.csv").open(newline="")))
    csv_per_burn = {}
    for r in csv_rows:
        bid = r.get("burn_id", "").strip()
        cls = r.get("classification", "").strip()
        if cls not in CLS: cd_errs.append(f"burn_timeseries.csv: classification '{cls}' out of vocab")
        csv_per_burn.setdefault(bid, []).append(r)
    if set(csv_per_burn.keys()) != set(burn_ids_geo):
        cd_errs.append(f"burn_timeseries.csv burn_id set != GeoJSON: {sorted(set(csv_per_burn.keys()) ^ set(burn_ids_geo))[:8]}")
    expected_dates = sorted(d["target"] for d in gt["dates_meta"])
    for bid, rows in csv_per_burn.items():
        ds = sorted(r["date"].strip() for r in rows)
        if ds != expected_dates:
            cd_errs.append(f"burn_timeseries.csv burn_id={bid}: dates {ds} != {expected_dates}")

    feats_utm = [{"geom": sh_transform(TO_UTM, f["geom"]).buffer(0), "props": f["props"]} for f in feats4326]
    geo_total = sum(f["geom"].area for f in feats_utm) / 1e6

    memo = (out / "response_memo.md").read_text(errors="ignore")
    n_words = len(re.findall(r"\b[\w'-]+\b", memo))
    if n_words > WORD_CAP + WORD_BUFFER:
        cd_errs.append(f"response_memo.md: {n_words} words > {WORD_CAP}+{WORD_BUFFER} cap")
    nums = [float(m.group(1)) for m in re.finditer(r"(\d+(?:\.\d{1,3})?)\s*km(?:²|2|\^2)?", memo)]
    memo_ok = any(abs(n - geo_total) / max(geo_total, 1.0) <= AREA_TOL for n in nums)
    if not memo_ok:
        cd_errs.append(f"response_memo.md: total burn area not within ±5% of {geo_total:.2f} km^2 (numbers: {nums[:8]})")

    gt_burns = [{"burn_id": b["burn_id"], "geom": sh_transform(TO_UTM, shape(json.loads(b["polygon_geojson"]))).buffer(0),
                 "severity": b["severity"], "classification": b["classification"],
                 "area_km2": b["area_km2"]} for b in gt["burns"]]
    gt_decoys = [{"decoy_id": d["decoy_id"], "category": d["category"],
                  "geom": sh_transform(TO_UTM, shape(json.loads(d["polygon_geojson"]))).buffer(0)}
                 for d in gt["decoys"]]
    gt_total = sum(b["area_km2"] for b in gt_burns)
    area_mae = abs(geo_total - gt_total)

    cost = [[1.0 - iou(af["geom"], gb["geom"]) for gb in gt_burns] for af in feats_utm]
    pairs = greedy_match(cost, 0.3) if cost and cost[0] else []
    matched_gt = set()
    n_correct = 0
    for i, j, _sim in pairs:
        matched_gt.add(j)
        bid = feats_utm[i]["props"].get("burn_id")
        rows = csv_per_burn.get(bid, [])
        if not rows: continue
        cnt = {}
        for r in rows:
            c = r.get("classification", "").strip()
            cnt[c] = cnt.get(c, 0) + 1
        dom = max(cnt.items(), key=lambda x: x[1])[0] if cnt else ""
        if dom == gt_burns[j]["classification"]:
            n_correct += 1
    class_acc = n_correct / max(1, len(gt_burns))

    n_decoy_hits = 0
    union_a = unary_union([f["geom"] for f in feats_utm]) if feats_utm else None
    if union_a is not None and not union_a.is_empty:
        for d in gt_decoys:
            inter = union_a.intersection(d["geom"]).area
            if d["geom"].area > 0 and inter / d["geom"].area > 0.3:
                n_decoy_hits += 1
    decoy_ratio = (len(gt_decoys) - n_decoy_hits) / max(1, len(gt_decoys))

    consistent = 0
    for af in feats_utm:
        s = sample_nbr_series(af["geom"], expected_dates, fixtures)
        if s and temporal_ok(s, gt["fire_start_idx"]):
            consistent += 1
    temp_ratio = consistent / max(1, len(feats_utm))

    return {
        "burn_area_mae_km2": {"value": round(area_mae, 3),
                               "ratio": max(0.0, 1.0 - area_mae / 30.0),
                               "agent_total": round(geo_total, 3),
                               "gt_total": round(gt_total, 3),
                               "ok": area_mae <= 15.0},
        "new_vs_old_classification": {"value": round(class_acc, 3), "ratio": class_acc,
                                       "matched": len(matched_gt), "n_gt": len(gt_burns),
                                       "ok": class_acc >= 0.75},
        "decoy_rejection": {"value": round(decoy_ratio, 3), "ratio": decoy_ratio,
                             "hits": n_decoy_hits, "n_decoys": len(gt_decoys),
                             "ok": decoy_ratio >= 0.80},
        "temporal_consistency": {"value": round(temp_ratio, 3), "ratio": temp_ratio,
                                  "consistent": consistent, "n_agent": len(feats_utm),
                                  "ok": temp_ratio >= 0.85},
        "cross_doc_consistency": {"value": 1.0 if not cd_errs else 0.0,
                                   "ratio": 1.0 if not cd_errs else 0.0,
                                   "errors": cd_errs[:8], "n_errors": len(cd_errs),
                                   "ok": not cd_errs},
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", required=True)
    ap.add_argument("--gt", required=True)
    ap.add_argument("--fixtures", default="fixtures")
    args = ap.parse_args()
    out = Path(args.output); fix = Path(args.fixtures)
    ok, errs = gating(out)
    if not ok:
        print(json.dumps({"pass": False, "stage": "gating", "errors": errs}, indent=2)); return
    gt = json.load(open(args.gt))
    res = evaluate(out, gt, fix)

    # Build ground data for memo axes
    try:
        gj = json.load((out / "burn_perimeters.geojson").open())
        burn_ids = [f.get("properties", {}).get("burn_id") for f in gj.get("features", [])]
    except Exception:
        burn_ids = []
    ground_for_memo = {
        "agent_total_km2": res["burn_area_mae_km2"].get("agent_total"),
        "gt_total_km2": res["burn_area_mae_km2"].get("gt_total"),
        "n_burns_matched": res["new_vs_old_classification"].get("matched"),
        "n_burns_gt": res["new_vs_old_classification"].get("n_gt"),
        "n_decoys_total": res["decoy_rejection"].get("n_decoys"),
        "n_decoys_hits": res["decoy_rejection"].get("hits"),
        "burn_ids_in_geojson": burn_ids[:20],
    }
    memo_text = (out / "response_memo.md").read_text(encoding="utf-8", errors="ignore")
    payload = (
        f"GROUND_DATA:\n{json.dumps(ground_for_memo, ensure_ascii=False)}\n\n"
        f"RESPONSE_MEMO.MD:\n{memo_text[:6000]}\n"
    )
    mq = axes_yes_no_score(payload, RESPONSE_MEMO_AXES, threshold=0.70)
    mq_score = float(mq.get("score", 0.0))
    mq_ok = bool(mq.get("ok", False))
    res["response_memo_quality_axes"] = {
        "axes": mq.get("axes", []),
        "score": round(mq_score, 3),
        "ok": mq_ok,
        "threshold": 0.70,
    }
    # MLLM image content checklist on rendered comparison figure.
    ca = mllm_image_axes(out / "comparison.png", COMPARISON_AXES,
                         threshold=0.70)
    res["comparison_image_axes"] = {
        "axes": ca.get("axes", []),
        "score": round(float(ca.get("score", 0.0)), 3),
        "ok": bool(ca.get("ok", False)),
        "threshold": 0.70,
    }
    all_ok = all(v["ok"] for v in res.values())
    weights = {"burn_area_mae_km2": 0.18, "new_vs_old_classification": 0.18,
               "decoy_rejection": 0.18, "temporal_consistency": 0.13,
               "cross_doc_consistency": 0.13, "response_memo_quality_axes": 0.10,
               "comparison_image_axes": 0.10}
    score = 0.0
    for k, w in weights.items():
        if k == "response_memo_quality_axes":
            score += w * res[k]["score"]
        elif k == "comparison_image_axes":
            score += w * res[k]["score"]
        else:
            score += w * res[k]["ratio"]
    score = round(score, 3)
    print(json.dumps({"pass": all_ok, "checks": {k: v["ok"] for k, v in res.items()},
                      "score": score, "weights": weights, "dims": res}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
