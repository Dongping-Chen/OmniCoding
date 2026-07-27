"""Grader for T75a LiDAR Street Tree Inventory (v3).

v3 redesign:
  - Closed-vocab fields (species / health / decoy reason) graded by
    deterministic strict match (already strict).
  - Free-form `audit_memo.md` graded by 5-axis yes/no
    (`audit_quality_axes`), replacing single-shot VLM judge.
  - `species_vocab.json` and `decoy_categories.json` augmented with
    decoys (CLOSED_WITH_DECOYS).

6 dimensions (weights sum 1.0):
  tree_position_recall     >= 0.70   (weight 0.20)
  tree_position_precision  >= 0.65   (weight 0.15)
  height_mae               <= 1.5 m  (weight 0.10)
  decoy_rejection          >= 0.85   (weight 0.15)
  cross_doc_consistency    == 1.0    (weight 0.20)
  audit_quality_axes       >= 0.70   (weight 0.20) LLM yes/no axes
"""
import argparse
import csv
import json
import os
import re
import sys
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT / "_lib"))
from open_vocab_judge import axes_yes_no_score, mllm_image_axes


# Multimodal checklist for `canopy_visualization.png` (top-down +
# perspective Open3D / matplotlib render of detected canopies, trees
# green and decoy poles red). Anti-cheat content checklist — single-
# image yes/no across 4 axes.
CANOPY_AXES = [
    ("real_visualization_not_placeholder",
     "Is this image a real scientific visualization — NOT a single "
     "solid color, blank canvas, all-black image, or a single text "
     "caption?"),
    ("contains_canopy_or_point_cloud_render",
     "Does the image contain a canopy / tree-crown render or a point-"
     "cloud-style visualization (top-down or perspective view), "
     "rather than an unrelated photograph or chart?"),
    ("has_legend_or_color_coding",
     "Does the image include a visible legend or distinct colour "
     "coding (e.g. green tree markers vs red decoy / pole markers) "
     "to distinguish categories?"),
    ("scientific_plot_aesthetic_not_random_image",
     "Does the image look like a scientific plot / 3D render output "
     "(axes, structured layout, plotted markers) rather than a random "
     "photograph or stock image?"),
]


AUDIT_AXES = [
    ("count_and_breakdown",
     "Does the memo state a tree count whose order of magnitude is "
     "plausible relative to GROUND_DATA.csv_row_count (the held-out "
     "GT tree population), and break it down by health (healthy / "
     "stressed / dead / unknown)?"),
    ("specific_tree_ids",
     "Does the memo cite >=3 specific tree_id values (T001, T012, ...) "
     "from the inventory with concrete location (x, y) or species, "
     "not generic adjectives?"),
    ("decoy_effort_cited",
     "Does the memo mention the decoy rejection effort (lamp_post / "
     "power_pole / traffic_sign / sign_gantry / fence_post) with a "
     "rejected count whose order of magnitude is plausible relative "
     "to GROUND_DATA.decoy_total?"),
    ("arborist_tone",
     "Does the memo read as coherent municipal-arborist English with "
     "at least one sentence on which trees need re-inspection or "
     "follow-up?"),
    ("no_fabrication",
     "Is the memo free of fabricated tree_ids or species not in the "
     "supplied closed vocabulary, and not contradicting GROUND_DATA "
     "numerics?"),
]

REQUIRED_FILES = [
    "tree_inventory.csv",
    "pole_rejections.json",
    "canopy_visualization.png",
    "audit_memo.md",
]
TREE_COLS = ["tree_id", "x_m", "y_m", "height_m", "canopy_radius_m",
             "species", "health", "confidence"]
HEALTH_VOCAB = {"healthy", "stressed", "dead", "unknown"}
POLE_KEYS = {"cluster_id", "centroid_x", "centroid_y", "centroid_z", "reason"}

WEIGHTS = {
    "tree_position_recall": 0.18,
    "tree_position_precision": 0.13,
    "height_mae": 0.09,
    "decoy_rejection": 0.13,
    "cross_doc_consistency": 0.18,
    "audit_quality_axes": 0.17,
    "canopy_image_axes": 0.12,
}
THRESH = {
    "tree_position_recall": 0.70,
    "tree_position_precision": 0.65,
    "height_mae": 1.5,
    "decoy_rejection": 0.85,
}
MATCH_RADIUS_M = 2.0


def png_dims(p: Path):
    with open(p, "rb") as f:
        sig = f.read(8)
        if sig[:8] != b"\x89PNG\r\n\x1a\n":
            return None
        f.read(4)
        if f.read(4) != b"IHDR":
            return None
        w = int.from_bytes(f.read(4), "big")
        h = int.from_bytes(f.read(4), "big")
        return w, h


def gating(out: Path):
    errs = []
    for f in REQUIRED_FILES:
        p = out / f
        if not p.exists():
            errs.append(f"missing: {f}")
        elif p.stat().st_size == 0:
            errs.append(f"empty: {f}")
    if errs:
        return False, errs
    with (out / "tree_inventory.csv").open() as f:
        reader = csv.reader(f)
        try:
            header = next(reader)
        except StopIteration:
            return False, ["tree_inventory.csv has no header"]
        if header != TREE_COLS:
            return False, [f"tree_inventory.csv header mismatch: got {header}"]
        rows = list(reader)
    if len(rows) < 15:
        return False, [f"tree_inventory.csv has {len(rows)} rows (< 15)"]
    try:
        json.loads((out / "pole_rejections.json").read_text())
    except Exception as e:
        return False, [f"pole_rejections.json not valid JSON: {e}"]
    return True, []


def parse_inventory(out: Path, vocab):
    rows = []
    with (out / "tree_inventory.csv").open() as f:
        reader = csv.DictReader(f)
        for r in reader:
            try:
                row = {
                    "tree_id": r["tree_id"].strip(),
                    "x": float(r["x_m"]),
                    "y": float(r["y_m"]),
                    "height": float(r["height_m"]),
                    "canopy_r": float(r["canopy_radius_m"]),
                    "species": r["species"].strip(),
                    "health": r["health"].strip(),
                    "confidence": float(r["confidence"]),
                }
            except (ValueError, KeyError) as e:
                row = {"_err": str(e)}
            rows.append(row)
    return rows


def parse_poles(out: Path):
    raw = json.loads((out / "pole_rejections.json").read_text())
    if not isinstance(raw, list):
        return None, "pole_rejections.json must be a JSON array"
    return raw, None


def memo_count_and_words(out: Path):
    text = (out / "audit_memo.md").read_text()
    words = len(text.split())
    nums = [int(m.group(0)) for m in re.finditer(r"\b\d+\b", text)]
    return text, words, nums


def evaluate(out: Path, gt: dict, fixtures: Path):
    species_vocab = set(json.loads((fixtures / "species_vocab.json").read_text())["species"])
    decoy_vocab = set(json.loads((fixtures / "decoy_categories.json").read_text())["categories"])

    inv = parse_inventory(out, species_vocab)
    poles, pole_err = parse_poles(out)
    text, n_words, nums = memo_count_and_words(out)

    inv_ok = [r for r in inv if "_err" not in r]
    pred_xy = np.array([[r["x"], r["y"]] for r in inv_ok]) if inv_ok else np.zeros((0, 2))
    pred_h = np.array([r["height"] for r in inv_ok]) if inv_ok else np.zeros((0,))

    gt_trees = gt["trees"]
    gt_xy = np.array([[t["x"], t["y"]] for t in gt_trees])
    gt_h = np.array([t["height"] for t in gt_trees])

    # tree_position_recall: each GT has any prediction in 2m
    tp_recall_ratio = 0.0
    matched_gt_idx = []
    matched_pred_idx = []
    if len(pred_xy):
        tree_pred = cKDTree(pred_xy)
        recall_hits = 0
        for i, g in enumerate(gt_xy):
            d, idx = tree_pred.query(g, k=1)
            if d <= MATCH_RADIUS_M:
                recall_hits += 1
                matched_gt_idx.append(i)
                matched_pred_idx.append(idx)
        tp_recall_ratio = recall_hits / max(len(gt_xy), 1)
    recall_ok = tp_recall_ratio >= THRESH["tree_position_recall"]

    # tree_position_precision: each prediction has any GT in 2m
    tp_prec_ratio = 0.0
    if len(pred_xy):
        tree_gt = cKDTree(gt_xy)
        prec_hits = 0
        for p in pred_xy:
            d, _ = tree_gt.query(p, k=1)
            if d <= MATCH_RADIUS_M:
                prec_hits += 1
        tp_prec_ratio = prec_hits / max(len(pred_xy), 1)
    prec_ok = tp_prec_ratio >= THRESH["tree_position_precision"]

    # height_mae over TPs (recall-matched pairs); MAE le threshold
    if matched_gt_idx:
        diffs = np.abs(pred_h[matched_pred_idx] - gt_h[matched_gt_idx])
        h_mae = float(diffs.mean())
        h_ratio = max(0.0, 1.0 - h_mae / THRESH["height_mae"])
    else:
        h_mae = float("inf")
        h_ratio = 0.0
    h_ok = h_mae <= THRESH["height_mae"]

    # decoy_rejection: each GT decoy has a pole_rejections entry within 2m with valid reason
    gt_decoys = gt["decoys"]
    gd_xy = np.array([[d["centroid"][0], d["centroid"][1]] for d in gt_decoys])
    rej_xy = []
    valid_reason = []
    rej_errs = []
    for i, p in enumerate(poles or []):
        if not isinstance(p, dict):
            rej_errs.append(f"entry {i} not an object")
            continue
        if not POLE_KEYS.issubset(p.keys()):
            rej_errs.append(f"entry {i} missing keys: {POLE_KEYS - p.keys()}")
            continue
        try:
            x = float(p["centroid_x"]); y = float(p["centroid_y"])
        except (TypeError, ValueError):
            rej_errs.append(f"entry {i} non-numeric centroid")
            continue
        rej_xy.append([x, y])
        valid_reason.append(p["reason"] in decoy_vocab)
    rej_xy = np.array(rej_xy) if rej_xy else np.zeros((0, 2))

    decoy_hits = 0
    if len(rej_xy) and len(gd_xy):
        rej_tree = cKDTree(rej_xy)
        for g in gd_xy:
            d, idx = rej_tree.query(g, k=1)
            if d <= MATCH_RADIUS_M and valid_reason[idx]:
                decoy_hits += 1
    decoy_ratio = decoy_hits / max(len(gd_xy), 1)
    decoy_ok = decoy_ratio >= THRESH["decoy_rejection"]

    # cross_doc_consistency: schema + closed vocab + ID + memo + PNG size
    cd_errs = []
    if any("_err" in r for r in inv):
        cd_errs.append(f"{sum(1 for r in inv if '_err' in r)} bad CSV rows")
    bad_species = [r["species"] for r in inv_ok if r["species"] not in species_vocab]
    if bad_species:
        cd_errs.append(f"species out of vocab: {sorted(set(bad_species))[:5]}")
    bad_health = [r["health"] for r in inv_ok if r["health"] not in HEALTH_VOCAB]
    if bad_health:
        cd_errs.append(f"health out of vocab: {sorted(set(bad_health))[:5]}")
    ids = [r["tree_id"] for r in inv_ok]
    if len(ids) != len(set(ids)):
        cd_errs.append("tree_id not unique")
    if any(not re.fullmatch(r"T\d{3,}", i) for i in ids):
        cd_errs.append("tree_id format must be T### (>=3 digits)")
    pos_pairs = [(round(r["x"], 1), round(r["y"], 1)) for r in inv_ok]
    if len(pos_pairs) != len(set(pos_pairs)):
        cd_errs.append("duplicate (x,y) within 0.1 m bin")
    rej_errs.extend(rej_errs)
    if rej_errs:
        cd_errs.extend(rej_errs[:5])
    bad_reasons = [p.get("reason") for p in (poles or []) if isinstance(p, dict) and p.get("reason") not in decoy_vocab]
    if bad_reasons:
        cd_errs.append(f"reason out of vocab: {sorted(set(bad_reasons))[:5]}")
    if len(poles or []) < 8:
        cd_errs.append(f"pole_rejections has {len(poles or [])} entries (< 8)")
    if n_words > 400:
        cd_errs.append(f"audit_memo.md word count = {n_words} (> 400)")
    if len(inv_ok) not in nums:
        cd_errs.append(f"audit_memo.md does not state CSV row count {len(inv_ok)}")
    png = png_dims(out / "canopy_visualization.png")
    if png is None:
        cd_errs.append("canopy_visualization.png not a PNG")
    elif png[0] < 1280 or png[1] < 960:
        cd_errs.append(f"canopy_visualization.png too small: {png}")

    return {
        "tree_position_recall": {"ratio": round(tp_recall_ratio, 3),
                                 "n_gt": len(gt_xy), "n_hit": len(matched_gt_idx),
                                 "ok": bool(recall_ok)},
        "tree_position_precision": {"ratio": round(tp_prec_ratio, 3),
                                    "n_pred": len(pred_xy),
                                    "ok": bool(prec_ok)},
        "height_mae": {"mae": None if not np.isfinite(h_mae) else round(h_mae, 3),
                       "ratio": round(h_ratio, 3),
                       "n_matched": len(matched_gt_idx),
                       "ok": bool(h_ok)},
        "decoy_rejection": {"ratio": round(decoy_ratio, 3),
                            "n_gt": len(gd_xy), "n_hit": decoy_hits,
                            "ok": bool(decoy_ok)},
        "cross_doc_consistency": {"errors": cd_errs[:8],
                                  "n_errors": len(cd_errs),
                                  "ok": len(cd_errs) == 0},
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
        print(json.dumps({"pass": False, "stage": "gating", "errors": errs}, indent=2))
        return
    gt = json.loads(Path(args.gt).read_text())
    res = evaluate(out, gt, fix)
    # IMPORTANT: only ship held-out GT-side numbers to the LLM judge.
    # Including grader-recomputed agent-derived counts (n_hit / n_pred /
    # decoy_hits) lets a sloppy memo echo those numbers verbatim and get
    # credit despite wrong underlying outputs (circular grading).
    ground_for_memo = {
        "csv_row_count": len(gt.get("trees", [])),
        "decoy_total": len(gt.get("decoys", [])),
    }
    memo_text = (out / "audit_memo.md").read_text(errors="ignore")
    payload = (
        f"GROUND_DATA:\n{json.dumps(ground_for_memo, ensure_ascii=False)}\n\n"
        f"AUDIT_MEMO.MD:\n{memo_text[:6000]}\n"
    )
    aq = axes_yes_no_score(payload, AUDIT_AXES, threshold=0.70)
    aq_ok = bool(aq.get("ok")) or bool(aq.get("skipped"))
    aq_score = float(aq.get("score", 0.0))
    res["audit_quality_axes"] = {"score": round(aq_score, 3),
                                 "axes": aq.get("axes", []),
                                 "skipped": bool(aq.get("skipped")),
                                 "ok": aq_ok,
                                 "threshold": 0.70}
    ca = mllm_image_axes(out / "canopy_visualization.png", CANOPY_AXES,
                          threshold=0.70)
    ca_ok = bool(ca.get("ok")) or bool(ca.get("skipped"))
    ca_score = float(ca.get("score", 0.0))
    res["canopy_image_axes"] = {"score": round(ca_score, 3),
                                "axes": ca.get("axes", []),
                                "skipped": bool(ca.get("skipped")),
                                "ok": ca_ok,
                                "threshold": 0.70}
    all_ok = all(v["ok"] for v in res.values())
    score = 0.0
    for k, w in WEIGHTS.items():
        if k == "cross_doc_consistency":
            score += w * (1.0 if res[k]["ok"] else 0.0)
        elif k == "audit_quality_axes":
            score += w * aq_score
        elif k == "canopy_image_axes":
            score += w * ca_score
        else:
            score += w * res[k]["ratio"]
    score = round(max(0.0, min(score, 1.0)), 3)
    print(json.dumps({"pass": all_ok,
                      "checks": {k: v["ok"] for k, v in res.items()},
                      **res,
                      "score": score, "weights": WEIGHTS},
                     indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
