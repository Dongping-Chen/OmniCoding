"""Grader for T63a Dashcam Privacy Redaction (v3 — deterministic +
summary-only-LLM).

v3 redesign:
  - Capability dims (target_recall, false_blur_rate, coverage_temporal,
    audit_completeness, cross_doc) all deterministic. class is closed
    {license_plate, face}; OOV class fails cross_doc.
  - compliance_score switches from a single VLM rubric to
    `axes_yes_no_score` over 5 yes/no axes.
  - Reweighted: deterministic dims ~0.82; LLM-axes 0.18.
"""
import argparse, csv, json, os, random, re, sys
from pathlib import Path
import cv2
import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT / "_lib"))

from open_vocab_judge import axes_yes_no_score, mllm_video_axes  # noqa: E402


# MLLM checklist for `redacted.mp4` content quality (privacy + utility).
# 4 evenly-spaced frames, majority vote across frames per axis.
REDACTED_VIDEO_AXES = [
    ("driving_scene_intact",
     "Does this frame still show a recognisable driving scene "
     "(road / vehicles / surroundings) — i.e. the redaction did NOT "
     "blank out the entire frame, and core dashcam content is still "
     "visible?"),
    ("faces_obscured_or_absent",
     "Are any human faces in this frame either NOT visible OR "
     "clearly blurred / pixelated / masked (no sharp identifiable "
     "face features)?"),
    ("license_plates_obscured_or_absent",
     "Are any visible vehicle license plates either NOT readable "
     "(blurred / pixelated / blacked out) OR not present at all in "
     "this frame? Plates should NOT be sharply legible."),
    ("not_completely_blanked",
     "Is this frame NOT entirely solid black / single colour / "
     "fully blurred — driving context (road, vehicles, sky, "
     "buildings) is still discernible somewhere in the frame?"),
]


COMPLIANCE_AXES = [
    ("class_counts_consistent",
     "Does compliance_notes report per-class redacted target counts "
     "(license_plate vs face) consistent with GROUND_DATA "
     "`class_counts` within +/-1?"),
    ("high_freq_target_named",
     "Does compliance_notes call out at least one high-frequency "
     "target by target_id from GROUND_DATA `high_freq_targets` "
     "(last_frame - first_frame >= 30)? If high_freq_targets is empty "
     "answer yes."),
    ("residual_risk_disclaimer",
     "Does compliance_notes include a short residual-privacy-risk "
     "disclaimer (building numbers, reflective surfaces, off-target "
     "identifiers, etc.)?"),
    ("compliance_prose_under_cap",
     "Does compliance_notes read as cohesive compliance English under "
     "~200 words — not a CSV row dump, lorem-ipsum, or placeholder "
     "bullets?"),
    ("self_consistent_with_audit",
     "Are numbers and target_ids cited in compliance_notes consistent "
     "with GROUND_DATA / the audit log (no contradictions)?"),
]

VOCAB_CLASS = {"license_plate", "face"}
DELIVS = ["detections.jsonl", "redacted.mp4",
          "audit_log.csv", "compliance_notes.md"]
AUDIT_FIELDS = {"target_id", "class", "first_frame", "last_frame",
                "max_confidence", "n_frames_present"}
PATCH = 32
LAP_THR = 30.0
NON_GT_PATCHES = 8
COVERAGE_IOU = 0.30


def gating(out: Path):
    errs = []
    for f in DELIVS:
        p = out / f
        if not p.exists():
            errs.append(f"missing {f}")
        elif p.stat().st_size == 0:
            errs.append(f"empty {f}")
    return not errs, errs


def lap_var(p):
    g = cv2.cvtColor(p, cv2.COLOR_BGR2GRAY) if p.ndim == 3 else p
    return float(cv2.Laplacian(g, cv2.CV_64F).var())


def iou(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    inter = (ix2 - ix1) * (iy2 - iy1)
    aa = max(1, (a[2] - a[0]) * (a[3] - a[1]))
    bb = max(1, (b[2] - b[0]) * (b[3] - b[1]))
    return inter / (aa + bb - inter)


def video_meta(path):
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return None
    m = {"width":  int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
         "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
         "fps":    float(cap.get(cv2.CAP_PROP_FPS)),
         "n_frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT))}
    cap.release()
    return m


def load_video(path):
    cap = cv2.VideoCapture(str(path))
    out = []
    while True:
        ok, fr = cap.read()
        if not ok: break
        out.append(fr)
    cap.release()
    return out


def load_dets(path: Path):
    rows, errs = [], []
    for ln, raw in enumerate(path.read_text().splitlines(), 1):
        if not raw.strip(): continue
        try:
            d = json.loads(raw)
        except json.JSONDecodeError as e:
            errs.append(f"detections.jsonl L{ln}: {e.msg}"); continue
        if not all(k in d for k in ("frame", "class", "bbox", "confidence", "target_id")):
            errs.append(f"detections.jsonl L{ln}: missing fields"); continue
        rows.append(d)
    return rows, errs


def sample_non_gt(W, H, gtb, k, rng):
    out, tries = [], 0
    while len(out) < k and tries < k * 30:
        tries += 1
        x = rng.randint(0, W - PATCH); y = rng.randint(0, H - PATCH)
        ok = True
        for bb in gtb:
            if not (x + PATCH <= bb[0] or bb[2] <= x or
                    y + PATCH <= bb[1] or bb[3] <= y):
                ok = False; break
        if ok: out.append((x, y))
    return out


def evaluate(out: Path, gt: dict, fixtures: Path):
    rng = random.Random(2026)
    res, cd_errs = {}, []
    n_frames, W, H = gt["n_frames"], gt["width"], gt["height"]

    dets, det_errs = load_dets(out / "detections.jsonl")
    cd_errs.extend(det_errs)
    pred_by_frame, pred_by_target = {}, {}
    for d in dets:
        cl = d.get("class", "")
        if cl not in VOCAB_CLASS: cd_errs.append(f"class oov: {cl}")
        bb = d.get("bbox")
        if not (isinstance(bb, list) and len(bb) == 4):
            cd_errs.append("bbox not 4-list"); continue
        try: x1, y1, x2, y2 = [int(v) for v in bb]
        except (TypeError, ValueError): cd_errs.append("bbox non-int"); continue
        if not (0 <= x1 < x2 <= W and 0 <= y1 < y2 <= H):
            cd_errs.append(f"bbox oob: {bb}")
        try: fi = int(d["frame"])
        except (TypeError, ValueError): cd_errs.append("frame non-int"); continue
        if not (0 <= fi < n_frames):
            cd_errs.append(f"frame oob: {fi}"); continue
        try:
            cf = float(d["confidence"])
            if not 0.0 <= cf <= 1.0: cd_errs.append("conf oob")
        except (TypeError, ValueError): cd_errs.append("conf nan"); cf = 0.0
        tid = str(d.get("target_id", "")).strip()
        if not tid: cd_errs.append("missing target_id"); continue
        pred_by_frame.setdefault(fi, []).append({
            "class": cl, "bbox": [x1, y1, x2, y2], "tid": tid, "conf": cf})
        pred_by_target.setdefault(tid, []).append(fi)

    # ---- Audit log ----
    audit_rows = list(csv.DictReader((out / "audit_log.csv").open()))
    audit_cols = set(audit_rows[0].keys()) if audit_rows else set()
    miss = AUDIT_FIELDS - audit_cols
    if miss:
        cd_errs.append(f"audit_log.csv missing cols: {sorted(miss)}")

    # ---- Video meta ----
    fix_meta = video_meta(fixtures / "dashcam.mp4")
    red_meta = video_meta(out / "redacted.mp4")
    if red_meta is None:
        cd_errs.append("redacted.mp4 cannot open")
    else:
        if abs(red_meta["n_frames"] - fix_meta["n_frames"]) > 2:
            cd_errs.append(f"redacted n_frames {red_meta['n_frames']} != "
                           f"fixture {fix_meta['n_frames']}")
        if (red_meta["width"], red_meta["height"]) != (fix_meta["width"], fix_meta["height"]):
            cd_errs.append(f"redacted res mismatch: "
                           f"{red_meta['width']}x{red_meta['height']}")
        if abs(red_meta["fps"] - fix_meta["fps"]) > 0.5:
            cd_errs.append(f"redacted fps {red_meta['fps']} != {fix_meta['fps']}")

    # ---- Notes word cap ----
    notes = (out / "compliance_notes.md").read_text(errors="ignore")
    n_words = len(re.findall(r"\b\w+\b", notes))
    if n_words > 200:
        cd_errs.append(f"compliance_notes word count {n_words} > 200")

    # ---- detections.jsonl must have content ----
    if not dets:
        cd_errs.append("detections.jsonl has zero entries")

    # ---- audit_completeness ----
    unique_tids = {tid for tid in pred_by_target if tid}
    a_errs, a_ok = [], bool(unique_tids)
    if not unique_tids: a_errs.append("no detections")
    if len(audit_rows) != len(unique_tids):
        a_errs.append(f"audit_rows={len(audit_rows)} != unique_tids={len(unique_tids)}"); a_ok = False
    audit_tids = set()
    for r in audit_rows:
        tid = (r.get("target_id") or "").strip()
        audit_tids.add(tid)
        try:
            ff, lf, n_p = int(r.get("first_frame", -1)), int(r.get("last_frame", -1)), int(r.get("n_frames_present", -1))
        except ValueError:
            a_errs.append(f"audit {tid}: non-int field"); a_ok = False; continue
        if ff > lf: a_errs.append(f"audit {tid}: first>last"); a_ok = False
        actual = len(set(pred_by_target.get(tid, [])))
        if n_p != actual: a_errs.append(f"audit {tid}: n_present {n_p}!=actual {actual}"); a_ok = False
        if (r.get("class") or "").strip() not in VOCAB_CLASS:
            a_errs.append(f"audit {tid}: bad class"); a_ok = False
    if audit_tids != unique_tids:
        a_errs.append(f"audit/det tid set mismatch (miss {len(unique_tids - audit_tids)}, extra {len(audit_tids - unique_tids)})"); a_ok = False
    res["audit_completeness"] = {"value": 1.0 if a_ok else 0.0,
                                 "n_unique_targets": len(unique_tids),
                                 "n_audit_rows": len(audit_rows),
                                 "errors": a_errs[:5], "ok": a_ok}

    # ---- target_recall + false_blur_rate (decode videos) ----
    if red_meta is None:
        res["target_recall"] = {"value": 0.0, "ok": False, "note": "no video"}
        res["false_blur_rate"] = {"value": 1.0, "ok": False, "note": "no video"}
    else:
        fix_frames = load_video(fixtures / "dashcam.mp4")
        red_frames = load_video(out / "redacted.mp4")
        n_eval = min(len(fix_frames), len(red_frames), n_frames)
        gt_by_frame = {}
        for g in gt["ground_truth"]:
            gt_by_frame.setdefault(g["frame"], []).append(g["bbox"])

        # target_recall: crop full GT bbox, resize to PATCH x PATCH, LapVar < thr
        tr_total = tr_blur = 0
        for g in gt["ground_truth"]:
            fi = g["frame"]
            if fi >= n_eval: continue
            x1, y1, x2, y2 = g["bbox"]
            x1 = max(0, x1); y1 = max(0, y1)
            x2 = min(W, x2); y2 = min(H, y2)
            if x2 - x1 < 3 or y2 - y1 < 3: continue
            crop = red_frames[fi][y1:y2, x1:x2]
            if crop.shape[0] < 3 or crop.shape[1] < 3: continue
            patch = cv2.resize(crop, (PATCH, PATCH), interpolation=cv2.INTER_AREA)
            tr_total += 1
            if lap_var(patch) < LAP_THR: tr_blur += 1
        tr = tr_blur / max(1, tr_total)
        res["target_recall"] = {"value": round(tr, 3),
                                "n_gt": tr_total, "n_blurred": tr_blur,
                                "ok": tr >= 0.70}

        # false_blur_rate (paired)
        fb_pool = fb_blur = 0
        stride = max(1, n_eval // 200)
        for fi in range(0, n_eval, stride):
            for x, y in sample_non_gt(W, H, gt_by_frame.get(fi, []),
                                      NON_GT_PATCHES, rng):
                fxp = fix_frames[fi][y:y + PATCH, x:x + PATCH]
                rxp = red_frames[fi][y:y + PATCH, x:x + PATCH]
                if fxp.shape[:2] != (PATCH, PATCH): continue
                if lap_var(fxp) < LAP_THR: continue   # naturally smooth
                fb_pool += 1
                if lap_var(rxp) < LAP_THR: fb_blur += 1
        fb = fb_blur / max(1, fb_pool)
        res["false_blur_rate"] = {"value": round(fb, 3),
                                  "n_pool": fb_pool, "n_blurred": fb_blur,
                                  "ok": fb <= 0.10}

    # ---- coverage_temporal ----
    # Pre-build (track, frame) -> bbox lookup
    gt_track_box = {(g["track_id"], g["frame"]): g["bbox"] for g in gt["ground_truth"]}
    cov_tot = cov_hit = 0
    for t in gt["tracks"]:
        cls = t["class"]
        for fi in t["frames"]:
            preds = [p for p in pred_by_frame.get(fi, []) if p["class"] == cls]
            ref = gt_track_box.get((t["track_id"], fi), [0, 0, 0, 0])
            best = max((iou(p["bbox"], ref) for p in preds), default=0.0)
            if best > COVERAGE_IOU: cov_hit += 1
        cov_tot += t["n_frames"]
    cov = cov_hit / max(1, cov_tot)
    res["coverage_temporal"] = {"value": round(cov, 3),
                                "n_tracks": len(gt["tracks"]),
                                "covered_frames": cov_hit,
                                "total_track_frames": cov_tot,
                                "ok": cov >= 0.85}

    res["cross_doc_consistency"] = {"value": 1.0 if not cd_errs else 0.0,
                                    "n_errors": len(cd_errs),
                                    "errors": cd_errs[:8],
                                    "ok": not cd_errs}

    # Memo grading
    class_counts = {"license_plate": 0, "face": 0}
    high_freq = []
    for r in audit_rows:
        c = (r.get("class") or "").strip()
        if c in class_counts:
            class_counts[c] += 1
        try:
            ff = int(r.get("first_frame", -1))
            lf = int(r.get("last_frame", -1))
            if lf - ff >= 30:
                high_freq.append((r.get("target_id") or "").strip())
        except (TypeError, ValueError):
            continue
    ground_for_memo = {
        "class_counts": class_counts,
        "high_freq_targets": high_freq[:10],
        "n_unique_targets": len(unique_tids),
    }
    notes_text = (out / "compliance_notes.md").read_text(errors="ignore")
    payload = (
        f"GROUND_DATA:\n{json.dumps(ground_for_memo, ensure_ascii=False)}\n\n"
        f"COMPLIANCE_NOTES.MD:\n{notes_text[:6000]}\n"
    )
    cs = axes_yes_no_score(payload, COMPLIANCE_AXES, threshold=0.70)
    res["compliance_score"] = {
        "axes": cs.get("axes", []),
        "score": float(cs.get("score", 0.0)),
        "value": float(cs.get("score", 0.0)),
        "threshold": 0.70,
        "ok": bool(cs.get("ok", False)),
    }
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", required=True)
    ap.add_argument("--gt", required=True)
    ap.add_argument("--fixtures", default="fixtures")
    a = ap.parse_args()
    out, fix = Path(a.output), Path(a.fixtures)
    ok, errs = gating(out)
    if not ok:
        print(json.dumps({"pass": False, "stage": "gating", "errors": errs}, indent=2))
        return
    gt = json.load(open(a.gt))
    res = evaluate(out, gt, fix)
    # MLLM video content checklist on rendered redacted.mp4 — verifies
    # both (a) faces / plates actually obscured visually (privacy)
    # and (b) the rest of the driving scene is preserved (utility).
    rv = mllm_video_axes(out / "redacted.mp4", REDACTED_VIDEO_AXES,
                         n_frames=4, threshold=0.70)
    res["redacted_video_axes"] = {
        "axes": rv.get("axes", []),
        "score": float(rv.get("score", 0.0)),
        "value": float(rv.get("score", 0.0)),
        "ok": bool(rv.get("ok", False)),
        "threshold": 0.70,
    }
    all_ok = all(v["ok"] for v in res.values())
    # v3 reweight: deterministic ~0.65, LLM-axes 0.18, MLLM-video 0.17.
    weights = {"target_recall": 0.18, "false_blur_rate": 0.13,
               "coverage_temporal": 0.13, "audit_completeness": 0.10,
               "cross_doc_consistency": 0.13, "compliance_score": 0.18,
               "redacted_video_axes": 0.15}
    parts = {}
    for k, w in weights.items():
        if k == "false_blur_rate":
            v = max(0.0, 1.0 - res[k]["value"] / 0.10) if res[k]["value"] <= 0.10 else 0.0
        elif k in ("audit_completeness", "cross_doc_consistency"):
            v = res[k]["value"]
        elif k in ("compliance_score", "redacted_video_axes"):
            v = float(res[k].get("value", res[k].get("score", 0.0)))
        else:
            v = max(0.0, min(1.0, res[k]["value"]))
        parts[k] = w * v
    score = round(sum(parts.values()), 3)
    print(json.dumps({"pass": all_ok,
                      "checks": {k: v["ok"] for k, v in res.items()},
                      "score": score, "weights": weights, "dims": res},
                     indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
