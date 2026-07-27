"""Grader for T65a Yoga Pose Form Audit (v3 — deterministic +
summary-only-LLM).

v3 redesign:
  - Capability dims (pose recall, decoy rejection, keyframe relevance,
    form rule F1, cross_doc) deterministic. pose vocab is closed
    target ∪ decoy; OOV fails cross_doc; decoys directly tested.
  - feedback_quality switches from a single VLM rubric to
    `axes_yes_no_score` over 5 yes/no axes.
  - Reweighted: deterministic dims ~0.82; LLM-axes 0.18.
"""
import argparse
import csv
import json
import os
import re
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT / "_lib"))

from open_vocab_judge import axes_yes_no_score, mllm_image_axes  # noqa: E402

DELIVERABLES = ["pose_timeline.json", "form_audit.csv", "feedback.md"]
# v3 reweight: deterministic dims dominate (~0.82); LLM-axes 0.18.
# +0.07 mllm-axes on sampled keyframe images to defend against blank /
# placeholder / non-yoga keyframe files; trimmed from feedback_quality
# (0.18 -> 0.13) so weights remain summed to 1.0.
WEIGHTS = {"pose_recall": 0.21, "decoy_rejection": 0.12,
           "keyframe_pose_relevance": 0.16, "form_rule_f1": 0.16,
           "cross_doc_consistency": 0.15, "feedback_quality": 0.13,
           "keyframe_content_axes": 0.07}

KEYFRAME_AXES = [
    ("real_frame_not_placeholder",
     "Is this image a real video frame / photograph — NOT a single "
     "solid color, blank canvas, all-black image, or single-text "
     "caption?"),
    ("person_visible",
     "Is at least one person (full or partial body) visible in the "
     "image, occupying a meaningful portion of the frame?"),
    ("recognizable_yoga_pose",
     "Does the person's body posture resemble a recognizable yoga "
     "pose (e.g. warrior, downward-dog, tree, cobra, plank, lunge), "
     "rather than standing idle or sitting in a chair?"),
]

FEEDBACK_AXES = [
    ("pose_form_scores_named",
     "Does feedback.md mention every target pose's average form score "
     "(3 = no rule violated, 0 = all 3 violated) using the pose names "
     "in GROUND_DATA `target_poses`?"),
    ("two_poses_needing_work",
     "Does feedback.md call out the 2 poses needing the most work, "
     "with reasoning grounded in form_audit findings?"),
    ("one_or_two_poses_done_well",
     "Does feedback.md call out 1-2 poses done well, with concrete "
     "praise (alignment, breath, hold, etc.)?"),
    ("coaching_prose",
     "Does feedback.md read as cohesive coaching English — not a CSV "
     "row dump, lorem-ipsum, or placeholder bullets — and stay under "
     "~200 words?"),
    ("self_consistent_with_audit",
     "Are pose names and counts cited in feedback.md consistent with "
     "GROUND_DATA / form_audit (no contradictions)?"),
]
WORD_CAP = 200
VIOLATED_VOCAB = {"yes", "no"}


def gating(out: Path):
    errs = []
    for f in DELIVERABLES:
        p = out / f
        if not p.exists():
            errs.append(f"missing: {f}")
        elif p.stat().st_size == 0:
            errs.append(f"empty: {f}")
    if not (out / "keyframes").exists():
        errs.append("missing: keyframes/ directory")
    return len(errs) == 0, errs


def time_iou(a0, a1, b0, b1):
    inter = max(0.0, min(a1, b1) - max(a0, b0))
    union = max(a1, b1) - min(a0, b0)
    return inter / union if union > 0 else 0.0


def best_match(gt_seg, pred_segs):
    best, best_iou = None, 0.0
    for ps in pred_segs:
        try:
            iou = time_iou(float(gt_seg["t_start"]), float(gt_seg["t_end"]),
                           float(ps["t_start"]), float(ps["t_end"]))
        except (TypeError, ValueError, KeyError):
            continue
        if iou > best_iou:
            best, best_iou = ps, iou
    return best, best_iou


def load_pred_timeline(out: Path):
    try:
        data = json.loads((out / "pose_timeline.json").read_text())
    except json.JSONDecodeError as e:
        return None, [f"pose_timeline.json: invalid JSON ({e})"]
    if not isinstance(data, list):
        return None, ["pose_timeline.json: top-level must be a JSON array"]
    return data, []


def evaluate(out: Path, gt: dict, fixtures: Path):
    pred_segs, parse_errs = load_pred_timeline(out)
    if pred_segs is None:
        return None, parse_errs
    cross_errs = list(parse_errs)
    pose_vocab = set(gt["vocab_target"]) | set(gt["vocab_decoy"])
    target_vocab = set(gt["vocab_target"])
    decoy_vocab = set(gt["vocab_decoy"])
    rule_universe = set(gt["rule_id_universe"])
    duration = float(gt["video_duration_sec"])

    # ---- 1) cross_doc: vocab + bounds + non-overlap on pose_timeline ----
    norm_pred = []
    for i, ps in enumerate(pred_segs):
        if not isinstance(ps, dict):
            cross_errs.append(f"timeline[{i}]: not an object"); continue
        try:
            t0 = float(ps.get("t_start"))
            t1 = float(ps.get("t_end"))
        except (TypeError, ValueError):
            cross_errs.append(f"timeline[{i}]: bad t_start/t_end"); continue
        pose = (ps.get("pose") or "").strip()
        if pose and pose not in pose_vocab:
            cross_errs.append(f"timeline[{i}]: pose out of vocab: {pose!r}")
        if not (0.0 <= t0 < t1 <= duration + 0.5):
            cross_errs.append(f"timeline[{i}]: out-of-range [{t0}, {t1}]")
        sid = ps.get("segment_id")
        norm_pred.append({"segment_id": sid, "pose": pose, "t_start": t0,
                          "t_end": t1, "is_decoy": bool(ps.get("is_decoy")),
                          "keyframe_file": (ps.get("keyframe_file") or "").strip(),
                          "keyframe_t": ps.get("keyframe_t")})
    sorted_p = sorted(norm_pred, key=lambda x: x["t_start"])
    for a, b in zip(sorted_p, sorted_p[1:]):
        if b["t_start"] + 1e-6 < a["t_end"]:
            cross_errs.append(
                f"overlap: seg ending {a['t_end']} > seg starting {b['t_start']}")

    # ---- 2) pose_recall (target poses) ----
    gt_targets = [s for s in gt["segments"] if not s["is_decoy"]]
    gt_decoys = [s for s in gt["segments"] if s["is_decoy"]]
    n_target_hits = 0
    keyframe_hits = 0
    matched_targets = []
    for gs in gt_targets:
        bm, biou = best_match(gs, norm_pred)
        if bm is not None and biou > 0.5 and bm["pose"] == gs["pose"]:
            n_target_hits += 1
            matched_targets.append((gs, bm))
        # keyframe-relevance count
        if bm is not None and biou > 0.5:
            kt = bm["keyframe_t"]
            kf = bm["keyframe_file"]
            try:
                kt_f = float(kt)
            except (TypeError, ValueError):
                kt_f = None
            kf_ok = bool(kf) and (out / kf).exists()
            if (kt_f is not None and gs["t_start"] - 0.05 <= kt_f
                    <= gs["t_end"] + 0.05 and kf_ok):
                keyframe_hits += 1
    pose_recall = n_target_hits / max(1, len(gt_targets))
    keyframe_rel = keyframe_hits / max(1, len(gt_targets))

    # ---- 3) decoy_rejection ----
    decoy_misses = 0
    for gd in gt_decoys:
        bm, biou = best_match(gd, norm_pred)
        if bm is None or biou <= 0.0:
            decoy_misses += 1; continue
        named_decoy = bm["pose"] in decoy_vocab
        if not (bm["is_decoy"] or named_decoy):
            decoy_misses += 1
    decoy_rate = decoy_misses / max(1, len(gt_decoys))

    # ---- 4) form_rule_f1 ----
    fa_rows = []
    try:
        with (out / "form_audit.csv").open(newline="") as fh:
            reader = csv.DictReader(fh)
            cols = set(reader.fieldnames or [])
            need_cols = {"segment_id", "pose", "rule_id", "violated"}
            if not need_cols.issubset(cols):
                cross_errs.append(f"form_audit.csv missing cols: "
                                  f"{sorted(need_cols - cols)}")
            for r in reader:
                fa_rows.append({k: (r.get(k) or "").strip()
                                for k in ("segment_id", "pose", "rule_id",
                                          "violated")})
    except Exception as e:
        cross_errs.append(f"form_audit.csv: {e}")

    # rule_id and violated vocab + segment_id/pose consistency
    pred_lookup = {(ps["segment_id"], ps["pose"]): ps for ps in norm_pred
                   if ps["segment_id"] is not None}
    pred_pose_by_seg = {ps["segment_id"]: ps["pose"] for ps in norm_pred
                        if ps["segment_id"] is not None}
    pred_audit = {}     # (segment_id, rule_id) -> violated bool
    for r in fa_rows:
        rid = r["rule_id"]
        if rid and rid not in rule_universe:
            cross_errs.append(f"form_audit rule_id out of vocab: {rid!r}")
        v = r["violated"].lower()
        if v and v not in VIOLATED_VOCAB:
            cross_errs.append(f"form_audit violated out of vocab: {v!r}")
        try:
            sid = int(r["segment_id"])
        except ValueError:
            cross_errs.append(f"form_audit bad segment_id: {r['segment_id']!r}")
            continue
        if sid in pred_pose_by_seg and r["pose"] != pred_pose_by_seg[sid]:
            cross_errs.append(
                f"form_audit seg {sid} pose mismatch: {r['pose']!r} vs "
                f"{pred_pose_by_seg[sid]!r}")
        pred_audit[(sid, rid)] = (v == "yes")

    # build GT cells: only target segments, only the 3 rules per pose
    tp = fp = fn = 0
    for gs in gt_targets:
        bm, biou = best_match(gs, norm_pred)
        if bm is None or biou <= 0.5:
            for rid in gs["rules_present"]:
                if rid in gs["rules_violated"]:
                    fn += 1
            continue
        pred_sid = bm["segment_id"]
        for rid in gs["rules_present"]:
            gt_v = rid in gs["rules_violated"]
            pr_v = pred_audit.get((pred_sid, rid), False)
            if gt_v and pr_v:
                tp += 1
            elif gt_v and not pr_v:
                fn += 1
            elif (not gt_v) and pr_v:
                fp += 1
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0

    # ---- 5) more cross_doc: feedback word count ----
    fb_text = (out / "feedback.md").read_text(errors="ignore")
    n_words = len(re.findall(r"\S+", fb_text))
    if n_words > WORD_CAP:
        cross_errs.append(f"feedback.md exceeds {WORD_CAP} words: got {n_words}")

    # keyframes referenced by timeline must exist
    for ps in norm_pred:
        kf = ps["keyframe_file"]
        if kf and not (out / kf).exists():
            cross_errs.append(f"missing keyframe file: {kf}")

    cross_ok = len(cross_errs) == 0

    # Memo grading
    pose_scores = {}
    for gs in gt_targets:
        pose = gs["pose"]
        violated_n = 0
        for rid in gs["rules_present"]:
            bm, biou = best_match(gs, norm_pred)
            if bm is not None and biou > 0.5:
                if pred_audit.get((bm["segment_id"], rid), False):
                    violated_n += 1
        pose_scores[pose] = 3 - violated_n
    ground_for_memo = {
        "target_poses": sorted(target_vocab),
        "pose_form_scores": pose_scores,
    }
    payload = (
        f"GROUND_DATA:\n{json.dumps(ground_for_memo, ensure_ascii=False)}\n\n"
        f"FEEDBACK.MD:\n{fb_text[:6000]}\n"
    )
    fb = axes_yes_no_score(payload, FEEDBACK_AXES, threshold=0.70)
    fb["score"] = float(fb.get("score", 0.0))
    fb["ok"] = bool(fb.get("ok", False))

    # Keyframe content axes: sample 1-2 existing keyframe files (those
    # tied to target-pose segments) and run a small VLM yes/no checklist
    # to defend against placeholder / blank / non-yoga keyframe outputs.
    keyframe_candidates = []
    for ps in norm_pred:
        kf = ps.get("keyframe_file", "")
        if kf and not ps.get("is_decoy") and (out / kf).exists():
            keyframe_candidates.append(out / kf)
    sampled = keyframe_candidates[:2]
    if sampled:
        per = []
        for kf_path in sampled:
            r = mllm_image_axes(kf_path, KEYFRAME_AXES, threshold=0.70)
            per.append(r)
        # Average yes_fraction across sampled frames; ok if mean >= 0.70
        scores = [float(r.get("score", 0.0)) for r in per]
        skipped_all = all(bool(r.get("skipped")) for r in per)
        mean_score = sum(scores) / len(scores) if scores else 0.0
        kf_axes_result = {
            "samples": [{"file": str(sampled[i].name),
                          "score": round(scores[i], 3),
                          "axes": per[i].get("axes", []),
                          "skipped": bool(per[i].get("skipped"))}
                         for i in range(len(sampled))],
            "score": round(mean_score, 3),
            "n_sampled": len(sampled),
            "ok": skipped_all or mean_score >= 0.70,
            "threshold": 0.70,
        }
    else:
        kf_axes_result = {"samples": [], "score": 0.0, "n_sampled": 0,
                           "ok": False, "threshold": 0.70,
                           "error": "no usable keyframe files to sample"}

    return {
        "pose_recall":             {"hits": n_target_hits, "n_gt": len(gt_targets),
                                     "value": round(pose_recall, 3),
                                     "ok": pose_recall >= 0.60},
        "decoy_rejection":         {"misses": decoy_misses, "n_decoys": len(gt_decoys),
                                     "value": round(decoy_rate, 3),
                                     "ok": decoy_rate <= 0.34},
        "keyframe_pose_relevance": {"hits": keyframe_hits, "n_gt": len(gt_targets),
                                     "value": round(keyframe_rel, 3),
                                     "ok": keyframe_rel >= 0.70},
        "form_rule_f1":            {"tp": tp, "fp": fp, "fn": fn,
                                     "precision": round(prec, 3),
                                     "recall": round(rec, 3),
                                     "value": round(f1, 3),
                                     "ok": f1 >= 0.50},
        "cross_doc_consistency":   {"errors": cross_errs[:8],
                                     "n_errors": len(cross_errs),
                                     "value": 1.0 if cross_ok else 0.0,
                                     "ok": cross_ok,
                                     "n_words_feedback": n_words},
        "feedback_quality":        {**fb, "value": float(fb.get("score", 0.0))},
        "keyframe_content_axes":   {**kf_axes_result,
                                     "value": float(kf_axes_result.get("score", 0.0))},
    }, []


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", required=True)
    ap.add_argument("--gt", required=True)
    ap.add_argument("--fixtures", default="fixtures")
    a = ap.parse_args()
    out = Path(a.output); fix = Path(a.fixtures)
    ok, errs = gating(out)
    if not ok:
        print(json.dumps({"pass": False, "stage": "gating", "errors": errs},
                          indent=2))
        return
    res, parse_errs = evaluate(out, json.load(open(a.gt)), fix)
    if res is None:
        print(json.dumps({"pass": False, "stage": "parse",
                          "errors": parse_errs}, indent=2))
        return
    all_ok = all(v["ok"] for v in res.values())
    score = round(sum(WEIGHTS[k] * (max(0.0, 1.0 - v["value"])
                                     if k == "decoy_rejection"
                                     else v["value"])
                      for k, v in res.items()), 3)
    print(json.dumps({"pass": all_ok,
                      "checks": {k: v["ok"] for k, v in res.items()},
                      "score": score,
                      "weights": WEIGHTS,
                      "dims": res}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
