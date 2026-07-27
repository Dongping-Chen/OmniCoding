"""Grader for T49a Action Recognition Dataset Build (v3).

v3 pattern:
  - Closed-vocab action labels (CLOSED_WITH_DECOYS): 8 canonical UCF
    classes that actually appear + 5 decoys (Biking, Bowling,
    BoxingPunchingBag, BreastStroke, CleanAndJerk) that are valid
    label strings but never appear in any GT window. label_vocabulary
    accepts decoys; spotcheck_oracle penalises them.
  - VLM `check_windows_visual` removed: closed-vocab GT windows are
    authoritative; VLM verification was redundant + cross-model bias.
  - Reweighted (sum = 1.000): deterministic structured checks plus one
    VLM clip-content checklist to catch blank/placeholder clips.

Pass criteria:
  - label_vocabulary == 1.0      (every label in allowed set)
  - per-class min count >= 5
  - split integrity == 1.0       (leak-free + train/total near [0.7, 0.85],
                                  with a one-clip integer tolerance)
  - temporal_disjoint >= 0.95    (≤ 5% of pairs overlap > 0.5s on source)
  - spotcheck_oracle >= 6/8      (dominant overlapping label matches GT)
"""
import argparse
import json
import math
import os
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT / "_lib"))
from open_vocab_judge import mllm_video_axes  # noqa: E402

# Multimodal checklist for one action clip (clips/clip_0001.mp4 — first
# clip the records list iterates over). Sample 4 evenly-spaced frames;
# majority vote per axis across frames.
CLIP_CONTENT_AXES = [
    ("real_action_footage_not_placeholder",
     "Is this frame natural action-recognition footage of a human "
     "doing something — NOT a black screen, color bars, test pattern, "
     "or single-color placeholder?"),
    ("human_subject_visible",
     "Is at least one human subject (full body, partial body, or face) "
     "visible in this frame?"),
    ("single_dominant_action_visible",
     "Does this frame show a single dominant action being performed "
     "(e.g. running, lifting, throwing, swimming) rather than an "
     "ambiguous static scene with no clear action?"),
    ("ucf_style_clip_quality",
     "Does the frame look like a typical UCF-101 style action clip "
     "(reasonable resolution, lighting, framing on the actor) rather "
     "than a heavily blurred / pixelated / corrupted frame?"),
]


def ffprobe_dur(p):
    try:
        return float(subprocess.check_output(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "default=nw=1:nk=1", str(p)]).strip())
    except Exception:
        return None


def gating(out: Path, allowed: set):
    errs = []
    for p in ["clips", "labels.jsonl", "splits/train.txt",
              "splits/val.txt", "manifest.json"]:
        if not (out / p).exists():
            errs.append(f"missing {p}")
    if (out / "clips").exists():
        clips = sorted((out / "clips").glob("clip_*.mp4"))
        if len(clips) < 60:
            errs.append(f"only {len(clips)} clips (need ≥ 60)")
        bad_dur = 0
        for c in clips[:200]:
            d = ffprobe_dur(c)
            if d is None or not (2.0 <= d <= 8.0):
                bad_dur += 1
        if bad_dur > 0:
            errs.append(f"{bad_dur} clips with duration outside [2,8]s")
    if (out / "labels.jsonl").exists():
        try:
            with (out / "labels.jsonl").open() as f:
                lines = [json.loads(l) for l in f if l.strip()]
            for r in lines:
                for k in ("clip", "label", "source_start_sec", "source_end_sec"):
                    if k not in r:
                        errs.append(f"labels.jsonl missing {k}")
                        break
        except Exception as e:
            errs.append(f"labels.jsonl parse: {e}")
    return len(errs) == 0, errs


def label_vocabulary(records, allowed):
    used = {r["label"] for r in records}
    extras = used - allowed
    return {"used": sorted(used), "extras": sorted(extras),
            "ratio": 0.0 if extras else 1.0,
            "ok": not extras}


def per_class_count(records):
    c = Counter(r["label"] for r in records)
    min_cnt = min(c.values()) if c else 0
    return {"distribution": dict(c), "min": min_cnt, "ok": min_cnt >= 5}


def split_integrity(out: Path, records):
    train = (out / "splits" / "train.txt").read_text().split()
    val = (out / "splits" / "val.txt").read_text().split()
    train_set, val_set = set(train), set(val)
    overlap = train_set & val_set
    all_clips = {r["clip"] for r in records}
    missing = all_clips - (train_set | val_set)
    extras = (train_set | val_set) - all_clips
    total = len(train_set) + len(val_set)
    ratio = len(train_set) / max(1, total)
    lower_n = max(0, math.ceil(0.70 * total) - 1)
    upper_n = min(total, math.floor(0.85 * total) + 1)
    ratio_ok = lower_n <= len(train_set) <= upper_n
    return {"n_train": len(train_set), "n_val": len(val_set),
            "leak": sorted(overlap), "missing": sorted(missing),
            "extras": sorted(extras), "train_ratio": round(ratio, 3),
            "train_count_bounds": [lower_n, upper_n],
            "ok": (not overlap and not missing and not extras
                   and ratio_ok)}


def temporal_disjoint(records):
    n = len(records)
    if n < 2:
        return {"max_overlap": 0.0, "n_pairs": 0,
                "n_violations": 0, "ratio": 1.0, "ok": True}
    # O(n^2); n is small (60-200)
    violations = 0
    pairs = 0
    max_ov = 0.0
    intervals = sorted([(r["source_start_sec"], r["source_end_sec"], r["clip"])
                        for r in records])
    for i in range(n):
        s_i, e_i, _ = intervals[i]
        for j in range(i + 1, n):
            s_j, e_j, _ = intervals[j]
            if s_j >= e_i:
                break
            ov = min(e_i, e_j) - max(s_i, s_j)
            pairs += 1
            if ov > max_ov:
                max_ov = ov
            if ov > 0.5:
                violations += 1
    ratio = 1.0 - (violations / max(1, pairs))
    return {"max_overlap_sec": round(max_ov, 3),
            "n_pairs": pairs, "n_violations": violations,
            "ratio": round(ratio, 3), "ok": ratio >= 0.95}


def spotcheck_oracle(records, gt_windows):
    matched = 0
    details = []
    for w in gt_windows:
        ws, we, wlbl = w["start_sec"], w["end_sec"], w["label"]
        # find clips overlapping [ws, we]; vote the dominant label by
        # overlap-duration weight
        wt = defaultdict(float)
        for r in records:
            ov = min(r["source_end_sec"], we) - max(r["source_start_sec"], ws)
            if ov > 0:
                wt[r["label"]] += ov
        if not wt:
            details.append({"window": [ws, we], "gt": wlbl,
                            "dominant": None, "ok": False})
            continue
        dom = max(wt.items(), key=lambda x: x[1])[0]
        ok = dom == wlbl
        if ok:
            matched += 1
        details.append({"window": [ws, we], "gt": wlbl,
                        "dominant": dom, "ok": ok})
    return {"matched": matched, "n_gt": len(gt_windows),
            "ratio": matched / max(1, len(gt_windows)),
            "details": details, "ok": matched >= 6}


# v3: broad VLM `check_windows_visual` removed. spotcheck_oracle
# (deterministic GT match) checks action/window semantics; the remaining
# MLLM clip checklist only guards against blank or placeholder video files.


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", required=True)
    ap.add_argument("--label_defs", required=True)
    ap.add_argument("--gt", required=True)
    args = ap.parse_args()
    out = Path(args.output)
    allowed = set(json.load(open(args.label_defs))["labels"])
    ok, errs = gating(out, allowed)
    if not ok:
        print(json.dumps({"pass": False, "stage": "gating",
                           "errors": errs}, indent=2))
        return
    records = [json.loads(l) for l in (out / "labels.jsonl").open() if l.strip()]
    gt_windows = json.load(open(args.gt))["windows"]

    lv = label_vocabulary(records, allowed)
    bal = per_class_count(records)
    si = split_integrity(out, records)
    td = temporal_disjoint(records)
    sc = spotcheck_oracle(records, gt_windows)

    sample_clip_rel = (records[0].get("clip") if records else None) or "clips/clip_0001.mp4"
    cv = mllm_video_axes(out / sample_clip_rel, CLIP_CONTENT_AXES,
                          n_frames=4, threshold=0.70)
    cv_ok = bool(cv.get("ok", False))

    # v3 weights (sum = 1.000) — deterministic dims + MLLM clip checklist.
    score = (
        0.15 * lv["ratio"]
        + 0.10 * (1 if bal["ok"] else bal["min"] / 5)
        + 0.15 * (1 if si["ok"] else 0)
        + 0.15 * td["ratio"]
        + 0.30 * sc["ratio"]
        + 0.15 * cv.get("score", 0.0)
    )
    print(json.dumps({
        "pass": (lv["ok"] and bal["ok"] and si["ok"] and td["ok"]
                 and sc["ok"] and cv_ok),
        "checks": {"label_vocabulary_ok": lv["ok"],
                    "balance_ok": bal["ok"],
                    "split_integrity_ok": si["ok"],
                    "temporal_disjoint_ok": td["ok"],
                    "spotcheck_oracle_ok": sc["ok"],
                    "clip_video_axes_ok": cv_ok},
        "weights": {
            "label_vocabulary": 0.15,
            "balance": 0.10,
            "split_integrity": 0.15,
            "temporal_disjoint": 0.15,
            "spotcheck_oracle": 0.30,
            "clip_video_axes": 0.15,
        },
        "label_vocabulary": lv,
        "balance": bal,
        "split_integrity": si,
        "temporal_disjoint": td,
        "spotcheck_oracle": sc,
        "clip_video_axes": {
            "axes": cv.get("axes", []),
            "score": cv.get("score", 0.0),
            "ok": cv_ok,
            "threshold": 0.70,
            "sampled_clip": sample_clip_rel,
        },
        "score_so_far": round(score, 3),
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
