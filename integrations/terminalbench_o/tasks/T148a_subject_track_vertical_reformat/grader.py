#!/usr/bin/env python3
"""T148a — deterministic grader for subject-tracking vertical reformat.

Inputs (default --output-dir = /workspace/output):
  vertical/seq_NNN.mp4    portrait-cropped video (270×480)
  crop_path.csv           per (seq_id, frame_idx, crop_x, crop_y, crop_w, crop_h)

Reference:
  reference/gt_bbox.csv   per-frame subject bbox in source coords

Fixtures:
  fixtures/wide/seq_NNN.mp4    source 854×480 video
  fixtures/seqs.json           manifest

Score dimensions (matching task.yaml):
  subject_in_frame_rate   ≥ 0.90  weight 0.30
  subject_containment     ≥ 0.90  weight 0.20
  crop_smoothness         ≤ 12.0  weight 0.20
  bounds_compliance       ≥ 0.99  weight 0.10
  output_continuity       ≥ 0.95  weight 0.10
  format_compliance       ≥ 0.99  weight 0.10
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import cv2  # type: ignore
import numpy as np

TASK_DIR = Path(__file__).resolve().parent

DIMS = [
    ("subject_in_frame_rate", 0.30, 0.90, "ge"),
    ("subject_containment", 0.20, 0.90, "ge"),
    ("crop_smoothness", 0.20, 12.0, "le"),
    ("bounds_compliance", 0.10, 0.99, "ge"),
    ("output_continuity", 0.10, 0.95, "ge"),
    ("format_compliance", 0.10, 0.99, "ge"),
]

EXPECTED_FPS = 24
EXPECTED_OUT_W = 270
EXPECTED_OUT_H = 480
SRC_W = 854
SRC_H = 480


# ---------- helpers --------------------------------------------------------


def _read_json(p: Path) -> dict:
    return json.loads(p.read_text())


def parse_gt_bbox(p: Path) -> Dict[str, Dict[int, Tuple[int, int, int, int]]]:
    res: Dict[str, Dict[int, Tuple[int, int, int, int]]] = {}
    with p.open() as f:
        rdr = csv.DictReader(f)
        for row in rdr:
            sid = row["seq_id"]
            try:
                fi = int(row["frame_idx"])
            except ValueError:
                continue
            xs, ys, ws, hs = row["x"], row["y"], row["w"], row["h"]
            if not (xs and ys and ws and hs):
                continue
            res.setdefault(sid, {})[fi] = (
                int(xs), int(ys), int(ws), int(hs)
            )
    return res


def parse_crop_csv(p: Path) -> Dict[str, Dict[int, Tuple[float, float, float, float]]]:
    res: Dict[str, Dict[int, Tuple[float, float, float, float]]] = {}
    with p.open() as f:
        rdr = csv.DictReader(f)
        for row in rdr:
            sid = row.get("seq_id") or row.get("id")
            if not sid:
                continue
            try:
                fi = int(row["frame_idx"])
                cx = float(row["crop_x"])
                cy = float(row["crop_y"])
                cw = float(row["crop_w"])
                ch = float(row["crop_h"])
            except (KeyError, ValueError):
                continue
            res.setdefault(sid, {})[fi] = (cx, cy, cw, ch)
    return res


def ffprobe_video(p: Path) -> dict:
    """Return {'width','height','fps','nb_frames','has_audio'}."""
    cmd = ["ffprobe", "-v", "error", "-show_entries",
           "stream=index,codec_type,width,height,r_frame_rate,nb_frames,duration",
           "-of", "json", str(p)]
    try:
        out = subprocess.check_output(cmd, timeout=30)
        info = json.loads(out)
    except Exception as e:
        return {"error": str(e)}
    res = {"width": 0, "height": 0, "fps": 0.0, "nb_frames": 0,
           "has_audio": False}
    for st in info.get("streams", []):
        if st.get("codec_type") == "video":
            res["width"] = int(st.get("width", 0))
            res["height"] = int(st.get("height", 0))
            num, _, den = st.get("r_frame_rate", "0/1").partition("/")
            try:
                res["fps"] = float(num) / float(den) if float(den) else 0.0
            except Exception:
                pass
            try:
                res["nb_frames"] = int(st.get("nb_frames", 0))
            except Exception:
                pass
        elif st.get("codec_type") == "audio":
            res["has_audio"] = True
    return res


def gating(out_dir: Path, expected_ids: List[str]) -> List[str]:
    errs: List[str] = []
    if not (out_dir / "vertical").is_dir():
        errs.append("output/vertical/ missing")
    if not (out_dir / "crop_path.csv").exists():
        errs.append("output/crop_path.csv missing")
    missing = [s for s in expected_ids
               if not (out_dir / "vertical" / f"{s}.mp4").exists()]
    if missing:
        errs.append(
            f"{len(missing)} mp4(s) missing under output/vertical/ "
            f"(first: {missing[:3]})"
        )
    return errs


# ---------- metrics --------------------------------------------------------


def containment(subject: Tuple[float, float, float, float],
                crop: Tuple[float, float, float, float]) -> float:
    """Fraction of subject bbox area that lies inside the crop window."""
    sx0, sy0, sw, sh = subject
    cx0, cy0, cw, ch = crop
    sx1, sy1 = sx0 + sw, sy0 + sh
    cx1, cy1 = cx0 + cw, cy0 + ch
    ix0, iy0 = max(sx0, cx0), max(sy0, cy0)
    ix1, iy1 = min(sx1, cx1), min(sy1, cy1)
    iw = max(0.0, ix1 - ix0)
    ih = max(0.0, iy1 - iy0)
    inter = iw * ih
    s_area = sw * sh
    return inter / s_area if s_area > 0 else 0.0


def smoothness_residual_sigma(crop_x: List[float], fps: float) -> float:
    """σ of crop_x[t] minus a 1 Hz Butterworth low-pass version."""
    from scipy.signal import butter, filtfilt  # type: ignore

    x = np.asarray(crop_x, dtype=np.float64)
    if len(x) < 16:
        return 0.0
    nyq = 0.5 * fps
    cutoff = min(1.0 / nyq, 0.95)
    if cutoff <= 0 or cutoff >= 1:
        return float(np.std(x))
    b, a = butter(2, cutoff, btype="low")
    try:
        lp = filtfilt(b, a, x)
    except Exception:
        return float(np.std(x))
    return float(np.std(x - lp))


def consecutive_frame_stats(video_path: Path,
                            max_frames: int = 200) -> dict:
    """For anti-cheat continuity check. Returns:
      - 'median_ssim'  : median SSIM between consecutive frames
      - 'frac_static'  : fraction of pairs with SSIM > 0.999 (duplicate frames)
      - 'mean_var'     : mean per-frame pixel variance (catches all-black)
    """
    from skimage.metrics import structural_similarity as ssim  # type: ignore

    cap = cv2.VideoCapture(str(video_path))
    prev = None
    ssims: List[float] = []
    var_vals: List[float] = []
    while len(ssims) < max_frames:
        ok, frame = cap.read()
        if not ok:
            break
        small = cv2.resize(frame, (128, 128), interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        var_vals.append(float(np.var(gray)))
        if prev is not None:
            try:
                v = ssim(prev, gray, data_range=255)
            except Exception:
                v = 0.0
            ssims.append(float(v))
        prev = gray
    cap.release()
    if not ssims:
        return {"median_ssim": 0.0, "frac_static": 1.0, "mean_var": 0.0}
    arr = np.array(ssims)
    return {
        "median_ssim": float(np.median(arr)),
        "frac_static": float(np.mean(arr > 0.999)),
        "mean_var": float(np.mean(var_vals)) if var_vals else 0.0,
    }


# ---------- main grader ----------------------------------------------------


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-dir", type=Path,
                    default=Path("/workspace/output"))
    ap.add_argument("--reference-dir", type=Path,
                    default=TASK_DIR / "reference")
    ap.add_argument("--fixtures-dir", type=Path,
                    default=TASK_DIR / "fixtures")
    ap.add_argument("--report", type=Path,
                    default=TASK_DIR / "grader_report.json")
    args = ap.parse_args()

    out = args.output_dir
    ref = args.reference_dir
    fix = args.fixtures_dir

    seqs = _read_json(fix / "seqs.json")["sequences"]
    expected_ids = [s["id"] for s in seqs]
    seqs_by_id = {s["id"]: s for s in seqs}

    gating_errs = gating(out, expected_ids)
    if gating_errs:
        report = {"pass": False, "score": 0.0,
                  "gating_errors": gating_errs, "dims": {}}
        args.report.write_text(json.dumps(report, indent=2))
        print(json.dumps(report, indent=2))
        return 1

    gt_bboxes = parse_gt_bbox(ref / "gt_bbox.csv")
    crops = parse_crop_csv(out / "crop_path.csv")

    in_frame_fracs: List[float] = []
    contain_means: List[float] = []
    smoothness_vals: List[float] = []
    bounds_fracs: List[float] = []
    continuity_pass: List[bool] = []
    fmt_pass = 0
    fmt_total = len(seqs)

    per_seq: List[dict] = []

    for seq_meta in seqs:
        sid = seq_meta["id"]
        rec = {"id": sid}
        out_mp4 = out / "vertical" / f"{sid}.mp4"
        info = ffprobe_video(out_mp4)
        rec["video_info"] = info
        n_expected = seq_meta["frame_count"]
        fps = float(seq_meta["fps"])

        fmt_ok = (
            info.get("width", 0) == EXPECTED_OUT_W
            and info.get("height", 0) == EXPECTED_OUT_H
            and abs(info.get("fps", 0.0) - fps) < 0.5
            and info.get("has_audio", False)
            and abs(info.get("nb_frames", 0) - n_expected) <= 2
        )
        if fmt_ok:
            fmt_pass += 1
        rec["format_ok"] = fmt_ok

        gt = gt_bboxes.get(sid, {})
        cr = crops.get(sid, {})
        rec["n_gt_frames"] = len(gt)
        rec["n_crop_frames"] = len(cr)

        # iterate over frames the agent provided that have GT
        common_frames = sorted(set(gt.keys()) & set(cr.keys()))
        if not common_frames:
            rec["error"] = "no overlap between crop_path frames and GT frames"
            in_frame_fracs.append(0.0)
            contain_means.append(0.0)
            smoothness_vals.append(100.0)
            bounds_fracs.append(0.0)
            continuity_pass.append(False)
            per_seq.append(rec)
            continue

        in_frame_count = 0
        contains: List[float] = []
        bounds_count = 0
        crop_xs: List[float] = []
        for fi in common_frames:
            gx, gy, gw, gh = gt[fi]
            cx, cy, cw, ch = cr[fi]
            crop_xs.append(cx)
            # bounds: crop entirely inside [0, SRC_W) × [0, SRC_H)
            if (cx >= 0 and cy >= 0
                    and cx + cw <= SRC_W and cy + ch <= SRC_H
                    and cw > 0 and ch > 0):
                bounds_count += 1
            # subject in central 70%
            sx = gx + gw / 2.0
            sy = gy + gh / 2.0
            x_lo = cx + 0.15 * cw
            x_hi = cx + 0.85 * cw
            y_lo = cy + 0.15 * ch
            y_hi = cy + 0.85 * ch
            if x_lo <= sx <= x_hi and y_lo <= sy <= y_hi:
                in_frame_count += 1
            contains.append(containment((gx, gy, gw, gh), (cx, cy, cw, ch)))

        in_frac = in_frame_count / len(common_frames)
        bounds_frac = bounds_count / len(common_frames)
        mean_contain = float(np.mean(contains)) if contains else 0.0
        sigma = smoothness_residual_sigma(crop_xs, fps)

        in_frame_fracs.append(in_frac)
        contain_means.append(mean_contain)
        smoothness_vals.append(sigma)
        bounds_fracs.append(bounds_frac)

        # continuity (anti-cheat: no all-black frames, no all-duplicate
        # frames, no random noise). Heavy motion in source content is
        # expected — we test for clear pathologies rather than smoothness.
        if fmt_ok:
            cs = consecutive_frame_stats(out_mp4, max_frames=200)
        else:
            cs = {"median_ssim": 0.0, "frac_static": 1.0, "mean_var": 0.0}
        rec["consecutive_stats"] = {
            "median_ssim": round(cs["median_ssim"], 4),
            "frac_static": round(cs["frac_static"], 4),
            "mean_var": round(cs["mean_var"], 2),
        }
        # Pass if: median SSIM > 0.10 (rules out random noise),
        #          frac_static < 0.50 (rules out blatant duplicate-frame
        #            cheating; 0.50 tolerates naturally-low-motion source
        #            content like distant slow vehicles), and
        #          mean_var > 25.0 (rules out all-black or constant frames).
        continuity_pass.append(
            cs["median_ssim"] > 0.10
            and cs["frac_static"] < 0.50
            and cs["mean_var"] > 25.0
        )

        rec.update(
            {
                "in_frame_frac": round(in_frac, 4),
                "mean_containment": round(mean_contain, 4),
                "smoothness_sigma": round(sigma, 4),
                "bounds_frac": round(bounds_frac, 4),
                "common_frames": len(common_frames),
            }
        )
        per_seq.append(rec)

    def _avg(xs: List[float], default: float = 0.0) -> float:
        return float(np.mean(xs)) if xs else default

    agg = {
        "subject_in_frame_rate": _avg(in_frame_fracs, 0.0),
        "subject_containment": _avg(contain_means, 0.0),
        "crop_smoothness": _avg(smoothness_vals, 100.0),
        "bounds_compliance": _avg(bounds_fracs, 0.0),
        "output_continuity": (sum(continuity_pass) / max(len(continuity_pass), 1))
                             if continuity_pass else 0.0,
        "format_compliance": fmt_pass / max(fmt_total, 1),
    }

    dims_report: Dict[str, dict] = {}
    score = 0.0
    for name, weight, thr, direction in DIMS:
        v = float(agg[name])
        ok = v >= thr if direction == "ge" else v <= thr
        if direction == "ge":
            cred = max(0.0, min(1.0, v / max(thr, 1e-9)))
        else:
            cred = max(0.0, min(1.0, thr / max(abs(v), 1e-9)))
        score += weight * (1.0 if ok else cred)
        dims_report[name] = {
            "value": round(v, 4),
            "threshold": thr,
            "direction": direction,
            "pass": bool(ok),
            "weight": weight,
        }

    overall_pass = all(d["pass"] for d in dims_report.values())
    report = {
        "pass": overall_pass,
        "score": round(score, 4),
        "dims": dims_report,
        "per_seq": per_seq,
        "n_sequences": len(seqs),
    }
    args.report.write_text(json.dumps(report, indent=2))
    print(json.dumps({"pass": overall_pass, "score": report["score"],
                      "dims": dims_report}, indent=2))
    return 0 if overall_pass else 1


if __name__ == "__main__":
    sys.exit(main())
