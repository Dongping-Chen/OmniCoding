#!/usr/bin/env python3
"""Deterministic grader for T153a video stabilization."""
from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
from pathlib import Path
from typing import Dict

import cv2  # type: ignore
import numpy as np
from skimage.metrics import structural_similarity as ssim  # type: ignore

TASK_DIR = Path(__file__).resolve().parent
W, H = 426, 240
FPS = 24.0

DIMS = {
    "reference_alignment_ssim": (0.22, 0.88, "ge"),
    "motion_path_rmse": (0.16, 4.0, "le"),
    "jitter_reduction": (0.16, 0.45, "ge"),
    "crop_retention": (0.12, 0.82, "ge"),
    "border_artifacts": (0.10, 0.96, "ge"),
    "content_preservation": (0.08, 0.97, "ge"),
    "crop_path_consistency": (0.08, 0.98, "ge"),
    "format_compliance": (0.08, 0.99, "ge"),
}


def read_json(p: Path) -> dict:
    return json.loads(p.read_text())


def read_frames(p: Path) -> list[np.ndarray]:
    cap = cv2.VideoCapture(str(p))
    out: list[np.ndarray] = []
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        if fr.shape[1] != W or fr.shape[0] != H:
            fr = cv2.resize(fr, (W, H), interpolation=cv2.INTER_AREA)
        out.append(fr)
    cap.release()
    return out


def ffprobe_video(p: Path) -> dict:
    try:
        raw = subprocess.check_output([
            "ffprobe", "-v", "error", "-show_entries",
            "stream=codec_type,width,height,r_frame_rate,nb_frames",
            "-of", "json", str(p)
        ], timeout=30)
        info = json.loads(raw)
    except Exception:
        return {"width": 0, "height": 0, "fps": 0.0, "frames": 0}
    res = {"width": 0, "height": 0, "fps": 0.0, "frames": 0}
    for st in info.get("streams", []):
        if st.get("codec_type") == "video":
            res["width"] = int(st.get("width", 0))
            res["height"] = int(st.get("height", 0))
            n, _, d = st.get("r_frame_rate", "0/1").partition("/")
            res["fps"] = float(n) / max(float(d), 1.0)
            try:
                res["frames"] = int(st.get("nb_frames", 0))
            except Exception:
                res["frames"] = 0
    return res


def parse_motion(p: Path) -> Dict[tuple[str, int], dict]:
    res: Dict[tuple[str, int], dict] = {}
    if not p.exists():
        return res
    with p.open() as f:
        for row in csv.DictReader(f):
            try:
                res[(row["seq_id"], int(row["frame_idx"]))] = row
            except Exception:
                continue
    return res


def parse_crop(p: Path) -> Dict[tuple[str, int], dict]:
    res: Dict[tuple[str, int], dict] = {}
    if not p.exists():
        return res
    with p.open() as f:
        for row in csv.DictReader(f):
            try:
                res[(row["seq_id"], int(row["frame_idx"]))] = row
            except Exception:
                continue
    return res


def frame_ssim(a: np.ndarray, b: np.ndarray) -> float:
    ag = cv2.cvtColor(a, cv2.COLOR_BGR2GRAY)
    bg = cv2.cvtColor(b, cv2.COLOR_BGR2GRAY)
    return float(ssim(ag, bg, data_range=255))


def global_shifts(frames: list[np.ndarray]) -> list[float]:
    vals: list[float] = []
    prev = None
    for fr in frames:
        gray = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY).astype(np.float32)
        gray = cv2.resize(gray, (W // 2, H // 2), interpolation=cv2.INTER_AREA)
        if prev is not None:
            shift, response = cv2.phaseCorrelate(prev, gray)
            vals.append(float(np.hypot(shift[0], shift[1])))
        prev = gray
    if len(vals) < 5:
        return vals
    arr = np.asarray(vals)
    # High-frequency component after subtracting a small moving average.
    k = 5
    smooth = np.convolve(arr, np.ones(k) / k, mode="same")
    return list(np.abs(arr - smooth))


def border_score(frames: list[np.ndarray]) -> float:
    vals = []
    for fr in frames[::max(1, len(frames) // 30)]:
        edge = np.concatenate([
            fr[:4, :, :].reshape(-1, 3), fr[-4:, :, :].reshape(-1, 3),
            fr[:, :4, :].reshape(-1, 3), fr[:, -4:, :].reshape(-1, 3)
        ], axis=0)
        dark = np.mean(np.mean(edge, axis=1) < 5)
        vals.append(1.0 - float(dark))
    return float(np.mean(vals)) if vals else 0.0


def content_preservation(frames: list[np.ndarray]) -> float:
    if len(frames) < 3:
        return 0.0
    vals = []
    static = []
    var = []
    prev = None
    for fr in frames[::max(1, len(frames) // 80)]:
        g = cv2.cvtColor(cv2.resize(fr, (128, 72)), cv2.COLOR_BGR2GRAY)
        var.append(float(np.var(g)))
        if prev is not None:
            v = float(ssim(prev, g, data_range=255))
            vals.append(v)
            static.append(v > 0.999)
        prev = g
    checks = [
        (float(np.median(vals)) if vals else 0.0) > 0.10,
        (float(np.mean(static)) if static else 1.0) < 0.25,
        (float(np.mean(var)) if var else 0.0) > 20.0,
    ]
    return sum(1.0 for x in checks if x) / len(checks)


def metric_ok(name: str, value: float) -> bool:
    _, thr, direction = DIMS[name]
    return value >= thr if direction == "ge" else value <= thr


def metric_credit(name: str, value: float) -> float:
    _, thr, direction = DIMS[name]
    if direction == "ge":
        return max(0.0, min(1.0, value / max(thr, 1e-9)))
    return max(0.0, min(1.0, thr / max(value, 1e-9)))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-dir", type=Path, default=Path("/workspace/output"))
    ap.add_argument("--fixtures-dir", type=Path, default=TASK_DIR / "fixtures")
    ap.add_argument("--reference-dir", type=Path, default=TASK_DIR / "reference")
    ap.add_argument("--report", type=Path, default=TASK_DIR / "grader_report.json")
    args = ap.parse_args()

    seqs = read_json(args.fixtures_dir / "sequences.json")["sequences"]
    pred_motion = parse_motion(args.output_dir / "motion_path.csv")
    gt_motion = parse_motion(args.reference_dir / "motion_gt.csv")
    crops = parse_crop(args.output_dir / "crop_window.csv")

    align_vals = []
    rmse_terms = []
    jitter_vals = []
    crop_area_vals = []
    crop_valid_vals = []
    border_vals = []
    content_vals = []
    format_vals = []

    for s in seqs:
        sid = s["id"]
        clean = read_frames(args.reference_dir / "clean" / f"{sid}.mp4")
        shaky = read_frames(args.fixtures_dir / "shaky" / f"{sid}.mp4")
        out = read_frames(args.output_dir / "stabilized" / f"{sid}.mp4")
        n = min(len(clean), len(out), int(s["frame_count"]))
        if n == 0:
            align_vals.append(0.0)
            jitter_vals.append(0.0)
            border_vals.append(0.0)
            content_vals.append(0.0)
            format_vals.append(0.0)
            continue
        idxs = list(range(0, n, max(1, n // 48)))
        align_vals.extend(frame_ssim(clean[i], out[i]) for i in idxs)
        in_j = np.mean(global_shifts(shaky[:n])) if len(shaky) else 0.0
        out_j = np.mean(global_shifts(out[:n])) if len(out) else 1e9
        jitter_vals.append(max(0.0, min(1.0, 1.0 - out_j / max(in_j, 1e-6))))
        border_vals.append(border_score(out[:n]))
        content_vals.append(content_preservation(out[:n]))
        for i in idxs:
            g = gt_motion.get((sid, i))
            p = pred_motion.get((sid, i))
            if not g or not p:
                rmse_terms.append(100.0)
            else:
                try:
                    rmse_terms.extend([
                        float(p["dx"]) - float(g["dx"]),
                        float(p["dy"]) - float(g["dy"]),
                        2.0 * (float(p["rotation_deg"]) - float(g["rotation_deg"])),
                        40.0 * (float(p["scale"]) - float(g["scale"])),
                    ])
                except Exception:
                    rmse_terms.append(100.0)
            c = crops.get((sid, i))
            if not c:
                crop_valid_vals.append(0.0)
            else:
                try:
                    x, y, w, h = float(c["x"]), float(c["y"]), float(c["w"]), float(c["h"])
                    valid = x >= 0 and y >= 0 and w > 0 and h > 0 and x + w <= W and y + h <= H
                    crop_valid_vals.append(1.0 if valid else 0.0)
                    crop_area_vals.append(max(0.0, min(1.0, (w * h) / (W * H))))
                except Exception:
                    crop_valid_vals.append(0.0)
        info = ffprobe_video(args.output_dir / "stabilized" / f"{sid}.mp4")
        checks = [
            info["width"] == W,
            info["height"] == H,
            abs(info["fps"] - FPS) <= 0.2,
            abs(info["frames"] - int(s["frame_count"])) <= 1,
        ]
        format_vals.append(sum(1.0 for x in checks if x) / len(checks))

    values = {
        "reference_alignment_ssim": float(np.mean(align_vals)) if align_vals else 0.0,
        "motion_path_rmse": float(np.sqrt(np.mean(np.square(rmse_terms)))) if rmse_terms else 1e9,
        "jitter_reduction": float(np.mean(jitter_vals)) if jitter_vals else 0.0,
        "crop_retention": float(np.mean(crop_area_vals)) if crop_area_vals else 0.0,
        "border_artifacts": float(np.mean(border_vals)) if border_vals else 0.0,
        "content_preservation": float(np.mean(content_vals)) if content_vals else 0.0,
        "crop_path_consistency": float(np.mean(crop_valid_vals)) if crop_valid_vals else 0.0,
        "format_compliance": float(np.mean(format_vals)) if format_vals else 0.0,
    }

    dims = {}
    score = 0.0
    all_ok = True
    for name, value in values.items():
        weight, threshold, direction = DIMS[name]
        ok = metric_ok(name, value)
        all_ok = all_ok and ok
        score += weight * metric_credit(name, value)
        dims[name] = {"value": round(value, 4), "threshold": threshold,
                      "direction": direction, "ok": ok, "weight": weight}
    report = {"pass": bool(all_ok), "score": round(score, 4), "dims": dims}
    args.report.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
