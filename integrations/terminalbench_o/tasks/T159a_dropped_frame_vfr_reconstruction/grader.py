#!/usr/bin/env python3
"""Deterministic grader for T159a dropped-frame/VFR reconstruction."""
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
DEFECTS = {"dropped_frame", "duplicate_frame", "bad_frame", "timestamp_gap"}

DIMS = {
    "frame_mapping_accuracy": (0.14, 0.75, "ge"),
    "missing_bad_frame_quality": (0.16, 24.0, "ge"),
    "overall_reconstruction_ssim": (0.13, 0.84, "ge"),
    "valid_frame_preservation": (0.12, 28.0, "ge"),
    "timestamp_rmse_ms": (0.10, 8.0, "le"),
    "dropout_detection_f1": (0.10, 0.60, "ge"),
    "stutter_removal_score": (0.09, 0.90, "ge"),
    "temporal_flow_consistency": (0.08, 0.82, "ge"),
    "format_compliance": (0.08, 0.99, "ge"),
}

CRITICAL_DIMS = {
    "frame_mapping_accuracy",
    "overall_reconstruction_ssim",
    "valid_frame_preservation",
    "timestamp_rmse_ms",
    "dropout_detection_f1",
    "stutter_removal_score",
    "format_compliance",
}
PASS_SCORE_THRESHOLD = 0.92


def read_json(p: Path) -> dict:
    return json.loads(p.read_text())


def read_frames(p: Path) -> list[np.ndarray]:
    cap = cv2.VideoCapture(str(p))
    out = []
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
                pass
    return res


def psnr(a: np.ndarray, b: np.ndarray) -> float:
    mse = float(np.mean((a.astype(np.float32) - b.astype(np.float32)) ** 2))
    if mse <= 1e-9:
        return 60.0
    return min(60.0, 20 * math.log10(255.0 / math.sqrt(mse)))


def read_frame_map(p: Path) -> Dict[tuple[str, int], dict]:
    rows: Dict[tuple[str, int], dict] = {}
    if not p.exists():
        return rows
    with p.open() as f:
        for r in csv.DictReader(f):
            try:
                rows[(str(r["clip_id"]), int(r["output_frame_idx"]))] = {
                    "input_frame_idx": int(float(r["input_frame_idx"])),
                    "status": str(r["status"]),
                }
            except Exception:
                continue
    return rows


def frame_map_accuracy(output_dir: Path, reference_dir: Path) -> float:
    gt = read_frame_map(reference_dir / "frame_map_gt.csv")
    pred = read_frame_map(output_dir / "frame_map.csv")
    if not gt:
        return 0.0
    vals = []
    for key, g in gt.items():
        p = pred.get(key)
        if not p:
            vals.append(0.0)
            continue
        status_score = 1.0 if p["status"] == g["status"] else 0.0
        if g["input_frame_idx"] < 0:
            idx_score = 1.0 if p["input_frame_idx"] < 0 else 0.0
        else:
            idx_score = 1.0 if abs(p["input_frame_idx"] - g["input_frame_idx"]) <= 1 else 0.0
        vals.append(0.6 * status_score + 0.4 * idx_score)
    return float(np.mean(vals)) if vals else 0.0


def read_segments(p: Path) -> set[tuple[str, int, str]]:
    rows: set[tuple[str, int, str]] = set()
    if not p.exists():
        return rows
    with p.open() as f:
        for r in csv.DictReader(f):
            try:
                typ = str(r["defect_type"])
                if typ not in DEFECTS:
                    continue
                for i in range(int(r["frame_start"]), int(r["frame_end"]) + 1):
                    rows.add((str(r["clip_id"]), i, typ))
            except Exception:
                continue
    return rows


def segment_f1(output_dir: Path, reference_dir: Path) -> float:
    gt = read_segments(reference_dir / "dropout_segments_gt.csv")
    pred = read_segments(output_dir / "dropout_segments.csv")
    if not gt:
        return 1.0 if not pred else 0.0
    tp = len(gt & pred)
    prec = tp / len(pred) if pred else 0.0
    rec = tp / len(gt) if gt else 0.0
    return 2 * prec * rec / (prec + rec) if prec + rec else 0.0


def issue_frame_sets(reference_dir: Path) -> tuple[set[tuple[str, int]], set[tuple[str, int]]]:
    all_issue = set()
    normal_issue = set()
    with (reference_dir / "dropout_segments_gt.csv").open() as f:
        for r in csv.DictReader(f):
            try:
                typ = str(r["defect_type"])
                frames = range(int(r["frame_start"]), int(r["frame_end"]) + 1)
                for i in frames:
                    all_issue.add((str(r["clip_id"]), i))
                    if typ in {"dropped_frame", "bad_frame"}:
                        normal_issue.add((str(r["clip_id"]), i))
            except Exception:
                continue
    return all_issue, normal_issue


def timestamp_rmse(output_dir: Path, reference_dir: Path) -> float:
    gt = {}
    with (reference_dir / "timestamp_repair_gt.csv").open() as f:
        for r in csv.DictReader(f):
            try:
                gt[(str(r["clip_id"]), int(r["output_frame_idx"]))] = float(r["corrected_pts_ms"])
            except Exception:
                continue
    pred = {}
    p = output_dir / "timestamp_repair.csv"
    if p.exists():
        with p.open() as f:
            for r in csv.DictReader(f):
                try:
                    pred[(str(r["clip_id"]), int(r["output_frame_idx"]))] = float(r["corrected_pts_ms"])
                except Exception:
                    continue
    if not gt:
        return 999.0
    vals = []
    for key, val in gt.items():
        vals.append((pred.get(key, val + 1000.0) - val) ** 2)
    return float(math.sqrt(np.mean(vals))) if vals else 999.0


def stutter_score(frames: list[np.ndarray]) -> float:
    if len(frames) < 3:
        return 0.0
    vals = []
    for a, b in zip(frames, frames[1:]):
        ag = cv2.cvtColor(cv2.resize(a, (128, 72)), cv2.COLOR_BGR2GRAY)
        bg = cv2.cvtColor(cv2.resize(b, (128, 72)), cv2.COLOR_BGR2GRAY)
        vals.append(float(ssim(ag, bg, data_range=255)))
    duplicate_rate = float(np.mean([v > 0.9985 for v in vals])) if vals else 1.0
    return max(0.0, min(1.0, 1.0 - duplicate_rate / 0.12))


def temporal_flow_score(clean: list[np.ndarray], out: list[np.ndarray], idxs: list[int]) -> float:
    vals = []
    for i, j in zip(idxs, idxs[1:]):
        cd = np.abs(clean[j].astype(np.float32) - clean[i].astype(np.float32))
        od = np.abs(out[j].astype(np.float32) - out[i].astype(np.float32))
        err = float(np.mean(np.abs(cd - od)))
        vals.append(max(0.0, min(1.0, 1.0 - err / 34.0)))
    return float(np.mean(vals)) if vals else 0.0


def metric_ok(name: str, value: float) -> bool:
    _, thr, direction = DIMS[name]
    return value >= thr if direction == "ge" else value <= thr


def metric_credit(name: str, value: float) -> float:
    _, thr, direction = DIMS[name]
    if direction == "ge":
        return max(0.0, min(1.0, value / max(thr, 1e-9)))
    if value <= 1e-9:
        return 1.0
    return max(0.0, min(1.0, thr / value))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-dir", type=Path, default=Path("/workspace/output"))
    ap.add_argument("--fixtures-dir", type=Path, default=TASK_DIR / "fixtures")
    ap.add_argument("--reference-dir", type=Path, default=TASK_DIR / "reference")
    ap.add_argument("--report", type=Path, default=TASK_DIR / "grader_report.json")
    args = ap.parse_args()

    clips = read_json(args.fixtures_dir / "clips.json")["clips"]
    issue_all, issue_repair = issue_frame_sets(args.reference_dir)
    missing_psnr = []
    valid_psnr = []
    ssims = []
    stutters = []
    temporals = []
    formats = []
    for c in clips:
        cid = c["id"]
        clean = read_frames(args.reference_dir / "clean" / f"{cid}.mp4")
        out = read_frames(args.output_dir / "repaired" / f"{cid}.mp4")
        n = min(len(clean), len(out), int(c["frame_count"]))
        if n == 0:
            continue
        idxs = list(range(0, n, max(1, n // 72)))
        for i in idxs:
            g1 = cv2.cvtColor(clean[i], cv2.COLOR_BGR2GRAY)
            g2 = cv2.cvtColor(out[i], cv2.COLOR_BGR2GRAY)
            ssims.append(float(ssim(g1, g2, data_range=255)))
            if (cid, i) in issue_repair:
                missing_psnr.append(psnr(clean[i], out[i]))
            elif (cid, i) not in issue_all:
                valid_psnr.append(psnr(clean[i], out[i]))
        stutters.append(stutter_score(out[:n]))
        temporals.append(temporal_flow_score(clean[:n], out[:n], idxs))
        info = ffprobe_video(args.output_dir / "repaired" / f"{cid}.mp4")
        checks = [
            info["width"] == W,
            info["height"] == H,
            abs(info["fps"] - FPS) <= 0.2,
            abs(info["frames"] - int(c["frame_count"])) <= 1,
        ]
        formats.append(sum(1.0 for x in checks if x) / len(checks))

    values = {
        "frame_mapping_accuracy": frame_map_accuracy(args.output_dir, args.reference_dir),
        "missing_bad_frame_quality": float(np.mean(missing_psnr)) if missing_psnr else 0.0,
        "overall_reconstruction_ssim": float(np.mean(ssims)) if ssims else 0.0,
        "valid_frame_preservation": float(np.mean(valid_psnr)) if valid_psnr else 0.0,
        "timestamp_rmse_ms": timestamp_rmse(args.output_dir, args.reference_dir),
        "dropout_detection_f1": segment_f1(args.output_dir, args.reference_dir),
        "stutter_removal_score": float(np.mean(stutters)) if stutters else 0.0,
        "temporal_flow_consistency": float(np.mean(temporals)) if temporals else 0.0,
        "format_compliance": float(np.mean(formats)) if formats else 0.0,
    }

    dims = {}
    score = 0.0
    critical_ok = True
    for name, value in values.items():
        weight, threshold, direction = DIMS[name]
        ok = metric_ok(name, value)
        if name in CRITICAL_DIMS:
            critical_ok = critical_ok and ok
        score += weight * metric_credit(name, value)
        dims[name] = {
            "value": round(value, 4),
            "threshold": threshold,
            "direction": direction,
            "ok": ok,
            "weight": weight,
            "critical": name in CRITICAL_DIMS,
        }
    score = round(score, 4)
    passed = bool(critical_ok and score >= PASS_SCORE_THRESHOLD)
    report = {
        "pass": passed,
        "score": score,
        "dims": dims,
        "pass_policy": {
            "critical_dims": sorted(CRITICAL_DIMS),
            "non_critical_dims": sorted(set(DIMS) - CRITICAL_DIMS),
            "score_threshold": PASS_SCORE_THRESHOLD,
        },
    }
    args.report.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
