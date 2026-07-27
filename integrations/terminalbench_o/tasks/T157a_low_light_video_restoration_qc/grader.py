#!/usr/bin/env python3
"""Deterministic grader for T157a low-light video restoration."""
from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
from pathlib import Path
from typing import Dict, List, Tuple

import cv2  # type: ignore
import numpy as np
from skimage.metrics import structural_similarity as ssim  # type: ignore

TASK_DIR = Path(__file__).resolve().parent
W, H = 426, 240
FPS = 24.0
VOCAB = {"low_light", "sensor_noise", "motion_blur", "compression", "duplicate_frame"}

DIMS = {
    "restoration_psnr": (0.18, 24.0, "ge"),
    "restoration_ssim": (0.16, 0.78, "ge"),
    "color_exposure_score": (0.12, 0.82, "ge"),
    "detail_preservation": (0.10, 0.72, "ge"),
    "temporal_consistency": (0.12, 0.80, "ge"),
    "defect_detection_f1": (0.12, 0.60, "ge"),
    "curve_plausibility": (0.08, 0.90, "ge"),
    "output_continuity": (0.05, 0.97, "ge"),
    "format_compliance": (0.07, 0.99, "ge"),
}


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


def detail_score(a: np.ndarray, b: np.ndarray) -> float:
    ag = cv2.cvtColor(a, cv2.COLOR_BGR2GRAY)
    bg = cv2.cvtColor(b, cv2.COLOR_BGR2GRAY)
    la = cv2.Laplacian(ag, cv2.CV_32F).var()
    lb = cv2.Laplacian(bg, cv2.CV_32F).var()
    if la <= 1e-6:
        return 1.0
    return max(0.0, min(1.0, 1.0 - abs(lb - la) / max(la, 1e-6)))


def color_score(a: np.ndarray, b: np.ndarray) -> float:
    am = a.reshape(-1, 3).mean(axis=0)
    bm = b.reshape(-1, 3).mean(axis=0)
    err = float(np.mean(np.abs(am - bm))) / 255.0
    return max(0.0, min(1.0, 1.0 - err * 4.0))


def temporal_score(clean: list[np.ndarray], out: list[np.ndarray], idxs: list[int]) -> float:
    vals = []
    for i, j in zip(idxs, idxs[1:]):
        cd = np.abs(clean[j].astype(np.float32) - clean[i].astype(np.float32))
        od = np.abs(out[j].astype(np.float32) - out[i].astype(np.float32))
        err = float(np.mean(np.abs(cd - od)))
        vals.append(max(0.0, min(1.0, 1.0 - err / 35.0)))
    return float(np.mean(vals)) if vals else 0.0


def read_defects(p: Path) -> set[tuple[str, int, str]]:
    rows: set[tuple[str, int, str]] = set()
    if not p.exists():
        return rows
    for line in p.read_text().splitlines():
        if not line.strip():
            continue
        try:
            r = json.loads(line)
            typ = str(r["defect_type"])
            if typ not in VOCAB:
                continue
            for i in range(int(r["frame_start"]), int(r["frame_end"]) + 1):
                rows.add((str(r["clip_id"]), i, typ))
        except Exception:
            continue
    return rows


def defect_f1(pred_p: Path, gt_p: Path) -> float:
    pred = read_defects(pred_p)
    gt = read_defects(gt_p)
    if not gt:
        return 1.0 if not pred else 0.0
    tp = len(pred & gt)
    prec = tp / len(pred) if pred else 0.0
    rec = tp / len(gt) if gt else 0.0
    return 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0


def curve_score(p: Path, clips: list[dict]) -> float:
    if not p.exists():
        return 0.0
    rows: Dict[str, list[dict]] = {}
    with p.open() as f:
        for row in csv.DictReader(f):
            rows.setdefault(row.get("clip_id", ""), []).append(row)
    vals = []
    for c in clips:
        cid = c["id"]
        rr = rows.get(cid, [])
        if len(rr) < int(c["frame_count"]) * 0.95:
            vals.append(0.0)
            continue
        ok = 0
        for r in rr:
            try:
                vals_num = [float(r["exposure_gain"]), float(r["denoise_strength"]),
                            float(r["sharpness_strength"]), float(r.get("confidence", 0.5))]
                if 0.0 <= vals_num[1] <= 1.0 and 0.0 <= vals_num[2] <= 1.0 and 0.0 <= vals_num[3] <= 1.0 and 0.5 <= vals_num[0] <= 8.0:
                    ok += 1
            except Exception:
                pass
        vals.append(ok / max(1, len(rr)))
    return float(np.mean(vals)) if vals else 0.0


def output_continuity(frames: list[np.ndarray]) -> float:
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
    return max(0.0, min(1.0, value / max(thr, 1e-9))) if direction == "ge" else max(0.0, min(1.0, thr / max(value, 1e-9)))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-dir", type=Path, default=Path("/workspace/output"))
    ap.add_argument("--fixtures-dir", type=Path, default=TASK_DIR / "fixtures")
    ap.add_argument("--reference-dir", type=Path, default=TASK_DIR / "reference")
    ap.add_argument("--report", type=Path, default=TASK_DIR / "grader_report.json")
    args = ap.parse_args()

    clips = read_json(args.fixtures_dir / "clips.json")["clips"]
    psnrs = []
    ssims = []
    colors = []
    details = []
    temporals = []
    cont = []
    formats = []
    for c in clips:
        cid = c["id"]
        clean = read_frames(args.reference_dir / "clean" / f"{cid}.mp4")
        out = read_frames(args.output_dir / "restored" / f"{cid}.mp4")
        n = min(len(clean), len(out), int(c["frame_count"]))
        if n == 0:
            continue
        idxs = list(range(0, n, max(1, n // 48)))
        for i in idxs:
            psnrs.append(psnr(clean[i], out[i]))
            ssims.append(float(ssim(cv2.cvtColor(clean[i], cv2.COLOR_BGR2GRAY),
                                    cv2.cvtColor(out[i], cv2.COLOR_BGR2GRAY), data_range=255)))
            colors.append(color_score(clean[i], out[i]))
            details.append(detail_score(clean[i], out[i]))
        temporals.append(temporal_score(clean[:n], out[:n], idxs))
        cont.append(output_continuity(out[:n]))
        info = ffprobe_video(args.output_dir / "restored" / f"{cid}.mp4")
        checks = [
            info["width"] == W,
            info["height"] == H,
            abs(info["fps"] - FPS) <= 0.2,
            abs(info["frames"] - int(c["frame_count"])) <= 1,
        ]
        formats.append(sum(1.0 for x in checks if x) / len(checks))

    values = {
        "restoration_psnr": float(np.mean(psnrs)) if psnrs else 0.0,
        "restoration_ssim": float(np.mean(ssims)) if ssims else 0.0,
        "color_exposure_score": float(np.mean(colors)) if colors else 0.0,
        "detail_preservation": float(np.mean(details)) if details else 0.0,
        "temporal_consistency": float(np.mean(temporals)) if temporals else 0.0,
        "defect_detection_f1": defect_f1(args.output_dir / "defect_map.jsonl", args.reference_dir / "defects_gt.jsonl"),
        "curve_plausibility": curve_score(args.output_dir / "enhancement_curve.csv", clips),
        "output_continuity": float(np.mean(cont)) if cont else 0.0,
        "format_compliance": float(np.mean(formats)) if formats else 0.0,
    }
    dims = {}
    score = 0.0
    all_ok = True
    for name, value in values.items():
        weight, threshold, direction = DIMS[name]
        ok = metric_ok(name, value)
        all_ok = all_ok and ok
        score += weight * metric_credit(name, value)
        dims[name] = {"value": round(value, 4), "threshold": threshold, "direction": direction, "ok": ok, "weight": weight}
    report = {"pass": bool(all_ok), "score": round(score, 4), "dims": dims}
    args.report.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
