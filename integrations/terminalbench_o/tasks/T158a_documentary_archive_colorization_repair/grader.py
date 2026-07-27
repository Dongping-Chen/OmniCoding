#!/usr/bin/env python3
"""Deterministic grader for T158a archive colorization and defect repair."""
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
    "color_psnr": (0.16, 23.5, "ge"),
    "color_ssim": (0.11, 0.80, "ge"),
    "deltae_score": (0.09, 0.72, "ge"),
    "defect_mask_f1": (0.12, 0.55, "ge"),
    "defect_region_repair": (0.12, 0.78, "ge"),
    "flicker_curve_rmse": (0.10, 0.08, "le"),
    "lut_param_rmse": (0.08, 0.08, "le"),
    "temporal_color_consistency": (0.09, 0.82, "ge"),
    "structure_preservation": (0.06, 0.76, "ge"),
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


def deltae_score(a: np.ndarray, b: np.ndarray) -> float:
    al = cv2.cvtColor(a, cv2.COLOR_BGR2LAB).astype(np.float32)
    bl = cv2.cvtColor(b, cv2.COLOR_BGR2LAB).astype(np.float32)
    err = float(np.mean(np.linalg.norm(al - bl, axis=2)))
    return max(0.0, min(1.0, 1.0 - err / 42.0))


def structure_score(a: np.ndarray, b: np.ndarray) -> float:
    ag = cv2.cvtColor(a, cv2.COLOR_BGR2GRAY)
    bg = cv2.cvtColor(b, cv2.COLOR_BGR2GRAY)
    ea = cv2.Laplacian(ag, cv2.CV_32F)
    eb = cv2.Laplacian(bg, cv2.CV_32F)
    denom = max(float(np.mean(np.abs(ea))), 1.0)
    err = float(np.mean(np.abs(ea - eb))) / denom
    return max(0.0, min(1.0, 1.0 - err / 2.5))


def temporal_color_score(clean: list[np.ndarray], out: list[np.ndarray], idxs: list[int]) -> float:
    vals = []
    for i, j in zip(idxs, idxs[1:]):
        cd = clean[j].astype(np.float32) - clean[i].astype(np.float32)
        od = out[j].astype(np.float32) - out[i].astype(np.float32)
        err = float(np.mean(np.abs(cd - od)))
        vals.append(max(0.0, min(1.0, 1.0 - err / 36.0)))
    return float(np.mean(vals)) if vals else 0.0


def read_mask(p: Path) -> np.ndarray:
    m = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
    if m is None:
        return np.zeros((H, W), dtype=np.uint8)
    if m.shape[1] != W or m.shape[0] != H:
        m = cv2.resize(m, (W, H), interpolation=cv2.INTER_NEAREST)
    return (m > 127).astype(np.uint8)


def mask_f1(output_dir: Path, reference_dir: Path, clips: list[dict]) -> float:
    tp = fp = fn = 0
    for c in clips:
        cid = c["id"]
        n = int(c["frame_count"])
        step = max(1, n // 72)
        for i in range(0, n, step):
            gt = read_mask(reference_dir / "defect_masks" / cid / f"{i:04d}.png")
            pred = read_mask(output_dir / "defect_masks" / cid / f"{i:04d}.png")
            tp += int(np.logical_and(gt == 1, pred == 1).sum())
            fp += int(np.logical_and(gt == 0, pred == 1).sum())
            fn += int(np.logical_and(gt == 1, pred == 0).sum())
    if tp + fp + fn == 0:
        return 1.0
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    return 2 * prec * rec / (prec + rec) if prec + rec else 0.0


def defect_region_repair(output_dir: Path, reference_dir: Path, clips: list[dict]) -> float:
    vals = []
    for c in clips:
        cid = c["id"]
        clean = read_frames(reference_dir / "clean" / f"{cid}.mp4")
        out = read_frames(output_dir / "restored_color" / f"{cid}.mp4")
        n = min(len(clean), len(out), int(c["frame_count"]))
        if n == 0:
            continue
        for i in range(0, n, max(1, n // 48)):
            mask = read_mask(reference_dir / "defect_masks" / cid / f"{i:04d}.png").astype(bool)
            if int(mask.sum()) < 8:
                continue
            err = float(np.mean(np.abs(clean[i][mask].astype(np.float32) - out[i][mask].astype(np.float32))))
            vals.append(max(0.0, min(1.0, 1.0 - err / 55.0)))
    return float(np.mean(vals)) if vals else 1.0


def read_flicker_csv(p: Path) -> Dict[tuple[str, int], float]:
    rows: Dict[tuple[str, int], float] = {}
    if not p.exists():
        return rows
    with p.open() as f:
        for r in csv.DictReader(f):
            try:
                rows[(str(r["clip_id"]), int(r["frame_idx"]))] = float(r["brightness_correction"])
            except Exception:
                continue
    return rows


def flicker_rmse(output_dir: Path, reference_dir: Path) -> float:
    gt = read_flicker_csv(reference_dir / "flicker_gt.csv")
    pred = read_flicker_csv(output_dir / "flicker_curve.csv")
    if not gt:
        return 999.0
    vals = []
    for key, val in gt.items():
        if key not in pred:
            vals.append(2.0)
        else:
            vals.append(pred[key] - val)
    return float(math.sqrt(np.mean(np.square(vals)))) if vals else 999.0


def lut_rmse(output_dir: Path, reference_dir: Path) -> float:
    try:
        gt = read_json(reference_dir / "color_model_gt.json")["clips"]
        pred = read_json(output_dir / "color_lut.json")["clips"]
    except Exception:
        return 999.0
    vals = []
    for cid, g in gt.items():
        p = pred.get(cid)
        if not p:
            vals.extend([10.0] * 12)
            continue
        try:
            gm = np.array(g["inverse_matrix_bgr"], dtype=np.float32)
            pm = np.array(p["inverse_matrix_bgr"], dtype=np.float32)
            gb = np.array(g["inverse_bias_bgr"], dtype=np.float32)
            pb = np.array(p["inverse_bias_bgr"], dtype=np.float32)
            vals.extend((gm - pm).reshape(-1).tolist())
            vals.extend((gb - pb).reshape(-1).tolist())
        except Exception:
            vals.extend([10.0] * 12)
    return float(math.sqrt(np.mean(np.square(vals)))) if vals else 999.0


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
    psnrs = []
    ssims = []
    deltas = []
    structs = []
    temporals = []
    formats = []
    for c in clips:
        cid = c["id"]
        clean = read_frames(args.reference_dir / "clean" / f"{cid}.mp4")
        out = read_frames(args.output_dir / "restored_color" / f"{cid}.mp4")
        n = min(len(clean), len(out), int(c["frame_count"]))
        if n == 0:
            continue
        idxs = list(range(0, n, max(1, n // 48)))
        for i in idxs:
            psnrs.append(psnr(clean[i], out[i]))
            ssims.append(float(ssim(cv2.cvtColor(clean[i], cv2.COLOR_BGR2GRAY),
                                    cv2.cvtColor(out[i], cv2.COLOR_BGR2GRAY), data_range=255)))
            deltas.append(deltae_score(clean[i], out[i]))
            structs.append(structure_score(clean[i], out[i]))
        temporals.append(temporal_color_score(clean[:n], out[:n], idxs))
        info = ffprobe_video(args.output_dir / "restored_color" / f"{cid}.mp4")
        checks = [
            info["width"] == W,
            info["height"] == H,
            abs(info["fps"] - FPS) <= 0.2,
            abs(info["frames"] - int(c["frame_count"])) <= 1,
        ]
        formats.append(sum(1.0 for x in checks if x) / len(checks))

    values = {
        "color_psnr": float(np.mean(psnrs)) if psnrs else 0.0,
        "color_ssim": float(np.mean(ssims)) if ssims else 0.0,
        "deltae_score": float(np.mean(deltas)) if deltas else 0.0,
        "defect_mask_f1": mask_f1(args.output_dir, args.reference_dir, clips),
        "defect_region_repair": defect_region_repair(args.output_dir, args.reference_dir, clips),
        "flicker_curve_rmse": flicker_rmse(args.output_dir, args.reference_dir),
        "lut_param_rmse": lut_rmse(args.output_dir, args.reference_dir),
        "temporal_color_consistency": float(np.mean(temporals)) if temporals else 0.0,
        "structure_preservation": float(np.mean(structs)) if structs else 0.0,
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
