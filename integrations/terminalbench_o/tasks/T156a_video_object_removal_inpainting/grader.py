#!/usr/bin/env python3
"""Deterministic grader for T156a video object removal/inpainting."""
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
EXPECTED_W = 426
EXPECTED_H = 240
EXPECTED_FPS = 24.0

DIMS = {
    "mask_iou": (0.15, 0.65, "ge"),
    "masked_region_psnr": (0.18, 26.0, "ge"),
    "masked_region_ssim": (0.12, 0.72, "ge"),
    "boundary_consistency": (0.10, 0.82, "ge"),
    "temporal_flicker": (0.12, 0.82, "ge"),
    "unmasked_preservation": (0.13, 0.97, "ge"),
    "track_consistency": (0.08, 0.95, "ge"),
    "output_continuity": (0.05, 0.97, "ge"),
    "format_compliance": (0.07, 0.99, "ge"),
}


def read_json(p: Path) -> dict:
    return json.loads(p.read_text())


def ffprobe_video(p: Path) -> dict:
    try:
        out = subprocess.check_output([
            "ffprobe", "-v", "error", "-show_entries",
            "stream=codec_type,width,height,r_frame_rate,nb_frames,duration",
            "-of", "json", str(p)
        ], timeout=30)
        info = json.loads(out)
    except Exception as e:
        return {"error": str(e)}
    res = {"width": 0, "height": 0, "fps": 0.0, "frames": 0}
    for st in info.get("streams", []):
        if st.get("codec_type") != "video":
            continue
        res["width"] = int(st.get("width", 0))
        res["height"] = int(st.get("height", 0))
        num, _, den = st.get("r_frame_rate", "0/1").partition("/")
        try:
            res["fps"] = float(num) / float(den)
        except Exception:
            pass
        try:
            res["frames"] = int(st.get("nb_frames", 0))
        except Exception:
            res["frames"] = 0
    return res


def read_frames(p: Path, max_frames: int | None = None) -> list[np.ndarray]:
    cap = cv2.VideoCapture(str(p))
    frames: list[np.ndarray] = []
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        if fr.shape[1] != EXPECTED_W or fr.shape[0] != EXPECTED_H:
            fr = cv2.resize(fr, (EXPECTED_W, EXPECTED_H), interpolation=cv2.INTER_AREA)
        frames.append(fr)
        if max_frames is not None and len(frames) >= max_frames:
            break
    cap.release()
    return frames


def read_mask(p: Path) -> np.ndarray:
    if not p.exists():
        return np.zeros((EXPECTED_H, EXPECTED_W), dtype=np.uint8)
    m = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
    if m is None:
        return np.zeros((EXPECTED_H, EXPECTED_W), dtype=np.uint8)
    if m.shape[1] != EXPECTED_W or m.shape[0] != EXPECTED_H:
        m = cv2.resize(m, (EXPECTED_W, EXPECTED_H), interpolation=cv2.INTER_NEAREST)
    return (m > 127).astype(np.uint8)


def mask_bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def bbox_iou(a: tuple[int, int, int, int] | None,
             b: tuple[int, int, int, int] | None) -> float:
    if a is None or b is None:
        return 0.0
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    aa = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    ba = max(0, bx2 - bx1) * max(0, by2 - by1)
    return inter / max(1, aa + ba - inter)


def mask_iou(pred: np.ndarray, gt: np.ndarray) -> float:
    inter = np.logical_and(pred > 0, gt > 0).sum()
    union = np.logical_or(pred > 0, gt > 0).sum()
    return float(inter / union) if union else 1.0


def psnr_region(a: np.ndarray, b: np.ndarray, mask: np.ndarray) -> float:
    pix = mask > 0
    if not pix.any():
        return 60.0
    diff = a.astype(np.float32)[pix] - b.astype(np.float32)[pix]
    mse = float(np.mean(diff * diff))
    if mse <= 1e-9:
        return 60.0
    return min(60.0, float(20 * math.log10(255.0 / math.sqrt(mse))))


def masked_crop_ssim(clean: np.ndarray, out: np.ndarray, mask: np.ndarray) -> float:
    bb = mask_bbox(mask)
    if bb is None:
        return 1.0
    x1, y1, x2, y2 = bb
    pad = 8
    x1, y1 = max(0, x1 - pad), max(0, y1 - pad)
    x2, y2 = min(EXPECTED_W, x2 + pad), min(EXPECTED_H, y2 + pad)
    if x2 - x1 < 8 or y2 - y1 < 8:
        return 0.0
    c = cv2.cvtColor(clean[y1:y2, x1:x2], cv2.COLOR_BGR2GRAY)
    o = cv2.cvtColor(out[y1:y2, x1:x2], cv2.COLOR_BGR2GRAY)
    return float(ssim(c, o, data_range=255))


def boundary_score(clean: np.ndarray, out: np.ndarray, mask: np.ndarray) -> float:
    kernel = np.ones((9, 9), np.uint8)
    dil = cv2.dilate(mask, kernel, iterations=1)
    ring = np.logical_and(dil > 0, mask == 0)
    if not ring.any():
        return 1.0
    diff = np.abs(clean.astype(np.float32)[ring] - out.astype(np.float32)[ring])
    mae = float(np.mean(diff))
    return float(max(0.0, min(1.0, 1.0 - mae / 50.0)))


def unmasked_score(clean: np.ndarray, out: np.ndarray, mask: np.ndarray) -> float:
    pix = mask == 0
    if not pix.any():
        return 1.0
    diff = np.abs(clean.astype(np.float32)[pix] - out.astype(np.float32)[pix])
    mae = float(np.mean(diff))
    return float(max(0.0, min(1.0, 1.0 - mae / 30.0)))


def temporal_flicker_score(clean_frames: list[np.ndarray], out_frames: list[np.ndarray],
                           masks: list[np.ndarray], idxs: list[int]) -> float:
    vals: list[float] = []
    for prev_i, i in zip(idxs, idxs[1:]):
        if prev_i < 0 or i <= prev_i or i >= len(clean_frames):
            continue
        union = np.logical_or(masks[i] > 0, masks[prev_i] > 0)
        if not union.any():
            continue
        cd = np.abs(clean_frames[i].astype(np.float32) - clean_frames[prev_i].astype(np.float32))
        od = np.abs(out_frames[i].astype(np.float32) - out_frames[prev_i].astype(np.float32))
        err = float(np.mean(np.abs(cd[union] - od[union])))
        vals.append(max(0.0, min(1.0, 1.0 - err / 40.0)))
    return float(np.mean(vals)) if vals else 0.0


def parse_tracks(p: Path) -> Dict[tuple[str, int], dict]:
    res: Dict[tuple[str, int], dict] = {}
    if not p.exists():
        return res
    with p.open() as f:
        rdr = csv.DictReader(f)
        for row in rdr:
            try:
                res[(row["video_id"], int(row["frame_idx"]))] = row
            except Exception:
                continue
    return res


def track_consistency(out_dir: Path, tracks: Dict[tuple[str, int], dict],
                      video_id: str, idxs: list[int]) -> float:
    vals: list[float] = []
    for i in idxs:
        row = tracks.get((video_id, i))
        mask_path = out_dir / "nuisance_masks" / video_id / f"frame_{i:06d}.png"
        pred_mask = read_mask(mask_path)
        pred_bb = mask_bbox(pred_mask)
        if row is None or pred_bb is None or not mask_path.exists():
            vals.append(0.0)
            continue
        try:
            row_bb = (int(float(row["x1"])), int(float(row["y1"])),
                      int(float(row["x2"])), int(float(row["y2"])))
            conf = float(row.get("confidence", 0.0))
        except Exception:
            vals.append(0.0)
            continue
        vals.append(1.0 if bbox_iou(pred_bb, row_bb) >= 0.90 and 0.0 <= conf <= 1.0 else 0.0)
    return float(np.mean(vals)) if vals else 0.0


def output_continuity(frames: list[np.ndarray]) -> float:
    if len(frames) < 3:
        return 0.0
    vals = []
    statics = []
    variances = []
    prev = None
    for fr in frames[:: max(1, len(frames) // 80)]:
        gray = cv2.cvtColor(cv2.resize(fr, (128, 72)), cv2.COLOR_BGR2GRAY)
        variances.append(float(np.var(gray)))
        if prev is not None:
            v = float(ssim(prev, gray, data_range=255))
            vals.append(v)
            statics.append(v > 0.999)
        prev = gray
    median_ssim = float(np.median(vals)) if vals else 0.0
    frac_static = float(np.mean(statics)) if statics else 1.0
    mean_var = float(np.mean(variances)) if variances else 0.0
    checks = [
        median_ssim > 0.10,
        frac_static < 0.25,
        mean_var > 20.0,
    ]
    return sum(1.0 for x in checks if x) / len(checks)


def format_score(out_dir: Path, videos: list[dict]) -> float:
    vals = []
    for v in videos:
        info = ffprobe_video(out_dir / "inpainted" / f"{v['id']}.mp4")
        checks = [
            info.get("width") == EXPECTED_W,
            info.get("height") == EXPECTED_H,
            abs(float(info.get("fps", 0.0)) - EXPECTED_FPS) <= 0.2,
            abs(int(info.get("frames", 0)) - int(v["frame_count"])) <= 1,
        ]
        vals.append(sum(1.0 for x in checks if x) / len(checks))
    return float(np.mean(vals)) if vals else 0.0


def metric_ok(name: str, value: float) -> bool:
    _, thr, direction = DIMS[name]
    if direction == "ge":
        return value >= thr
    return value <= thr


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

    videos = read_json(args.fixtures_dir / "videos.json")["videos"]
    tracks = parse_tracks(args.output_dir / "object_tracks.csv")

    vals: Dict[str, list[float]] = {k: [] for k in DIMS}
    for v in videos:
        vid = v["id"]
        clean_frames = read_frames(args.reference_dir / "clean" / f"{vid}.mp4")
        out_frames = read_frames(args.output_dir / "inpainted" / f"{vid}.mp4")
        n = min(len(clean_frames), len(out_frames), int(v["frame_count"]))
        if n <= 0:
            for k in vals:
                vals[k].append(0.0)
            continue
        step = max(1, n // 48)
        idxs = list(range(0, n, step))
        ref_masks = [read_mask(args.reference_dir / "masks" / vid / f"frame_{i:06d}.png")
                     for i in range(n)]

        for i in idxs:
            gt_m = ref_masks[i]
            pred_m = read_mask(args.output_dir / "nuisance_masks" / vid / f"frame_{i:06d}.png")
            clean = clean_frames[i]
            out = out_frames[i]
            vals["mask_iou"].append(mask_iou(pred_m, gt_m))
            vals["masked_region_psnr"].append(psnr_region(clean, out, gt_m))
            vals["masked_region_ssim"].append(masked_crop_ssim(clean, out, gt_m))
            vals["boundary_consistency"].append(boundary_score(clean, out, gt_m))
            vals["unmasked_preservation"].append(unmasked_score(clean, out, gt_m))

        vals["temporal_flicker"].append(temporal_flicker_score(clean_frames[:n], out_frames[:n], ref_masks, idxs))
        vals["track_consistency"].append(track_consistency(args.output_dir, tracks, vid, idxs))
        vals["output_continuity"].append(output_continuity(out_frames[:n]))

    vals["format_compliance"].append(format_score(args.output_dir, videos))

    values = {k: float(np.mean(v)) if v else 0.0 for k, v in vals.items()}
    dims = {}
    score = 0.0
    all_ok = True
    for name, value in values.items():
        weight, threshold, direction = DIMS[name]
        ok = metric_ok(name, value)
        all_ok = all_ok and ok
        score += weight * metric_credit(name, value)
        dims[name] = {
            "value": round(value, 4),
            "threshold": threshold,
            "direction": direction,
            "ok": ok,
            "weight": weight,
        }

    report = {"pass": bool(all_ok), "score": round(score, 4), "dims": dims}
    args.report.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
