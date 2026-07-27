#!/usr/bin/env python3
"""Deterministic grader for T152a AV sync drift repair."""
from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, List

import cv2  # type: ignore
import numpy as np
import soundfile as sf  # type: ignore
from scipy.signal import correlate, correlation_lags  # type: ignore
from skimage.metrics import structural_similarity as ssim  # type: ignore

TASK_DIR = Path(__file__).resolve().parent
SR = 16000
EXPECTED_W = 426
EXPECTED_H = 240
EXPECTED_FPS = 25.0

DIMS = {
    "av_offset_mae_ms": (0.20, 50.0, "le"),
    "drift_curve_rmse_ms": (0.15, 80.0, "le"),
    "audio_snr_db": (0.15, 14.0, "ge"),
    "speech_window_corr": (0.10, 0.80, "ge"),
    "dropout_recovery": (0.10, 0.75, "ge"),
    "video_preservation_ssim": (0.10, 0.95, "ge"),
    "curve_plausibility": (0.05, 0.90, "ge"),
    "format_compliance": (0.08, 0.99, "ge"),
    "cross_file_consistency": (0.07, 1.0, "eq"),
}


def run(cmd: list[str]) -> None:
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL,
                   stderr=subprocess.DEVNULL)


def read_json(p: Path) -> dict:
    return json.loads(p.read_text())


def extract_wav(mp4: Path, wav: Path) -> None:
    run([
        "ffmpeg", "-y", "-i", str(mp4), "-vn", "-ar", str(SR),
        "-ac", "1", "-c:a", "pcm_s16le", str(wav)
    ])


def read_audio(p: Path) -> np.ndarray:
    x, sr = sf.read(p)
    if sr != SR:
        raise ValueError(f"unexpected sample rate {sr}: {p}")
    if x.ndim > 1:
        x = x.mean(axis=1)
    return x.astype(np.float32)


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
    res = {"width": 0, "height": 0, "fps": 0.0, "duration": 0.0,
           "has_audio": False}
    for st in info.get("streams", []):
        if st.get("codec_type") == "video":
            res["width"] = int(st.get("width", 0))
            res["height"] = int(st.get("height", 0))
            num, _, den = st.get("r_frame_rate", "0/1").partition("/")
            try:
                res["fps"] = float(num) / float(den)
            except Exception:
                pass
            res["duration"] = float(st.get("duration") or 0.0)
        elif st.get("codec_type") == "audio":
            res["has_audio"] = True
    return res


def norm_corr(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(np.float64)
    b = b.astype(np.float64)
    a = a - np.mean(a)
    b = b - np.mean(b)
    den = float(np.linalg.norm(a) * np.linalg.norm(b))
    if den <= 1e-9:
        return 0.0
    return float(np.dot(a, b) / den)


def estimate_window_offsets(ref: np.ndarray, out: np.ndarray,
                            duration_s: float) -> tuple[list[float], list[float]]:
    offsets_ms: list[float] = []
    corrs: list[float] = []
    win = int(1.25 * SR)
    max_lag = int(0.80 * SR)
    centers = np.arange(1.0, max(1.1, duration_s - 0.5), 1.0)
    for c in centers:
        mid = int(c * SR)
        s = max(0, mid - win // 2)
        e = min(len(ref), mid + win // 2)
        if e - s < int(0.75 * SR):
            continue
        r = ref[s:e]
        # Give the output window extra context so lag can be estimated.
        os = max(0, s - max_lag)
        oe = min(len(out), e + max_lag)
        y = out[os:oe]
        if len(y) < len(r) or float(np.std(r)) < 1e-4:
            continue
        cc = correlate(y - np.mean(y), r - np.mean(r), mode="valid")
        if len(cc) == 0:
            continue
        best = int(np.argmax(cc))
        lag_samples = (os + best) - s
        aligned = y[best:best + len(r)]
        offsets_ms.append(1000.0 * lag_samples / SR)
        corrs.append(norm_corr(r, aligned))
    return offsets_ms, corrs


def snr_db(ref: np.ndarray, out: np.ndarray) -> float:
    n = min(len(ref), len(out))
    if n <= SR:
        return 0.0
    r = ref[:n].astype(np.float64)
    y = out[:n].astype(np.float64)
    err = r - y
    return float(10.0 * np.log10((np.mean(r * r) + 1e-9) /
                                 (np.mean(err * err) + 1e-9)))


def dropout_score(ref: np.ndarray, out: np.ndarray, rows: list[dict]) -> float:
    vals: list[float] = []
    for row in rows:
        s = int(float(row["start_s"]) * SR)
        e = int(float(row["end_s"]) * SR)
        if e <= s or e > min(len(ref), len(out)):
            continue
        r = ref[s:e]
        y = out[s:e]
        c = max(0.0, norm_corr(r, y))
        e_ratio = float(np.sqrt(np.mean(y * y) + 1e-9) /
                        (np.sqrt(np.mean(r * r) + 1e-9)))
        e_score = max(0.0, min(1.0, 1.0 - abs(1.0 - e_ratio)))
        vals.append(0.7 * c + 0.3 * e_score)
    return float(np.mean(vals)) if vals else 1.0


def load_curve(p: Path) -> Dict[str, list[tuple[float, float]]]:
    res: Dict[str, list[tuple[float, float]]] = {}
    with p.open() as f:
        rdr = csv.DictReader(f)
        for row in rdr:
            try:
                cid = row["clip_id"]
                t = float(row["t_seconds"])
                off = float(row["audio_offset_ms"])
            except Exception:
                continue
            res.setdefault(cid, []).append((t, off))
    for v in res.values():
        v.sort()
    return res


def curve_rmse(pred_p: Path, ref_p: Path, clip_ids: list[str]) -> float:
    if not pred_p.exists():
        return 1e9
    pred = load_curve(pred_p)
    ref = load_curve(ref_p)
    errs: list[float] = []
    for cid in clip_ids:
        if cid not in pred or cid not in ref:
            errs.append(1000.0)
            continue
        pt = np.array([x[0] for x in pred[cid]], dtype=np.float64)
        pv = np.array([x[1] for x in pred[cid]], dtype=np.float64)
        if len(pt) < 10:
            errs.append(1000.0)
            continue
        for t, rv in ref[cid]:
            val = float(np.interp(t, pt, pv))
            errs.append(val - rv)
    return float(np.sqrt(np.mean(np.square(errs)))) if errs else 1e9


def curve_plausibility_score(p: Path, clip_ids: list[str], duration_s: float) -> float:
    if not p.exists():
        return 0.0
    curves = load_curve(p)
    vals: list[float] = []
    for cid in clip_ids:
        rows = curves.get(cid, [])
        if len(rows) < int(duration_s / 0.5):
            vals.append(0.0)
            continue
        ts = np.array([r[0] for r in rows])
        offs = np.array([r[1] for r in rows])
        dense = float(np.mean(np.diff(ts) <= 0.75)) if len(ts) > 1 else 0.0
        bounded = float(np.mean(np.abs(offs) <= 1000.0))
        smooth = 1.0
        if len(offs) > 2:
            smooth = float(np.mean(np.abs(np.diff(offs, n=2)) <= 80.0))
        vals.append(0.45 * dense + 0.35 * bounded + 0.20 * smooth)
    return float(np.mean(vals)) if vals else 0.0


def video_ssim(ref_mp4: Path, out_mp4: Path, n_samples: int = 8) -> float:
    rcap = cv2.VideoCapture(str(ref_mp4))
    ycap = cv2.VideoCapture(str(out_mp4))
    n = int(min(rcap.get(cv2.CAP_PROP_FRAME_COUNT),
                ycap.get(cv2.CAP_PROP_FRAME_COUNT)))
    if n <= 0:
        return 0.0
    idxs = np.linspace(0, n - 1, num=min(n_samples, n), dtype=int)
    vals: list[float] = []
    for idx in idxs:
        rcap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ycap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ok1, fr = rcap.read()
        ok2, fy = ycap.read()
        if not ok1 or not ok2:
            continue
        fr = cv2.cvtColor(cv2.resize(fr, (EXPECTED_W, EXPECTED_H)), cv2.COLOR_BGR2GRAY)
        fy = cv2.cvtColor(cv2.resize(fy, (EXPECTED_W, EXPECTED_H)), cv2.COLOR_BGR2GRAY)
        vals.append(float(ssim(fr, fy, data_range=255)))
    rcap.release()
    ycap.release()
    return float(np.mean(vals)) if vals else 0.0


def format_score(out_dir: Path, clips: list[dict]) -> float:
    vals: list[float] = []
    for c in clips:
        p = out_dir / "repaired" / f"{c['id']}.mp4"
        info = ffprobe_video(p)
        checks = [
            info.get("width") == EXPECTED_W,
            info.get("height") == EXPECTED_H,
            abs(float(info.get("fps", 0.0)) - EXPECTED_FPS) <= 0.2,
            abs(float(info.get("duration", 0.0)) - float(c["duration_seconds"])) <= 0.12,
            bool(info.get("has_audio")),
        ]
        vals.append(sum(1.0 for x in checks if x) / len(checks))
    return float(np.mean(vals)) if vals else 0.0


def cross_file_score(out_dir: Path, clip_ids: list[str]) -> float:
    checks: list[bool] = []
    checks.extend((out_dir / "repaired" / f"{cid}.mp4").exists() for cid in clip_ids)
    checks.append((out_dir / "sync_curve.csv").exists())
    checks.append((out_dir / "repair_log.json").exists())
    checks.append((out_dir / "processing_manifest.json").exists())
    try:
        log = read_json(out_dir / "repair_log.json")
        checks.append(int(log.get("clips_processed", -1)) == len(clip_ids))
    except Exception:
        checks.append(False)
    try:
        manifest = read_json(out_dir / "processing_manifest.json")
        checks.append(int(manifest.get("inputs_processed", -1)) == len(clip_ids))
    except Exception:
        checks.append(False)
    curve = load_curve(out_dir / "sync_curve.csv") if (out_dir / "sync_curve.csv").exists() else {}
    checks.extend(cid in curve and len(curve[cid]) >= 10 for cid in clip_ids)
    return sum(1.0 for x in checks if x) / len(checks) if checks else 0.0


def metric_ok(name: str, value: float) -> bool:
    _, thr, direction = DIMS[name]
    if direction == "le":
        return value <= thr
    if direction == "ge":
        return value >= thr
    return abs(value - thr) <= 1e-9


def metric_credit(name: str, value: float) -> float:
    _, thr, direction = DIMS[name]
    if direction == "le":
        return max(0.0, min(1.0, thr / max(value, 1e-9)))
    if direction == "ge":
        return max(0.0, min(1.0, value / max(thr, 1e-9)))
    return 1.0 if metric_ok(name, value) else 0.0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-dir", type=Path, default=Path("/workspace/output"))
    ap.add_argument("--fixtures-dir", type=Path, default=TASK_DIR / "fixtures")
    ap.add_argument("--reference-dir", type=Path, default=TASK_DIR / "reference")
    ap.add_argument("--report", type=Path, default=TASK_DIR / "grader_report.json")
    args = ap.parse_args()

    clips = read_json(args.fixtures_dir / "clips.json")["clips"]
    clip_ids = [c["id"] for c in clips]
    out_dir = args.output_dir
    ref_dir = args.reference_dir

    dropout_by_clip: Dict[str, list[dict]] = {cid: [] for cid in clip_ids}
    with (ref_dir / "dropouts.csv").open() as f:
        for row in csv.DictReader(f):
            dropout_by_clip.setdefault(row["clip_id"], []).append(row)

    offset_abs: list[float] = []
    snrs: list[float] = []
    corrs: list[float] = []
    drops: list[float] = []
    vssims: list[float] = []

    with tempfile.TemporaryDirectory() as td:
        td_path = Path(td)
        for c in clips:
            cid = c["id"]
            ref_mp4 = ref_dir / "clean" / f"{cid}.mp4"
            out_mp4 = out_dir / "repaired" / f"{cid}.mp4"
            if not out_mp4.exists():
                offset_abs.append(1000.0)
                snrs.append(0.0)
                corrs.append(0.0)
                drops.append(0.0)
                vssims.append(0.0)
                continue
            ref_wav = td_path / f"{cid}_ref.wav"
            out_wav = td_path / f"{cid}_out.wav"
            try:
                extract_wav(ref_mp4, ref_wav)
                extract_wav(out_mp4, out_wav)
                ref_a = read_audio(ref_wav)
                out_a = read_audio(out_wav)
                offs, wcorrs = estimate_window_offsets(ref_a, out_a, float(c["duration_seconds"]))
                offset_abs.extend(abs(x) for x in offs)
                corrs.append(float(np.median(wcorrs)) if wcorrs else 0.0)
                snrs.append(snr_db(ref_a, out_a))
                drops.append(dropout_score(ref_a, out_a, dropout_by_clip.get(cid, [])))
            except Exception:
                offset_abs.append(1000.0)
                snrs.append(0.0)
                corrs.append(0.0)
                drops.append(0.0)
            try:
                vssims.append(video_ssim(ref_mp4, out_mp4))
            except Exception:
                vssims.append(0.0)

    values = {
        "av_offset_mae_ms": float(np.mean(offset_abs)) if offset_abs else 1000.0,
        "drift_curve_rmse_ms": curve_rmse(out_dir / "sync_curve.csv",
                                          ref_dir / "sync_curve.csv", clip_ids),
        "audio_snr_db": float(np.mean(snrs)) if snrs else 0.0,
        "speech_window_corr": float(np.mean(corrs)) if corrs else 0.0,
        "dropout_recovery": float(np.mean(drops)) if drops else 0.0,
        "video_preservation_ssim": float(np.mean(vssims)) if vssims else 0.0,
        "curve_plausibility": curve_plausibility_score(out_dir / "sync_curve.csv",
                                                       clip_ids, float(clips[0]["duration_seconds"])),
        "format_compliance": format_score(out_dir, clips),
        "cross_file_consistency": cross_file_score(out_dir, clip_ids),
    }

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
