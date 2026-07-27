#!/usr/bin/env python3
"""Deterministic grader for T154a music stem remix."""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Dict

import numpy as np
import soundfile as sf  # type: ignore

TASK_DIR = Path(__file__).resolve().parent
SR = 44100
STEMS = ["vocals", "drums", "bass", "other"]
SECTIONS = {"intro", "verse", "chorus", "bridge", "solo", "outro", "unknown"}

DIMS = {
    "stem_sisdr": (0.26, 5.0, "ge"),
    "mixture_consistency": (0.14, 0.90, "ge"),
    "karaoke_quality": (0.14, 8.0, "ge"),
    "vocal_suppression": (0.10, 0.70, "ge"),
    "beat_alignment": (0.10, 0.70, "ge"),
    "promo_audio_quality": (0.08, 0.90, "ge"),
    "manifest_consistency": (0.08, 0.98, "ge"),
    "format_compliance": (0.10, 0.99, "ge"),
}


def read_json(p: Path) -> dict:
    return json.loads(p.read_text())


def read_wav(p: Path) -> tuple[np.ndarray, int]:
    x, sr = sf.read(p)
    if x.ndim == 1:
        x = np.stack([x, x], axis=1)
    return x.astype(np.float32), sr


def align_len(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    n = min(len(a), len(b))
    return a[:n], b[:n]


def si_sdr(ref: np.ndarray, est: np.ndarray) -> float:
    ref = ref.reshape(-1).astype(np.float64)
    est = est.reshape(-1).astype(np.float64)
    n = min(len(ref), len(est))
    ref = ref[:n] - np.mean(ref[:n])
    est = est[:n] - np.mean(est[:n])
    alpha = np.dot(est, ref) / (np.dot(ref, ref) + 1e-9)
    target = alpha * ref
    noise = est - target
    return float(10 * np.log10((np.sum(target ** 2) + 1e-9) / (np.sum(noise ** 2) + 1e-9)))


def snr_db(ref: np.ndarray, est: np.ndarray) -> float:
    ref, est = align_len(ref, est)
    return float(10 * np.log10((np.mean(ref ** 2) + 1e-9) / (np.mean((ref - est) ** 2) + 1e-9)))


def read_beats(p: Path) -> Dict[str, list[float]]:
    res: Dict[str, list[float]] = {}
    with p.open() as f:
        for row in csv.DictReader(f):
            res.setdefault(row["song_id"], []).append(float(row["beat_time_s"]))
    return res


def beat_alignment(plan_p: Path, beat_p: Path) -> float:
    if not plan_p.exists():
        return 0.0
    beats = read_beats(beat_p)
    checks = []
    with plan_p.open() as f:
        for row in csv.DictReader(f):
            sid = row.get("song_id", "")
            if row.get("section") not in SECTIONS:
                checks.append(False)
                continue
            bt = beats.get(sid, [])
            if not bt:
                checks.append(False)
                continue
            for key in ["src_start_s", "src_end_s"]:
                try:
                    t = float(row[key])
                except Exception:
                    checks.append(False)
                    continue
                checks.append(min(abs(t - b) for b in bt) <= 0.15)
    return sum(1.0 for x in checks if x) / len(checks) if checks else 0.0


def promo_quality(p: Path) -> float:
    try:
        x, sr = read_wav(p)
    except Exception:
        return 0.0
    dur = len(x) / sr
    peak = float(np.max(np.abs(x))) if len(x) else 0.0
    rms = float(np.sqrt(np.mean(x ** 2))) if len(x) else 0.0
    checks = [2.0 <= dur <= 5.0, peak <= 0.99, peak > 0.02, rms > 0.005]
    return sum(1.0 for z in checks if z) / len(checks)


def manifest_score(out: Path, songs: list[dict]) -> float:
    checks = []
    checks.append((out / "stem_manifest.json").exists())
    checks.append((out / "processing_manifest.json").exists())
    for s in songs:
        sid = s["id"]
        checks.extend((out / "stems" / sid / f"{stem}.wav").exists() for stem in STEMS)
        checks.append((out / "karaoke" / f"{sid}.wav").exists())
        checks.append((out / "promo" / f"{sid}_teaser.wav").exists())
    try:
        mf = read_json(out / "processing_manifest.json")
        checks.append(int(mf.get("inputs_processed", -1)) == len(songs))
    except Exception:
        checks.append(False)
    return sum(1.0 for x in checks if x) / len(checks) if checks else 0.0


def format_score(out: Path, songs: list[dict]) -> float:
    vals = []
    for s in songs:
        sid = s["id"]
        for rel in [*(f"stems/{sid}/{stem}.wav" for stem in STEMS), f"karaoke/{sid}.wav", f"promo/{sid}_teaser.wav"]:
            try:
                x, sr = read_wav(out / rel)
                vals.append(1.0 if sr == SR and x.ndim == 2 and x.shape[1] == 2 else 0.0)
            except Exception:
                vals.append(0.0)
    return float(np.mean(vals)) if vals else 0.0


def metric_ok(name: str, value: float) -> bool:
    _, thr, direction = DIMS[name]
    if direction == "eq":
        return abs(value - thr) <= 1e-9
    return value >= thr if direction == "ge" else value <= thr


def metric_credit(name: str, value: float) -> float:
    _, thr, direction = DIMS[name]
    if direction == "eq":
        return 1.0 if metric_ok(name, value) else value
    return max(0.0, min(1.0, value / max(thr, 1e-9))) if direction == "ge" else max(0.0, min(1.0, thr / max(value, 1e-9)))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-dir", type=Path, default=Path("/workspace/output"))
    ap.add_argument("--fixtures-dir", type=Path, default=TASK_DIR / "fixtures")
    ap.add_argument("--reference-dir", type=Path, default=TASK_DIR / "reference")
    ap.add_argument("--report", type=Path, default=TASK_DIR / "grader_report.json")
    args = ap.parse_args()

    songs = read_json(args.fixtures_dir / "songs.json")["songs"]
    stem_scores = []
    mix_scores = []
    karaoke_scores = []
    vocal_supp = []
    promo_scores = []

    for s in songs:
        sid = s["id"]
        pred_stems = {}
        ref_stems = {}
        for stem in STEMS:
            try:
                pred_stems[stem], _ = read_wav(args.output_dir / "stems" / sid / f"{stem}.wav")
                ref_stems[stem], _ = read_wav(args.reference_dir / "stems" / sid / f"{stem}.wav")
                stem_scores.append(si_sdr(ref_stems[stem], pred_stems[stem]))
            except Exception:
                stem_scores.append(-30.0)
        try:
            mix, _ = read_wav(args.fixtures_dir / "mixes" / f"{sid}.wav")
            pred_sum = sum(pred_stems.values())
            mix_scores.append(max(0.0, min(1.0, snr_db(mix, pred_sum) / 30.0)))
        except Exception:
            mix_scores.append(0.0)
        try:
            karaoke, _ = read_wav(args.output_dir / "karaoke" / f"{sid}.wav")
            accomp = ref_stems["drums"] + ref_stems["bass"] + ref_stems["other"]
            karaoke_scores.append(snr_db(accomp, karaoke))
            mix, _ = read_wav(args.fixtures_dir / "mixes" / f"{sid}.wav")
            v = ref_stems["vocals"]
            n = min(len(karaoke), len(v), len(mix))
            leak = abs(np.mean(karaoke[:n] * v[:n])) / (np.sqrt(np.mean(karaoke[:n] ** 2)) * np.sqrt(np.mean(v[:n] ** 2)) + 1e-9)
            vocal_supp.append(max(0.0, min(1.0, 1.0 - float(leak))))
        except Exception:
            karaoke_scores.append(-30.0)
            vocal_supp.append(0.0)
        promo_scores.append(promo_quality(args.output_dir / "promo" / f"{sid}_teaser.wav"))

    values = {
        "stem_sisdr": float(np.mean(stem_scores)) if stem_scores else -30.0,
        "mixture_consistency": float(np.mean(mix_scores)) if mix_scores else 0.0,
        "karaoke_quality": float(np.mean(karaoke_scores)) if karaoke_scores else -30.0,
        "vocal_suppression": float(np.mean(vocal_supp)) if vocal_supp else 0.0,
        "beat_alignment": beat_alignment(args.output_dir / "beat_edit_plan.csv", args.reference_dir / "beat_gt.csv"),
        "promo_audio_quality": float(np.mean(promo_scores)) if promo_scores else 0.0,
        "manifest_consistency": manifest_score(args.output_dir, songs),
        "format_compliance": format_score(args.output_dir, songs),
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
