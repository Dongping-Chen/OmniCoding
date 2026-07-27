#!/usr/bin/env python3
"""T146a — deterministic grader for audio bandwidth extension.

Inputs (under --output-dir, default /workspace/output):
  hi/utt_NNN.wav             restored 24 kHz mono PCM s16
  process_log.json           method + parameters

Reference (under --reference-dir, default ./reference):
  hi_gt/utt_NNN.wav          held-out 24 kHz GT
  transcripts.json           reference transcripts

Fixtures (under --fixtures-dir, default ./fixtures):
  lo/utt_NNN.mp3             8 kHz / 32 kbps degraded input
  files.json                 expected durations

Score dims (matching task.yaml):
  hf_band_energy_ratio       (ge 6.0,  weight 0.20)
  log_spectral_distance_hf   (le 3.0,  weight 0.30)
  estoi                      (ge 0.85, weight 0.20)
  content_wer                (ge 0.92, weight 0.15)  — i.e. 1 - WER
  format_compliance          (ge 0.99, weight 0.15)
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from math import gcd
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly, stft

GT_SR = 24000
LO_SR = 8000
HF_BAND = (8000.0, 12000.0)  # 8-12 kHz upper band (GT contains up to 12 kHz)
TASK_DIR = Path(__file__).resolve().parent

DIMS = [
    ("hf_band_energy_ratio", 0.20, 6.0, "ge"),
    ("log_spectral_distance_hf", 0.30, 3.0, "le"),
    ("estoi", 0.20, 0.85, "ge"),
    ("content_wer", 0.15, 0.92, "ge"),
    ("format_compliance", 0.15, 0.99, "ge"),
]


def _load_wav(p: Path) -> Tuple[np.ndarray, int, int, int]:
    """Returns (mono_audio, sr, channels, bit_depth_or_0)."""
    info = sf.info(str(p))
    audio, sr = sf.read(str(p), dtype="float32", always_2d=True)
    ch = audio.shape[1]
    mono = audio.mean(axis=1) if ch > 1 else audio[:, 0]
    bd = 0
    sub = (info.subtype or "").upper()
    if "PCM_16" in sub:
        bd = 16
    elif "PCM_24" in sub:
        bd = 24
    elif "PCM_32" in sub or "FLOAT" in sub:
        bd = 32
    return mono, sr, ch, bd


def _read_lo_mp3(p: Path) -> Tuple[np.ndarray, int]:
    """Decode the MP3 fixture for HF-energy denominator."""
    audio, sr = sf.read(str(p), dtype="float32", always_2d=False)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    return audio, sr


def _resample_to(audio: np.ndarray, sr_from: int, sr_to: int) -> np.ndarray:
    if sr_from == sr_to:
        return audio.astype(np.float32)
    g = gcd(sr_from, sr_to)
    return resample_poly(audio, sr_to // g, sr_from // g).astype(np.float32)


# ---------- HF band energy ------------------------------------------------


def band_energy(audio: np.ndarray, sr: int,
                f_lo: float, f_hi: float) -> float:
    """Mean-square spectral energy in [f_lo, f_hi] Hz via STFT."""
    n_fft = 1024
    hop = 256
    if len(audio) < n_fft:
        return 0.0
    win = np.hanning(n_fft).astype(np.float32)
    n_frames = 1 + (len(audio) - n_fft) // hop
    if n_frames < 1:
        return 0.0
    freqs = np.fft.rfftfreq(n_fft, d=1.0 / sr)
    band = (freqs >= f_lo) & (freqs <= f_hi)
    if not band.any():
        return 0.0
    e = 0.0
    for k in range(n_frames):
        s = k * hop
        spec = np.fft.rfft(audio[s : s + n_fft] * win)
        mag2 = (np.abs(spec) ** 2)
        e += float(mag2[band].sum())
    return e / max(n_frames, 1)


# ---------- log spectral distance restricted to HF band -------------------


def log_spectral_distance_hf(
    estimate: np.ndarray, target: np.ndarray, sr: int
) -> float:
    """Mean per-frame RMS dB difference of log-magnitude STFT in HF_BAND."""
    n_fft = 1024
    hop = 256
    L = min(len(estimate), len(target))
    if L < n_fft:
        return 100.0
    win = np.hanning(n_fft).astype(np.float32)
    freqs = np.fft.rfftfreq(n_fft, d=1.0 / sr)
    band = (freqs >= HF_BAND[0]) & (freqs <= HF_BAND[1])
    if not band.any():
        return 100.0
    n_frames = 1 + (L - n_fft) // hop
    if n_frames < 1:
        return 100.0
    eps = 1e-7
    diffs = []
    for k in range(n_frames):
        s = k * hop
        e_spec = np.abs(np.fft.rfft(estimate[s : s + n_fft] * win))
        t_spec = np.abs(np.fft.rfft(target[s : s + n_fft] * win))
        e_log = 20.0 * np.log10(e_spec[band] + eps)
        t_log = 20.0 * np.log10(t_spec[band] + eps)
        d = e_log - t_log
        diffs.append(float(np.sqrt(np.mean(d * d))))
    return float(np.mean(diffs))


# ---------- ESTOI / local intelligibility proxy ---------------------------


def _fallback_estoi_proxy(estimate: np.ndarray, target: np.ndarray,
                          sr: int) -> float:
    """Dependency-free intelligibility proxy used when pystoi is unavailable.

    The score is a speech-band, short-time envelope correlation. It is not a
    byte-for-byte reimplementation of ESTOI, but it preserves the intended
    role of this dimension: rewarding speech structure preservation instead of
    silently assigning zero when an optional package is missing.
    """
    L = min(len(estimate), len(target))
    if L < sr:
        return 0.0
    e16 = _resample_to(estimate[:L], sr, 16000).astype(np.float64)
    t16 = _resample_to(target[:L], sr, 16000).astype(np.float64)
    L2 = min(len(e16), len(t16))
    if L2 < 16000:
        return 0.0
    e16 = e16[:L2] - float(np.mean(e16[:L2]))
    t16 = t16[:L2] - float(np.mean(t16[:L2]))
    er = float(np.sqrt(np.mean(e16 * e16) + 1e-12))
    tr = float(np.sqrt(np.mean(t16 * t16) + 1e-12))
    if er < 1e-5 or tr < 1e-5:
        return 0.0
    e16 /= er
    t16 /= tr

    n_fft = 512
    hop = 128
    if L2 < n_fft:
        return 0.0
    win = np.hanning(n_fft).astype(np.float64)
    freqs = np.fft.rfftfreq(n_fft, d=1.0 / 16000)
    # Approximate third-octave speech bands from 150 Hz to 7.5 kHz.
    edges = np.array(
        [150, 250, 400, 630, 1000, 1600, 2500, 4000, 6300, 7500],
        dtype=np.float64,
    )
    bands = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (freqs >= lo) & (freqs < hi)
        if mask.any():
            bands.append(mask)
    if not bands:
        return 0.0

    n_frames = 1 + (L2 - n_fft) // hop
    e_env = np.zeros((n_frames, len(bands)), dtype=np.float64)
    t_env = np.zeros_like(e_env)
    for k in range(n_frames):
        s = k * hop
        e_spec = np.abs(np.fft.rfft(e16[s : s + n_fft] * win)) ** 2
        t_spec = np.abs(np.fft.rfft(t16[s : s + n_fft] * win)) ** 2
        for b, mask in enumerate(bands):
            e_env[k, b] = math.log(float(e_spec[mask].sum()) + 1e-8)
            t_env[k, b] = math.log(float(t_spec[mask].sum()) + 1e-8)

    target_band_energy = np.exp(np.mean(t_env, axis=0))
    weights = target_band_energy / max(float(target_band_energy.sum()), 1e-12)
    scores = []
    score_weights = []
    for b in range(len(bands)):
        x = e_env[:, b]
        y = t_env[:, b]
        active = y >= np.percentile(y, 20.0)
        if active.sum() < 8:
            active = np.ones_like(y, dtype=bool)
        x = x[active]
        y = y[active]
        xs = float(np.std(x))
        ys = float(np.std(y))
        if xs < 1e-6 or ys < 1e-6:
            scores.append(0.0)
        else:
            corr = float(np.mean(((x - x.mean()) / xs) * ((y - y.mean()) / ys)))
            scores.append(max(0.0, min(1.0, 0.5 + 0.5 * corr)))
        score_weights.append(float(weights[b]))
    return float(np.clip(np.average(scores, weights=score_weights), 0.0, 1.0))


def compute_estoi(estimate: np.ndarray, target: np.ndarray, sr: int) -> float:
    """Extended STOI on 16 kHz versions of estimate/target."""
    try:
        from pystoi import stoi
    except Exception:  # pragma: no cover
        return _fallback_estoi_proxy(estimate, target, sr)
    L = min(len(estimate), len(target))
    if L < sr:
        return 0.0
    e16 = _resample_to(estimate[:L], sr, 16000)
    t16 = _resample_to(target[:L], sr, 16000)
    L2 = min(len(e16), len(t16))
    return float(stoi(t16[:L2], e16[:L2], 16000, extended=True))


# ---------- WER via faster-whisper ----------------------------------------


def normalise_text(s: str) -> str:
    import re

    s = s.lower()
    s = re.sub(r"[^a-z0-9\s']", " ", s)
    return " ".join(s.split())


def word_error_rate(hyp: str, ref: str) -> float:
    h = normalise_text(hyp).split()
    r = normalise_text(ref).split()
    if not r:
        return 1.0 if h else 0.0
    m, n = len(h), len(r)
    dp = np.zeros((m + 1, n + 1), dtype=np.int32)
    for i in range(m + 1):
        dp[i, 0] = i
    for j in range(n + 1):
        dp[0, j] = j
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if h[i - 1] == r[j - 1]:
                dp[i, j] = dp[i - 1, j - 1]
            else:
                dp[i, j] = 1 + min(dp[i - 1, j], dp[i, j - 1], dp[i - 1, j - 1])
    return float(dp[m, n]) / max(n, 1)


_WHISPER = None


def _whisper_model():
    global _WHISPER
    if _WHISPER is not None:
        return _WHISPER
    try:
        from faster_whisper import WhisperModel  # type: ignore

        _WHISPER = WhisperModel("base", device="cpu", compute_type="int8")
        return _WHISPER
    except Exception as e:
        print(f"[grader] faster-whisper unavailable ({e}); WER skipped.",
              file=sys.stderr)
        return None


def transcribe(audio: np.ndarray, sr: int) -> str:
    m = _whisper_model()
    if m is None:
        return ""
    a16 = _resample_to(audio, sr, 16000)
    segs, _info = m.transcribe(a16, language="en", beam_size=1)
    return " ".join(s.text for s in segs)


# ---------- gating ---------------------------------------------------------


def gating(out_dir: Path, expected_ids: List[str]) -> List[str]:
    errs: List[str] = []
    if not (out_dir / "hi").is_dir():
        errs.append("output/hi/ directory missing")
    if not (out_dir / "process_log.json").exists():
        errs.append("output/process_log.json missing")
    missing = [
        u for u in expected_ids if not (out_dir / "hi" / f"{u}.wav").exists()
    ]
    if missing:
        errs.append(
            f"{len(missing)} restored wav file(s) missing under output/hi/ "
            f"(first: {missing[:3]})"
        )
    return errs


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
    ap.add_argument("--skip-whisper", action="store_true")
    args = ap.parse_args()

    out = args.output_dir
    ref = args.reference_dir
    fix = args.fixtures_dir

    files_meta = json.loads((fix / "files.json").read_text())["files"]
    expected_ids = [f["id"] for f in files_meta]
    expected_dur = {f["id"]: f["duration_seconds"] for f in files_meta}
    transcripts = {
        u["id"]: u["transcript"]
        for u in json.loads((ref / "transcripts.json").read_text())["utterances"]
    }

    gating_errs = gating(out, expected_ids)
    if gating_errs:
        report = {"pass": False, "score": 0.0,
                  "gating_errors": gating_errs, "dims": {}}
        args.report.write_text(json.dumps(report, indent=2))
        print(json.dumps(report, indent=2))
        return 1

    fmt_pass = 0
    hf_ratios: List[float] = []
    lsd_vals: List[float] = []
    estoi_vals: List[float] = []
    wer_vals: List[float] = []
    per_utt: List[dict] = []

    for uid in expected_ids:
        rec = {"id": uid}
        try:
            hi, sr_hi, ch_hi, bd_hi = _load_wav(out / "hi" / f"{uid}.wav")
            gt, sr_gt, _, _ = _load_wav(ref / "hi_gt" / f"{uid}.wav")
            lo, sr_lo = _read_lo_mp3(fix / "lo" / f"{uid}.mp3")
        except Exception as e:
            rec["error"] = f"load failed: {e}"
            per_utt.append(rec)
            continue

        # format compliance: 24 kHz / mono / 16-bit / duration drift ≤ 50 ms
        exp_d = expected_dur[uid]
        actual_d = len(hi) / max(sr_hi, 1)
        fmt_ok = (
            sr_hi == GT_SR
            and ch_hi == 1
            and bd_hi == 16
            and abs(actual_d - exp_d) <= 0.050
        )
        if fmt_ok:
            fmt_pass += 1
        rec["format_ok"] = fmt_ok
        rec["sr"] = sr_hi
        rec["dur_s"] = round(actual_d, 4)
        rec["bit_depth"] = bd_hi

        # HF band energy ratio: hi vs lo, both at GT_SR
        # lo is 8 kHz so it has no content above ~4 kHz; resample to 24 kHz
        # and measure the same 8-12 kHz band.
        lo_24 = _resample_to(lo, sr_lo, GT_SR)
        e_hi = band_energy(hi, GT_SR, *HF_BAND) if fmt_ok else 0.0
        e_lo = band_energy(lo_24, GT_SR, *HF_BAND)
        ratio = e_hi / max(e_lo, 1e-12)
        hf_ratios.append(ratio)
        rec["hf_band_energy_hi"] = float(e_hi)
        rec["hf_band_energy_lo"] = float(e_lo)
        rec["hf_band_energy_ratio"] = round(float(ratio), 4)

        # LSD restricted to HF band
        if fmt_ok:
            lsd = log_spectral_distance_hf(hi, gt, GT_SR)
        else:
            lsd = 100.0
        lsd_vals.append(min(lsd, 100.0))
        rec["lsd_hf_db"] = round(float(lsd), 4)

        # ESTOI on 16 kHz
        if fmt_ok:
            es = compute_estoi(hi, gt, GT_SR)
        else:
            es = 0.0
        estoi_vals.append(es)
        rec["estoi"] = round(es, 4)

        # WER
        if not args.skip_whisper and fmt_ok:
            try:
                hyp = transcribe(hi, GT_SR)
                wer = word_error_rate(hyp, transcripts[uid])
                wer_vals.append(wer)
                rec["wer"] = round(wer, 4)
                rec["hyp"] = hyp[:200]
            except Exception as e:
                rec["wer_error"] = str(e)
        per_utt.append(rec)

    def _avg(xs: List[float], default: float) -> float:
        return float(np.mean(xs)) if xs else default

    agg = {
        "hf_band_energy_ratio": _avg(hf_ratios, 0.0),
        "log_spectral_distance_hf": _avg(lsd_vals, 100.0),
        "estoi": _avg(estoi_vals, 0.0),
        "content_wer": (1.0 - _avg(wer_vals, 1.0)) if wer_vals else 0.0,
        "format_compliance": fmt_pass / max(len(expected_ids), 1),
    }

    dims_report: Dict[str, dict] = {}
    score = 0.0
    for name, weight, thr, direction in DIMS:
        v = float(agg[name])
        if direction == "ge":
            ok = v >= thr
            cred = max(0.0, min(1.0, v / max(thr, 1e-9)))
        else:
            ok = v <= thr
            cred = max(0.0, min(1.0, thr / max(v, 1e-9)))
        score += weight * (1.0 if ok else cred)
        dims_report[name] = {
            "value": round(v, 4), "threshold": thr,
            "direction": direction, "pass": bool(ok), "weight": weight,
        }

    overall = all(d["pass"] for d in dims_report.values())
    report = {
        "pass": overall,
        "score": round(score, 4),
        "dims": dims_report,
        "n_utterances": len(expected_ids),
        "per_utt": per_utt,
    }
    args.report.write_text(json.dumps(report, indent=2))
    print(json.dumps({"pass": overall, "score": report["score"],
                      "dims": dims_report}, indent=2))
    return 0 if overall else 1


if __name__ == "__main__":
    sys.exit(main())
