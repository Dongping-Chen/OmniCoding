#!/usr/bin/env python3
"""T147a — deterministic grader for click/pop removal.

Inputs (under --output-dir, default /workspace/output):
  clean/utt_NNN.wav          restored audio (16 kHz mono s16)
  clicks.json                agent's own click detections
  process_log.json           method + parameters

Reference (under --reference-dir, default ./reference):
  clean/utt_NNN.wav          held-out original clean audio
  click_log.json             GT click events
  transcripts.json           GT transcripts

Fixtures (under --fixtures-dir, default ./fixtures):
  noisy/utt_NNN.wav          corrupted input the agent saw

Score dims (matching task.yaml):
  click_residual_count   (le 0.10, weight 0.25)  — residual / injected
  si_sdr_vs_clean        (ge 15.0, weight 0.25)  — dB
  wer_preservation       (le 0.03, weight 0.20)  — WER delta vs noisy baseline
  click_detection_f1     (ge 0.70, weight 0.15)
  over_smoothing_penalty (ge 0.95, weight 0.10)  — flux ratio
  format_compliance      (ge 0.99, weight 0.05)

Usage:
  python3 grader.py --output-dir /workspace/output
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import soundfile as sf

SR = 16000
TASK_DIR = Path(__file__).resolve().parent

DIMS = [
    ("click_residual_count", 0.25, 0.10, "le"),
    ("si_sdr_vs_clean", 0.25, 15.0, "ge"),
    ("wer_preservation", 0.20, 0.03, "le"),
    ("click_detection_f1", 0.15, 0.70, "ge"),
    ("over_smoothing_penalty", 0.10, 0.95, "ge"),
    ("format_compliance", 0.05, 0.99, "ge"),
]


# ---------- I/O helpers ----------------------------------------------------


def _load_wav(p: Path) -> Tuple[np.ndarray, int, int]:
    """Returns (audio, sr, channels)."""
    audio, sr = sf.read(str(p), dtype="float32", always_2d=True)
    ch = audio.shape[1]
    if ch > 1:
        mono = audio.mean(axis=1)
    else:
        mono = audio[:, 0]
    return mono, sr, ch


def _read_json(p: Path) -> dict:
    return json.loads(p.read_text())


# ---------- transient detection -------------------------------------------


def detect_clicks(
    audio: np.ndarray, sr: int, min_amp: float = 0.30, max_dur_ms: float = 5.0
) -> List[Tuple[float, float]]:
    """Median-based outlier on the 1st-order difference. Returns list of
    (t_seconds, duration_ms) for each detected click. Deterministic.
    """
    if len(audio) < sr // 100:
        return []
    diff = np.diff(audio, prepend=audio[:1])
    abs_d = np.abs(diff)
    # robust threshold via median absolute deviation
    med = float(np.median(abs_d))
    mad = float(np.median(np.abs(abs_d - med))) + 1e-9
    thr = max(med + 6.0 * 1.4826 * mad, min_amp / 2.0)
    above = abs_d > thr

    # group consecutive `above` samples; allow up to 2-sample gaps
    events: List[Tuple[float, float]] = []
    i = 0
    n = len(above)
    max_w = int(round(max_dur_ms * sr / 1000.0)) + 4
    while i < n:
        if above[i]:
            j = i
            gap = 0
            while j < n and (above[j] or gap < 2):
                if above[j]:
                    gap = 0
                else:
                    gap += 1
                j += 1
                if j - i > max_w:
                    break
            seg = audio[i:j]
            peak = float(np.max(np.abs(seg))) if len(seg) else 0.0
            dur_ms = (j - i) * 1000.0 / sr
            if peak >= min_amp and dur_ms <= max_dur_ms + 1.0:
                t = (i + int(np.argmax(np.abs(seg)))) / sr
                events.append((t, dur_ms))
            i = j
        else:
            i += 1
    return events


# ---------- metrics --------------------------------------------------------


def si_sdr(estimate: np.ndarray, target: np.ndarray) -> float:
    """Scale-invariant SDR in dB (Le Roux 2018)."""
    n = min(len(estimate), len(target))
    if n == 0:
        return -math.inf
    e = estimate[:n].astype(np.float64)
    t = target[:n].astype(np.float64)
    e = e - e.mean()
    t = t - t.mean()
    denom = float(np.dot(t, t)) + 1e-12
    alpha = float(np.dot(e, t)) / denom
    s_target = alpha * t
    e_noise = e - s_target
    num = float(np.dot(s_target, s_target)) + 1e-12
    den = float(np.dot(e_noise, e_noise)) + 1e-12
    return 10.0 * math.log10(num / den)


def spectral_flux(audio: np.ndarray, sr: int) -> float:
    """Sum of L2 norms of positive spectral differences across frames."""
    n_fft = 512
    hop = 128
    if len(audio) < n_fft:
        return 0.0
    win = np.hanning(n_fft).astype(np.float32)
    n_frames = 1 + (len(audio) - n_fft) // hop
    if n_frames < 2:
        return 0.0
    mag_prev = None
    flux = 0.0
    for k in range(n_frames):
        s = k * hop
        frame = audio[s : s + n_fft] * win
        spec = np.fft.rfft(frame)
        mag = np.abs(spec)
        if mag_prev is not None:
            d = mag - mag_prev
            d[d < 0] = 0
            flux += float(np.linalg.norm(d))
        mag_prev = mag
    return flux


def click_pr_f1(
    pred: List[Tuple[float, float]],
    gt: List[Tuple[float, float]],
    tol_s: float = 0.025,
) -> float:
    """F1 with greedy 1-to-1 matching by nearest time, tolerance ±25 ms."""
    if not pred and not gt:
        return 1.0
    if not pred or not gt:
        return 0.0
    used_gt = set()
    tp = 0
    pred_sorted = sorted(pred, key=lambda x: x[0])
    for pt, _pd in pred_sorted:
        best = -1
        best_dt = tol_s + 1
        for j, (gt_t, _gd) in enumerate(gt):
            if j in used_gt:
                continue
            dt = abs(pt - gt_t)
            if dt < best_dt:
                best_dt = dt
                best = j
        if best >= 0 and best_dt <= tol_s:
            used_gt.add(best)
            tp += 1
    fp = len(pred) - tp
    fn = len(gt) - tp
    prec = tp / max(tp + fp, 1)
    rec = tp / max(tp + fn, 1)
    if prec + rec == 0:
        return 0.0
    return 2 * prec * rec / (prec + rec)


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
    # Levenshtein
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
        print(f"[grader] faster-whisper unavailable ({e}); WER will be skipped.",
              file=sys.stderr)
        return None


def transcribe(audio: np.ndarray, sr: int) -> str:
    m = _whisper_model()
    if m is None:
        return ""
    if sr != 16000:
        # cheap polyphase resample fallback
        from scipy.signal import resample_poly  # type: ignore

        from math import gcd

        g = gcd(sr, 16000)
        audio = resample_poly(audio, 16000 // g, sr // g).astype(np.float32)
        sr = 16000
    segs, _info = m.transcribe(audio, language="en", beam_size=1)
    return " ".join(s.text for s in segs)


# ---------- gating ---------------------------------------------------------


def gating(out_dir: Path, expected_ids: List[str]) -> List[str]:
    """Returns list of human-readable gating errors. Empty list = pass."""
    errs: List[str] = []
    if not (out_dir / "clean").is_dir():
        errs.append("output/clean/ directory missing")
    if not (out_dir / "clicks.json").exists():
        errs.append("output/clicks.json missing")
    if not (out_dir / "process_log.json").exists():
        errs.append("output/process_log.json missing")
    missing = []
    for uid in expected_ids:
        if not (out_dir / "clean" / f"{uid}.wav").exists():
            missing.append(uid)
    if missing:
        errs.append(
            f"{len(missing)} restored wav file(s) missing under output/clean/ "
            f"(first: {missing[:3]})"
        )
    return errs


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
    ap.add_argument("--skip-whisper", action="store_true",
                    help="If set, skip Whisper-based WER (debugging).")
    args = ap.parse_args()

    out = args.output_dir
    ref = args.reference_dir
    fix = args.fixtures_dir

    click_log = _read_json(ref / "click_log.json")
    trans = {
        u["id"]: u["transcript"]
        for u in _read_json(ref / "transcripts.json")["utterances"]
    }
    expected_ids = [u["id"] for u in click_log["utterances"]]

    gating_errs = gating(out, expected_ids)
    if gating_errs:
        report = {
            "pass": False,
            "score": 0.0,
            "gating_errors": gating_errs,
            "dims": {},
        }
        args.report.write_text(json.dumps(report, indent=2))
        print(json.dumps(report, indent=2))
        return 1

    # Per-utt scoring.
    per_utt: List[dict] = []
    fmt_pass = 0
    fmt_total = len(expected_ids)
    sisdr_vals: List[float] = []
    residual_ratios: List[float] = []
    flux_ratios: List[float] = []
    wer_deltas: List[float] = []
    f1_vals: List[float] = []

    agent_clicks: Dict[str, List[Tuple[float, float]]] = {}
    try:
        ac = _read_json(out / "clicks.json")
        for u in ac.get("utterances", []):
            uid = u.get("id")
            if uid:
                agent_clicks[uid] = [
                    (float(d.get("t", 0)), float(d.get("duration_ms", 0)))
                    for d in u.get("detections", [])
                ]
    except Exception as e:
        print(f"[grader] clicks.json parse failed: {e}", file=sys.stderr)

    gt_clicks_by_id = {
        u["id"]: [(e["t_seconds"], e["duration_ms"]) for e in u["events"]]
        for u in click_log["utterances"]
    }
    n_injected_by_id = {u["id"]: u["n_clicks"] for u in click_log["utterances"]}

    for uid in expected_ids:
        rec = {"id": uid}
        try:
            restored, sr_r, ch_r = _load_wav(out / "clean" / f"{uid}.wav")
            clean, sr_c, _ = _load_wav(ref / "clean" / f"{uid}.wav")
            noisy, sr_n, _ = _load_wav(fix / "noisy" / f"{uid}.wav")
        except Exception as e:
            rec["error"] = f"load failed: {e}"
            per_utt.append(rec)
            continue

        # format compliance
        fmt_ok = (
            sr_r == SR
            and ch_r == 1
            and abs(len(restored) - len(noisy)) <= int(0.020 * SR)
        )
        if fmt_ok:
            fmt_pass += 1
        rec["format_ok"] = fmt_ok

        # length-align for SI-SDR
        L = min(len(restored), len(clean))
        sisdr = si_sdr(restored[:L], clean[:L])
        sisdr_vals.append(sisdr)
        rec["si_sdr_db"] = round(sisdr, 3)

        # residual click count (after restoration) / injected count
        det = detect_clicks(restored, sr_r) if fmt_ok else []
        n_after = len(det)
        n_inj = max(n_injected_by_id[uid], 1)
        rr = n_after / n_inj
        residual_ratios.append(rr)
        rec["n_residual"] = n_after
        rec["n_injected"] = n_inj
        rec["residual_ratio"] = round(rr, 4)

        # spectral flux ratio (over-smoothing)
        flux_r = spectral_flux(restored, sr_r) if fmt_ok else 0.0
        flux_g = spectral_flux(clean, sr_c)
        ratio = flux_r / max(flux_g, 1e-9)
        flux_ratios.append(min(ratio, 1.5))
        rec["flux_ratio"] = round(ratio, 4)

        # click detection F1 (agent vs GT)
        f1 = click_pr_f1(agent_clicks.get(uid, []), gt_clicks_by_id[uid])
        f1_vals.append(f1)
        rec["click_f1"] = round(f1, 4)

        # WER preservation
        if not args.skip_whisper:
            try:
                hyp_clean = transcribe(restored.astype(np.float32), sr_r)
                hyp_noisy = transcribe(noisy.astype(np.float32), sr_n)
                wer_clean = word_error_rate(hyp_clean, trans[uid])
                wer_noisy = word_error_rate(hyp_noisy, trans[uid])
                delta = wer_clean - wer_noisy
                wer_deltas.append(delta)
                rec["wer_clean"] = round(wer_clean, 4)
                rec["wer_noisy"] = round(wer_noisy, 4)
                rec["wer_delta"] = round(delta, 4)
            except Exception as e:
                rec["wer_error"] = str(e)
        per_utt.append(rec)

    # Aggregate.
    def _avg(xs: List[float], default: float = 0.0) -> float:
        return float(np.mean(xs)) if xs else default

    agg = {
        "click_residual_count": _avg(residual_ratios, 1.0),
        "si_sdr_vs_clean": _avg(sisdr_vals, -10.0),
        "wer_preservation": _avg(wer_deltas, 1.0) if wer_deltas else 0.0,
        "click_detection_f1": _avg(f1_vals, 0.0),
        "over_smoothing_penalty": _avg(flux_ratios, 0.0),
        "format_compliance": fmt_pass / max(fmt_total, 1),
    }

    # Pass/fail per dim & weighted score.
    dims_report: Dict[str, dict] = {}
    score = 0.0
    for name, weight, thr, direction in DIMS:
        v = float(agg[name])
        if direction == "ge":
            ok = v >= thr
        else:
            ok = v <= thr
        # normalised contribution: 1.0 if pass else partial credit
        if direction == "ge":
            cred = max(0.0, min(1.0, v / max(thr, 1e-9)))
        else:
            cred = max(0.0, min(1.0, thr / max(v, 1e-9)))
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
        "per_utt": per_utt,
        "n_utterances": len(expected_ids),
    }
    args.report.write_text(json.dumps(report, indent=2))
    print(json.dumps({"pass": overall_pass, "score": report["score"],
                      "dims": dims_report}, indent=2))
    return 0 if overall_pass else 1


if __name__ == "__main__":
    sys.exit(main())
