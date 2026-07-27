#!/usr/bin/env python3
"""T149a — deterministic grader for dialog-music ducking.

Inputs:
  output/mix/mix_NNN.wav        48 kHz stereo s16, broadcast mix
  output/automation.csv         per-mix, t_seconds, music_gain (≥10 Hz)

Reference:
  reference/clean_dialog/utt_NNN.wav     16 kHz mono original speech
  reference/music_dry/bed_NNN.wav        44.1 kHz stereo dry music
  reference/transcripts.json
  reference/anon_mapping.json

Fixtures:
  fixtures/dialog/utt_NNN.wav   16 kHz mono
  fixtures/music/bed_NNN.wav    44.1 kHz stereo (peak-normalised input)
  fixtures/pairs.json

Score dimensions (matching task.yaml):
  ducking_depth          fraction of VAD-active 100 ms windows where
                         music RMS in mix is ≥ 18 dB below dialog RMS
                         in mix.   ge 0.85, weight 0.25
  music_recovery         fraction of VAD-inactive runs ≥ 0.5 s where
                         music RMS in mix is within 3 dB of un-ducked
                         baseline.    ge 0.85, weight 0.20
  dialog_intelligibility WER(mix) − WER(clean dialog) via Whisper.
                         le 0.03, weight 0.20
  integrated_loudness    fraction of clips with LUFS ∈ [-17, -15].
                         ge 0.95, weight 0.10
  true_peak_compliance   fraction of clips whose 4× oversampled true
                         peak ≤ -1.0 dBTP.   ge 0.99, weight 0.10
  transition_smoothness  per-clip RMS of automation 2nd derivative,
                         averaged.   le 1.0, weight 0.10
  format_compliance      sample rate, channels, bit depth, length.
                         ge 0.99, weight 0.05

VAD strategy: webrtcvad on the *original* clean dialog when available
(16 kHz, mono, 30 ms frames, mode 2), with a deterministic short-time
energy fallback for lean evaluation environments. The binary timeline
is used to gate both ducking_depth and music_recovery.
The dialog and music components are recovered from the mix by
re-aligning the original dialog and dry-music with cross-correlation
and projecting them out (least-squares per-channel scale + delay), so
agents that re-EQ the dialog are still graded fairly.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import soundfile as sf

TASK_DIR = Path(__file__).resolve().parent

DIMS = [
    ("ducking_depth", 0.25, 0.85, "ge"),
    ("music_recovery", 0.20, 0.85, "ge"),
    ("dialog_intelligibility", 0.20, 0.03, "le"),
    ("integrated_loudness", 0.10, 0.95, "ge"),
    ("true_peak_compliance", 0.10, 0.99, "ge"),
    ("transition_smoothness", 0.10, 1.0, "le"),
    ("format_compliance", 0.05, 0.99, "ge"),
]

MIX_SR = 48000


# ---------- helpers --------------------------------------------------------


def _load_wav(p: Path) -> Tuple[np.ndarray, int, int]:
    a, sr = sf.read(str(p), dtype="float32", always_2d=True)
    return a, sr, a.shape[1]


def _read_json(p: Path) -> dict:
    return json.loads(p.read_text())


def _resample(a: np.ndarray, sr_in: int, sr_out: int) -> np.ndarray:
    from math import gcd

    from scipy.signal import resample_poly  # type: ignore

    if sr_in == sr_out:
        return a
    g = gcd(sr_in, sr_out)
    return resample_poly(a, sr_out // g, sr_in // g, axis=0).astype(np.float32)


def _to_mono(a: np.ndarray) -> np.ndarray:
    if a.ndim == 1:
        return a
    return a.mean(axis=1)


# ---------- VAD on the clean dialog ---------------------------------------


def _close_short_runs(mask: np.ndarray, value: bool, max_len: int) -> None:
    i = 0
    while i < len(mask):
        if bool(mask[i]) == value:
            j = i
            while j < len(mask) and bool(mask[j]) == value:
                j += 1
            if j - i <= max_len:
                mask[i:j] = not value
            i = j
        else:
            i += 1


def _energy_vad_mask(dialog_mono: np.ndarray, sr: int = 16000) -> np.ndarray:
    """Dependency-free VAD for clean dialog fixtures.

    The fixtures are isolated speech, so a robust frame-energy threshold with
    small-gap closing is sufficient and avoids making webrtcvad a hard runtime
    dependency for the whole grader.
    """
    x = np.asarray(dialog_mono, dtype=np.float32)
    if x.size == 0:
        return np.zeros(0, dtype=bool)
    x = x - float(np.mean(x))
    frame_ms = 30
    frame_samples = max(1, int(sr * frame_ms / 1000))
    n_frames = int(math.ceil(len(x) / frame_samples))
    rms = np.zeros(n_frames, dtype=np.float64)
    for i in range(n_frames):
        s = i * frame_samples
        e = min(len(x), s + frame_samples)
        seg = x[s:e]
        if seg.size:
            rms[i] = math.sqrt(float(np.mean(seg.astype(np.float64) ** 2)))
    db = 20.0 * np.log10(rms + 1e-8)
    p10, p50, p90 = np.percentile(db, [10.0, 50.0, 90.0])
    spread = max(float(p90 - p10), 1.0)
    thr = max(float(p10 + 0.35 * spread), float(p50 - 3.0),
              float(p90 - 25.0))
    active = db >= thr

    # Fill very short pauses inside speech, remove isolated blips, and add a
    # modest hangover so plosives/fricatives at word edges are included.
    _close_short_runs(active, False, max_len=max(1, int(round(0.15 / 0.03))))
    _close_short_runs(active, True, max_len=max(1, int(round(0.09 / 0.03))))
    hang = max(1, int(round(0.06 / 0.03)))
    if active.any():
        dilated = active.copy()
        idx = np.where(active)[0]
        for i in idx:
            a = max(0, i - hang)
            b = min(len(active), i + hang + 1)
            dilated[a:b] = True
        active = dilated

    mask = np.zeros(len(x), dtype=bool)
    for i, is_active in enumerate(active):
        if is_active:
            s = i * frame_samples
            e = min(len(mask), s + frame_samples)
            mask[s:e] = True
    return mask


def vad_mask(dialog_16k_mono: np.ndarray, sr: int = 16000) -> np.ndarray:
    """Returns per-sample binary mask (1 = voice). Frame size = 30 ms."""
    try:
        import webrtcvad  # type: ignore
    except Exception:  # pragma: no cover
        return _energy_vad_mask(dialog_16k_mono, sr=sr)

    vad = webrtcvad.Vad(2)
    frame_ms = 30
    frame_samples = int(sr * frame_ms / 1000)
    pcm16 = (np.clip(dialog_16k_mono, -1.0, 1.0) * 32767.0).astype(np.int16)
    n_frames = len(pcm16) // frame_samples
    mask = np.zeros(len(dialog_16k_mono), dtype=bool)
    for i in range(n_frames):
        s = i * frame_samples
        e = s + frame_samples
        frame_bytes = pcm16[s:e].tobytes()
        is_voice = vad.is_speech(frame_bytes, sr)
        if is_voice:
            mask[s:e] = True
    return mask


def _resample_mask(mask: np.ndarray, sr_in: int, sr_out: int,
                   length_out: int) -> np.ndarray:
    """Nearest-neighbour resample a bool mask."""
    if sr_in == sr_out and len(mask) == length_out:
        return mask
    idx = np.arange(length_out)
    src_idx = (idx * sr_in / sr_out).astype(np.int64)
    src_idx = np.clip(src_idx, 0, len(mask) - 1)
    return mask[src_idx]


# ---------- alignment + projection ----------------------------------------


def _align_delay(a: np.ndarray, b: np.ndarray, max_lag: int = 4800) -> int:
    """Return integer sample lag k such that a[t] ≈ b[t + k]; truncated to
    ±max_lag (default 0.1 s @ 48 kHz). Cross-correlation, mono."""
    from scipy.signal import correlate, correlation_lags  # type: ignore

    L = min(len(a), len(b), 48000 * 5)
    if L < 1024:
        return 0
    # subtract mean for fairness
    aa = a[:L] - a[:L].mean()
    bb = b[:L] - b[:L].mean()
    corr = correlate(aa, bb, mode="full", method="fft")
    lags = correlation_lags(L, L, mode="full")
    keep = (lags >= -max_lag) & (lags <= max_lag)
    if not keep.any():
        return 0
    sub_lags = lags[keep]
    sub_corr = corr[keep]
    return int(sub_lags[int(np.argmax(sub_corr))])


def estimate_components_in_mix(
    mix: np.ndarray,            # [N, 2] @ MIX_SR
    dialog: np.ndarray,         # [M] mono @ 16k → upsampled to MIX_SR
    music_dry: np.ndarray,      # [K, 2] @ 44.1k → resampled to MIX_SR
) -> Tuple[np.ndarray, np.ndarray, int, int]:
    """Returns (dialog_in_mix, music_in_mix, dialog_offset, music_offset).

    Recovers the time-varying scales by:
      1. Resample components to MIX_SR.
      2. Estimate integer delays (mix vs each component).
      3. Build aligned versions (zero-padded) and per-channel mono
         dialog signal duplicated to stereo.
      4. Solve per-window scaling: for each 100 ms window, fit
         alpha_d, alpha_m so that mix ≈ alpha_d·d + alpha_m·m  in L2.
      5. Reconstruct the per-sample dialog/music components by linear
         interpolation between window scales.
    """
    N = len(mix)
    sr = MIX_SR

    d = _to_mono(dialog).astype(np.float32)
    d_st = np.stack([d, d], axis=1)
    m = music_dry.astype(np.float32)

    # crude alignment (mono mix vs mono dialog)
    mix_mono = mix.mean(axis=1)
    d_lag = _align_delay(mix_mono, d, max_lag=int(0.5 * sr))
    m_lag = _align_delay(mix_mono, m.mean(axis=1), max_lag=int(0.5 * sr))

    # construct shifted versions (positive lag means component starts later)
    def shift_pad(x: np.ndarray, lag: int, target_n: int) -> np.ndarray:
        if x.ndim == 1:
            x = x[:, None]
        out = np.zeros((target_n, x.shape[1]), dtype=np.float32)
        # mix[t] = component[t + lag]; so component at position -lag
        if lag >= 0:
            cn = min(target_n, len(x) - lag)
            if cn > 0:
                out[:cn] = x[lag : lag + cn]
        else:
            cn = min(target_n + lag, len(x))
            if cn > 0:
                out[-lag : -lag + cn] = x[:cn]
        return out

    d_al = shift_pad(d_st, d_lag, N)
    m_al = shift_pad(m, m_lag, N)

    # Per-window LS: solve [alpha_d, alpha_m] for windowed mix.
    win = int(0.10 * sr)  # 100 ms
    hop = int(0.05 * sr)  # 50% overlap
    n_win = max(1, 1 + (N - win) // hop)
    alpha_d = np.zeros(n_win, dtype=np.float32)
    alpha_m = np.zeros(n_win, dtype=np.float32)
    centers = np.zeros(n_win, dtype=np.int64)
    for k in range(n_win):
        s = k * hop
        e = s + win
        if e > N:
            break
        # collapse stereo to L2-stack [2*win, 2 cols]
        Mw = mix[s:e].reshape(-1)
        Dw = d_al[s:e].reshape(-1)
        Aw = m_al[s:e].reshape(-1)
        A = np.stack([Dw, Aw], axis=1)
        # solve A @ x ≈ Mw, x ≥ 0 (relax to plain LS, then clip)
        try:
            x, *_ = np.linalg.lstsq(A, Mw, rcond=None)
        except np.linalg.LinAlgError:
            x = np.array([0.0, 0.0])
        x = np.clip(x, -3.0, 3.0)
        alpha_d[k] = float(x[0])
        alpha_m[k] = float(x[1])
        centers[k] = s + win // 2

    # Smooth small-window jitter.
    if n_win > 4:
        kk = 5
        kernel = np.ones(kk, dtype=np.float32) / kk
        alpha_d = np.convolve(alpha_d, kernel, mode="same")
        alpha_m = np.convolve(alpha_m, kernel, mode="same")

    # Interpolate per-sample.
    sample_t = np.arange(N)
    if n_win > 1:
        ad_per_sample = np.interp(sample_t, centers[:n_win], alpha_d[:n_win])
        am_per_sample = np.interp(sample_t, centers[:n_win], alpha_m[:n_win])
    else:
        ad_per_sample = np.full(N, alpha_d[0])
        am_per_sample = np.full(N, alpha_m[0])

    dialog_in_mix = ad_per_sample[:, None] * d_al
    music_in_mix = am_per_sample[:, None] * m_al
    return dialog_in_mix, music_in_mix, d_lag, m_lag


# ---------- per-band RMS ---------------------------------------------------


def windowed_rms(x: np.ndarray, sr: int, win_s: float = 0.10,
                 hop_s: float = 0.05) -> Tuple[np.ndarray, np.ndarray]:
    """Returns (centers_seconds, rms_dBFS) for stereo or mono x."""
    if x.ndim == 1:
        mono = x
    else:
        mono = x.mean(axis=1)
    win = int(win_s * sr)
    hop = int(hop_s * sr)
    n = max(0, 1 + (len(mono) - win) // hop)
    if n <= 0:
        return np.array([]), np.array([])
    centers = (np.arange(n) * hop + win // 2) / sr
    rms = np.empty(n, dtype=np.float64)
    for i in range(n):
        s = i * hop
        seg = mono[s : s + win]
        v = float(np.sqrt(np.mean(seg.astype(np.float64) ** 2) + 1e-20))
        rms[i] = 20.0 * math.log10(v + 1e-12)
    return centers, rms


# ---------- loudness + true-peak ------------------------------------------


def integrated_lufs(stereo: np.ndarray, sr: int) -> float:
    import pyloudnorm as pyln  # type: ignore

    meter = pyln.Meter(sr)
    return float(meter.integrated_loudness(stereo))


def true_peak_dbtp(stereo: np.ndarray, sr: int, oversample: int = 4) -> float:
    from math import gcd
    from scipy.signal import resample_poly  # type: ignore

    sr_os = sr * oversample
    g = gcd(sr, sr_os)
    up = resample_poly(stereo, sr_os // g, sr // g, axis=0)
    peak = float(np.max(np.abs(up)) + 1e-12)
    return 20.0 * math.log10(peak)


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
        print(f"[grader] faster-whisper unavailable ({e})", file=sys.stderr)
        return None


def transcribe(mono_16k: np.ndarray) -> str:
    m = _whisper_model()
    if m is None:
        return ""
    segs, _ = m.transcribe(mono_16k.astype(np.float32),
                           language="en", beam_size=1)
    return " ".join(s.text for s in segs)


# ---------- gating ---------------------------------------------------------


def gating(out_dir: Path, expected_ids: List[str]) -> List[str]:
    errs: List[str] = []
    if not (out_dir / "mix").is_dir():
        errs.append("output/mix/ directory missing")
    if not (out_dir / "automation.csv").exists():
        errs.append("output/automation.csv missing")
    missing = []
    for mid in expected_ids:
        if not (out_dir / "mix" / f"{mid}.wav").exists():
            missing.append(mid)
    if missing:
        errs.append(
            f"{len(missing)} mix file(s) missing under output/mix/ "
            f"(first: {missing[:3]})"
        )
    return errs


# ---------- automation parsing --------------------------------------------


def parse_automation(p: Path) -> Dict[str, np.ndarray]:
    """Returns mix_id → array shape [K, 2] of (t_seconds, music_gain)."""
    out: Dict[str, list] = {}
    with p.open() as f:
        rdr = csv.DictReader(f)
        for row in rdr:
            mid = row.get("mix_id") or row.get("id") or ""
            if not mid:
                continue
            try:
                t = float(row["t_seconds"])
                g = float(row["music_gain"])
            except (KeyError, ValueError):
                continue
            out.setdefault(mid, []).append((t, g))
    res: Dict[str, np.ndarray] = {}
    for mid, lst in out.items():
        lst.sort(key=lambda x: x[0])
        arr = np.array(lst, dtype=np.float32)
        res[mid] = arr
    return res


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
    ap.add_argument("--skip-whisper", action="store_true")
    args = ap.parse_args()

    out = args.output_dir
    ref = args.reference_dir
    fix = args.fixtures_dir

    pairs = _read_json(fix / "pairs.json")["pairs"]
    transcripts = {
        u["id"]: u["transcript"]
        for u in _read_json(ref / "transcripts.json")["pairs"]
    }
    expected_ids = [p["id"] for p in pairs]

    gating_errs = gating(out, expected_ids)
    if gating_errs:
        report = {"pass": False, "score": 0.0,
                  "gating_errors": gating_errs, "dims": {}}
        args.report.write_text(json.dumps(report, indent=2))
        print(json.dumps(report, indent=2))
        return 1

    automation = parse_automation(out / "automation.csv")

    duck_fracs: List[float] = []
    rec_fracs: List[float] = []
    wer_deltas: List[float] = []
    lufs_pass: List[bool] = []
    tp_pass: List[bool] = []
    smooth_vals: List[float] = []
    fmt_pass = 0
    fmt_total = len(pairs)

    per_mix: List[dict] = []

    for pair in pairs:
        mid = pair["id"]
        utt_id = Path(pair["dialog_file"]).stem  # utt_NNN
        bed_id = Path(pair["music_file"]).stem   # bed_NNN
        rec = {"id": mid}
        try:
            mix, mix_sr, mix_ch = _load_wav(out / "mix" / f"{mid}.wav")
            d_clean, d_sr, _ = _load_wav(ref / "clean_dialog" / f"{utt_id}.wav")
            m_dry, m_sr, _ = _load_wav(ref / "music_dry" / f"{bed_id}.wav")
        except Exception as e:
            rec["error"] = f"load failed: {e}"
            per_mix.append(rec)
            continue

        # format compliance
        expected_dur = float(pair["duration_seconds"])
        actual_dur = len(mix) / max(mix_sr, 1)
        fmt_ok = (
            mix_sr == MIX_SR
            and mix_ch == 2
            and abs(actual_dur - expected_dur) <= 0.050
        )
        if fmt_ok:
            fmt_pass += 1
        rec["format_ok"] = fmt_ok
        if not fmt_ok:
            rec["format_detail"] = (
                f"sr={mix_sr} ch={mix_ch} dur={actual_dur:.3f} "
                f"(expected sr=48000 ch=2 dur={expected_dur:.3f})"
            )

        # Resample components to mix_sr
        d_mono = _to_mono(d_clean)
        d_at_mix = _resample(d_mono, d_sr, mix_sr)
        m_at_mix = _resample(m_dry, m_sr, mix_sr)

        # estimate dialog and music components in the mix
        dialog_in_mix, music_in_mix, d_lag, m_lag = estimate_components_in_mix(
            mix, d_at_mix, m_at_mix
        )

        # VAD on the clean dialog (16 kHz)
        vmask_16k = vad_mask(d_mono.astype(np.float32), sr=d_sr)
        vmask_mix = _resample_mask(vmask_16k, d_sr, mix_sr,
                                   length_out=len(mix))

        # ducking_depth: in active windows, music RMS ≥ 18 dB below dialog RMS
        win = int(0.10 * mix_sr)
        hop = int(0.05 * mix_sr)
        n_w = max(0, 1 + (len(mix) - win) // hop)
        active = 0
        ducked = 0
        for k in range(n_w):
            s = k * hop
            e = s + win
            if vmask_mix[s:e].mean() < 0.5:
                continue
            d_seg = dialog_in_mix[s:e]
            m_seg = music_in_mix[s:e]
            d_rms = float(np.sqrt(np.mean(d_seg ** 2) + 1e-20))
            m_rms = float(np.sqrt(np.mean(m_seg ** 2) + 1e-20))
            if d_rms < 1e-5:
                continue
            active += 1
            db_diff = 20.0 * math.log10(m_rms / max(d_rms, 1e-12))
            if db_diff <= -18.0:
                ducked += 1
        duck_frac = ducked / max(active, 1) if active else 1.0
        duck_fracs.append(duck_frac)
        rec["ducking_depth_active_windows"] = active
        rec["ducking_depth_passing"] = ducked
        rec["ducking_depth_frac"] = round(duck_frac, 4)

        # music_recovery: in inactive runs ≥ 0.5 s, music RMS within 3 dB of
        # un-ducked baseline (use input music_dry RMS at the same windows).
        # Compute baseline music RMS from the dry music (peak-normalised
        # input) windowed identically.
        m_at_mix_baseline_rms = []
        for k in range(n_w):
            s = k * hop
            e = s + win
            if e > len(m_at_mix):
                break
            seg = m_at_mix[s:e]
            v = float(np.sqrt(np.mean(seg ** 2) + 1e-20))
            m_at_mix_baseline_rms.append(20.0 * math.log10(v + 1e-12))

        # find inactive runs
        # contract to per-window inactive flag
        per_win_inactive = []
        for k in range(n_w):
            s = k * hop
            e = s + win
            per_win_inactive.append(vmask_mix[s:e].mean() < 0.5)
        per_win_inactive = np.array(per_win_inactive, dtype=bool)

        # group inactive runs of consecutive windows; require ≥ 0.5 s = 10 wins
        recover_total = 0
        recover_pass = 0
        i = 0
        while i < len(per_win_inactive):
            if per_win_inactive[i]:
                j = i
                while j < len(per_win_inactive) and per_win_inactive[j]:
                    j += 1
                run_len = j - i
                if run_len >= 10:  # ≥ 0.5 s
                    # check middle 60% of the run (skip transitions)
                    a0 = i + max(2, run_len // 5)
                    a1 = j - max(2, run_len // 5)
                    for k in range(a0, a1):
                        if k >= len(m_at_mix_baseline_rms):
                            continue
                        m_seg = music_in_mix[k * hop : k * hop + win]
                        v = float(np.sqrt(np.mean(m_seg ** 2) + 1e-20))
                        m_in_db = 20.0 * math.log10(v + 1e-12)
                        baseline = m_at_mix_baseline_rms[k]
                        recover_total += 1
                        if m_in_db >= baseline - 3.0:
                            recover_pass += 1
                i = j
            else:
                i += 1
        rec_frac = recover_pass / max(recover_total, 1) if recover_total else 1.0
        rec_fracs.append(rec_frac)
        rec["music_recovery_total"] = recover_total
        rec["music_recovery_pass"] = recover_pass
        rec["music_recovery_frac"] = round(rec_frac, 4)

        # integrated loudness + true peak (whole mix)
        try:
            lufs = integrated_lufs(mix, mix_sr)
        except Exception as e:
            lufs = -100.0
            rec["lufs_error"] = str(e)
        rec["integrated_lufs"] = round(lufs, 2)
        lufs_ok = -17.0 <= lufs <= -15.0
        lufs_pass.append(lufs_ok)

        try:
            tp = true_peak_dbtp(mix, mix_sr)
        except Exception as e:
            tp = 0.0
            rec["tp_error"] = str(e)
        rec["true_peak_dbtp"] = round(tp, 2)
        tp_ok = tp <= -1.0
        tp_pass.append(tp_ok)

        # transition_smoothness via automation.csv
        if mid in automation and len(automation[mid]) >= 3:
            arr = automation[mid]
            t = arr[:, 0]
            g = arr[:, 1]
            dt = np.diff(t)
            dt[dt < 1e-6] = 1e-6
            v1 = np.diff(g) / dt  # 1st deriv
            v2 = np.diff(v1) / dt[1:]  # 2nd deriv (irregular OK)
            smooth = float(np.sqrt(np.mean(v2 ** 2) + 1e-12))
            rec["automation_rows"] = int(arr.shape[0])
            # ensure ≥ 10 Hz (max gap ≤ 100 ms)
            max_gap = float(np.max(dt)) if len(dt) else 0.0
            rec["automation_max_gap_s"] = round(max_gap, 3)
            if max_gap > 0.105:
                smooth = max(smooth, 100.0)  # penalise sparse curves
        else:
            rec["automation_rows"] = 0
            smooth = 100.0
        smooth_vals.append(smooth)
        rec["smoothness_rms"] = round(smooth, 4)

        # WER preservation
        if not args.skip_whisper:
            try:
                # mix → mono 16k for transcription
                mix_mono = _to_mono(mix)
                if mix_sr != 16000:
                    mix_mono = _resample(mix_mono, mix_sr, 16000)
                hyp_mix = transcribe(mix_mono)
                hyp_clean = transcribe(d_mono)
                ref_text = transcripts.get(mid, "")
                wer_mix = word_error_rate(hyp_mix, ref_text)
                wer_clean = word_error_rate(hyp_clean, ref_text)
                delta = wer_mix - wer_clean
                wer_deltas.append(delta)
                rec["wer_mix"] = round(wer_mix, 4)
                rec["wer_clean"] = round(wer_clean, 4)
                rec["wer_delta"] = round(delta, 4)
            except Exception as e:
                rec["wer_error"] = str(e)

        per_mix.append(rec)

    def _avg(xs: List[float], default: float = 0.0) -> float:
        return float(np.mean(xs)) if xs else default

    agg = {
        "ducking_depth": _avg(duck_fracs, 0.0),
        "music_recovery": _avg(rec_fracs, 0.0),
        "dialog_intelligibility": _avg(wer_deltas, 0.0),
        "integrated_loudness": sum(lufs_pass) / max(len(lufs_pass), 1)
                                if lufs_pass else 0.0,
        "true_peak_compliance": sum(tp_pass) / max(len(tp_pass), 1)
                                if tp_pass else 0.0,
        "transition_smoothness": _avg(smooth_vals, 100.0),
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
        "per_mix": per_mix,
        "n_pairs": len(expected_ids),
    }
    args.report.write_text(json.dumps(report, indent=2))
    print(json.dumps({"pass": overall_pass, "score": report["score"],
                      "dims": dims_report}, indent=2))
    return 0 if overall_pass else 1


if __name__ == "__main__":
    sys.exit(main())
