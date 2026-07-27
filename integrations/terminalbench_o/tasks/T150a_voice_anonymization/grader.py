#!/usr/bin/env python3
"""T150a — deterministic grader for voice anonymisation.

Inputs (under --output-dir):
  anon/utt_NNN.wav        16 kHz mono PCM s16
  transform.json          method + params

Reference (under --reference-dir):
  ecapa/spk_X.npy         per-speaker mean ECAPA embedding (192-d, normalised)
  transcripts.json
  anon_to_real_mapping.json

Fixtures (under --fixtures-dir):
  raw/utt_NNN.wav         original 16 kHz clip the agent saw
  utterances.json         id, duration_seconds, anon_speaker_tag

Score dims (matching task.yaml):
  content_wer             ge 0.94      0.25
  identity_cosine_drop    le 0.30      0.30
  mcd_naturalness_band    in [3,8] dB  0.15  (reported as in_band fraction
                                              vs threshold 1.0; pass when
                                              every utt is in band)
  f0_or_formant_shift     ge 0.95 frac  0.15
  mp3_robustness          le 0.03 wer_delta 0.10
  format_compliance       ge 0.99      0.05
"""
from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
from math import gcd
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

OUT_SR = 16000
TASK_DIR = Path(__file__).resolve().parent

DIMS = [
    ("content_wer", 0.25, 0.94, "ge"),
    ("identity_cosine_drop", 0.30, 0.30, "le"),
    ("mcd_naturalness_band", 0.15, 1.0, "ge"),
    ("f0_or_formant_shift", 0.15, 0.95, "ge"),
    ("mp3_robustness", 0.10, 0.03, "le"),
    ("format_compliance", 0.05, 0.99, "ge"),
]


# ---------- I/O ----------------------------------------------------------


def _load_wav(p: Path) -> Tuple[np.ndarray, int, int, int]:
    info = sf.info(str(p))
    audio, sr = sf.read(str(p), dtype="float32", always_2d=True)
    ch = audio.shape[1]
    mono = audio.mean(axis=1) if ch > 1 else audio[:, 0]
    sub = (info.subtype or "").upper()
    bd = 16 if "PCM_16" in sub else (24 if "PCM_24" in sub else 0)
    return mono, sr, ch, bd


def _resample(audio: np.ndarray, sr_from: int, sr_to: int) -> np.ndarray:
    if sr_from == sr_to:
        return audio.astype(np.float32)
    g = gcd(sr_from, sr_to)
    return resample_poly(audio, sr_to // g, sr_from // g).astype(np.float32)


# ---------- ECAPA --------------------------------------------------------


_ECAPA = None


def ecapa_model():
    global _ECAPA
    if _ECAPA is not None:
        return _ECAPA
    os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "120")
    from speechbrain.inference.speaker import EncoderClassifier  # type: ignore

    save_dir = Path.home() / ".cache" / "claw_bench" / "ecapa-voxceleb"
    save_dir.mkdir(parents=True, exist_ok=True)
    _ECAPA = EncoderClassifier.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb",
        savedir=str(save_dir),
        run_opts={"device": "cpu"},
    )
    return _ECAPA


def ecapa_embed(audio_16k: np.ndarray) -> np.ndarray:
    import torch  # type: ignore

    m = ecapa_model()
    with torch.no_grad():
        x = torch.from_numpy(audio_16k.astype(np.float32)).unsqueeze(0)
        emb = m.encode_batch(x).squeeze().cpu().numpy()
    n = float(np.linalg.norm(emb))
    return emb / max(n, 1e-9)


# ---------- WER (faster-whisper) -----------------------------------------


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


def whisper_model():
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


def transcribe(audio: np.ndarray, sr: int) -> str:
    m = whisper_model()
    if m is None:
        return ""
    a16 = _resample(audio, sr, 16000)
    segs, _info = m.transcribe(a16, language="en", beam_size=1)
    return " ".join(s.text for s in segs)


# ---------- F0 + formant via librosa / pyworld ---------------------------


def median_f0(audio: np.ndarray, sr: int) -> float:
    """Voiced-frame median F0 via librosa.pyin."""
    import librosa  # type: ignore

    try:
        f0, _vflag, _vprob = librosa.pyin(
            audio.astype(np.float32),
            fmin=70, fmax=400, sr=sr, frame_length=1024,
        )
    except Exception:
        return 0.0
    f0 = f0[~np.isnan(f0)]
    if len(f0) < 5:
        return 0.0
    return float(np.median(f0))


def first_three_formants(audio: np.ndarray, sr: int) -> List[float]:
    """LPC-based formant tracking. Returns [F1, F2, F3] medians (Hz).

    This is a deterministic, dependency-light estimate: pre-emphasise,
    frame the audio, fit LPC, find roots, sort by frequency.
    """
    import librosa  # type: ignore

    if len(audio) < sr // 4:
        return [0.0, 0.0, 0.0]
    pre = librosa.effects.preemphasis(audio.astype(np.float32))
    frame_len = int(0.025 * sr)
    hop = int(0.010 * sr)
    n_frames = max(1, 1 + (len(pre) - frame_len) // hop)
    order = 2 + sr // 1000
    F: List[List[float]] = [[], [], []]
    for k in range(n_frames):
        s = k * hop
        frame = pre[s : s + frame_len]
        if len(frame) < order + 2:
            continue
        # voicing gate: skip silent / very-low-energy frames
        if float(np.sqrt(np.mean(frame * frame))) < 0.005:
            continue
        win = np.hanning(len(frame)).astype(np.float32)
        try:
            a = librosa.lpc(frame * win, order=order)
        except Exception:
            continue
        roots = np.roots(a)
        roots = roots[np.imag(roots) > 0]
        if len(roots) == 0:
            continue
        ang = np.angle(roots)
        freqs = ang * (sr / (2 * math.pi))
        bws = -0.5 * (sr / (2 * math.pi)) * np.log(np.abs(roots) + 1e-12)
        keep = (freqs > 90) & (freqs < sr / 2 - 100) & (bws < 600)
        formants = sorted(freqs[keep].tolist())
        for j in range(min(3, len(formants))):
            F[j].append(formants[j])
    return [float(np.median(x)) if x else 0.0 for x in F]


# ---------- MCD with DTW -------------------------------------------------


def proper_mcep(audio: np.ndarray, sr: int,
                alpha: float = 0.42, n_mcep: int = 24) -> np.ndarray:
    """Mel-cepstrum coefficients via pyworld spectral envelope + pysptk
    sp2mc. Returns (T, n_mcep+1). C0 is index 0; drop it for MCD."""
    import pyworld as pw  # type: ignore
    import pysptk  # type: ignore

    if len(audio) < int(0.025 * sr):
        return np.zeros((1, n_mcep + 1), dtype=np.float32)
    a = np.ascontiguousarray(audio.astype(np.float64))
    f0, t = pw.dio(a, sr)
    f0 = pw.stonemask(a, f0, t, sr)
    sp = pw.cheaptrick(a, f0, t, sr)
    mc = pysptk.sp2mc(sp, order=n_mcep, alpha=alpha)
    return mc.astype(np.float32)


def dtw_mcd(a: np.ndarray, b: np.ndarray) -> float:
    """Mel-Cepstral Distortion in dB after DTW alignment, using proper
    mel-cepstrum coefficients (pyworld + pysptk).

    MCD = (10 / ln(10)) * sqrt(2) * mean over aligned frames of
    sqrt(sum_{d=1}^{D} (a_d - b_d)^2). C0 is dropped before computing.
    """
    if len(a) == 0 or len(b) == 0:
        return 100.0
    from librosa.sequence import dtw  # type: ignore

    a_drop = a[:, 1:]
    b_drop = b[:, 1:]
    cost = np.linalg.norm(
        a_drop[:, None, :] - b_drop[None, :, :], axis=-1
    ).astype(np.float32)
    D, wp = dtw(C=cost, subseq=False, backtrack=True)
    diffs = [float(cost[i, j]) for (i, j) in wp]
    if not diffs:
        return 100.0
    rms = float(np.mean(diffs))
    return rms * (10.0 / math.log(10)) * math.sqrt(2.0)


# ---------- MP3 round-trip -----------------------------------------------


def mp3_roundtrip(audio_16k: np.ndarray) -> np.ndarray:
    """16 kHz mono float32 → MP3 64 kbps → back to 16 kHz mono float32."""
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        in_wav = td / "in.wav"
        mid_mp3 = td / "mid.mp3"
        out_wav = td / "out.wav"
        sf.write(str(in_wav), audio_16k, OUT_SR, subtype="PCM_16")
        subprocess.run(
            ["ffmpeg", "-y", "-loglevel", "error", "-i", str(in_wav),
             "-c:a", "libmp3lame", "-b:a", "64k", "-ac", "1", "-ar",
             str(OUT_SR), str(mid_mp3)],
            check=True, capture_output=True,
        )
        subprocess.run(
            ["ffmpeg", "-y", "-loglevel", "error", "-i", str(mid_mp3),
             "-ac", "1", "-ar", str(OUT_SR), "-f", "wav", str(out_wav)],
            check=True, capture_output=True,
        )
        out, _sr = sf.read(str(out_wav), dtype="float32", always_2d=False)
    if out.ndim > 1:
        out = out.mean(axis=1)
    return out.astype(np.float32)


# ---------- gating -------------------------------------------------------


def gating(out_dir: Path, expected_ids: List[str]) -> List[str]:
    errs: List[str] = []
    if not (out_dir / "anon").is_dir():
        errs.append("output/anon/ directory missing")
    if not (out_dir / "transform.json").exists():
        errs.append("output/transform.json missing")
    missing = [
        u for u in expected_ids if not (out_dir / "anon" / f"{u}.wav").exists()
    ]
    if missing:
        errs.append(
            f"{len(missing)} anon wav file(s) missing under output/anon/ "
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
    ap.add_argument("--skip-mp3-robustness", action="store_true")
    args = ap.parse_args()

    out = args.output_dir
    ref = args.reference_dir
    fix = args.fixtures_dir

    utts = json.loads((fix / "utterances.json").read_text())["utterances"]
    expected_ids = [u["id"] for u in utts]
    expected_dur = {u["id"]: u["duration_seconds"] for u in utts}
    spk_tag_for = {u["id"]: u["anon_speaker_tag"] for u in utts}
    transcripts = {
        u["id"]: u["transcript"]
        for u in json.loads((ref / "transcripts.json").read_text())["utterances"]
    }
    # Load reference embeddings.
    ref_embs: Dict[str, np.ndarray] = {}
    for npy in (ref / "ecapa").glob("*.npy"):
        v = np.load(npy).astype(np.float32)
        v = v / (np.linalg.norm(v) + 1e-9)
        ref_embs[npy.stem] = v

    gating_errs = gating(out, expected_ids)
    if gating_errs:
        report = {"pass": False, "score": 0.0,
                  "gating_errors": gating_errs, "dims": {}}
        args.report.write_text(json.dumps(report, indent=2))
        print(json.dumps(report, indent=2))
        return 1

    fmt_pass = 0
    wer_vals: List[float] = []
    cos_vals: List[float] = []
    mcd_in_band: List[int] = []
    mcd_vals: List[float] = []
    shift_pass: List[int] = []
    mp3_deltas: List[float] = []
    per_utt: List[dict] = []

    for uid in expected_ids:
        rec = {"id": uid}
        try:
            anon, sr_a, ch_a, bd_a = _load_wav(out / "anon" / f"{uid}.wav")
            orig, sr_o, _, _ = _load_wav(fix / "raw" / f"{uid}.wav")
        except Exception as e:
            rec["error"] = f"load failed: {e}"
            per_utt.append(rec)
            continue

        exp_d = expected_dur[uid]
        actual_d = len(anon) / max(sr_a, 1)
        fmt_ok = (
            sr_a == OUT_SR and ch_a == 1 and bd_a == 16
            and abs(actual_d - exp_d) <= 0.050
        )
        if fmt_ok:
            fmt_pass += 1
        rec["format_ok"] = fmt_ok
        rec["sr"] = sr_a
        rec["bit_depth"] = bd_a
        rec["dur_s"] = round(actual_d, 4)

        # Identity cosine
        try:
            emb = ecapa_embed(anon)
            ref_emb = ref_embs.get(spk_tag_for[uid])
            cos = float(emb @ ref_emb) if ref_emb is not None else 1.0
        except Exception as e:
            cos = 1.0
            rec["cosine_error"] = str(e)
        cos_vals.append(cos)
        rec["identity_cosine"] = round(cos, 4)

        # MCD
        try:
            mfcc_a = proper_mcep(anon, sr_a)
            mfcc_o = proper_mcep(orig, sr_o)
            mcd = dtw_mcd(mfcc_a, mfcc_o)
        except Exception as e:
            mcd = 100.0
            rec["mcd_error"] = str(e)
        mcd_vals.append(mcd)
        in_band = 1 if 3.0 <= mcd <= 8.0 else 0
        mcd_in_band.append(in_band)
        rec["mcd_db"] = round(mcd, 4)
        rec["mcd_in_band"] = bool(in_band)

        # F0 / formant shift
        try:
            f0_a = median_f0(anon, sr_a)
            f0_o = median_f0(orig, sr_o)
            F_a = first_three_formants(anon, sr_a)
            F_o = first_three_formants(orig, sr_o)
            f0_shift = abs(f0_a - f0_o)
            f_changes = []
            for fa, fo in zip(F_a, F_o):
                if fo > 0 and fa > 0:
                    f_changes.append(abs(fa - fo) / fo)
            mean_f_change = float(np.mean(f_changes)) if f_changes else 0.0
            ok_shift = (f0_shift >= 30.0) or (mean_f_change >= 0.10)
        except Exception as e:
            f0_a = f0_o = mean_f_change = f0_shift = 0.0
            ok_shift = False
            rec["f0_error"] = str(e)
        shift_pass.append(1 if ok_shift else 0)
        rec.update({
            "f0_anon": round(f0_a, 2), "f0_orig": round(f0_o, 2),
            "f0_shift_hz": round(f0_shift, 2),
            "formant_change_frac": round(mean_f_change, 4),
            "shift_ok": bool(ok_shift),
        })

        # Content WER
        if not args.skip_whisper and fmt_ok:
            try:
                hyp = transcribe(anon, sr_a)
                wer = word_error_rate(hyp, transcripts[uid])
                wer_vals.append(wer)
                rec["wer"] = round(wer, 4)
                rec["hyp"] = hyp[:200]
            except Exception as e:
                rec["wer_error"] = str(e)

        # MP3 robustness: WER on round-tripped anon vs WER on anon
        if (not args.skip_whisper and not args.skip_mp3_robustness
                and fmt_ok and "wer" in rec):
            try:
                anon_mp3 = mp3_roundtrip(anon)
                hyp_mp3 = transcribe(anon_mp3, OUT_SR)
                wer_mp3 = word_error_rate(hyp_mp3, transcripts[uid])
                delta = max(0.0, wer_mp3 - rec["wer"])
                mp3_deltas.append(delta)
                rec["wer_mp3"] = round(wer_mp3, 4)
                rec["wer_mp3_delta"] = round(delta, 4)
            except Exception as e:
                rec["mp3_error"] = str(e)

        per_utt.append(rec)

    def _avg(xs: List[float], default: float) -> float:
        return float(np.mean(xs)) if xs else default

    agg = {
        "content_wer": (1.0 - _avg(wer_vals, 1.0)) if wer_vals else 0.0,
        "identity_cosine_drop": _avg(cos_vals, 1.0),
        "mcd_naturalness_band": _avg([float(x) for x in mcd_in_band], 0.0),
        "f0_or_formant_shift": _avg([float(x) for x in shift_pass], 0.0),
        "mp3_robustness": _avg(mp3_deltas, 1.0) if mp3_deltas else 0.0,
        "format_compliance": fmt_pass / max(len(expected_ids), 1),
    }
    # mcd_naturalness_band threshold semantics: pass when in_band fraction
    # ≥ 1.0 (i.e. all utts in band). We treat the dim "value" as the
    # fraction; threshold 1.0 ge.

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
        "mcd_mean_db": round(_avg(mcd_vals, 0.0), 4),
    }
    args.report.write_text(json.dumps(report, indent=2))
    print(json.dumps({"pass": overall, "score": report["score"],
                      "dims": dims_report}, indent=2))
    return 0 if overall else 1


if __name__ == "__main__":
    sys.exit(main())
