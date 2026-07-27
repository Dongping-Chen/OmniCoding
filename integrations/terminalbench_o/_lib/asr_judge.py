"""ASR-as-judge helper for Claw-Bench graders.

Wraps faster-whisper to provide cached transcription plus CER/WER scoring of
agent-produced audio against reference text. Used by audio-output tasks where
the deliverable's *spoken content* must be verified, not just its file shape.

Conventions
-----------
- Default model: small (configurable via CLAW_BENCH_ASR_MODEL). Override to
  `large-v3` etc. when tighter accuracy is needed.
- Cache: ~/.cache/claw_bench/asr/{sha256}-{model}-{lang}.json so repeated
  grader runs (and oracle-vs-cheat self-tests) don't re-decode audio.
- CER for languages without word boundaries (zh/ja/ko); WER otherwise.

Public API
----------
- transcribe(audio_path, language=None, model=None) -> dict
- cer(ref, hyp) / wer(ref, hyp) -> float in [0, +inf)
- asr_score(audio_path, reference_text, language=None, metric="auto",
            model=None) -> {transcript, cer, wer, score, language}

`score` is `max(0.0, 1.0 - error_rate)` so callers can plug it straight into
a grader dimension.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import unicodedata
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Iterable

DEFAULT_MODEL = os.environ.get("CLAW_BENCH_ASR_MODEL", "small")
CACHE_DIR = Path(os.environ.get(
    "CLAW_BENCH_ASR_CACHE",
    str(Path.home() / ".cache" / "claw_bench" / "asr"),
))
CACHE_DIR.mkdir(parents=True, exist_ok=True)

CJK_LANGS = {"zh", "ja", "ko", "yue"}


# --------------------------------------------------------------------------- #
# Whisper model loader (cached per-process)
# --------------------------------------------------------------------------- #
@lru_cache(maxsize=4)
def _load_model(name: str):
    from faster_whisper import WhisperModel

    device = os.environ.get("CLAW_BENCH_ASR_DEVICE", "auto")
    compute_type = os.environ.get("CLAW_BENCH_ASR_COMPUTE", "auto")
    if device == "auto":
        try:
            import torch  # type: ignore
            device = "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            device = "cpu"
    if compute_type == "auto":
        compute_type = "float16" if device == "cuda" else "int8"
    return WhisperModel(name, device=device, compute_type=compute_type)


# --------------------------------------------------------------------------- #
# Caching
# --------------------------------------------------------------------------- #
def _file_hash(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _cache_path(audio_hash: str, model: str, language: str | None) -> Path:
    safe_model = model.replace("/", "_")
    lang = language or "auto"
    return CACHE_DIR / f"{audio_hash}-{safe_model}-{lang}.json"


# --------------------------------------------------------------------------- #
# Public: transcribe
# --------------------------------------------------------------------------- #
def transcribe(
    audio_path: str | Path,
    language: str | None = None,
    model: str | None = None,
    beam_size: int = 1,
    vad_filter: bool = True,
) -> dict:
    """Transcribe an audio file. Cached by content hash + model + language.

    Returns {text, segments, language, duration}.
    """
    p = Path(audio_path)
    if not p.exists():
        raise FileNotFoundError(p)
    model_name = model or DEFAULT_MODEL
    h = _file_hash(p)
    cache_file = _cache_path(h, model_name, language)
    if cache_file.exists():
        try:
            return json.loads(cache_file.read_text())
        except Exception:
            cache_file.unlink(missing_ok=True)

    m = _load_model(model_name)
    segments_iter, info = m.transcribe(
        str(p),
        language=language,
        beam_size=beam_size,
        vad_filter=vad_filter,
        condition_on_previous_text=False,
    )
    segments = []
    for seg in segments_iter:
        segments.append({
            "start": float(seg.start),
            "end": float(seg.end),
            "text": seg.text.strip(),
        })
    text = " ".join(s["text"] for s in segments).strip()
    out = {
        "text": text,
        "segments": segments,
        "language": info.language,
        "duration": float(info.duration),
    }
    try:
        cache_file.write_text(json.dumps(out, ensure_ascii=False))
    except Exception:
        pass
    return out


# --------------------------------------------------------------------------- #
# Text normalization (loose: punct strip + casefold + NFKC)
# --------------------------------------------------------------------------- #
_PUNCT_RE = re.compile(r"[\s　]+|[\.,!?;:\"'`~\-_/\\(){}\[\]<>《》「」『』、，。！？；：]")


def _normalize(s: str, language: str | None) -> str:
    s = unicodedata.normalize("NFKC", s).casefold()
    if language in CJK_LANGS:
        return _PUNCT_RE.sub("", s)
    s = _PUNCT_RE.sub(" ", s)
    return re.sub(r"\s+", " ", s).strip()


# --------------------------------------------------------------------------- #
# CER / WER (jiwer wrappers with normalisation)
# --------------------------------------------------------------------------- #
def cer(reference: str, hypothesis: str, language: str | None = "zh") -> float:
    from jiwer import cer as _cer

    r = _normalize(reference, language)
    h = _normalize(hypothesis, language)
    if not r:
        return 0.0 if not h else 1.0
    return float(_cer(r, h))


def wer(reference: str, hypothesis: str, language: str | None = "en") -> float:
    from jiwer import wer as _wer

    r = _normalize(reference, language)
    h = _normalize(hypothesis, language)
    if not r:
        return 0.0 if not h else 1.0
    return float(_wer(r, h))


# --------------------------------------------------------------------------- #
# One-shot: ASR + score
# --------------------------------------------------------------------------- #
@dataclass
class AsrScoreResult:
    transcript: str
    cer: float
    wer: float
    score: float
    language: str

    def to_dict(self) -> dict:
        return {
            "transcript": self.transcript,
            "cer": self.cer,
            "wer": self.wer,
            "score": self.score,
            "language": self.language,
        }


def asr_score(
    audio_path: str | Path,
    reference_text: str,
    language: str | None = None,
    metric: str = "auto",
    model: str | None = None,
) -> dict:
    """Transcribe `audio_path`, compare to `reference_text`, return scoring dict.

    metric:
      - "cer" — character error rate
      - "wer" — word error rate
      - "auto" — CER for CJK, WER otherwise (chosen via detected language if
        `language` is None).
    """
    res = transcribe(audio_path, language=language, model=model)
    detected = (language or res["language"] or "").lower()
    use_cer = metric == "cer" or (metric == "auto" and detected in CJK_LANGS)

    c = cer(reference_text, res["text"], language=detected or "zh")
    w = wer(reference_text, res["text"], language=detected or "en")
    err = c if use_cer else w
    score = max(0.0, 1.0 - err)
    return AsrScoreResult(
        transcript=res["text"],
        cer=c,
        wer=w,
        score=score,
        language=detected,
    ).to_dict()


def asr_batch_score(
    items: Iterable[tuple[str | Path, str]],
    language: str | None = None,
    metric: str = "auto",
    model: str | None = None,
) -> list[dict]:
    """Run asr_score over many (audio_path, reference_text) pairs.

    Transcribe results are content-hash-cached, so concurrency only helps
    on first-time scoring of distinct files. Whisper itself is heavy
    (CPU/GPU compute) so we cap workers low (default 2 via
    CLAW_BENCH_ASR_CONCURRENCY) to avoid OOM / GPU contention.
    """
    pairs = list(items)
    if not pairs:
        return []

    def _one(pair):
        a, t = pair
        try:
            return asr_score(a, t, language=language, metric=metric, model=model)
        except Exception as e:
            return {"error": f"{type(e).__name__}: {str(e)[:160]}",
                    "transcript": "", "cer": 1.0, "wer": 1.0,
                    "score": 0.0, "language": language or ""}

    workers_raw = os.environ.get("CLAW_BENCH_ASR_CONCURRENCY", "2")
    try:
        workers = max(1, min(8, int(workers_raw)))
    except ValueError:
        workers = 2
    if workers <= 1 or len(pairs) <= 1:
        return [_one(p) for p in pairs]
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=min(workers, len(pairs))) as ex:
        return list(ex.map(_one, pairs))


# --------------------------------------------------------------------------- #
# Smoke test
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    import sys

    if len(sys.argv) < 3:
        print("usage: asr_judge.py <audio> <reference_text> [language]")
        sys.exit(1)
    lang = sys.argv[3] if len(sys.argv) > 3 else None
    out = asr_score(sys.argv[1], sys.argv[2], language=lang)
    print(json.dumps(out, ensure_ascii=False, indent=2))
