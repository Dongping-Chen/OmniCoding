#!/usr/bin/env python3
"""Small Whisper-compatible CLI backed by faster-whisper.

The coding agent uses the common subset of OpenAI Whisper's CLI:

    whisper input.wav --model small --output_format txt --output_dir artifacts

Keeping the model in the read-only worker image avoids one download per
isolated rollout container.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from faster_whisper import WhisperModel


def _default_device() -> str:
    """Match OpenAI Whisper's CUDA-when-available CLI default."""
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
    except (ImportError, RuntimeError):
        pass
    return "cpu"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="whisper")
    parser.add_argument("audio", nargs="+")
    parser.add_argument("--model", default="small")
    parser.add_argument("--device", default=None)
    parser.add_argument("--fp16", default=None)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--output_format", default="txt")
    parser.add_argument("--output_dir", default=".")
    parser.add_argument("--language", default=None)
    return parser


def main() -> None:
    args, _unknown = _parser().parse_known_args()
    model_ref = "/opt/whisper-small" if args.model == "small" else args.model
    device = args.device or _default_device()
    if device not in {"cpu", "cuda"}:
        raise SystemExit(f"unsupported --device {device!r}; choose cpu or cuda")
    compute_type = "float16" if device == "cuda" else "int8"
    model = WhisperModel(
        model_ref,
        device=device,
        compute_type=compute_type,
        cpu_threads=max(1, args.threads),
    )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for audio_name in args.audio:
        segments, _info = model.transcribe(
            audio_name,
            language=args.language,
            vad_filter=True,
        )
        transcript = "".join(segment.text for segment in segments).strip() + "\n"
        output_path = output_dir / f"{Path(audio_name).stem}.txt"
        output_path.write_text(transcript)
        print(f"[whisper] wrote {output_path}")


if __name__ == "__main__":
    main()
