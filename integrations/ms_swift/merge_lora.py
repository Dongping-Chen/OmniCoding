"""Merge a PEFT LoRA adapter into a multimodal Hugging Face checkpoint."""

from __future__ import annotations

import argparse
from pathlib import Path


def main() -> int:
    import torch
    from peft import PeftModel
    from transformers import AutoProcessor, Qwen3_5ForConditionalGeneration

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", required=True, help="Base model id or local path.")
    parser.add_argument("--adapter", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--dtype",
        default="bfloat16",
        choices=["bfloat16", "float16", "float32"],
    )
    args = parser.parse_args()

    dtype = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[args.dtype]
    args.output.mkdir(parents=True, exist_ok=True)

    base = Qwen3_5ForConditionalGeneration.from_pretrained(
        args.base,
        torch_dtype=dtype,
        device_map="auto",
        trust_remote_code=True,
    )
    merged = PeftModel.from_pretrained(base, args.adapter).merge_and_unload()
    merged.save_pretrained(args.output, safe_serialization=True, max_shard_size="5GB")
    AutoProcessor.from_pretrained(args.base, trust_remote_code=True).save_pretrained(
        args.output
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
