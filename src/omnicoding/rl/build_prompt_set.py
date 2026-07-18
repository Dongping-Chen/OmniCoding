#!/usr/bin/env python3
"""Convert ``coding-agent-rl/processed/rl_train.jsonl`` → a Parquet that
Relax's ``StreamingDataset`` can load.

Schema:
- ``prompt: str``    the question text (with options inlined for MCQ). Used by
                    Relax for prompt-length filtering and logging only — our
                    custom rollout function ignores it and POSTs ``task_id`` to
                    the local coordinator instead.
- ``label: str``    JSON-encoded ``ground_truth`` list, used by the Modal-side
                    grader fallback (and as the canonical Relax label field).
- ``metadata: dict`` populated into ``Sample.metadata``: ``{task_id,
                    answer_type, source_dataset, category, options}``.

Usage:
    omnicoding-rl-build-prompts \\
        --input /path/to/rl_train.jsonl \\
        --output /path/to/rl_prompts.parquet
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

LOG = logging.getLogger("build_prompt_set")


def _format_prompt(row: dict) -> str:
    """Bake the question (and options for MCQ) into a single string for
    Relax-side bookkeeping. The actual user-message kira sees is built in the
    coordinator; this is just for logging + length filtering."""
    parts = [row["question"].strip()]
    if row.get("options"):
        parts.append("")
        parts.append("Options:")
        parts.extend(f"  {opt}" for opt in row["options"])
    return "\n".join(parts)


def _row_to_record(row: dict) -> dict:
    return {
        "prompt": _format_prompt(row),
        "label": json.dumps(row["ground_truth"], ensure_ascii=False),
        "metadata": json.dumps(
            {
                "task_id": row["id"],
                "answer_type": row["answer_type"],
                "options": row.get("options"),
                "source_dataset": row.get("source_dataset", ""),
                "category": row.get("category", ""),
            },
            ensure_ascii=False,
        ),
    }


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default=os.environ.get("RL_TRAIN_JSONL"))
    ap.add_argument("--output", default=os.environ.get("RL_PROMPT_PARQUET"))
    ap.add_argument("--max-records", type=int, default=None, help="Smoke-test cap.")
    args = ap.parse_args()

    if not args.input:
        LOG.error("--input required (or set RL_TRAIN_JSONL)")
        return 2
    if not args.output:
        LOG.error("--output required (or set RL_PROMPT_PARQUET)")
        return 2

    in_path = Path(args.input)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    with in_path.open() as f:
        for ln, raw in enumerate(f, 1):
            raw = raw.strip()
            if not raw:
                continue
            try:
                src = json.loads(raw)
            except json.JSONDecodeError as exc:
                LOG.error("%s:%d invalid JSON: %s", in_path, ln, exc)
                return 1
            rows.append(_row_to_record(src))
            if args.max_records and len(rows) >= args.max_records:
                break

    LOG.info("converted %d records, writing %s", len(rows), out_path)
    table = pa.Table.from_pylist(rows)
    pq.write_table(table, out_path, compression="snappy")
    LOG.info("done — %s (%.1f KB)", out_path, out_path.stat().st_size / 1024)
    return 0


if __name__ == "__main__":
    sys.exit(main())
