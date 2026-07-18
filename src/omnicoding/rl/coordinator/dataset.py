"""rl_train.jsonl loader.

Records are keyed by their string id (e.g. ``"omnimodal:283"``). Media paths
are relative to the dataset root (e.g. ``coding-agent-rl/``), not the media
subdir.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

LOGGER = logging.getLogger("coordinator.dataset")


@dataclass(frozen=True, slots=True)
class Record:
    id: str
    question: str
    answer_type: str  # "mcq" | "open"
    ground_truth: list[str]
    options: list[str] | None
    media: dict[str, list[str]]  # {"videos": [...], "audios": [...], "images": [...]}
    source_dataset: str
    category: str


def _row_to_record(row: dict) -> Record:
    # NOTE: rl_train.jsonl has a legacy ``tools_required`` column from omnigaia's
    # old schema (read_audio / page_browser / web_search / code_executor). It does
    # NOT describe what kira's contract actually exposes (kira = bash + image_read +
    # task_complete only). Ignore it entirely — coordinator/instruction.py gives
    # the agent the correct media-handling guidance regardless.
    media = row.get("media") or {}
    return Record(
        id=row["id"],
        question=row["question"],
        answer_type=row["answer_type"],
        ground_truth=list(row["ground_truth"]),
        options=row.get("options"),
        media={k: list(media.get(k, [])) for k in ("videos", "audios", "images")},
        source_dataset=row.get("source_dataset", ""),
        category=row.get("category", ""),
    )


def load_records(jsonl_path: str | Path) -> dict[str, Record]:
    path = Path(jsonl_path)
    if not path.is_file():
        raise FileNotFoundError(f"rl_train jsonl not found at {path}")
    out: dict[str, Record] = {}
    with path.open() as f:
        for ln, raw in enumerate(f, 1):
            raw = raw.strip()
            if not raw:
                continue
            try:
                row = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{ln} not valid JSON: {exc}") from exc
            rec = _row_to_record(row)
            if rec.id in out:
                raise ValueError(f"{path}:{ln} duplicate id {rec.id}")
            out[rec.id] = rec
    LOGGER.info("loaded %d records from %s", len(out), path)
    return out
