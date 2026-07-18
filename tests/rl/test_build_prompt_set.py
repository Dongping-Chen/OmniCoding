from __future__ import annotations

import json
import sys
from pathlib import Path

import pyarrow.parquet as pq

from omnicoding.rl import build_prompt_set


def test_build_prompt_parquet_from_rl_jsonl(tmp_path: Path, monkeypatch) -> None:
    source = tmp_path / "rl_train.jsonl"
    output = tmp_path / "rl_prompts.parquet"
    source.write_text(
        json.dumps(
            {
                "id": "fixture:1",
                "question": "Which option?",
                "answer_type": "mcq",
                "ground_truth": ["A"],
                "options": ["A. first", "B. second"],
                "source_dataset": "fixture",
                "category": "test",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        sys,
        "argv",
        ["omnicoding-rl-build-prompts", "--input", str(source), "--output", str(output)],
    )

    assert build_prompt_set.main() == 0
    rows = pq.read_table(output).to_pylist()
    assert len(rows) == 1
    assert "A. first" in rows[0]["prompt"]
    assert json.loads(rows[0]["metadata"])["task_id"] == "fixture:1"
