from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from omnicoding.data.trajectory_inventory import (
    _parse_runs,
    _validate_output_location,
    inventory_run,
    load_judgements,
)


def _attempt(
    root: Path,
    directory: str,
    *,
    item_id: int,
    attempt: int,
    correct: bool | None,
    prediction: str,
    complete: bool = True,
) -> None:
    run = root / "videozerobench_kira" / directory
    run.mkdir(parents=True)
    row = {
        "source_index": 0,
        "question_id": item_id,
        "attempt": attempt,
        "is_correct": correct,
        "level3_answer": prediction,
        "completed": True,
        "error": None,
        "harness": "kira",
        "model": "test-model",
    }
    (run / "results.json").write_text(json.dumps([row]), encoding="utf-8")
    item_dir = run / "item_0000"
    item_dir.mkdir()
    (item_dir / "final_text.txt").write_text(prediction, encoding="utf-8")
    if complete:
        (item_dir / "messages.json").write_text('[{"role":"assistant"}]', encoding="utf-8")
        (item_dir / "trajectory.json").write_text('[{"step":1}]', encoding="utf-8")


def test_selects_later_correct_attempt_over_earlier_wrong(tmp_path: Path) -> None:
    _attempt(tmp_path, "shard_00", item_id=7, attempt=1, correct=False, prediction="wrong")
    _attempt(tmp_path, "job_00", item_id=7, attempt=2, correct=True, prediction="right")

    inventory = inventory_run("test", tmp_path)

    assert inventory["summary"]["videozerobench"]["candidate_rows"] == 2
    assert inventory["summary"]["videozerobench"]["correct_complete_selected"] == 1
    assert inventory["selected"][0]["attempt"] == 2
    assert inventory["selected"][0]["item_dir"].endswith("job_00/item_0000")


def test_rejects_correct_row_without_complete_messages(tmp_path: Path) -> None:
    _attempt(
        tmp_path,
        "job_00",
        item_id=7,
        attempt=2,
        correct=True,
        prediction="right",
        complete=False,
    )

    inventory = inventory_run("test", tmp_path)

    assert inventory["selected"] == []
    assert inventory["summary"]["videozerobench"]["unselected_items"] == 1


def test_complete_artifacts_survive_missing_task_complete_protocol(tmp_path: Path) -> None:
    _attempt(tmp_path, "job_00", item_id=7, attempt=2, correct=True, prediction="right")
    path = tmp_path / "videozerobench_kira" / "job_00" / "results.json"
    rows = json.loads(path.read_text())
    rows[0]["completed"] = False
    rows[0]["exit_reason"] = "no_tool_calls"
    path.write_text(json.dumps(rows), encoding="utf-8")

    inventory = inventory_run("test", tmp_path)

    assert len(inventory["selected"]) == 1
    assert inventory["selected"][0]["protocol_completed"] is False
    assert inventory["selected"][0]["complete"] is True


def test_external_judge_matches_item_and_prediction(tmp_path: Path) -> None:
    run = tmp_path / "omnigaia_kira" / "redo"
    run.mkdir(parents=True)
    row = {
        "source_index": 0,
        "id": 9,
        "attempt": 2,
        "is_correct": None,
        "predicted_answer": "  Forty two\n",
        "completed": True,
        "error": None,
    }
    (run / "results.json").write_text(json.dumps([row]), encoding="utf-8")
    item = run / "item_0000"
    item.mkdir()
    (item / "messages.json").write_text('[{"role":"assistant"}]', encoding="utf-8")
    (item / "final_text.txt").write_text("Forty two", encoding="utf-8")
    judge = tmp_path / "judge.json"
    judge.write_text(
        json.dumps([{"id": 9, "pred": "forty  two", "judge_correct": True}]),
        encoding="utf-8",
    )
    judgements = load_judgements([f"test:omnigaia:{judge}:id:pred:judge_correct"])

    inventory = inventory_run("test", tmp_path, judgements)

    assert len(inventory["selected"]) == 1
    assert inventory["selected"][0]["correctness_source"].startswith("external_judge")


def test_external_judge_cannot_leak_between_runs(tmp_path: Path) -> None:
    run = tmp_path / "omnigaia_kira" / "shard_00"
    run.mkdir(parents=True)
    row = {
        "source_index": 0,
        "id": 9,
        "is_correct": None,
        "predicted_answer": "same answer",
        "completed": True,
        "error": None,
    }
    (run / "results.json").write_text(json.dumps([row]), encoding="utf-8")
    item = run / "item_0000"
    item.mkdir()
    (item / "messages.json").write_text('[{"role":"assistant"}]', encoding="utf-8")
    (item / "final_text.txt").write_text("same answer", encoding="utf-8")
    judge = tmp_path / "judge.json"
    judge.write_text(
        json.dumps([{"id": 9, "pred": "same answer", "ok": True}]),
        encoding="utf-8",
    )
    judgements = load_judgements([f"other:omnigaia:{judge}:id:pred:ok"])

    inventory = inventory_run("test", tmp_path, judgements)

    assert inventory["selected"] == []


def test_zero_question_id_is_a_stable_id(tmp_path: Path) -> None:
    _attempt(tmp_path, "shard_00", item_id=0, attempt=1, correct=True, prediction="right")

    inventory = inventory_run("test", tmp_path)

    assert inventory["selected"][0]["item_id"] == "0"


def test_invalid_messages_json_is_not_complete(tmp_path: Path) -> None:
    _attempt(tmp_path, "shard_00", item_id=1, attempt=1, correct=True, prediction="right")
    messages = tmp_path / "videozerobench_kira" / "shard_00" / "item_0000" / "messages.json"
    messages.write_text("{truncated", encoding="utf-8")

    inventory = inventory_run("test", tmp_path)

    assert inventory["selected"] == []
    assert inventory["summary"]["videozerobench"]["correct_items_missing_complete_artifacts"] == 1
    assert inventory["summary"]["videozerobench"]["correct_item_ids_missing_complete_artifacts"] == ["1"]


def test_inventory_output_must_be_outside_read_only_run(tmp_path: Path) -> None:
    run = tmp_path / "run"
    run.mkdir()

    with pytest.raises(ValueError, match="outside read-only run root"):
        _validate_output_location(run / "manifest.json", [("test", run)])


def test_duplicate_run_labels_are_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="run labels must be unique"):
        _parse_runs([f"same={tmp_path / 'a'}", f"same={tmp_path / 'b'}"])


def test_tie_break_is_stable_path_not_filesystem_mtime(tmp_path: Path) -> None:
    _attempt(tmp_path, "a_job", item_id=7, attempt=1, correct=True, prediction="right")
    _attempt(tmp_path, "z_job", item_id=7, attempt=1, correct=True, prediction="right")
    a_result = tmp_path / "videozerobench_kira" / "a_job" / "results.json"
    z_result = tmp_path / "videozerobench_kira" / "z_job" / "results.json"
    os.utime(a_result, ns=(200, 200))
    os.utime(z_result, ns=(100, 100))

    inventory = inventory_run("test", tmp_path)

    assert inventory["selected"][0]["result_file"].endswith("z_job/results.json")
