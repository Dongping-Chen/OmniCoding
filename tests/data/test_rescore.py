"""Tests for the post-hoc rescore helpers in scripts/rescore_kira.

Focus: file I/O safety + counter accounting. We mock out the spec entirely
so the test doesn't depend on real bench packages or live LLM responses.
"""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from omnicoding.data import rescore as rk


def _make_shard(tmp_path: Path, bench: str, shard_idx: str,
                rows: list[dict], items: list[dict],
                final_texts: dict[int, str]) -> Path:
    """Lay out a fake run_root: results.json + per-item dirs + _shards input."""
    shard_dir = tmp_path / f"{bench}_kira" / f"shard_{shard_idx}"
    shard_dir.mkdir(parents=True)
    (shard_dir / "results.json").write_text(json.dumps(rows), encoding="utf-8")
    for si, ft in final_texts.items():
        item_dir = shard_dir / f"item_{si:04d}"
        item_dir.mkdir()
        (item_dir / "final_text.txt").write_text(ft, encoding="utf-8")
    shards_dir = tmp_path / "_shards" / bench
    shards_dir.mkdir(parents=True)
    (shards_dir / f"shard_{shard_idx}.json").write_text(
        json.dumps(items), encoding="utf-8",
    )
    return shard_dir


def _fake_spec(extract_fn, is_correct_fn):
    return SimpleNamespace(extract_prediction=extract_fn, is_correct=is_correct_fn)


def test_rescore_recovers_when_extractor_now_finds_answer(tmp_path):
    """Pre: predicted=None, is_correct=False. Post: extractor now returns
    'X' and is_correct returns True. Counters reflect recovery."""
    rows = [{
        "source_index": 0,
        "predicted": None,
        "is_correct": False,
        "error": None,
    }]
    items = [{"__source_index__": 0, "answer": "X"}]
    final_texts = {0: "ramble ramble Final Answer: X\n"}
    shard_dir = _make_shard(tmp_path, "vzb", "00", rows, items, final_texts)

    spec = _fake_spec(
        extract_fn=lambda txt, item: "X" if "Final Answer: X" in txt else None,
        is_correct_fn=lambda item, pred: pred == item["answer"],
    )
    by_si = rk._items_by_source_index(items)
    cnt = rk._rescore_shard(spec, shard_dir / "results.json", by_si)

    assert cnt["recovered_to_correct"] == 1
    assert cnt["predicted_flipped"] == 1
    out = json.loads((shard_dir / "results_rescored.json").read_text())
    assert out[0]["predicted"] == "X"
    assert out[0]["is_correct"] is True
    # Original results.json is untouched.
    orig = json.loads((shard_dir / "results.json").read_text())
    assert orig[0]["predicted"] is None
    assert orig[0]["is_correct"] is False


def test_rescore_skips_errored_rows(tmp_path):
    """Rows with error set (Connection error etc.) aren't recovered by
    a smarter extractor — they had no rollout to extract from."""
    rows = [{
        "source_index": 0, "predicted": None, "is_correct": None,
        "error": "BlockTimeoutError: stuck",
    }]
    items = [{"__source_index__": 0, "answer": "X"}]
    shard_dir = _make_shard(tmp_path, "x", "00", rows, items, {})

    spec = _fake_spec(lambda *a: "X", lambda *a: True)
    cnt = rk._rescore_shard(spec, shard_dir / "results.json",
                             rk._items_by_source_index(items))
    assert cnt["skip_errored"] == 1
    assert cnt["recovered_to_correct"] == 0
    out = json.loads((shard_dir / "results_rescored.json").read_text())
    # Row written through unchanged.
    assert out[0]["predicted"] is None
    assert out[0]["error"] == "BlockTimeoutError: stuck"


def test_rescore_skips_when_final_text_missing(tmp_path):
    """If the per-item dir was never written (preempt mid-item), there's
    no text to rescore. Don't fabricate."""
    rows = [{"source_index": 0, "predicted": None, "is_correct": False, "error": None}]
    items = [{"__source_index__": 0, "answer": "X"}]
    shard_dir = _make_shard(tmp_path, "x", "00", rows, items, final_texts={})

    spec = _fake_spec(lambda *a: "X", lambda *a: True)
    cnt = rk._rescore_shard(spec, shard_dir / "results.json",
                             rk._items_by_source_index(items))
    assert cnt["skip_no_final_text"] == 1
    assert cnt["recovered_to_correct"] == 0


def test_rescore_picks_correct_predicted_field_by_existing_key(tmp_path):
    """Different specs use predicted_option / predicted_answer / predicted.
    The rescorer writes back to whichever key the original row used so the
    analyzer keeps its expectations."""
    rows = [
        {"source_index": 0, "predicted_option": "", "is_correct": False, "error": None},
        {"source_index": 1, "predicted_answer": "", "is_correct": False, "error": None},
        {"source_index": 2, "predicted": None, "is_correct": False, "error": None},
    ]
    items = [
        {"__source_index__": 0, "answer": "A"},
        {"__source_index__": 1, "answer": "B"},
        {"__source_index__": 2, "answer": "C"},
    ]
    final_texts = {0: "A", 1: "B", 2: "C"}
    shard_dir = _make_shard(tmp_path, "mixed", "00", rows, items, final_texts)

    spec = _fake_spec(
        extract_fn=lambda txt, item: txt.strip(),
        is_correct_fn=lambda item, pred: pred == item["answer"],
    )
    rk._rescore_shard(spec, shard_dir / "results.json",
                      rk._items_by_source_index(items))
    out = json.loads((shard_dir / "results_rescored.json").read_text())
    assert out[0]["predicted_option"] == "A"
    assert "predicted_answer" not in out[0]
    assert out[1]["predicted_answer"] == "B"
    assert "predicted_option" not in out[1]
    assert out[2]["predicted"] == "C"
    for r in out:
        assert r["is_correct"] is True


def test_rescore_atomic_write_no_partial_file(tmp_path):
    """Atomic tmp+rename: results_rescored.json.tmp must not survive."""
    rows = [{"source_index": 0, "predicted": "old", "is_correct": False, "error": None}]
    items = [{"__source_index__": 0, "answer": "old"}]
    final_texts = {0: "old"}
    shard_dir = _make_shard(tmp_path, "x", "00", rows, items, final_texts)

    spec = _fake_spec(lambda txt, item: txt.strip(),
                      lambda item, pred: pred == item["answer"])
    rk._rescore_shard(spec, shard_dir / "results.json",
                      rk._items_by_source_index(items))
    assert (shard_dir / "results_rescored.json").exists()
    assert not (shard_dir / "results_rescored.json.tmp").exists()


def test_rescore_records_regression(tmp_path):
    """If new extractor weakens an answer that was previously right,
    the regressed counter trips."""
    rows = [{"source_index": 0, "predicted": "X", "is_correct": True, "error": None}]
    items = [{"__source_index__": 0, "answer": "X"}]
    final_texts = {0: "no answer here"}
    shard_dir = _make_shard(tmp_path, "x", "00", rows, items, final_texts)

    spec = _fake_spec(lambda *a: None, lambda item, pred: pred == item["answer"])
    cnt = rk._rescore_shard(spec, shard_dir / "results.json",
                             rk._items_by_source_index(items))
    assert cnt["regressed_from_correct"] == 1
