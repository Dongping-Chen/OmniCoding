"""Regression tests for `omnigaia.omnigaia_prompting._build_question_block`.

Round-17.8 production bug: MCQ items in the unified SFT pool carry an
``options`` field (e.g. ``["A. 2.", "B. 3.", ...]``). The original
prompt builder was written for OmniGAIA's free-form items and ignored
``options`` — so MCQs were rendered as a question with the instruction
"Answer with the option letter only" but no choices visible to the
model. Across the 1000-item scale-up that collapsed MCQ accuracy to
27% (≈ random for 4-option) while open-ended stayed at 84%.

These tests pin the contract: every ``option`` string must appear
verbatim in the rendered prompt, under an ``Options:`` header. Open
items must be unchanged (no ``Options:`` header, no extra lines).
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
for path in (REPO_ROOT,):
    sys.path.insert(0, str(path))

from omnicoding.benchmarks.prompts.omnigaia_prompting import _build_question_block


def test_mcq_options_rendered_with_header():
    item = {
        "id": "ovb:390",
        "answer_type": "mcq",
        "question": "How many super fans did I meet?\n\nAnswer with the option letter only (e.g., A).",
        "options": ["A. 2.", "B. 3.", "C. 4.", "D. 5."],
    }
    rendered = "\n".join(_build_question_block(item))
    assert "Options:" in rendered, "MCQ prompt must include `Options:` header"
    for opt in item["options"]:
        assert opt in rendered, f"option {opt!r} missing from prompt:\n{rendered}"
    # Header should come before the options and after the question.
    q_idx = rendered.index("Question:")
    o_idx = rendered.index("Options:")
    a_idx = rendered.index("A. 2.")
    assert q_idx < o_idx < a_idx, "block order must be Question → Options → choices"


def test_mcq_options_with_long_descriptions():
    """Real ovb-style item with full sentence per option — every full
    sentence must land verbatim, no truncation/escaping."""
    item = {
        "id": "vmme2:222-1",
        "answer_type": "mcq",
        "question": "What is the video frame when the narrator says ...?",
        "options": [
            "A. The narrator holds a fully opened cocoa pod.",
            "B. The narrator takes a large bite directly from the entire white pulp structure.",
            "C. A close-up shot of one hand pinching a small piece of white cocoa pulp.",
            "D. A shot showing several whole cocoa pods of different colors.",
            "E. A hand breaks open a seed covered in pulp.",
            "F. A worker's gloved hands cut open a fresh cocoa pod.",
            "G. A large white bucket filled with freshly harvested cocoa pulp.",
            "H. A man sits in a cocoa grove.",
        ],
    }
    rendered = "\n".join(_build_question_block(item))
    for opt in item["options"]:
        assert opt in rendered, f"long option missing: {opt!r}"


def test_open_item_unchanged_no_options_header():
    """Open-ended items have ``options=None``. Renderer must NOT add an
    ``Options:`` header (keeps the historical prompt format intact for
    the 84%-accurate open trajectories — no need to re-run them)."""
    item = {
        "id": "omnimodal:1676",
        "answer_type": "open",
        "question": "Identify the model of the humanoid robot ...",
        "options": None,
    }
    rendered = "\n".join(_build_question_block(item))
    assert "Options:" not in rendered, (
        "open items must not get an Options: header; "
        f"got: {rendered}"
    )
    assert "Question:" in rendered
    assert item["question"] in rendered


def test_open_item_with_missing_options_key_unchanged():
    """Earlier datasets may not even include the ``options`` key. Same
    behaviour expected as ``options=None``."""
    item = {
        "id": "x",
        "answer_type": "open",
        "question": "What is X?",
    }
    rendered = "\n".join(_build_question_block(item))
    assert "Options:" not in rendered


def test_options_empty_list_treated_as_no_options():
    """Edge case: ``options=[]`` should not render an empty Options
    header — be permissive, treat the empty list as "no options"."""
    item = {
        "id": "x",
        "answer_type": "mcq",
        "question": "Q?",
        "options": [],
    }
    rendered = "\n".join(_build_question_block(item))
    assert "Options:" not in rendered


def test_mcq_question_block_no_longer_carries_wrapper_rule():
    """Round-17.11 (2026-04-30): the general "wrap your answer in
    <answer></answer> + emit as plain text + then task_complete"
    protocol moved from per-item user_question to role=system via
    ``common/spec.py:FINAL_ANSWER_PROTOCOL``. The omnigaia question
    block must no longer carry that rule (otherwise it gets duplicated
    on every item — wasted tokens, more prompt-cache busts).

    Per-item content rules (e.g. MCQ "Answer with the option letter
    only (e.g., A).") still live in ``item['question']`` text and are
    rendered verbatim — those are not affected.
    """
    item = {
        "id": "x",
        "answer_type": "mcq",
        "question": "Q?",
        "options": ["A. foo", "B. bar"],
    }
    rendered = "\n".join(_build_question_block(item))
    # The protocol-style wrapper sentence must be GONE from the per-item
    # block. Cross-bench coverage (system-level presence) is in
    # test_unified_prompt.py.
    assert "wrap your final answer" not in rendered.lower(), (
        "wrapper rule must move to role=system via FINAL_ANSWER_PROTOCOL"
    )
    assert "After you have finished using tools" not in rendered, (
        "redundant pre-protocol sentence must be removed"
    )


def test_mcq_non_string_option_coerced():
    """Defensive: if a future loader produces non-string options (int,
    dict), they should be str()-coerced rather than crashing the
    renderer mid-batch."""
    item = {
        "id": "x",
        "answer_type": "mcq",
        "question": "Q?",
        "options": [1, 2, "C. three"],
    }
    rendered = "\n".join(_build_question_block(item))
    assert "1" in rendered
    assert "2" in rendered
    assert "C. three" in rendered
