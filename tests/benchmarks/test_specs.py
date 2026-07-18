"""BenchSpec extractor tests.

Each test case covers the four production input shapes:
  1. Clean: model emits the answer tag exactly as requested.
  2. Think-leak: tag is inside `<think>...</think>` (Qwen3.6 + thinking ON).
  3. Bare: model emits the bare letter without the XML wrapper.
  4. None: model never commits an answer.

These tests document which path is taken so we can delete the
defensive fallbacks they prove to be unnecessary.

Run from the repo root:
    cd coding_agent_benchmarks_1/benchmarks
    PYTHONPATH=.:./lvomnibench/src:./socialomni:./videozerobench \
        python -m pytest tests/ -q
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
for path in (
    REPO_ROOT,
    REPO_ROOT / "omnigaia",
    REPO_ROOT / "lvomnibench" / "src",
    REPO_ROOT / "socialomni",
    REPO_ROOT / "videozerobench",
):
    sys.path.insert(0, str(path))

from omnicoding.benchmarks import specs  # noqa: E402


# ---------------------------------------------------------------------------
# LVOmniBench
# ---------------------------------------------------------------------------


@pytest.fixture
def lvo_item() -> dict:
    return {
        "question_id": "q0",
        "video_id": "vid0",
        "options": ["A. apple", "B. banana", "C. cherry", "D. date"],
        "correct_option": "C",
    }


def test_lvo_clean_tag(lvo_item):
    spec = specs.get("lvomnibench")
    assert spec.extract_prediction("Found it. <answer>C</answer>", lvo_item) == "C"
    assert spec.is_correct(lvo_item, "C") is True


def test_lvo_think_leak(lvo_item):
    spec = specs.get("lvomnibench")
    text = "<think>after frame inspection the answer is <answer>C</answer></think>"
    assert spec.extract_prediction(text, lvo_item) == "C"


def test_lvo_bare_letter(lvo_item):
    spec = specs.get("lvomnibench")
    assert spec.extract_prediction("The answer is C.", lvo_item) == "C"


def test_lvo_no_answer(lvo_item):
    spec = specs.get("lvomnibench")
    assert spec.extract_prediction("Let me try one more thing...", lvo_item) == ""
    assert spec.is_correct(lvo_item, "") is False


def test_lvo_wrong_answer(lvo_item):
    spec = specs.get("lvomnibench")
    assert spec.extract_prediction("<answer>A</answer>", lvo_item) == "A"
    assert spec.is_correct(lvo_item, "A") is False


# ---------------------------------------------------------------------------
# SocialOmni Level 1
# ---------------------------------------------------------------------------


@pytest.fixture
def l1_item() -> dict:
    return {
        "id": "s1_0",
        "video_path": "level_1/videos/foo.mp4",
        "options": ["A. Alice", "B. Bob", "C. Carol", "D. Dave"],
        "correct_answer": "B",
    }


def test_l1_clean_tag(l1_item):
    spec = specs.get("socialomni_l1")
    assert spec.extract_prediction("<answer>B</answer>", l1_item) == "B"
    assert spec.is_correct(l1_item, "B") is True


def test_l1_think_leak(l1_item):
    spec = specs.get("socialomni_l1")
    text = "<think>I think the speaker is Bob, so <answer>B</answer></think>"
    assert spec.extract_prediction(text, l1_item) == "B"


def test_l1_bare_letter_in_prose(l1_item):
    spec = specs.get("socialomni_l1")
    assert spec.extract_prediction("Therefore the answer is: B.", l1_item) == "B"


def test_l1_no_answer(l1_item):
    spec = specs.get("socialomni_l1")
    assert spec.extract_prediction("Hmm, the audio is unclear.", l1_item) == ""


# ---------------------------------------------------------------------------
# SocialOmni Level 2 question 1 (yes/no)
# ---------------------------------------------------------------------------


@pytest.fixture
def l2_item() -> dict:
    return {
        "video_id": "v0",
        "video_file": "level_2/videos/v0.mp4",
        "question_1": {
            "question": "Was the speaker interrupted?",
            "option_A": "YES",
            "option_B": "NO",
            "correct_answer": "A",
        },
    }


def test_l2_clean_tag(l2_item):
    spec = specs.get("socialomni_l2")
    assert spec.extract_prediction("<answer>A</answer>", l2_item) == "A"
    assert spec.is_correct(l2_item, "A") is True


def test_l2_think_leak(l2_item):
    spec = specs.get("socialomni_l2")
    text = "<think>looks interrupted around 0:14, so the answer is <answer>A</answer></think>"
    assert spec.extract_prediction(text, l2_item) == "A"


def test_l2_yes_no_fallback(l2_item):
    spec = specs.get("socialomni_l2")
    assert spec.extract_prediction("Yes, the speaker was interrupted.", l2_item) == "A"
    assert spec.extract_prediction("No, the speaker was not interrupted.", l2_item) == "B"


def test_l2_no_answer(l2_item):
    spec = specs.get("socialomni_l2")
    assert spec.extract_prediction("Need more analysis.", l2_item) == ""


# ---------------------------------------------------------------------------
# VideoZeroBench (Level-3 only, single answer-tag)
# ---------------------------------------------------------------------------


@pytest.fixture
def vzb_item() -> dict:
    return {
        "question_id": 0,
        "video": "vid0.mp4",
        "video_id": "vid0",
        "language": "en",
        "category": "object",
        "answer": "front right",
    }


def test_vzb_clean_tag(vzb_item):
    spec = specs.get("videozerobench")
    text = "Tools showed the speaker is to the right. <answer>front right</answer>"
    assert spec.extract_prediction(text, vzb_item) == "front right"
    assert spec.is_correct(vzb_item, "front right") is True


def test_vzb_think_leak(vzb_item):
    spec = specs.get("videozerobench")
    text = (
        "<think>example tag <answer>placeholder</answer> never use this</think>\n"
        "After inspecting frame 390.7, <answer>front right</answer>"
    )
    assert spec.extract_prediction(text, vzb_item) == "front right"


def test_vzb_no_tag(vzb_item):
    spec = specs.get("videozerobench")
    assert spec.extract_prediction("Cannot determine.", vzb_item) == ""
    assert spec.is_correct(vzb_item, "") is False


def test_vzb_numeric_normalization():
    spec = specs.get("videozerobench")
    assert spec.is_correct({"answer": "8"}, "8.0") is True
    assert spec.is_correct({"answer": "8"}, "08") is True
    assert spec.is_correct({"answer": "8"}, "8") is True
    assert spec.is_correct({"answer": "8"}, "9") is False


def test_vzb_text_normalization():
    spec = specs.get("videozerobench")
    assert spec.is_correct({"answer": "front right"}, "Front Right") is True
    assert spec.is_correct({"answer": "front right"}, "front-right") is True
    assert spec.is_correct({"answer": "front right"}, "front right.") is True


def test_vzb_missing_gold_returns_none():
    spec = specs.get("videozerobench")
    assert spec.is_correct({}, "8") is None
    assert spec.is_correct({"answer": ""}, "8") is None


# ---------------------------------------------------------------------------
# VZB BUG-X1 lenient-fallback (model wrote answer in prose only,
# typically because the shell echo hop hung — see
# build_completion_checklist in local_model/kira/tools.py and the
# matching pattern set in videozerobench_prompting._FINAL_ANSWER_RE).
# ---------------------------------------------------------------------------


def test_vzb_lenient_fallback_final_answer_header(vzb_item):
    """When the strict ``<answer>`` tag is absent but the model
    explicitly wrote ``Final Answer: X`` in its closing prose, recover
    X. This is the failure mode BUG-X1 documents: model knows the
    answer, signposts it clearly, but never wraps it because the shell
    echo path failed."""
    spec = specs.get("videozerobench")
    text = (
        "I extracted frames at 4:21 and counted ducks visually.\n"
        "After careful analysis the count is consistent across "
        "neighbouring frames.\n\n"
        "Final Answer: 5"
    )
    # No strict tag present.
    assert "<answer>" not in text
    assert spec.extract_prediction(text, vzb_item) == "5"


def test_vzb_lenient_fallback_the_answer_is(vzb_item):
    spec = specs.get("videozerobench")
    text = "Looking at the spinning carousel, the answer is: clockwise."
    assert spec.extract_prediction(text, vzb_item) == "clockwise"


def test_vzb_lenient_fallback_bolded(vzb_item):
    spec = specs.get("videozerobench")
    text = "Counting carefully... **Final Answer:** 7 ducks."
    assert spec.extract_prediction(text, vzb_item) == "7 ducks"


def test_vzb_lenient_fallback_last_match_wins(vzb_item):
    """When the model rehearses one ``Final Answer:`` early then commits
    a different one later, the LAST match wins — mirrors the strict
    extractor's last-match policy for ``<answer>`` tags."""
    spec = specs.get("videozerobench")
    text = (
        "Initially I thought the answer is: 5.\n"
        "Then I rechecked: actually one of those was a pigeon.\n"
        "Final Answer: 4"
    )
    assert spec.extract_prediction(text, vzb_item) == "4"


def test_vzb_lenient_fallback_strict_tag_takes_precedence(vzb_item):
    """If both a strict ``<answer>`` tag AND a ``Final Answer:`` prose
    cue exist, strict wins. This protects against a model that writes
    a rehearsal "Final Answer: 5" inside its <think> CoT before
    committing the real ``<answer>front right</answer>`` outside —
    the strict-tag branch must short-circuit before the lenient pass
    runs."""
    spec = specs.get("videozerobench")
    text = (
        "<think>I think the Final Answer: 5 is wrong, let me recount</think>\n"
        "<answer>front right</answer>"
    )
    assert spec.extract_prediction(text, vzb_item) == "front right"


def test_vzb_lenient_fallback_rejects_filler(vzb_item):
    """Hedge words that mean "I don't know" are rejected so we count
    them as no-answer rather than scoring a junk string. Without this
    guard, a Qwen "the answer is unclear" hedge would surface as
    prediction=``unclear from the evidence`` and confuse the analyzer's
    is_correct=False bucket."""
    spec = specs.get("videozerobench")
    for text in (
        "the answer is unclear from this evidence",
        "the answer is unknown",
        "Final Answer: I cannot determine",
        "the answer is yes",  # one-word filler
        "Final Answer: above",
    ):
        assert spec.extract_prediction(text, vzb_item) == "", text


def test_vzb_lenient_fallback_rejects_terminal_noise(vzb_item):
    """The lenient fallback must NEVER pick up an "answer" that came
    from terminal output (ffprobe / ls / find). The "Final Answer" /
    "the answer is" / "Answer:" anchor is what protects this — these
    tokens essentially never appear in the shell's stdout."""
    spec = specs.get("videozerobench")
    for text in (
        "ffprobe: duration=626.521 seconds, bitrate=917 kb/s",
        "size=    1875KiB time=00:01:00.00 bitrate= 256.0kbits/s",
        "-rw-rw-r--. 1 dongping  157 Apr 27 03:43 short.txt",
        "[image2 @ 0x55b611317f40] video:81KiB audio:0KiB",
        "Frame 5 shows a duck. Let me check more frames.",
    ):
        assert spec.extract_prediction(text, vzb_item) == "", text


def test_vzb_lenient_fallback_only_searches_tail(vzb_item):
    """The lenient pass searches only the LAST ~4 KB. A "Final Answer:"
    cue that appears in the model's first-step monologue (which the
    model later contradicts but doesn't restate at the end) must NOT
    be promoted. We pad with 5 KB of plausible early-trajectory junk
    to push the early cue out of the search window."""
    spec = specs.get("videozerobench")
    early = "Final Answer: 99\n"
    padding = ("Frame analysis output line\n" * 250)  # ~6.5 KB filler
    text = early + padding + "Tail prose without answer.\n"
    assert len(text) > 4000
    assert spec.extract_prediction(text, vzb_item) == ""


def test_vzb_no_groups_by():
    """VZB is per-question; the harness must not branch into _run_batch_loop."""
    spec = specs.get("videozerobench")
    assert spec.groups_by is None


def test_vzb_result_row_shape(vzb_item):
    """The official evaluator reads ``level3_answer`` at top level. Make
    sure ``result_row`` writes it there, and does not write level1/2/4/5
    fields that would trick ``has_prediction_field`` into reporting
    them as attempted."""
    spec = specs.get("videozerobench")
    from omnicoding.benchmarks.common.spec import ResultRowCtx
    row = spec.result_row(
        ResultRowCtx(
            item=vzb_item,
            item_index=0,
            prediction="front right",
            is_correct=True,
            raw_model_output="<answer>front right</answer>",
            tool_call_num=3,
            return_code=0,
            timed_out=False,
            stdout_text="",
            stderr_text="",
            workspace_dir=Path("/tmp/vzb_test"),
            keep_workdirs=False,
            include_gold_fields=False,
            extra={},
        )
    )
    assert row["level3_answer"] == "front right"
    assert row["is_correct"] is True
    assert row["question_id"] == 0
    for forbidden in ("level1_answer", "level2_answer", "level4_temporal", "level5_spatial",
                      "predicted", "levels_requested", "runs"):
        assert forbidden not in row, f"row leaks {forbidden!r} which would mislead the evaluator"


def test_vzb_eval_pipeline(tmp_path):
    """End-to-end: feed a few harness rows into ``evaluate_videozerobench``
    and assert that L3 is the only attempted level and that
    ``level3_accuracy`` matches the simulated correctness pattern."""
    eval_path = REPO_ROOT / "videozerobench"
    sys.path.insert(0, str(eval_path))
    from omnicoding.benchmarks.evaluation import videozerobench as ev

    annotations = [
        {"question_id": 0, "answer": "8", "video": "v0.mp4", "video_id": "v0",
         "category": "Daily Vlogs", "language": "en", "evidence_span": "short-term",
         "annotation_capabilities": ["counting"],
         "evidence_windows": [{"start": 1, "end": 2}], "evidence_boxes": []},
        {"question_id": 1, "answer": "front right", "video": "v0.mp4", "video_id": "v0",
         "category": "Daily Vlogs", "language": "en", "evidence_span": "short-term",
         "annotation_capabilities": ["spatial orientation discrimination"],
         "evidence_windows": [{"start": 3, "end": 4}], "evidence_boxes": []},
        {"question_id": 2, "answer": "clockwise", "video": "v1.mp4", "video_id": "v1",
         "category": "Daily Vlogs", "language": "en", "evidence_span": "single-frame",
         "annotation_capabilities": ["spatial orientation discrimination"],
         "evidence_windows": [], "evidence_boxes": []},
    ]
    # Predictions in the exact shape ``_result_row`` writes.
    predictions = ev.build_prediction_index([
        {"question_id": 0, "level3_answer": "8", "is_correct": True},
        {"question_id": 1, "level3_answer": "front right", "is_correct": True},
        {"question_id": 2, "level3_answer": "counterclockwise", "is_correct": False},
    ])
    report, _rows = ev.evaluate(annotations, predictions, time_tolerance=0.2)
    overall = report["overall"]

    # 2 of 3 correct ⇒ 66.67% (rounded by evaluator).
    assert overall["level3_attempted_count"] == 3
    assert overall["level3_correct_count"] == 2
    assert overall["level3_accuracy"] == pytest.approx(66.67, abs=0.01)

    # Other levels must be reported as never attempted (no level{1,2,4,5}
    # field present in our rows, no ``runs`` block, no ``levels_requested``).
    assert overall["level1_attempted_count"] == 0
    assert overall["level2_attempted_count"] == 0
    assert overall["level4_attempted_count"] == 0
    assert overall["level5_attempted_count"] == 0


# ---------------------------------------------------------------------------
# omnigaia extract_answer — placeholder shadowing
# ---------------------------------------------------------------------------


def _omnigaia_extract():
    from omnicoding.benchmarks.prompts.omnigaia_prompting import extract_answer
    return extract_answer


def test_omnigaia_real_then_placeholder():
    extract = _omnigaia_extract()
    # Model wrote a real answer first, then kept thinking and emitted a
    # stray "..." — extractor must skip the placeholder.
    text = "<answer>1234</answer> ... thinking ... <answer>...</answer>"
    assert extract(text) == "1234"


def test_omnigaia_placeholder_then_real():
    extract = _omnigaia_extract()
    text = "<answer>YOUR_ANSWER</answer> wait <answer>5678</answer>"
    assert extract(text) == "5678"


def test_omnigaia_only_placeholder_falls_back():
    extract = _omnigaia_extract()
    assert extract("<answer>...</answer>") == "..."


def test_omnigaia_self_correct_last_wins():
    extract = _omnigaia_extract()
    text = "<answer>100</answer> oops <answer>200</answer>"
    assert extract(text) == "200"


def test_omnigaia_single_char_numeric_kept():
    # Legit short answers ("0", "5", "Y") must NOT be classified placeholder.
    extract = _omnigaia_extract()
    assert extract("<answer>some thinking</answer> <answer>0</answer>") == "0"
    assert extract("<answer>5</answer>") == "5"
    assert extract("<answer>Y</answer>") == "Y"


def test_omnigaia_punct_only_placeholder():
    # Strings of only punctuation ("---", "...", "..", ".") are placeholders.
    extract = _omnigaia_extract()
    assert extract("<answer>foo</answer> <answer>---</answer>") == "foo"


def test_omnigaia_no_tag_returns_empty():
    extract = _omnigaia_extract()
    assert extract("model never wrote an answer tag") == ""
    assert extract("") == ""


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


def test_registry_completeness():
    assert set(specs.names()) == {
        "lvomnibench", "omnigaia", "socialomni_l1", "socialomni_l2", "videozerobench",
    }


def test_registry_lookup_unknown():
    with pytest.raises(KeyError):
        specs.get("nope")
