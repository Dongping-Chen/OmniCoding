"""Unit tests for ``omnicoding.rl.reward``.

Coverage:
- ``extract_answer`` / ``extract_answer_from_messages``: tag walking, last-match,
  assistant-only filtering (avoid tool-reply pollution).
- ``normalize``: case + separator collapse.
- ``grade_outcome`` / ``grade_format`` / ``grade_trajectory``: per-trajectory grading.
"""

from __future__ import annotations

import pytest

from omnicoding.rl.reward import (
    DEFAULT_AUDIO_TOOLS,
    DEFAULT_IMAGE_TOOLS,
    DEFAULT_VIDEO_TOOLS,
    _audio_tools,
    _image_tools,
    _video_tools,
    apply_length_penalty,
    check_active_reward_nonzero_std,
    extract_answer,
    extract_answer_from_messages,
    grade_format,
    grade_outcome,
    grade_trajectory,
    normalize,
    reward_post_process,
    summarize_tool_calls,
)


def test_extract_answer_last():
    text = "<answer>A</answer> mid noise <answer>B</answer>"
    assert extract_answer(text) == "B"


def test_extract_answer_none():
    assert extract_answer("no tags here") is None
    assert extract_answer("") is None
    assert extract_answer("<answer></answer>") == ""  # empty answer is still extractable


def test_extract_from_messages_skips_tool_replies():
    """Tool replies that echo a prompt-file's <answer> tag must NOT pollute the
    extracted prediction."""
    msgs = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "Q. Options: <answer>BOGUS</answer> in prompt"},
        {"role": "assistant", "content": "Let me check"},
        {"role": "tool", "tool_call_id": "1", "content": "echo found <answer>FROM_TOOL</answer>"},
        {"role": "assistant", "content": "<answer>REAL</answer>"},
    ]
    assert extract_answer_from_messages(msgs) == "REAL"


def test_extract_from_messages_walks_reasoning_content():
    msgs = [
        {"role": "assistant", "content": "", "reasoning_content": "thinking <answer>X</answer> done"},
    ]
    assert extract_answer_from_messages(msgs) == "X"


def test_normalize_case_and_separators():
    assert normalize("Atlanta, Georgia") == "atlanta georgia"
    assert normalize("Atlanta and Georgia") == "atlanta georgia"
    assert normalize("  A.  ") == "a."
    assert normalize("(B)") == "(b)"


def test_grade_outcome_match():
    assert grade_outcome("A", ["A", "a", "A."], "mcq") == 1.0
    assert grade_outcome("a.", ["A.", "a"], "mcq") == 1.0
    assert grade_outcome("Z", ["A", "B"], "mcq") == 0.0
    assert grade_outcome(None, ["A"], "mcq") == 0.0


def test_grade_format_only_on_clean_completion():
    # Clean completion + answer → +0.2 bonus (was 0.0); penalty -0.2 otherwise.
    assert grade_format("task_complete", "A") == 0.2
    assert grade_format("task_complete", None) == -0.2
    assert grade_format("task_complete", "") == -0.2
    assert grade_format("step_limit", "A") == -0.2
    assert grade_format("no_tool_calls", "A") == -0.2
    assert grade_format(None, "A") == -0.2


def _assistant_answer(answer="A", tool_calls=None):
    msg = {"role": "assistant", "content": f"<answer>{answer}</answer>"}
    if tool_calls is not None:
        msg["tool_calls"] = tool_calls
    return msg


def _tool_call(name="task_complete", arguments=None):
    return {
        "id": "c1",
        "type": "function",
        "function": {"name": name, "arguments": "{}" if arguments is None else arguments},
    }


def test_grade_trajectory_requires_tool_for_correctness():
    # Format good but no tool used → correctness gated to 0; reward base
    # falls through to FORMAT_BONUS = 0.2.
    msgs = [{"role": "assistant", "content": "<answer>A</answer>"}]
    details = grade_trajectory(msgs, ["A"], "mcq", exit_reason="task_complete")
    assert details["raw_acc"] == 1.0
    assert details["correctness"] == 0.0
    assert details["score"] == 0.2  # FORMAT_BONUS (correctness=0)

    # Format good AND tool used AND answer correct → correctness=1, base=1.0,
    # NO format bonus stacking.
    msgs = [_assistant_answer(tool_calls=[_tool_call()])]
    details = grade_trajectory(msgs, ["A"], "mcq", exit_reason="task_complete")
    assert details["correctness"] == 1.0
    assert details["score"] == 1.0  # base = correctness, no fmt bonus


def test_grade_trajectory_can_use_agent_tool_count_fallback():
    msgs = [{"role": "assistant", "content": "<answer>A</answer>"}]
    details = grade_trajectory(msgs, ["A"], "mcq", exit_reason="task_complete", observed_n_tool_calls=1)
    assert details["n_tool"] == 1.0
    assert details["correctness"] == 1.0


def test_grade_trajectory_bad_tool_penalty_and_escape():
    msgs = [
        _assistant_answer(
            tool_calls=[
                _tool_call(
                    "execute_commands",
                    '{"keystrokes": "cat /tmp/ground_truth.json"}',
                )
            ]
        )
    ]
    details = grade_trajectory(msgs, ["A"], "mcq", exit_reason="task_complete")
    assert details["n_escape"] == 1.0
    assert details["p_bad_tool"] == 0.5
    assert details["correctness"] == 0.0
    # 0 (correctness, hard-zeroed by escape) + 0.2 (format bonus, exit clean
    # + answer present) + 0 (no modality) + (-0.5 × 0.5 = -0.25 bad_tool)
    # = -0.05
    assert details["score"] == pytest.approx(-0.05)


def test_modality_logic_is_configurable_via_env(monkeypatch):
    """Set env to a single tool; only that tool counts. Confirms env override
    fully replaces the default set (not merges with it)."""
    msgs = [_assistant_answer(tool_calls=[_tool_call("image_read")])]
    monkeypatch.setenv("RELAX_ROUTER_VIDEO_TOOLS", "some_other_tool")
    monkeypatch.setenv("RELAX_ROUTER_IMAGE_TOOLS", "")
    details = grade_trajectory(
        msgs,
        ["A"],
        "mcq",
        exit_reason="task_complete",
        media={"videos": ["v.mp4"], "audios": [], "images": []},
    )
    # image_read is in the default video set (via image_tools union) but env
    # narrowed it to "some_other_tool" only → no match → modality fails.
    assert details["raw_acc"] == 1.0
    assert details["modality_match"] == 0.0
    assert details["correctness"] == 0.0

    monkeypatch.setenv("RELAX_ROUTER_VIDEO_TOOLS", "image_read")
    details = grade_trajectory(
        msgs,
        ["A"],
        "mcq",
        exit_reason="task_complete",
        media={"videos": ["v.mp4"], "audios": [], "images": []},
    )
    assert details["modality_match"] == 1.0
    assert details["correctness"] == 1.0


def test_timeout_without_answer_is_removed():
    details = grade_trajectory([], ["A"], "mcq", exit_reason="timeout")
    assert details["removed"] == 1.0
    assert details["score"] == 0.0


def test_tool_summary_counts_unparseable_disallowed_and_syntax_failure():
    msgs = [
        {"role": "assistant", "content": "<tool_call>{bad json}</tool_call>"},
        {"role": "assistant", "content": "", "tool_calls": [_tool_call("forbidden_tool")]},
        {"role": "tool", "content": "SyntaxError: invalid syntax"},
    ]
    summary = summarize_tool_calls(msgs)
    assert summary["fail_counts"]["unparseable"] == 1
    assert summary["fail_counts"]["disallowed"] == 1
    assert summary["fail_counts"]["syntax-fail"] == 1
    assert summary["n_tool"] == 2


def test_length_penalty_scales_correct_samples():
    results = [
        {"score": 1.0, "correctness": 1.0, "modality_match": 1.0, "n_tool": 1.0, "num_steps": 3.0},
        {"score": 1.0, "correctness": 1.0, "modality_match": 1.0, "n_tool": 1.0, "num_steps": 6.0},
        {"score": -0.2, "correctness": 0.0, "modality_match": 1.0, "n_tool": 1.0, "num_steps": 10.0},
    ]
    apply_length_penalty(results)
    assert results[0]["length_factor"] == 1.0
    assert round(results[1]["length_factor"], 3) == 0.8
    assert results[1]["score"] == pytest.approx(0.8)
    assert results[2]["score"] == -0.2


class _Args:
    reward_key = "score"
    advantage_estimator = "gspo"
    rewards_normalization = True
    grpo_std_normalization = False
    n_samples_per_prompt = 4


class _Sample:
    def __init__(self, reward, remove_sample=False):
        self.reward = {"score": reward}
        self.remove_sample = remove_sample

    def get_reward_value(self, args):
        return self.reward[args.reward_key]


def test_reward_post_process_ignores_removed_samples():
    samples = [_Sample(1.0), _Sample(0.0, remove_sample=True), _Sample(0.0), _Sample(1.0)]
    raw, rewards = reward_post_process(_Args(), samples)
    assert raw == [1.0, 0.0, 0.0, 1.0]
    assert rewards == pytest.approx([1 / 3, 0.0, -2 / 3, 1 / 3])


def test_active_dynamic_filter_ignores_removed_samples():
    keep = check_active_reward_nonzero_std(_Args(), [_Sample(1.0), _Sample(0.0, remove_sample=True), _Sample(0.0)])
    assert keep.keep
    drop = check_active_reward_nonzero_std(_Args(), [_Sample(1.0), _Sample(0.0, remove_sample=True), _Sample(1.0)])
    assert not drop.keep


# ─── modality tool universe defaults ─────────────────────────────────────────
# Regression guard: empty defaults are a footgun — every video/audio task gets
# modality_match=False → gated correctness=0 → ~95% of dataset gets no positive
# reward. Lock the canonical lists so accidental refactors that revert to
# empty-set defaults break loudly.


def test_modality_default_sets_are_nonempty(monkeypatch):
    """Without env vars set, _video/_audio/_image_tools must return the
    canonical defaults (non-empty)."""
    monkeypatch.delenv("RELAX_ROUTER_VIDEO_TOOLS", raising=False)
    monkeypatch.delenv("RELAX_ROUTER_AUDIO_TOOLS", raising=False)
    monkeypatch.delenv("RELAX_ROUTER_IMAGE_TOOLS", raising=False)
    assert len(_video_tools()) > 0, "video defaults empty — gating will collapse all video tasks"
    assert len(_audio_tools()) > 0, "audio defaults empty — gating will collapse all audio tasks"
    assert len(_image_tools()) > 0, "image defaults empty"
    # The DEFAULT_* constants are the source of truth; the helper functions
    # should fall back to them when env unset.
    assert _video_tools() == set(DEFAULT_VIDEO_TOOLS)
    assert _audio_tools() == set(DEFAULT_AUDIO_TOOLS)
    assert _image_tools() == set(DEFAULT_IMAGE_TOOLS)


def test_modality_defaults_cover_canonical_helpers():
    """Sanity: the default sets must contain the most common shell helpers
    actually used in the dataset's reference solutions. If you add a new tool
    universe, extend these checks accordingly."""
    assert "ffmpeg" in DEFAULT_VIDEO_TOOLS
    assert "whisper" in DEFAULT_AUDIO_TOOLS
    assert "image_read" in DEFAULT_IMAGE_TOOLS  # kira's image_read tool
    # PIL.Image.open is an `execute_commands` keyword (Python script that opens
    # an image); kira-loop.py:_dispatch_image_read native path also routes here.
    assert "PIL.Image.open" in DEFAULT_IMAGE_TOOLS


def test_modality_env_var_overrides_default(monkeypatch):
    """Env var, when set, must override the default (lets users tighten /
    expand the universe per-experiment without code edits)."""
    monkeypatch.setenv("RELAX_ROUTER_VIDEO_TOOLS", "only_one_thing")
    assert _video_tools() == {"only_one_thing"}


def test_modality_env_empty_string_yields_empty_set(monkeypatch):
    """Explicit empty env (intentional disable) must return empty — the
    DEFAULT fallback only kicks in when env is UNSET, not when set to ''."""
    monkeypatch.setenv("RELAX_ROUTER_AUDIO_TOOLS", "")
    assert _audio_tools() == set()


def test_modality_grading_works_with_defaults(monkeypatch):
    """End-to-end: a video task whose trajectory uses ffmpeg via
    execute_commands must reach correctness=1 with just the defaults — no env
    var needed. This is what the live coordinator does."""
    monkeypatch.delenv("RELAX_ROUTER_VIDEO_TOOLS", raising=False)
    monkeypatch.delenv("RELAX_ROUTER_AUDIO_TOOLS", raising=False)
    monkeypatch.delenv("RELAX_ROUTER_IMAGE_TOOLS", raising=False)
    msgs = [
        {"role": "system", "content": "agent"},
        {"role": "user", "content": "Look at the video"},
        {"role": "assistant", "content": "I'll inspect it.",
         "tool_calls": [{"id": "c0", "type": "function",
                         "function": {"name": "execute_commands",
                                      "arguments": '{"keystrokes":"ffmpeg -i x.mp4 -vf fps=1 frames/%03d.jpg"}'}}]},
        {"role": "tool", "tool_call_id": "c0", "content": "frames extracted"},
        {"role": "assistant", "content": "<answer>blue</answer>"},
    ]
    res = grade_trajectory(
        msgs, ["blue"], "open",
        exit_reason="task_complete",
        media={"videos": ["x.mp4"]},
    )
    assert res["modality_match"] == 1.0, "ffmpeg in default video tools — should match"
    assert res["correctness"] == 1.0, "raw_acc=1 + tool_used + modality_ok → gated correctness=1"
    # base = correctness = 1.0 (no fmt bonus stacking); modality_penalty=0; bad_tool=0
    assert res["score"] == pytest.approx(1.0)
