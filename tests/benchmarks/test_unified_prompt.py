"""Cross-benchmark prompt-shape consistency tests.

The kira harness is the single inference pipeline used to generate
synthetic data AND run benchmark eval. Every benchmark's
``BenchSpec`` must produce a prompt with the same structural shape:

  role=system =
    [agent_md (optional)]
    workspace_instructions(...)
    Codex exec sandbox mode: <sandbox>
    [native_vision_restriction (optional)]
    network_instructions(...)
    [gpu_instructions (optional)]
    [shared_python_env_instructions (optional)]
    tool_workflow_instructions(...)
    [spec-specific extras: web_search hint, answer-format hints, ...]

  role=user =
    Available staged files:
    - <staged_path_0>
    ...
    [per-spec question block: Question + Options + answer-format text]

Together they guarantee:

1. ``build_system_prefix`` is byte-identical across items in a run
   (only depends on ``ctx.sandbox`` / ``ctx.allow_*`` / ``ctx.shared_
   python_env`` / ``ctx.disable_native_vision`` — none of which are
   per-item) → LLM provider's prompt cache holds it once.
2. ``build_user_question`` is per-item (Question + staged paths
   change) → small, encoded freshly per item.
3. ``build_codex_prompt(ctx) == build_system_prefix(ctx) + "\\n\\n"
   + build_user_question(ctx)`` exactly → codex-cli, mini-swe-agent,
   and kira see the same prompt content (only role assignment differs).

If any of these break, train/serve parity is broken: the model
trained on kira trajectories will see a different prompt structure
when evaluated through codex-cli, or worse, the prompt cache will
not catch and benchmark token costs will balloon.
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

from omnicoding.benchmarks.common.spec import BuildPromptCtx, FINAL_ANSWER_PROTOCOL  # noqa: E402
from omnicoding.benchmarks import specs  # noqa: E402


def _ctx(item: dict, staged_paths: list[Path]) -> BuildPromptCtx:
    return BuildPromptCtx(
        item=item,
        staged_paths=staged_paths,
        sandbox="workspace-write",
        allow_shell_network=True,
        allow_shell_gpu=True,
        shared_python_env="/tmp/v",
        disable_native_vision=False,
        extra_system_prompt="",
    )


# Minimal items that satisfy each spec's question_block builder.
_SAMPLE_ITEMS = {
    "omnigaia": [
        {"id": "ovb:1", "question": "Q1?", "options": ["A. x", "B. y"], "answer_type": "mcq"},
        {"id": "ovb:2", "question": "Q2?", "answer_type": "open"},
        {"id": "ovb:3", "question": "Q3?", "options": ["A. p", "B. q", "C. r"], "answer_type": "mcq"},
    ],
    "lvomnibench": [
        {"question_id": "q1", "video_id": "v1", "question": "Q1?",
         "options": ["A. red", "B. blue", "C. green", "D. yellow"]},
        {"question_id": "q2", "video_id": "v2", "question": "Q2?",
         "options": ["A. man", "B. woman", "C. child", "D. robot"]},
    ],
    "videozerobench": [
        {"question_id": "v1", "video": "v1.mp4", "question": "Length?"},
        {"question_id": "v2", "video": "v2.mp4", "question": "Setting?"},
    ],
    "socialomni_l1": [
        {"id": "s1", "video_path": "v1.mp4", "question": "Q1?",
         "options": ["A. yes", "B. no", "C. maybe", "D. unsure"]},
        {"id": "s2", "video_path": "v2.mp4", "question": "Q2?",
         "options": ["A. red", "B. blue", "C. green", "D. yellow"]},
    ],
    "socialomni_l2": [
        {"video_id": "v1", "video_file": "v1.mp4",
         "question_1": {"question": "Yes or no?", "option_A": "YES", "option_B": "NO"}},
        {"video_id": "v2", "video_file": "v2.mp4",
         "question_1": {"question": "Yes or no?", "option_A": "YES", "option_B": "NO"}},
    ],
}


@pytest.mark.parametrize("bench", list(_SAMPLE_ITEMS.keys()))
def test_split_builders_present(bench: str) -> None:
    """Every spec exposes both ``build_system_prefix`` and
    ``build_user_question`` (post-2026-04-29 unified split). The
    fallback path in run_bench_kira was removed; absence here would
    break dispatch."""
    spec = specs.get(bench)
    assert spec.build_system_prefix is not None, f"{bench} missing build_system_prefix"
    assert spec.build_user_question is not None, f"{bench} missing build_user_question"
    assert spec.build_codex_prompt is not None, f"{bench} missing build_codex_prompt"


@pytest.mark.parametrize("bench", list(_SAMPLE_ITEMS.keys()))
def test_system_prefix_is_byte_identical_across_items(bench: str) -> None:
    """The system_prefix depends only on ctx.sandbox / .allow_* — all
    items in a dispatch run produce the SAME bytes. This is what lets
    the LLM provider's prompt cache catch."""
    spec = specs.get(bench)
    items = _SAMPLE_ITEMS[bench]
    prefixes = [
        spec.build_system_prefix(_ctx(it, [Path("inputs/v.mp4")]))
        for it in items
    ]
    assert len(set(prefixes)) == 1, (
        f"{bench}: system_prefix should be byte-identical across items, "
        f"got {len(set(prefixes))} distinct."
    )


@pytest.mark.parametrize("bench", list(_SAMPLE_ITEMS.keys()))
def test_user_question_varies_per_item(bench: str) -> None:
    """Per-item content (Question + staged paths) must produce
    distinct user_question strings — otherwise we'd silently train on
    duplicate (input, output) pairs."""
    spec = specs.get(bench)
    items = _SAMPLE_ITEMS[bench]
    users = [
        spec.build_user_question(_ctx(it, [Path(f"inputs/v{i}.mp4")]))
        for i, it in enumerate(items)
    ]
    assert len(set(users)) == len(users), (
        f"{bench}: user_question collided across items "
        f"({len(set(users))}/{len(users)} distinct)"
    )


@pytest.mark.parametrize("bench", list(_SAMPLE_ITEMS.keys()))
def test_codex_prompt_is_concat_of_system_and_user(bench: str) -> None:
    """``build_codex_prompt(ctx) == build_system_prefix(ctx) + '\\n\\n'
    + build_user_question(ctx)`` exactly. This guarantees codex-cli /
    mini-swe-agent (which use the legacy single-string interface)
    see byte-identical content to kira (which uses the split)."""
    spec = specs.get(bench)
    items = _SAMPLE_ITEMS[bench]
    for it in items:
        ctx = _ctx(it, [Path("inputs/v.mp4")])
        codex = spec.build_codex_prompt(ctx)
        sp = spec.build_system_prefix(ctx)
        uq = spec.build_user_question(ctx)
        expected = sp + "\n\n" + uq
        assert codex == expected, (
            f"{bench} item={it.get('id') or it.get('question_id')}: "
            f"build_codex_prompt diverged from concat of split builders"
        )


@pytest.mark.parametrize("bench", list(_SAMPLE_ITEMS.keys()))
def test_system_prefix_has_no_per_item_leakage(bench: str) -> None:
    """system_prefix must NOT contain any per-item text — the question,
    options, or item id leaking into the static prefix would (a) bust
    cache, (b) cause cross-item answer hints if the cache hit but the
    question changed."""
    spec = specs.get(bench)
    items = _SAMPLE_ITEMS[bench]
    for it in items:
        sp = spec.build_system_prefix(_ctx(it, [Path("inputs/v.mp4")]))
        # Question text must NOT be there.
        q = (it.get("question") or "")
        if q and len(q) > 8:
            assert q not in sp, f"{bench}: item question leaked into system_prefix"
        # Item id must NOT be there.
        for id_field in ("id", "question_id", "video_id"):
            v = it.get(id_field)
            if isinstance(v, str) and len(v) > 4 and "vid" in v.lower():
                assert v not in sp, f"{bench}: id={v} leaked into system_prefix"
        # First option text (if any) must NOT be there.
        opts = it.get("options") or []
        for opt in opts:
            if isinstance(opt, str) and len(opt) > 8:
                assert opt not in sp, f"{bench}: option text leaked into system_prefix"


@pytest.mark.parametrize("bench", list(_SAMPLE_ITEMS.keys()))
def test_user_question_starts_with_staged_files_header(bench: str) -> None:
    """All specs use ``render_user_question`` so the first line is
    ``Available staged files:``. This is the structural anchor the
    SFT pipeline + post-hoc auditor look for."""
    spec = specs.get(bench)
    items = _SAMPLE_ITEMS[bench]
    for it in items:
        uq = spec.build_user_question(_ctx(it, [Path("inputs/v.mp4")]))
        assert uq.startswith("Available staged files:"), (
            f"{bench} user_question must start with 'Available staged files:', "
            f"got: {uq[:80]!r}"
        )


@pytest.mark.parametrize("bench", list(_SAMPLE_ITEMS.keys()))
def test_system_prefix_carries_final_answer_protocol(bench: str) -> None:
    """Round-17.11 (2026-04-30): the general "wrap answer in
    <answer></answer>, emit as plain assistant text, then call
    task_complete" protocol lives in role=system as a single shared
    constant (``FINAL_ANSWER_PROTOCOL``). Every spec must surface it
    so the model sees the same delivery contract regardless of which
    benchmark is running.

    Per-item content rules (MCQ letter set, free-text payload format)
    stay per-spec in build_user_question — those are NOT checked here.
    """
    spec = specs.get(bench)
    items = _SAMPLE_ITEMS[bench]
    sp = spec.build_system_prefix(_ctx(items[0], [Path("inputs/v.mp4")]))
    assert FINAL_ANSWER_PROTOCOL in sp, (
        f"{bench}: system_prefix missing FINAL_ANSWER_PROTOCOL — "
        f"every spec must inject it via render_system_prefix"
    )
    # Sanity: the protocol mentions both delivery paths so the model
    # has explicit guidance + backstop.
    assert "plain text" in sp.lower() or "plain assistant" in sp.lower(), (
        f"{bench}: protocol should mention plain-text delivery"
    )
    assert "task_complete" in sp, (
        f"{bench}: protocol should mention task_complete as the closer"
    )


@pytest.mark.parametrize("bench", list(_SAMPLE_ITEMS.keys()))
def test_user_question_does_not_duplicate_wrapper_rule(bench: str) -> None:
    """Per-item user_question must NOT re-state the general wrapper
    rule (that's now in role=system). Duplication wastes tokens and
    busts the prompt cache because the wrapper rule includes static
    text that would otherwise live in the cached prefix.

    Note: this checks for the specific verbose pre-2026-04-30 wording
    that DID land per-item ("After you have finished using tools, wrap
    your final answer in <answer></answer> tags. The tags must
    contain ONLY..."). Per-spec wrapper EXAMPLES like "Format:
    <answer>A</answer>." may still appear (socialomni / lvomnibench
    keep them so the single-string Claude path still works).
    """
    spec = specs.get(bench)
    items = _SAMPLE_ITEMS[bench]
    for it in items:
        uq = spec.build_user_question(_ctx(it, [Path("inputs/v.mp4")]))
        # Specific banned phrase (the pre-protocol verbose rule).
        assert "After you have finished using tools, wrap your final answer" not in uq, (
            f"{bench}: per-item user_question still carries the verbose "
            "wrapper rule — should be in FINAL_ANSWER_PROTOCOL only"
        )


def test_system_prefix_depends_only_on_dispatcher_flags() -> None:
    """Sanity: changing ``allow_shell_network`` / ``allow_shell_gpu``
    DOES change the prefix (so the cache is correctly busted when the
    dispatcher flips a flag mid-run), but changing the item does NOT.
    Test on omnigaia as the canonical reference."""
    spec = specs.get("omnigaia")
    item = _SAMPLE_ITEMS["omnigaia"][0]
    base = spec.build_system_prefix(_ctx(item, [Path("inputs/v.mp4")]))
    # Flip allow_shell_network → prefix MUST change.
    flipped_ctx = BuildPromptCtx(
        item=item, staged_paths=[Path("inputs/v.mp4")],
        sandbox="workspace-write", allow_shell_network=False,
        allow_shell_gpu=True, shared_python_env="/tmp/v",
        disable_native_vision=False, extra_system_prompt="",
    )
    flipped = spec.build_system_prefix(flipped_ctx)
    assert base != flipped, (
        "system_prefix must change when allow_shell_network flips — "
        "otherwise the trained model can't tell what mode it's in"
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
