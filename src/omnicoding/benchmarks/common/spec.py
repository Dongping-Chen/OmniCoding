"""Benchmark adapter: the only thing that changes per benchmark.

`BenchSpec` is the contract a benchmark exposes to the harness runners.
Everything that is *not* benchmark-specific (sandboxing, env, retry,
metrics) lives in the harness runner; everything that *is* benchmark-
specific (data shape, prompt body, answer extraction, gold comparison)
lives in a `BenchSpec` instance.

A new benchmark = one file under `benchmarks/specs/<name>.py` exporting
a `BenchSpec` and registered in `benchmarks/specs/__init__.py`.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Optional


@dataclass(frozen=True)
class BuildPromptCtx:
    """Inputs to a prompt builder. Everything the prompt body might need."""

    item: dict[str, Any]
    staged_paths: list[Path]
    sandbox: str
    allow_shell_network: bool
    allow_shell_gpu: bool
    shared_python_env: Optional[str]
    disable_native_vision: bool
    extra_system_prompt: str


@dataclass(frozen=True)
class ResultRowCtx:
    """Inputs to result_row(). Carries everything the row might want."""

    item: dict[str, Any]
    item_index: int
    prediction: str
    is_correct: Optional[bool]
    raw_model_output: str
    tool_call_num: int
    return_code: Optional[int]
    timed_out: bool
    stdout_text: str
    stderr_text: str
    workspace_dir: Path
    keep_workdirs: bool
    include_gold_fields: bool
    extra: dict[str, Any] = field(default_factory=dict)


_AGENT_MD_HEADER = (
    "Additional system prompt from the repository agent guide follows.\n"
    "Treat it as higher-priority behavioral guidance when it does not "
    "conflict with the benchmark's required final-answer format."
)


# Shared across every benchmark spec — single source of truth for HOW
# the model delivers its answer (plain assistant text, then task_complete).
# WHAT counts as a valid answer (option letter A/B/C/D, free text,
# yes/no, etc.) is per-item and stays in the user_question.
#
# Added 2026-04-30 after the post-unify smoke audit found the model
# echoing ``<answer>...</answer>`` via shell ~80% of the time —
# functional (parser walks every text segment, including tool output)
# but burns one execute_commands round-trip per item and risks shell
# quoting failures when the answer contains apostrophes / `<` / `$`.
# The protocol nudges the model toward plain assistant text while
# keeping echo as an explicit backstop, so legacy trajectories stay
# parseable.
FINAL_ANSWER_PROTOCOL = (
    "Final-answer protocol\n"
    "- When you have your answer, your NEXT assistant message must be plain "
    "text (no tool call, no shell `echo`) whose entire content is exactly:\n"
    "    <answer>YOUR_ANSWER</answer>\n"
    "- The wrapper must contain ONLY the answer text — no surrounding prose, "
    "no explanation, no units unless the question requires them.\n"
    "- Only AFTER that plain-text answer message lands, call task_complete in "
    "your following turn to end the task. task_complete is the last action.\n"
    "- For backwards compatibility the harness also accepts a <answer>...</answer> "
    "tag found anywhere earlier in the trajectory (including shell `echo` "
    "output), but plain assistant text is preferred: it saves an extra tool "
    "call and avoids shell-quoting hazards (apostrophes, `<`, `$` inside the "
    "answer)."
)


def render_system_prefix(
    *,
    ctx: "BuildPromptCtx",
    benchmark_name: str,
    staged_file_description: str,
    scope: str,
    max_commands: int,
    extras: Optional[list[str]] = None,
) -> str:
    """Single source of truth for the static (per-run, per-spec)
    portion of every benchmark's prompt.

    Every spec in this repo previously hand-assembled the same blocks
    (workspace_instructions → sandbox line → optional native-vision
    restriction → network_instructions → optional gpu_instructions →
    optional shared-python-env → tool_workflow_instructions) with
    minor wording drift. That divergence is what produced the kira
    "5 KB of boilerplate stuffed into role=user" wart and made it
    impossible for the LLM provider's prompt cache to hold the prefix
    across items.

    Specs now collapse into a thin call to this helper plus a list
    of ``extras`` (spec-specific guidance like WEB_SEARCH_HINT,
    GPU_PRECHECK, NETWORK_FORBIDDEN_TARGET, SERIAL_MEDIA constraints,
    etc.). The helper guarantees:

    - identical block ordering across benchmarks;
    - byte-identical output across items in a run (the only inputs
      are ``ctx.sandbox``, ``ctx.allow_shell_network`` etc., which
      come from CLI/dispatcher and don't change item-to-item);
    - cleanly slottable into ``role=system`` so the prompt cache
      catches it.

    ``extras`` are appended verbatim, separated by blank lines, AFTER
    ``tool_workflow_instructions``. Each entry is one block (string)
    — keep them static across items.
    """
    from omnicoding.benchmarks.common.agent_environment import (
        gpu_instructions,
        native_vision_restriction,
        network_instructions,
        shared_python_env_instructions,
        tool_workflow_instructions,
        workspace_instructions,
    )

    lines: list[str] = []
    if ctx.extra_system_prompt:
        lines.extend([_AGENT_MD_HEADER, "", ctx.extra_system_prompt, ""])
    lines.extend(
        [
            workspace_instructions(
                benchmark_name=benchmark_name,
                staged_file_description=staged_file_description,
                scope="item",
            ),
            "",
            f"Codex exec sandbox mode for this run: {ctx.sandbox}",
        ]
    )
    if ctx.disable_native_vision:
        lines.extend(["", native_vision_restriction()])
    if ctx.allow_shell_network:
        lines.extend(
            [
                "",
                network_instructions(
                    allow_shell_network=True,
                    sandbox=ctx.sandbox,
                    forbidden_target=(
                        "benchmark answers, leaked annotations, "
                        "dataset-specific solutions, or existing "
                        "evaluation outputs"
                    ),
                ),
            ]
        )
    else:
        lines.extend(
            ["", network_instructions(allow_shell_network=False, sandbox=ctx.sandbox)]
        )
    if ctx.allow_shell_gpu:
        lines.extend(["", gpu_instructions()])
    if ctx.shared_python_env:
        lines.extend(
            ["", shared_python_env_instructions(scope="item", env_path=ctx.shared_python_env)]
        )
    lines.extend(["", tool_workflow_instructions(scope=scope, max_commands=max_commands)])
    for extra in extras or []:
        if not extra:
            continue
        lines.extend(["", extra])
    # FINAL_ANSWER_PROTOCOL is appended LAST so it stays close to the
    # task description that follows in role=user — the proximity helps
    # the model treat "emit <answer> as plain text" as the very next
    # action after its tool exploration finishes.
    lines.extend(["", FINAL_ANSWER_PROTOCOL])
    return "\n".join(lines)


def render_user_question(
    *,
    staged_paths: list[Path],
    question_block: list[str],
) -> str:
    """Single source of truth for the per-item portion of every
    benchmark's prompt.

    Output: ``Available staged files:\\n- ...\\n<question_block>``.

    ``question_block`` is the spec-specific Question + Options +
    answer-format-instruction lines. Everything else (the staged-file
    enumeration shape) is uniform across benchmarks. Keep this output
    small (<2 KB typical) — that is the cost the LLM provider re-
    encodes per item; the static prefix from ``render_system_prefix``
    pays once and rides the prompt cache.
    """
    lines: list[str] = ["Available staged files:"]
    if staged_paths:
        for path in staged_paths:
            lines.append(f"- {path.as_posix()}")
    else:
        lines.append("- (none)")
    lines.extend(question_block)
    return "\n".join(lines)


# Type aliases keep the BenchSpec signature readable.
ItemLoader = Callable[[Path], list[dict[str, Any]]]
ItemFilter = Callable[[list[dict[str, Any]], Any], list[dict[str, Any]]]
StageInputs = Callable[[dict[str, Any], Path, Path], list[Path]]
PromptBuilder = Callable[[BuildPromptCtx], str]
PredictionExtractor = Callable[[str, dict[str, Any]], str]
GoldComparator = Callable[[dict[str, Any], str], Optional[bool]]
ResultRowBuilder = Callable[[ResultRowCtx], dict[str, Any]]
ItemIdFn = Callable[[dict[str, Any]], str]


@dataclass(frozen=True)
class BenchSpec:
    """One benchmark's adapter.

    `name` is the public registry key (used by `--bench`).

    `iterate_items(input_path)` loads + flattens the JSON file (handles
    `[...]` vs `{"data": [...]}` shapes, sets `__source_index__`).

    `filter_items(items, args)` applies CLI filters (question_ids, etc.).

    `stage_inputs(item, dataset_root, workspace)` copies the per-item
    media into `workspace/inputs/...` and returns a list of
    *workspace-relative* paths the prompt should reference.

    `build_codex_prompt(ctx)` and `build_claude_prompt(ctx)` produce the
    final prompt strings. The harness runner provides the surrounding
    workspace_instructions / network_instructions / etc. via the
    BuildPromptCtx — the spec only has to glue in benchmark-specific
    text (question, options, output-format contract).

    `extract_prediction(raw_text, item)` returns the canonical answer
    string (e.g. `"A"`). Empty string == no answer extracted.

    `is_correct(item, prediction)` compares vs. gold and returns
    True/False, or None when the bench has no gold (free-form).

    `result_row(ctx)` builds the final dict written to
    `results.json`. Spec controls schema entirely.

    `item_id(item)` is just for log/path naming.

    `answer_format_hint` is interpolated into the continue-retry prompt
    when the model stops without an answer (e.g. "<answer>A</answer>
    containing one option letter from A, B, C, or D").

    `groups_by` is non-None for benchmarks like VideoZeroBench that run
    one prompt per group (e.g. per video), not per item; the harness
    runner branches on this.
    """

    name: str
    iterate_items: ItemLoader
    filter_items: ItemFilter
    stage_inputs: StageInputs
    # Split-prompt builders for harnesses with a real ``role=system``
    # slot (kira, claude). ``build_system_prefix`` returns the static
    # per-run prefix (workspace rules, web_search hint, tool-workflow
    # rules) that stays byte-identical across all items in a dispatch
    # run — kira lands it in ``role=system`` so the LLM provider's
    # prompt cache holds it once. ``build_user_question`` returns the
    # per-item content (staged files + Question + Options) that lands
    # in ``role=user``. Both required: every spec migrated 2026-04-29
    # so we stopped carrying a fallback path.
    build_system_prefix: PromptBuilder
    build_user_question: PromptBuilder
    # ``build_codex_prompt`` is concat shape (system + "\n\n" + user)
    # for callers without a system slot — codex-cli's ``codex exec``
    # (single prompt arg) and mini-swe-agent's user-only flow.
    build_codex_prompt: PromptBuilder
    build_claude_prompt: PromptBuilder
    extract_prediction: PredictionExtractor
    is_correct: GoldComparator
    result_row: ResultRowBuilder
    item_id: ItemIdFn
    answer_format_hint: str
    groups_by: Optional[Callable[[list[dict[str, Any]]], list[tuple[str, list[dict[str, Any]]]]]] = None


def load_json_items(path: Path) -> list[dict[str, Any]]:
    """Default item-loader. Handles both `[...]` and `{"data": [...]}`
    shapes, and stamps `__source_index__` for stable ordering."""

    import json

    raw = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(raw, list):
        items = raw
    elif isinstance(raw, dict):
        for key in ("data", "items", "results", "questions"):
            value = raw.get(key)
            if isinstance(value, list):
                items = value
                break
        else:
            raise ValueError(f"Cannot find item list in {path}")
    else:
        raise ValueError(f"Top-level JSON must be list or dict, got {type(raw).__name__}")
    for index, item in enumerate(items):
        if isinstance(item, dict) and "__source_index__" not in item:
            item["__source_index__"] = index
    return [item for item in items if isinstance(item, dict)]


CONTINUE_RETRY_LIMIT = 10


def current_gpu_device() -> Optional[str]:
    """Return the GPU index (as a string) currently visible to this
    process via `CUDA_VISIBLE_DEVICES`, or `None` if not allocated.

    Sequential harnesses (mini_swe / opencode / kira) call this when
    building the result row's `extra.gpu_device_assigned` so every
    runner exposes the same field that claude/codex already populate
    via `gpu_slot_queue`. If multiple devices are visible (rare for
    scavenger jobs), the first index is returned; if the env var is
    unset or empty, `None` is returned so the analyzer can tell
    "no GPU allocated" apart from "GPU 0"."""
    raw = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if not raw:
        return None
    first = raw.split(",")[0].strip()
    return first or None


RELATIVE_PATH_HINT = (
    "\n\nPath usage: prefer **relative paths** from the current "
    "working directory (your workspace) over absolute paths. "
    "The staged inputs and any files you create live next to this "
    "prompt, so `inputs/foo.mp4` works the same as the full path. "
    "Long absolute paths sometimes get truncated by tool-call "
    "serializers and produce `cd: No such file or directory` "
    "failures — keep paths short."
)


def build_continue_prompt(spec: "BenchSpec") -> str:
    """Single source of truth for the continue/reminder prompt that
    kira / opencode / claude / codex all inject when the model stops
    without producing a valid answer in the format the bench expects.

    A prior smoke audit found several long-run
    items (mini_swe/omnigaia at step_limit, kira/socialomni_l2 +
    videozerobench at task_complete) exited with the *answer in prose*
    in the trajectory but no ``<answer>...</answer>`` wrapper. Fixed by
    making the format requirement non-negotiable: the reminder now
    spells out the tag, calls out that prose-only is rejected, and
    instructs the model to re-emit the answer in the wrapper if it
    already wrote prose.
    """
    return (
        "Please continue solving the task. If you need more evidence, run more tools. "
        "Once you have enough to decide, your FINAL response must end with the answer "
        f"tag in the exact format the task requested — {spec.answer_format_hint}. "
        "If you already wrote the answer in prose without the required wrapper, re-emit "
        "it now wrapped in the tag — prose without the tag is treated as no answer at all "
        "and the run is scored as a failure. Do not stop, do not call task_complete, and "
        "do not respond with plain text without the final tag."
    )


def filter_by_field_set(
    items: list[dict[str, Any]],
    field_name: str,
    wanted: Optional[Iterable[Any]],
) -> list[dict[str, Any]]:
    """Filter items by membership of `field_name` in `wanted`. No-op if
    `wanted` is falsy."""

    if not wanted:
        return items
    target = {str(value) for value in wanted}
    return [item for item in items if str(item.get(field_name)) in target]
