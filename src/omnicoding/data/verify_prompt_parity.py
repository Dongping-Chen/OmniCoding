"""Verify train/serve prompt parity across benchmarks + synthetic data.

Run this after touching any prompt-builder code (``common/spec.py``,
per-spec ``*_prompting.py``, or kira-core ``SYSTEM_PROMPT``) to catch
divergence before it bakes into a multi-thousand-item synthetic-data
run.

Three independent checks:

1. **Cross-spec consistency** — every spec's system_prefix carries the
   same byte-identical ``FINAL_ANSWER_PROTOCOL`` block. If a future
   refactor accidentally inlines the constant or fork-edits one spec,
   the model will see different rules per benchmark and SFT data /
   eval scoring drift.

2. **Within-spec self-consistency** (cache friendliness) — for each
   spec, render system_prefix on multiple sample items with the same
   ctx; the bytes must be identical (after stripping the per-job
   ``shared_python_env`` random path). If divergent, the LLM
   provider's prompt cache cannot hold the prefix → costs balloon
   on long runs.

3. **On-disk vs in-process parity** — read an actual trajectory's
   ``role=system`` and ``role=user`` content from a recent kira run,
   reconstruct the BuildPromptCtx from run_meta + the items file,
   re-render via the same builder in-process, and confirm byte
   match (modulo per-job random paths). This is the train/serve
   parity check: when this passes, the synthetic-data trajectories
   we trained on were produced by the SAME prompt the eval harness
   would generate today.

Synthetic data and benchmark eval are literally the same code path
(both go through ``run_bench_kira.py:_build_prompt`` calling
``spec.build_system_prefix(ctx) + spec.build_user_question(ctx)``),
so check 3 is the load-bearing one — if it passes for any
trajectory under ``outputs/<run>/out/item_*/messages.json``, train
and serve are byte-identical for that item.

Exit code: 0 on full pass, 1 on any divergence. Designed for CI.

Usage::

    .venv_harness/bin/python local_model/scripts/verify_prompt_parity.py \\
        [--trajectories OUT_DIR ...] \\
        [--items_file ITEMS.json] \\
        [--report-file PATH]
"""
from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from omnicoding.benchmarks import specs
from omnicoding.benchmarks.common.spec import BuildPromptCtx, FINAL_ANSWER_PROTOCOL


# Regex to strip per-job random paths from system prefix before
# byte-comparing — these encode the slurm job's tmp workspace and
# differ per item by design (kira creates a fresh /tmp/<rand>/.venv).
_PER_JOB_VENV_RE = re.compile(
    r"/tmp/kira_[a-z0-9_]+/\.venv"
)


def _strip_per_job_paths(text: str) -> str:
    """Replace per-job random paths with a stable placeholder so two
    renders that only differ in the random component compare equal."""
    return _PER_JOB_VENV_RE.sub("/tmp/kira_<RUN>/.venv", text)


_SAMPLE_ITEMS = {
    "omnigaia": [
        {"id": "ovb:1", "answer_type": "mcq",
         "question": "Q1?", "options": ["A. x", "B. y"]},
        {"id": "ovb:2", "answer_type": "open", "question": "Q2?"},
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
        {"id": "s1", "video_path": "v1.mp4", "question": "Q?",
         "options": ["A. yes", "B. no", "C. maybe", "D. unsure"]},
    ],
    "socialomni_l2": [
        {"video_id": "v", "video_file": "v.mp4",
         "question_1": {"question": "Y?", "option_A": "YES", "option_B": "NO"}},
    ],
}


def _ctx(item: dict[str, Any], staged: list[Path], shared_env: str) -> BuildPromptCtx:
    return BuildPromptCtx(
        item=item, staged_paths=staged,
        sandbox="workspace-write",
        allow_shell_network=True, allow_shell_gpu=True,
        shared_python_env=shared_env,
        disable_native_vision=False,
        extra_system_prompt="",
    )


@dataclass
class CheckResult:
    name: str
    passed: bool
    detail: str = ""


def check_cross_spec_protocol_identity() -> list[CheckResult]:
    """Every spec's system_prefix must carry the SAME bytes for
    FINAL_ANSWER_PROTOCOL — no spec gets a fork-edited copy."""
    results = []
    for bench in _SAMPLE_ITEMS:
        spec = specs.get(bench)
        item = _SAMPLE_ITEMS[bench][0]
        sp = spec.build_system_prefix(_ctx(item, [Path("inputs/v.mp4")], "/tmp/v"))
        if FINAL_ANSWER_PROTOCOL not in sp:
            results.append(CheckResult(
                name=f"protocol_present[{bench}]",
                passed=False,
                detail=f"FINAL_ANSWER_PROTOCOL missing from {bench}.system_prefix",
            ))
            continue
        # Find the constant verbatim — its presence guarantees identity
        # because we look up the same Python object that
        # render_system_prefix injected.
        results.append(CheckResult(
            name=f"protocol_present[{bench}]", passed=True,
            detail=f"present at offset {sp.find(FINAL_ANSWER_PROTOCOL)}",
        ))
    return results


def check_within_spec_cache_friendliness() -> list[CheckResult]:
    """For each spec, render multiple items with the SAME ctx (same
    shared_python_env path); system_prefix must be byte-identical so
    the LLM provider's prompt cache catches.

    Note: production runs supply a per-job random shared_python_env
    path, which by design makes prefixes diverge — this check uses a
    constant path to isolate "spec correctness" from "kira workspace
    randomness" (the latter is documented as an open issue in
    agent.md and not a regression of the prompt change)."""
    results = []
    for bench, items in _SAMPLE_ITEMS.items():
        if len(items) < 2:
            continue
        spec = specs.get(bench)
        prefixes = [
            spec.build_system_prefix(_ctx(it, [Path("inputs/v.mp4")], "/tmp/v"))
            for it in items
        ]
        if len(set(prefixes)) == 1:
            results.append(CheckResult(
                name=f"cache_identity[{bench}]", passed=True,
                detail=f"{len(items)} items → 1 distinct prefix ({len(prefixes[0])} chars)",
            ))
        else:
            results.append(CheckResult(
                name=f"cache_identity[{bench}]", passed=False,
                detail=f"{len(items)} items → {len(set(prefixes))} distinct prefixes",
            ))
    return results


def check_user_question_no_wrapper_dup() -> list[CheckResult]:
    """user_question must NOT re-state the verbose pre-protocol wrapper
    rule (now in role=system). Catches the regression where a future
    edit to a spec's question_block re-introduces the duplicate."""
    BANNED = "After you have finished using tools, wrap your final answer"
    results = []
    for bench, items in _SAMPLE_ITEMS.items():
        spec = specs.get(bench)
        for it in items:
            uq = spec.build_user_question(_ctx(it, [Path("inputs/v.mp4")], "/tmp/v"))
            if BANNED in uq:
                results.append(CheckResult(
                    name=f"no_wrapper_dup[{bench}/{it.get('id') or it.get('question_id')}]",
                    passed=False,
                    detail="user_question carries banned per-item wrapper sentence",
                ))
            else:
                results.append(CheckResult(
                    name=f"no_wrapper_dup[{bench}/{it.get('id') or it.get('question_id')}]",
                    passed=True,
                ))
    return results


def check_disk_vs_inprocess_parity(out_dir: Path,
                                    items_file: Path | None = None) -> list[CheckResult]:
    """For each item_*/messages.json under ``out_dir``, reconstruct
    the in-process system_prefix + user_question and byte-compare
    against the on-disk ``role=system`` (after the kira-core
    SYSTEM_PROMPT prefix) and the first ``role=user`` message.

    Train/serve parity: a passing check means the trajectory's prompt
    matches what the eval harness would emit today — so SFT data
    trained on this trajectory plays back identically at eval time.
    """
    results = []

    run_meta_path = out_dir / "run_meta.json"
    if not run_meta_path.exists():
        results.append(CheckResult(
            name=f"disk_parity[{out_dir.parent.name}]", passed=False,
            detail=f"no run_meta.json at {run_meta_path}",
        ))
        return results
    run_meta = json.loads(run_meta_path.read_text())
    bench = run_meta.get("bench")
    if not bench or specs.get(bench) is None:
        results.append(CheckResult(
            name=f"disk_parity[{out_dir.parent.name}]", passed=False,
            detail=f"unknown bench={bench}",
        ))
        return results
    spec = specs.get(bench)

    # Locate the items file. The dispatcher passes ``--items_file``;
    # the run dir typically symlinks or stores the items source under
    # ../items.json or ../dataset/items.json — we fall back to user-
    # supplied --items_file if neither is found.
    candidates = [
        items_file,
        out_dir.parent / "items.json",
        out_dir.parent / "dataset" / "items.json",
    ]
    items_path = next((c for c in candidates if c and c.exists()), None)
    if items_path is None:
        results.append(CheckResult(
            name=f"disk_parity[{out_dir.parent.name}]", passed=False,
            detail="cannot find items.json (pass --items_file)",
        ))
        return results
    items_raw = json.loads(items_path.read_text())
    if isinstance(items_raw, dict):
        for key in ("data", "items", "results", "questions"):
            value = items_raw.get(key)
            if isinstance(value, list):
                items_raw = value
                break
    items_by_si = {it.get("__source_index__"): it
                   for it in items_raw if isinstance(it, dict)}

    item_dirs = sorted(out_dir.glob("item_*"))
    if not item_dirs:
        results.append(CheckResult(
            name=f"disk_parity[{out_dir.parent.name}]", passed=False,
            detail=f"no item_* dirs under {out_dir}",
        ))
        return results

    for item_dir in item_dirs:
        si_str = item_dir.name.removeprefix("item_")
        si = int(si_str)
        msgs_path = item_dir / "messages.json"
        if not msgs_path.exists():
            continue
        msgs = json.loads(msgs_path.read_text())
        sys_msg = next((m for m in msgs if m.get("role") == "system"), None)
        first_user = next((m for m in msgs if m.get("role") == "user"), None)
        if sys_msg is None or first_user is None:
            results.append(CheckResult(
                name=f"disk_parity[{out_dir.parent.name}/si={si}]", passed=False,
                detail="trajectory missing role=system or first role=user",
            ))
            continue

        item = items_by_si.get(si)
        if item is None:
            # __source_index__ may not have been stamped — fall back
            # to position.
            if si < len(items_raw):
                item = items_raw[si]
            else:
                results.append(CheckResult(
                    name=f"disk_parity[{out_dir.parent.name}/si={si}]", passed=False,
                    detail=f"si={si} not in items file",
                ))
                continue

        # Reconstruct ctx. For shared_python_env, lift it from the
        # actual trajectory's system content (the only field that
        # varies per item) so we compare apples-to-apples.
        env_match = _PER_JOB_VENV_RE.search(sys_msg.get("content", ""))
        shared_env = env_match.group(0) if env_match else "/tmp/v"
        # The trajectory may carry per-run staged paths different from
        # the spec's stage_inputs (e.g. relative to the workspace);
        # extract from the user message's "Available staged files:"
        # block to keep the user_question render byte-identical.
        first_user_text = first_user.get("content", "")
        if isinstance(first_user_text, list):
            first_user_text = next(
                (p.get("text", "") for p in first_user_text if p.get("type") == "text"),
                "",
            )
        staged_paths: list[Path] = []
        for line in first_user_text.splitlines():
            if line.startswith("- "):
                staged_paths.append(Path(line[2:]))
            elif "Question:" in line or "Options:" in line or "Video id" in line:
                break
        ctx = BuildPromptCtx(
            item=item,
            staged_paths=staged_paths,
            sandbox=run_meta.get("sandbox", "workspace-write"),
            allow_shell_network=run_meta.get("allow_shell_network", True),
            allow_shell_gpu=run_meta.get("allow_shell_gpu", True),
            shared_python_env=shared_env,
            disable_native_vision=run_meta.get("disable_native_vision", False),
            extra_system_prompt=run_meta.get("extra_system_prompt", ""),
        )

        # Build expected = spec system_prefix (the tail of role=system
        # after the kira-core SYSTEM_PROMPT marker).
        expected_sys = spec.build_system_prefix(ctx)
        expected_user = spec.build_user_question(ctx)
        actual_sys = sys_msg.get("content", "")

        # Find where the spec system_prefix starts in the actual
        # role=system. The kira-core SYSTEM_PROMPT precedes it (added
        # by run_bench_kira via agent.run(system_prefix=spec_prefix +
        # WEB_SEARCH_PROMPT_HINT + RELATIVE_PATH_HINT)). We anchor on
        # the first line of expected_sys.
        # actual = kira_core + "\n\n" + spec_prefix + WEB_SEARCH_PROMPT_HINT + RELATIVE_PATH_HINT
        anchor = expected_sys.split("\n", 1)[0]
        anchor_idx = actual_sys.find(anchor)
        if anchor_idx < 0:
            results.append(CheckResult(
                name=f"disk_parity[{out_dir.parent.name}/si={si}]", passed=False,
                detail=f"cannot locate spec_prefix anchor in role=system",
            ))
            continue
        # The expected spec_prefix region: anchor_idx .. anchor_idx + len(expected_sys).
        actual_spec_region = actual_sys[anchor_idx:anchor_idx + len(expected_sys)]
        if _strip_per_job_paths(actual_spec_region) == _strip_per_job_paths(expected_sys):
            sys_match = True
            sys_detail = "byte-match (after per-job-path normalization)"
        else:
            sys_match = False
            # First-divergence offset.
            a = _strip_per_job_paths(actual_spec_region)
            e = _strip_per_job_paths(expected_sys)
            div = next(
                (i for i, (x, y) in enumerate(zip(a, e)) if x != y),
                min(len(a), len(e)),
            )
            sys_detail = (f"diverged at offset {div}: "
                          f"actual={a[max(0,div-30):div+30]!r} "
                          f"expected={e[max(0,div-30):div+30]!r}")

        if first_user_text == expected_user:
            user_match = True
            user_detail = f"byte-match ({len(expected_user)} chars)"
        else:
            user_match = False
            div = next(
                (i for i, (x, y) in enumerate(zip(first_user_text, expected_user)) if x != y),
                min(len(first_user_text), len(expected_user)),
            )
            user_detail = (f"diverged at offset {div}: "
                           f"actual={first_user_text[max(0,div-30):div+30]!r} "
                           f"expected={expected_user[max(0,div-30):div+30]!r}")

        results.append(CheckResult(
            name=f"disk_parity[{out_dir.parent.name}/si={si}]/system",
            passed=sys_match, detail=sys_detail,
        ))
        results.append(CheckResult(
            name=f"disk_parity[{out_dir.parent.name}/si={si}]/user",
            passed=user_match, detail=user_detail,
        ))
    return results


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--trajectories", type=Path, action="append", default=[],
                   help="Path to a kira run output dir (e.g. "
                        "outputs/<run>/out). Repeat to compare multiple "
                        "runs. Each dir must contain run_meta.json + "
                        "item_<si>/messages.json.")
    p.add_argument("--items_file", type=Path, default=None,
                   help="Path to items.json — only needed when the run "
                        "dir doesn't carry items.json next to it.")
    p.add_argument("--report-file", type=Path, default=None,
                   help="Optional file to write the report to (in addition "
                        "to stdout).")
    args = p.parse_args(argv)

    all_results: list[CheckResult] = []
    print("=" * 78)
    print("[1/4] Cross-spec FINAL_ANSWER_PROTOCOL identity")
    print("-" * 78)
    rs = check_cross_spec_protocol_identity()
    for r in rs:
        print(f"  {'PASS' if r.passed else 'FAIL'}  {r.name:50s}  {r.detail}")
    all_results += rs

    print("=" * 78)
    print("[2/4] Within-spec cache friendliness")
    print("-" * 78)
    rs = check_within_spec_cache_friendliness()
    for r in rs:
        print(f"  {'PASS' if r.passed else 'FAIL'}  {r.name:50s}  {r.detail}")
    all_results += rs

    print("=" * 78)
    print("[3/4] user_question carries no duplicate wrapper rule")
    print("-" * 78)
    rs = check_user_question_no_wrapper_dup()
    for r in rs:
        print(f"  {'PASS' if r.passed else 'FAIL'}  {r.name:50s}  {r.detail}")
    all_results += rs

    print("=" * 78)
    print("[4/4] On-disk vs in-process parity (synthetic data == benchmark eval)")
    print("-" * 78)
    if not args.trajectories:
        print("  SKIP  no --trajectories supplied")
    else:
        for out_dir in args.trajectories:
            rs = check_disk_vs_inprocess_parity(out_dir, args.items_file)
            for r in rs:
                print(f"  {'PASS' if r.passed else 'FAIL'}  {r.name:60s}  {r.detail[:80]}")
            all_results += rs

    # Summary.
    n_pass = sum(1 for r in all_results if r.passed)
    n_fail = sum(1 for r in all_results if not r.passed)
    print("=" * 78)
    summary = f"\nSUMMARY: {n_pass} passed, {n_fail} failed (of {len(all_results)} checks)"
    print(summary)

    if args.report_file:
        args.report_file.write_text(
            "\n".join(
                f"{'PASS' if r.passed else 'FAIL'}\t{r.name}\t{r.detail}"
                for r in all_results
            ) + summary + "\n",
            encoding="utf-8",
        )
        print(f"\nReport: {args.report_file}")

    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
