"""Unit tests for ``omnicoding.rl.coordinator.dispatcher.SlurmDispatcher`` coalescing path.

Tests use ``monkeypatch`` to replace ``_sbatch`` (the actual slurm submit) +
``_squeue_running`` so we can exercise the coalescer logic without slurm.
Result files are written by the test as if the sbatch worker produced them.

Coverage:
- 4 concurrent n_samples=1 submits → 2 sbatch calls, each with 2 tasks
- partial batch fires after coalesce_window expires (1 submit → 1 sbatch)
- different requests get isolated workspace dirs + result files
- futures resolve with the correct Trajectory per request
- n_samples > 1 bypasses coalescer (uses immediate path)

Run with ``python -m pytest tests/rl/test_dispatcher_coalesce.py -v``.
"""

from __future__ import annotations

import asyncio
import json
import os
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import pytest

from omnicoding.rl.coordinator.dispatcher import SlurmDispatcher  # noqa: E402
from omnicoding.rl.schemas import RolloutRequest, Trajectory  # noqa: E402


# ─── fixtures ────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class _FakeRecord:
    id: str
    question: str = "fake question"
    answer_type: str = "open"
    ground_truth: tuple = ("answer",)
    options: tuple | None = None
    media: dict = None
    source_dataset: str = ""
    category: str = ""

    def __post_init__(self):
        if self.media is None:
            object.__setattr__(self, "media", {"videos": [], "audios": [], "images": []})


def _make_req(n_samples: int = 1) -> RolloutRequest:
    return RolloutRequest(
        task_id="fake:0",
        n_samples=n_samples,
        sglang_base_url="http://invalid.test/v1",
        sglang_model_name="openai/Qwen/Qwen3.5-9B",
        max_turns=2,
        request_timeout_s=30,
        block_timeout_s=30,
    )


@pytest.fixture
def scratch(tmp_path):
    s = tmp_path / "scratch"
    s.mkdir()
    return s


@pytest.fixture
def fake_sbatch_script(tmp_path):
    """A no-op sbatch script (we monkeypatch ``_sbatch`` so it never runs)."""
    p = tmp_path / "sbatch_one_job.sh"
    p.write_text("#!/usr/bin/env bash\nexit 0\n")
    p.chmod(0o755)
    return p


@pytest.fixture
def relax_root(tmp_path):
    r = tmp_path / "relax-router"
    r.mkdir()
    return r


def _success_traj(sample_idx: int, reward: float = 1.0) -> dict:
    """Result-file dict shape that ``Trajectory.model_validate_json`` accepts."""
    return {
        "sample_index": sample_idx,
        "messages": [],
        "final_text": "",
        "extracted_answer": None,
        "prediction_normalized": None,
        "reward": reward,
        "outcome_reward": reward,
        "raw_outcome_reward": reward,
        "format_reward": 0.0,
        "modality_reward": 0.0,
        "bad_tool_reward": 0.0,
        "modality_match": 1.0,
        "p_bad_tool": 0.0,
        "n_unparseable": 0,
        "n_disallowed": 0,
        "n_escape": 0,
        "n_syntax_fail": 0,
        "removed": False,
        "reward_details": {},
        "exit_reason": "task_complete",
        "n_steps": 1,
        "n_tool_calls": 1,
        "cumulative_prompt_tokens": 0,
        "cumulative_completion_tokens": 0,
        "cumulative_reasoning_tokens": 0,
        "error": None,
    }


# ─── coalescing behavior ─────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_two_simultaneous_submits_coalesce_into_one_sbatch(
    scratch, fake_sbatch_script, relax_root, monkeypatch,
):
    """Two ``n_samples=1`` requests submitted within the coalesce window
    should produce ONE sbatch call carrying 2 task files."""
    sbatch_calls: list[tuple[list[Path], list[Path]]] = []
    written_results: list[Path] = []

    def fake_sbatch(self, req_paths, res_paths, job_dir, chunk_idx):
        sbatch_calls.append((list(req_paths), list(res_paths)))
        # Simulate the sbatch worker writing each result file.
        for i, p in enumerate(res_paths):
            p.write_text(json.dumps(_success_traj(i, reward=float(i + 1))))
            written_results.append(p)
        return 12345 + len(sbatch_calls)  # fake jid

    fake_running_state = {"jids": set()}

    def fake_squeue(self, jids):
        # First poll: report empty (job done) so coalescer reads results.
        return set()

    monkeypatch.setattr(SlurmDispatcher, "_sbatch", fake_sbatch)
    monkeypatch.setattr(SlurmDispatcher, "_squeue_running", fake_squeue)

    d = SlurmDispatcher(
        scratch_root=scratch,
        sbatch_script=fake_sbatch_script,
        tasks_per_job=2,
        poll_interval_s=0.05,        # fast for tests
        coalesce_window_s=0.05,
    )
    d.start()
    try:
        # Fire 2 concurrent n_samples=1 submits
        results = await asyncio.gather(
            d.submit_and_collect(_FakeRecord(id="t1"), _make_req(1), Path("/data"), deadline_s=10),
            d.submit_and_collect(_FakeRecord(id="t2"), _make_req(1), Path("/data"), deadline_s=10),
        )
    finally:
        await d.stop()

    # ONE sbatch with 2 tasks
    assert len(sbatch_calls) == 1, (
        f"expected 1 sbatch (coalesced), got {len(sbatch_calls)}"
    )
    req_paths, res_paths = sbatch_calls[0]
    assert len(req_paths) == 2
    assert len(res_paths) == 2

    # Each request got its own Trajectory list of len 1
    assert len(results) == 2
    for r in results:
        assert isinstance(r, list) and len(r) == 1
        assert isinstance(r[0], Trajectory)


@pytest.mark.asyncio
async def test_workspace_and_result_paths_are_isolated(
    scratch, fake_sbatch_script, relax_root, monkeypatch,
):
    """Two coalesced requests must write to DIFFERENT job_uuid scratch dirs
    and DIFFERENT compute-node /tmp workspaces — no aliasing."""
    captured_payloads: list[dict] = []

    def fake_sbatch(self, req_paths, res_paths, job_dir, chunk_idx):
        for rp in req_paths:
            captured_payloads.append(json.loads(rp.read_text()))
        for i, p in enumerate(res_paths):
            p.write_text(json.dumps(_success_traj(i)))
        return 99999

    monkeypatch.setattr(SlurmDispatcher, "_sbatch", fake_sbatch)
    monkeypatch.setattr(SlurmDispatcher, "_squeue_running", lambda self, jids: set())

    d = SlurmDispatcher(
        scratch_root=scratch,
        sbatch_script=fake_sbatch_script,
        tasks_per_job=2,
        poll_interval_s=0.05,
        coalesce_window_s=0.05,
    )
    d.start()
    try:
        await asyncio.gather(
            d.submit_and_collect(_FakeRecord(id="A"), _make_req(1), Path("/data"), deadline_s=10),
            d.submit_and_collect(_FakeRecord(id="B"), _make_req(1), Path("/data"), deadline_s=10),
        )
    finally:
        await d.stop()

    assert len(captured_payloads) == 2
    # Different job_id, different workspace_root
    job_ids = {p["job_id"] for p in captured_payloads}
    workspaces = {p["workspace"] for p in captured_payloads}
    record_ids = {p["record"]["id"] for p in captured_payloads}
    assert all("ground_truth" not in p["record"] for p in captured_payloads)
    assert all("dataset_root" not in p for p in captured_payloads)
    assert len(job_ids) == 2, f"job_ids should be unique: {job_ids}"
    assert len(workspaces) == 2, f"workspaces should be unique: {workspaces}"
    assert record_ids == {"A", "B"}, f"records mixed up: {record_ids}"


@pytest.mark.asyncio
async def test_partial_batch_fires_after_window(
    scratch, fake_sbatch_script, relax_root, monkeypatch,
):
    """One lonely n_samples=1 request should still get sbatched after
    coalesce_window_s expires (no peer to bundle with)."""
    sbatch_calls: list[int] = []

    def fake_sbatch(self, req_paths, res_paths, job_dir, chunk_idx):
        sbatch_calls.append(len(req_paths))
        for i, p in enumerate(res_paths):
            p.write_text(json.dumps(_success_traj(i)))
        return 12345

    monkeypatch.setattr(SlurmDispatcher, "_sbatch", fake_sbatch)
    monkeypatch.setattr(SlurmDispatcher, "_squeue_running", lambda self, jids: set())

    d = SlurmDispatcher(
        scratch_root=scratch,
        sbatch_script=fake_sbatch_script,
        tasks_per_job=4,             # batch target much larger than what we send
        poll_interval_s=0.05,
        coalesce_window_s=0.1,       # short so the test doesn't drag
    )
    d.start()
    try:
        # Single request — coalescer waits coalesce_window_s, then drains
        result = await d.submit_and_collect(
            _FakeRecord(id="solo"), _make_req(1), Path("/data"), deadline_s=10,
        )
    finally:
        await d.stop()

    assert sbatch_calls == [1], (
        f"expected 1 sbatch with 1 task (partial-window drain), got {sbatch_calls}"
    )
    assert len(result) == 1
    assert isinstance(result[0], Trajectory)


@pytest.mark.asyncio
async def test_four_simultaneous_yields_two_sbatch_pairs(
    scratch, fake_sbatch_script, relax_root, monkeypatch,
):
    """4 concurrent n=1 submits with tasks_per_job=2 → exactly 2 sbatch
    invocations, each carrying 2 tasks."""
    sbatch_call_sizes: list[int] = []

    def fake_sbatch(self, req_paths, res_paths, job_dir, chunk_idx):
        sbatch_call_sizes.append(len(req_paths))
        for i, p in enumerate(res_paths):
            p.write_text(json.dumps(_success_traj(i)))
        return 60000 + len(sbatch_call_sizes)

    monkeypatch.setattr(SlurmDispatcher, "_sbatch", fake_sbatch)
    monkeypatch.setattr(SlurmDispatcher, "_squeue_running", lambda self, jids: set())

    d = SlurmDispatcher(
        scratch_root=scratch,
        sbatch_script=fake_sbatch_script,
        tasks_per_job=2,
        poll_interval_s=0.05,
        coalesce_window_s=0.1,
    )
    d.start()
    try:
        await asyncio.gather(*[
            d.submit_and_collect(_FakeRecord(id=f"t{i}"), _make_req(1),
                                  Path("/data"), deadline_s=10)
            for i in range(4)
        ])
    finally:
        await d.stop()

    assert sorted(sbatch_call_sizes) == [2, 2], (
        f"expected two sbatch with 2 tasks each, got {sbatch_call_sizes}"
    )


# ─── n_samples>1 bypasses coalescer ──────────────────────────────────────────


@pytest.mark.asyncio
async def test_n_samples_gt_1_bypasses_coalescer(
    scratch, fake_sbatch_script, relax_root, monkeypatch,
):
    """A direct n_samples=4 request (rare; smoke tests use this) should NOT
    sit in the coalescer — dispatch immediately as ceil(4/2)=2 chunks."""
    sbatch_call_sizes: list[int] = []

    def fake_sbatch(self, req_paths, res_paths, job_dir, chunk_idx):
        sbatch_call_sizes.append(len(req_paths))
        for i, p in enumerate(res_paths):
            p.write_text(json.dumps(_success_traj(i)))
        return 70000 + len(sbatch_call_sizes)

    monkeypatch.setattr(SlurmDispatcher, "_sbatch", fake_sbatch)
    monkeypatch.setattr(SlurmDispatcher, "_squeue_running", lambda self, jids: set())

    d = SlurmDispatcher(
        scratch_root=scratch,
        sbatch_script=fake_sbatch_script,
        tasks_per_job=2,
        poll_interval_s=0.05,
        coalesce_window_s=10.0,        # very long; if coalescer was used, test would hang
    )
    d.start()
    try:
        result = await d.submit_and_collect(
            _FakeRecord(id="multi"), _make_req(4), Path("/data"), deadline_s=10,
        )
    finally:
        await d.stop()

    # 2 sbatch each with 2 tasks (chunks of tasks_per_job=2)
    assert sorted(sbatch_call_sizes) == [2, 2]
    assert len(result) == 4


# ─── failure modes ───────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_sbatch_submission_failure_resolves_failed_trajectory(
    scratch, fake_sbatch_script, relax_root, monkeypatch,
):
    """If sbatch itself fails (returns None jid), each pending future should
    resolve with a failed Trajectory, NOT hang."""
    monkeypatch.setattr(SlurmDispatcher, "_sbatch",
                         lambda self, *a, **kw: None)

    d = SlurmDispatcher(
        scratch_root=scratch,
        sbatch_script=fake_sbatch_script,
        tasks_per_job=2,
        poll_interval_s=0.05,
        coalesce_window_s=0.05,
    )
    d.start()
    try:
        results = await asyncio.gather(
            d.submit_and_collect(_FakeRecord(id="x"), _make_req(1), Path("/data"), deadline_s=5),
            d.submit_and_collect(_FakeRecord(id="y"), _make_req(1), Path("/data"), deadline_s=5),
        )
    finally:
        await d.stop()

    for r in results:
        assert len(r) == 1
        assert r[0].exit_reason in ("error", "timeout")
        assert r[0].removed is True


@pytest.mark.asyncio
async def test_dispatcher_stop_resolves_pending_futures(
    scratch, fake_sbatch_script, relax_root, monkeypatch,
):
    """If dispatcher.stop() is called while requests are queued (e.g. on
    coordinator shutdown), pending futures should resolve as failed —
    never hang the caller."""
    # Make _sbatch slow so the request enters the queue but doesn't fire
    # before we stop.
    monkeypatch.setattr(SlurmDispatcher, "_sbatch",
                         lambda self, *a, **kw: None)

    d = SlurmDispatcher(
        scratch_root=scratch,
        sbatch_script=fake_sbatch_script,
        tasks_per_job=4,                # so single submit waits for window
        poll_interval_s=0.05,
        coalesce_window_s=10.0,         # long → request sits in queue
    )
    d.start()
    # Submit but don't await yet
    task = asyncio.create_task(
        d.submit_and_collect(_FakeRecord(id="z"), _make_req(1), Path("/data"), deadline_s=30)
    )
    await asyncio.sleep(0.1)            # let it land in the queue
    await d.stop()
    result = await asyncio.wait_for(task, timeout=2)
    assert len(result) == 1
    assert result[0].exit_reason in ("error", "timeout")


def test_squeue_failure_is_treated_as_unknown_running_state(
    scratch, fake_sbatch_script, monkeypatch,
):
    dispatcher = SlurmDispatcher(scratch, fake_sbatch_script)
    monkeypatch.setattr(
        "omnicoding.rl.coordinator.dispatcher.subprocess.run",
        lambda *args, **kwargs: SimpleNamespace(
            returncode=1, stdout="", stderr="controller unavailable",
        ),
    )

    assert dispatcher._squeue_running([101, 202]) == {101, 202}


def test_coordinator_grades_worker_result_with_private_gold(
    scratch, fake_sbatch_script,
):
    dispatcher = SlurmDispatcher(scratch, fake_sbatch_script)
    payload = _success_traj(0, reward=0.0)
    payload["messages"] = [{
        "role": "assistant",
        "content": "<answer>answer</answer>",
        "tool_calls": [{
            "id": "call_1",
            "type": "function",
            "function": {"name": "task_complete", "arguments": "{}"},
        }],
    }]
    trajectory = Trajectory.model_validate(payload)

    graded = dispatcher._grade_result(
        trajectory,
        _FakeRecord(id="private-gold"),
        scratch / "workspace",
    )

    assert graded.raw_outcome_reward == 1.0
    assert graded.outcome_reward == 1.0
    assert graded.reward == 1.0


def test_gc_deletes_only_old_completed_job_dirs(
    scratch, fake_sbatch_script,
):
    dispatcher = SlurmDispatcher(scratch, fake_sbatch_script)
    active = scratch / "active-old-job"
    completed = scratch / "completed-old-job"
    active.mkdir()
    completed.mkdir()
    marker = completed / ".completed"
    marker.touch()
    old_timestamp = 1_000_000_000
    os.utime(active, (old_timestamp, old_timestamp))
    os.utime(marker, (old_timestamp, old_timestamp))

    assert dispatcher.cleanup_old_jobs(ttl_s=60) == 1
    assert active.is_dir()
    assert not completed.exists()
