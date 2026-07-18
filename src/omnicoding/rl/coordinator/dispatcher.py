"""Slurm-dispatch backend for the rollout coordinator.

Modal-side ``omnicoding.rl.rollout.generate`` fires ONE ``/rollout/submit``
per trajectory (Relax's stock per-sample async loop sets ``n_samples=1``).
Without coalescing, a single RL step (rollout_batch=16 × n_samples=8 = 128
trajectories) → 128 separate ``/rollout/submit`` calls → 128 sbatch jobs.

The dispatcher buffers incoming ``n_samples=1`` requests in a small window
and fires ONE sbatch job containing up to ``tasks_per_job`` trajectories.
With ``tasks_per_job=2`` and 128 incoming requests, the result is 64 sbatch
jobs (each running 2 trajectories concurrently on one allocated GPU node)
— halving slurm load while preserving per-trajectory isolation
(separate workspace dirs on local /tmp, separate result files on shared FS,
separate kira processes).

Requests with ``n_samples > 1`` (rare; e.g., direct curl tests) bypass the
coalescer and dispatch immediately as one sbatch.

Replaces the prior in-process ``asyncio.to_thread`` model, which hit the
default ThreadPoolExecutor cap (~8 on the 4-CPU coordinator host) and
silently bottlenecked at 8 concurrent trajectories.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import shutil
import subprocess
import sys
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from omnicoding.rl.reward import (
    extract_answer_from_messages,
    grade_trajectory,
    normalize,
)
from omnicoding.rl.schemas import RolloutRequest, Trajectory

from .media import stage_media

LOG = logging.getLogger("coordinator.dispatcher")
_SBATCH_RE = re.compile(r"Submitted batch job (\d+)")


@dataclass
class _PendingRequest:
    """One in-flight ``submit_and_collect`` waiting for the coalescer to
    bundle it into an sbatch job. ``future`` resolves with
    ``list[Trajectory]`` (length 1, since coalesced requests have
    ``n_samples=1``)."""

    record: Any
    req: RolloutRequest
    dataset_root: Path
    future: asyncio.Future
    enqueued_at: float = field(default_factory=time.monotonic)


class SlurmDispatcher:
    """Wraps sbatch submission + squeue polling + per-trajectory result IO,
    with optional coalescing of ``n_samples=1`` requests.

    Lifecycle::

        dispatcher = SlurmDispatcher(...)
        dispatcher.start()                     # spawn coalescer task
        # one POST /rollout/submit (n_samples=1):
        trajectories = await dispatcher.submit_and_collect(record, req,
                                                           dataset_root, deadline_s)
        # periodic GC:
        dispatcher.cleanup_old_jobs(ttl_s=3600)
        await dispatcher.stop()                # cancel coalescer

    Attrs:
        scratch_root:        where per-job dirs live (must be on shared FS so
                             compute nodes can read request files + write results)
        sbatch_script:       path to ``sbatch_one_job.sh``
        tasks_per_job:       max trajectories per sbatch job (coalesce target)
        poll_interval_s:     how often to ``squeue`` (default 15s)
        coalesce_window_s:   max time to wait for a partial batch to fill
                             (default 0.5s — 128 incoming requests typically
                             arrive within ~50ms so this rarely fires)
    """

    def __init__(
        self,
        scratch_root: Path,
        sbatch_script: Path,
        tasks_per_job: int = 2,
        poll_interval_s: float = 15.0,
        coalesce_window_s: float = 0.5,
    ):
        self.scratch_root = scratch_root
        self.sbatch_script = sbatch_script
        self.tasks_per_job = max(1, tasks_per_job)
        self.poll_interval_s = poll_interval_s
        self.coalesce_window_s = max(0.0, coalesce_window_s)
        scratch_root.mkdir(parents=True, exist_ok=True)
        if not sbatch_script.is_file():
            raise FileNotFoundError(f"sbatch script not found: {sbatch_script}")

        # Coalescer state
        self._pending: list[_PendingRequest] = []
        self._pending_cond: asyncio.Condition | None = None
        self._coalescer_task: asyncio.Task | None = None

    # ─── lifecycle ─────────────────────────────────────────────────────

    def start(self) -> None:
        """Spawn the background coalescer task. Must be called from inside
        an asyncio event loop (typically the FastAPI lifespan)."""
        if self._coalescer_task is not None:
            return
        self._pending_cond = asyncio.Condition()
        self._coalescer_task = asyncio.create_task(
            self._coalescer_loop(), name="dispatcher-coalescer",
        )
        LOG.info("dispatcher coalescer started: tasks_per_job=%d, window=%.2fs",
                 self.tasks_per_job, self.coalesce_window_s)

    async def stop(self) -> None:
        """Cancel the coalescer + fail-resolve any pending futures."""
        if self._coalescer_task is None:
            return
        self._coalescer_task.cancel()
        try:
            await self._coalescer_task
        except asyncio.CancelledError:
            pass
        self._coalescer_task = None
        # Fail-resolve anything still queued
        from .worker import _failed_trajectory  # noqa: PLC0415
        async with self._pending_cond:  # type: ignore[union-attr]
            for p in self._pending:
                if not p.future.done():
                    p.future.set_result([_failed_trajectory(0, "error", "dispatcher shutdown")])
            self._pending.clear()

    # ─── public entry point ────────────────────────────────────────────

    async def submit_and_collect(
        self,
        record: Any,
        req: RolloutRequest,
        dataset_root: Path,
        deadline_s: float,
    ) -> list[Trajectory]:
        """Submit ``req.n_samples`` trajectories.

        - ``n_samples == 1`` → enqueue into coalescer; coalescer fires one
          sbatch per ``tasks_per_job`` queued requests. Returns when this
          specific request's trajectory result is parsed.
        - ``n_samples > 1`` → bypass coalescer, dispatch immediately as
          chunks of ``tasks_per_job`` (legacy path; rare in production).

        Returns trajectories in sample-index order. Failed/missing samples
        get a ``_failed_trajectory`` shape (no exceptions raised).
        """
        if req.n_samples > 1 or self._coalescer_task is None or self.coalesce_window_s <= 0:
            return await self._submit_immediate(record, req, dataset_root, deadline_s)
        return await self._submit_via_coalescer(record, req, dataset_root, deadline_s)

    # ─── coalescer path (n_samples=1) ──────────────────────────────────

    async def _submit_via_coalescer(
        self,
        record: Any,
        req: RolloutRequest,
        dataset_root: Path,
        deadline_s: float,
    ) -> list[Trajectory]:
        from .worker import _failed_trajectory  # noqa: PLC0415

        future: asyncio.Future = asyncio.get_event_loop().create_future()
        pending = _PendingRequest(
            record=record, req=req, dataset_root=dataset_root, future=future,
        )
        async with self._pending_cond:  # type: ignore[union-attr]
            self._pending.append(pending)
            self._pending_cond.notify_all()  # wake coalescer

        try:
            return await asyncio.wait_for(future, timeout=deadline_s)
        except asyncio.TimeoutError:
            return [_failed_trajectory(0, "timeout", "coalescer/sbatch deadline")]

    async def _coalescer_loop(self) -> None:
        """Background drain loop. Sleeps until at least one request is
        pending; pops up to ``tasks_per_job`` and dispatches them as ONE
        sbatch. If pending count < tasks_per_job, waits up to
        ``coalesce_window_s`` for more before draining (so partial batches
        eventually go through)."""
        assert self._pending_cond is not None
        while True:
            try:
                async with self._pending_cond:
                    while not self._pending:
                        await self._pending_cond.wait()
                    # At least one pending. If < tasks_per_job, wait briefly
                    # for more to arrive.
                    if len(self._pending) < self.tasks_per_job:
                        try:
                            await asyncio.wait_for(
                                self._wait_for_full_batch(),
                                timeout=self.coalesce_window_s,
                            )
                        except asyncio.TimeoutError:
                            pass  # window expired — drain partial
                    batch = self._pending[: self.tasks_per_job]
                    self._pending = self._pending[self.tasks_per_job :]

                # Dispatch in background so coalescer keeps draining
                asyncio.create_task(
                    self._dispatch_coalesced_batch(batch),
                    name=f"dispatcher-batch-{len(batch)}",
                )
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001
                LOG.exception("coalescer loop error: %s", exc)
                await asyncio.sleep(1)

    async def _wait_for_full_batch(self) -> None:
        """Released by ``_pending_cond.notify_all()`` when new request enqueued.
        Used inside the window timeout to bail early once batch is full."""
        assert self._pending_cond is not None
        while len(self._pending) < self.tasks_per_job:
            await self._pending_cond.wait()

    async def _dispatch_coalesced_batch(self, batch: list[_PendingRequest]) -> None:
        """Write per-request scratch files (each in its own job_uuid dir for
        full isolation), submit ONE sbatch with N tasks, poll, resolve N
        futures with their respective Trajectory."""
        from .worker import _failed_trajectory  # noqa: PLC0415

        # Per-request: (PendingRequest, req_path, res_path, job_uuid, workspace)
        files: list[tuple[_PendingRequest, Path, Path, str, Path]] = []
        for p in batch:
            job_uuid = uuid.uuid4().hex[:12]
            job_dir = self.scratch_root / job_uuid
            job_dir.mkdir(parents=True, exist_ok=True)
            req_path = job_dir / "req_0.json"
            res_path = job_dir / "res_0.json"
            # Workspace under scratch/jobs/{uuid}/work/0/ (NFS, ~1.6 TB free)
            # rather than a small per-node /tmp shared with other jobs. Shared
            # storage is slower for small-file IO but avoids node-local
            # capacity failures during long media trajectories.
            workspace = job_dir / "work" / "0"
            try:
                payload = self._worker_payload(
                    job_id=job_uuid,
                    sample_index=0,
                    record=p.record,
                    request=p.req,
                    dataset_root=p.dataset_root,
                    workspace=workspace,
                )
                req_path.write_text(json.dumps(payload))
            except Exception as exc:  # noqa: BLE001
                LOG.exception("coalescer: task staging failed for %s", p.req.task_id)
                if not p.future.done():
                    p.future.set_result([
                        _failed_trajectory(0, "error", f"task staging failed: {exc}")
                    ])
                self._mark_completed(job_dir)
                continue
            files.append((p, req_path, res_path, job_uuid, workspace))

        if not files:
            return

        all_req_paths = [r for _, r, _, _, _ in files]
        all_res_paths = [r for _, _, r, _, _ in files]
        # sbatch logs go to first request's job_dir (informational only)
        log_dir = files[0][1].parent
        jid = self._sbatch(all_req_paths, all_res_paths, log_dir, chunk_idx=0)

        if jid is None:
            LOG.error("coalescer: sbatch FAILED for batch of %d trajectories",
                      len(batch))
            for p, _, _, _, _ in files:
                if not p.future.done():
                    p.future.set_result(
                        [_failed_trajectory(0, "error", "sbatch submission failed")]
                    )
            for _, req_path, _, _, _ in files:
                self._mark_completed(req_path.parent)
            return

        LOG.info("coalescer: submitted jid=%d running %d trajectories",
                 jid, len(batch))

        # Per-batch deadline = max of individual deadlines (the inner
        # wait_for at each future cap will trigger client-side timeout).
        per_req_deadlines = [
            p.req.request_timeout_s * (p.req.max_turns + 4) + 600
            for p, _, _, _, _ in files
        ]
        max_deadline = max(per_req_deadlines)
        end_at = time.monotonic() + max_deadline

        while time.monotonic() < end_at:
            await asyncio.sleep(self.poll_interval_s)
            # All futures already done (timed out client-side) → kill sbatch
            if all(p.future.done() for p, _, _, _, _ in files):
                LOG.info("coalescer: jid=%d — all callers gave up; scancelling",
                         jid)
                subprocess.run(["scancel", str(jid)], check=False, timeout=10)
                return
            running = self._squeue_running([jid])
            if jid in running:
                continue
            # Job finished — read results, resolve futures
            LOG.info("coalescer: jid=%d finished, reading %d results",
                     jid, len(files))
            for p, _, res_path, _, workspace in files:
                if p.future.done():
                    continue
                traj = self._read_result(
                    res_path,
                    sample_idx=0,
                    record=p.record,
                    workspace=workspace,
                )
                p.future.set_result([traj])
            for _, req_path, _, _, _ in files:
                self._mark_completed(req_path.parent)
            return

        # Deadline blew through
        LOG.warning("coalescer: jid=%d exceeded %.0fs deadline; scancelling",
                    jid, max_deadline)
        subprocess.run(["scancel", str(jid)], check=False, timeout=10)
        for p, _, _, _, _ in files:
            if not p.future.done():
                p.future.set_result(
                    [_failed_trajectory(0, "timeout", f"sbatch jid={jid} deadline")]
                )

    # ─── immediate path (n_samples > 1, no coalescing) ─────────────────

    async def _submit_immediate(
        self,
        record: Any,
        req: RolloutRequest,
        dataset_root: Path,
        deadline_s: float,
    ) -> list[Trajectory]:
        """Legacy path — chunk req.n_samples into tasks_per_job-sized chunks
        and submit one sbatch per chunk. Used when caller explicitly wants
        multiple samples in a single submit (rare; mostly direct curl)."""
        from .worker import _failed_trajectory  # noqa: PLC0415

        job_uuid = uuid.uuid4().hex[:12]
        job_dir = self.scratch_root / job_uuid
        job_dir.mkdir(parents=True)

        sample_indices = list(range(req.n_samples))
        chunks = [
            sample_indices[i : i + self.tasks_per_job]
            for i in range(0, len(sample_indices), self.tasks_per_job)
        ]

        results: dict[int, Trajectory] = {}
        jobs: list[tuple[int | None, list[int], list[Path], list[Path]]] = []
        for chunk_idx, indices in enumerate(chunks):
            req_paths: list[Path] = []
            res_paths: list[Path] = []
            workspaces: list[Path] = []
            staged_indices: list[int] = []
            for sample_idx in indices:
                req_path = job_dir / f"req_{sample_idx}.json"
                res_path = job_dir / f"res_{sample_idx}.json"
                workspace = job_dir / "work" / str(sample_idx)
                try:
                    payload = self._worker_payload(
                        job_id=job_uuid,
                        sample_index=sample_idx,
                        record=record,
                        request=req,
                        dataset_root=dataset_root,
                        workspace=workspace,
                    )
                    req_path.write_text(json.dumps(payload))
                except Exception as exc:  # noqa: BLE001
                    LOG.exception("immediate: task staging failed for sample %d", sample_idx)
                    results[sample_idx] = _failed_trajectory(
                        sample_idx, "error", f"task staging failed: {exc}",
                    )
                    continue
                req_paths.append(req_path)
                res_paths.append(res_path)
                workspaces.append(workspace)
                staged_indices.append(sample_idx)
            if not req_paths:
                continue
            jid = self._sbatch(req_paths, res_paths, job_dir, chunk_idx)
            jobs.append((jid, staged_indices, res_paths, workspaces))
            if jid is None:
                LOG.error("immediate: sbatch FAILED for chunk %d (samples %s)",
                          chunk_idx, staged_indices)
            else:
                LOG.info("immediate: submitted jid=%d chunk=%d samples=%s",
                         jid, chunk_idx, staged_indices)

        end_at = time.monotonic() + deadline_s
        for jid, indices, _, _ in jobs:
            if jid is not None:
                continue
            for sample_idx in indices:
                results[sample_idx] = _failed_trajectory(
                    sample_idx, "error", "sbatch submission failed",
                )

        pending_jids = {jid for jid, _, _, _ in jobs if jid is not None}
        jid_to_chunk = {
            jid: (indices, res_paths, workspaces)
            for jid, indices, res_paths, workspaces in jobs
            if jid is not None
        }

        while pending_jids and time.monotonic() < end_at:
            await asyncio.sleep(self.poll_interval_s)
            running = self._squeue_running(list(pending_jids))
            for jid in list(pending_jids):
                if jid in running:
                    continue
                indices, res_paths, workspaces = jid_to_chunk[jid]
                LOG.info("immediate: jid=%d finished, reading %d results",
                         jid, len(indices))
                for sample_idx, res_path, workspace in zip(
                    indices, res_paths, workspaces, strict=True,
                ):
                    if sample_idx in results:
                        continue
                    results[sample_idx] = self._read_result(
                        res_path,
                        sample_idx,
                        record=record,
                        workspace=workspace,
                    )
                pending_jids.discard(jid)

        completed_without_running_jobs = not pending_jids
        if pending_jids:
            LOG.warning("immediate: %d jids still running at deadline; scancelling",
                        len(pending_jids))
            for jid in pending_jids:
                subprocess.run(["scancel", str(jid)], check=False, timeout=10)
                indices, _, _ = jid_to_chunk[jid]
                for sample_idx in indices:
                    if sample_idx not in results:
                        results[sample_idx] = _failed_trajectory(
                            sample_idx, "timeout",
                            f"sbatch jid={jid} exceeded {deadline_s:.0f}s deadline",
                        )

        for sample_idx in sample_indices:
            if sample_idx not in results:
                results[sample_idx] = _failed_trajectory(
                    sample_idx, "error", "result missing post-deadline",
                )

        if completed_without_running_jobs:
            self._mark_completed(job_dir)

        return [results[i] for i in sorted(results)]

    # ─── periodic GC ───────────────────────────────────────────────────

    def cleanup_old_jobs(self, ttl_s: float) -> int:
        """Remove explicitly completed job dirs older than ``ttl_s``.

        Directories without a completion marker are conservatively retained:
        they may belong to a live Slurm job or to a coordinator-restart orphan
        whose scheduler state has not yet been reconciled.
        """
        now = time.time()
        dropped = 0
        for path in self.scratch_root.iterdir():
            if not path.is_dir():
                continue
            marker = path / ".completed"
            if not marker.is_file():
                continue
            try:
                age = now - marker.stat().st_mtime
            except OSError:
                continue
            if age > ttl_s:
                shutil.rmtree(path, ignore_errors=True)
                dropped += 1
        if dropped:
            LOG.info("dispatcher: GC dropped %d job dirs older than %.0fs",
                     dropped, ttl_s)
        return dropped

    # ─── helpers ───────────────────────────────────────────────────────

    @staticmethod
    def _mark_completed(job_dir: Path) -> None:
        """Make a job directory GC-eligible after scheduler completion."""
        (job_dir / ".completed").touch(exist_ok=True)

    @staticmethod
    def _record_to_dict(record: Any) -> dict:
        if hasattr(record, "__dataclass_fields__"):
            return asdict(record)
        if hasattr(record, "model_dump"):
            return record.model_dump(mode="json")
        raise TypeError(f"can't serialize record of type {type(record).__name__}")

    @classmethod
    def _record_for_worker(cls, record: Any) -> dict:
        """Serialize only task inputs; answer keys stay in the coordinator."""
        payload = cls._record_to_dict(record)
        payload.pop("ground_truth", None)
        return payload

    @classmethod
    def _worker_payload(
        cls,
        *,
        job_id: str,
        sample_index: int,
        record: Any,
        request: RolloutRequest,
        dataset_root: Path,
        workspace: Path,
    ) -> dict[str, Any]:
        """Stage task media and build a gold-free Slurm worker payload."""
        workspace.mkdir(parents=True, exist_ok=True)
        staged = stage_media(record, workspace, dataset_root)
        return {
            "job_id": job_id,
            "sample_index": sample_index,
            "record": cls._record_for_worker(record),
            "request": request.model_dump(mode="json"),
            "workspace": str(workspace.resolve()),
            "staged_media": staged,
        }

    @staticmethod
    def _grade_result(trajectory: Trajectory, record: Any, workspace: Path) -> Trajectory:
        """Apply gold-dependent reward components inside the coordinator."""
        if trajectory.error and not trajectory.messages:
            return trajectory
        extracted = extract_answer_from_messages(trajectory.messages)
        details = grade_trajectory(
            trajectory.messages,
            list(record.ground_truth),
            record.answer_type,
            exit_reason=trajectory.exit_reason,
            media=record.media,
            workspace=workspace,
            observed_n_tool_calls=trajectory.n_tool_calls,
        )
        details["num_steps"] = float(trajectory.n_steps)
        return trajectory.model_copy(update={
            "extracted_answer": extracted,
            "prediction_normalized": details.get("prediction_normalized")
            or (normalize(extracted) if extracted else None),
            "reward": details["score"],
            "outcome_reward": details["correctness"],
            "raw_outcome_reward": details["raw_acc"],
            "format_reward": details["format"],
            "modality_reward": details["modality_penalty"],
            "bad_tool_reward": details["bad_tool_penalty"],
            "modality_match": details["modality_match"],
            "p_bad_tool": details["p_bad_tool"],
            "n_unparseable": int(details["n_unparseable"]),
            "n_disallowed": int(details["n_disallowed"]),
            "n_escape": int(details["n_escape"]),
            "n_syntax_fail": int(details["n_syntax_fail"]),
            "removed": bool(details["removed"]),
            "reward_details": details,
        })

    def _sbatch(
        self,
        req_paths: list[Path],
        res_paths: list[Path],
        job_dir: Path,
        chunk_idx: int,
    ) -> int | None:
        allowed_environment = (
            "PATH",
            "PYTHONPATH",
            "LD_LIBRARY_PATH",
            "MODULEPATH",
            "CUDA_HOME",
            "HF_HOME",
            "HF_HUB_CACHE",
            "TRANSFORMERS_CACHE",
            "TOKENIZERS_PARALLELISM",
        )
        env = {
            key: value
            for key in allowed_environment
            if (value := os.environ.get(key)) is not None
        }
        env["REQUEST_FILES"] = ";".join(str(p) for p in req_paths)
        env["RESULT_FILES"] = ";".join(str(p) for p in res_paths)
        env["OMNICODING_PYTHON"] = sys.executable
        out_log = job_dir / f"chunk_{chunk_idx}.slurm.out"
        err_log = job_dir / f"chunk_{chunk_idx}.slurm.err"
        cmd = [
            "sbatch",
            "--export=" + ",".join(sorted(env)),
            "--output", str(out_log),
            "--error", str(err_log),
            "--job-name", f"relax-traj-{job_dir.name[:6]}-c{chunk_idx}",
            str(self.sbatch_script),
        ]
        try:
            proc = subprocess.run(
                cmd, env=env, capture_output=True, text=True, timeout=30,
            )
        except subprocess.TimeoutExpired:
            LOG.error("dispatcher: sbatch call timed out for chunk %d", chunk_idx)
            return None
        if proc.returncode != 0:
            LOG.error("dispatcher: sbatch rc=%d stderr=%s",
                      proc.returncode, proc.stderr.strip()[:500])
            return None
        m = _SBATCH_RE.search(proc.stdout)
        if not m:
            LOG.error("dispatcher: sbatch stdout unparseable: %r", proc.stdout)
            return None
        return int(m.group(1))

    def _squeue_running(self, jids: list[int]) -> set[int]:
        if not jids:
            return set()
        try:
            proc = subprocess.run(
                [
                    "squeue", "-h", "-o", "%i",
                    "-j", ",".join(str(j) for j in jids),
                ],
                capture_output=True, text=True, timeout=30,
            )
        except subprocess.TimeoutExpired:
            LOG.warning("dispatcher: squeue timeout, treating all as still running")
            return set(jids)
        if proc.returncode != 0:
            LOG.warning(
                "dispatcher: squeue rc=%d, treating requested jobs as still running: %s",
                proc.returncode,
                proc.stderr.strip()[:500],
            )
            return set(jids)
        return {int(line.strip()) for line in proc.stdout.splitlines() if line.strip().isdigit()}

    def _read_result(
        self,
        res_path: Path,
        sample_idx: int,
        record: Any,
        workspace: Path,
    ) -> Trajectory:
        from .worker import _failed_trajectory  # noqa: PLC0415

        if not res_path.is_file():
            LOG.warning("dispatcher: result file missing for sample %d at %s",
                        sample_idx, res_path)
            return _failed_trajectory(
                sample_idx, "error", f"no result file at {res_path}",
            )
        try:
            trajectory = Trajectory.model_validate_json(res_path.read_text())
            return self._grade_result(trajectory, record, workspace)
        except Exception as exc:  # noqa: BLE001
            LOG.error("dispatcher: result parse failed for %s: %s", res_path, exc)
            return _failed_trajectory(
                sample_idx, "error", f"result parse: {exc}",
            )
