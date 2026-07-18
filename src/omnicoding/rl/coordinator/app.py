"""FastAPI rollout coordinator.

Async submit/poll model — each HTTP call returns in under a second so a
Cloudflare quick-tunnel 100-second timeout never bites. Multi-minute kira
trajectories happen in background asyncio tasks.

Endpoints:
- ``GET  /health``                       liveness + dataset size + job stats
- ``POST /rollout/submit``               schedule N trajectories, return job_id
- ``GET  /rollout/result/{job_id}``      202 if pending, 200 with payload if done
- ``POST /rollout/run``                  legacy synchronous (used by smoke tests
                                           and direct curl); kept for back-compat.

Job lifecycle:
- submit creates an asyncio.Task in app.state.tasks[job_id]
- task awaits trajectory completion, stores result in app.state.results[job_id]
- a GC loop drops results older than ROLLOUT_RESULT_TTL_S (default 1h) and
  the matching task entry from app.state.tasks.
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import Depends, FastAPI, HTTPException
from fastapi.responses import JSONResponse

from omnicoding.rl.schemas import (
    RolloutRequest,
    RolloutResponse,
    RolloutResultResponse,
    RolloutSubmitResponse,
    Trajectory,
)
from omnicoding.rl.security import (
    coordinator_token,
    enforce_request_policy,
    require_coordinator_token,
    validate_policy_config,
)

from .dataset import Record, load_records
from .dispatcher import SlurmDispatcher

LOGGER = logging.getLogger("coordinator.app")


def _env_path(name: str, default: str | None = None) -> Path:
    val = os.environ.get(name, default)
    if not val:
        raise RuntimeError(f"env var {name} required (set in .env)")
    return Path(val)


# ─── lifespan: load dataset, spawn GC loop ───────────────────────────────────


async def _gc_loop(app: FastAPI) -> None:
    """Periodically drop completed results older than the TTL so memory
    usage stays bounded under long training runs. Also GC stale slurm
    job dirs on the shared scratch filesystem."""
    interval_s = int(os.environ.get("ROLLOUT_RESULT_GC_INTERVAL_S", "300"))
    ttl_s = int(os.environ.get("ROLLOUT_RESULT_TTL_S", "3600"))
    job_dir_ttl_s = int(os.environ.get("ROLLOUT_JOB_DIR_TTL_S", "7200"))
    while True:
        try:
            await asyncio.sleep(interval_s)
        except asyncio.CancelledError:
            return
        now = time.time()
        dropped = 0
        for jid, entry in list(app.state.results.items()):
            if now - entry["completed_at"] > ttl_s:
                app.state.results.pop(jid, None)
                app.state.tasks.pop(jid, None)
                app.state.submitted_at.pop(jid, None)
                dropped += 1
        if dropped:
            LOGGER.info("gc dropped %d expired result entries", dropped)
        try:
            n = app.state.dispatcher.cleanup_old_jobs(ttl_s=job_dir_ttl_s)
            if n:
                LOGGER.info("gc dropped %d stale slurm job dirs", n)
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning("gc dispatcher cleanup failed: %s", exc)


@asynccontextmanager
async def _lifespan(app: FastAPI):
    coordinator_token()
    validate_policy_config()
    rl_train = _env_path("RL_TRAIN_JSONL")
    dataset_root = _env_path("DATASET_ROOT")
    runtime_root = _env_path("OMNICODING_RUNTIME_ROOT", str(Path.cwd()))

    # Slurm-dispatch backend — replaces the prior in-process
    # ``asyncio.to_thread`` model that bottlenecked at the default
    # ThreadPoolExecutor cap (~8 on this 4-CPU host).
    scratch_root = Path(
        os.environ.get(
            "ROLLOUT_SCRATCH_ROOT",
            str(runtime_root / "scratch" / "rl_jobs"),
        )
    )
    sbatch_script = _env_path("ROLLOUT_SBATCH_SCRIPT")
    tasks_per_job = int(os.environ.get("ROLLOUT_TASKS_PER_JOB", "2"))
    poll_interval_s = float(os.environ.get("ROLLOUT_POLL_INTERVAL_S", "15"))
    coalesce_window_s = float(os.environ.get("ROLLOUT_COALESCE_WINDOW_S", "0.5"))
    dispatcher = SlurmDispatcher(
        scratch_root=scratch_root,
        sbatch_script=sbatch_script,
        tasks_per_job=tasks_per_job,
        poll_interval_s=poll_interval_s,
        coalesce_window_s=coalesce_window_s,
    )
    dispatcher.start()  # spawn background coalescer task

    app.state.records = load_records(rl_train)
    app.state.dataset_root = dataset_root
    app.state.dispatcher = dispatcher
    app.state.startup_ts = time.time()
    app.state.in_flight = 0
    app.state.max_in_flight = int(os.environ.get("ROLLOUT_MAX_IN_FLIGHT", "64"))
    app.state.max_queued_jobs = int(os.environ.get("ROLLOUT_MAX_QUEUED_JOBS", "256"))
    app.state.capacity_lock = asyncio.Lock()
    app.state.tasks: dict[str, asyncio.Task] = {}
    app.state.submitted_at: dict[str, float] = {}
    app.state.results: dict[str, dict[str, Any]] = {}
    gc_task = asyncio.create_task(_gc_loop(app), name="rollout-gc")
    LOGGER.info(
        "coordinator ready: %d records, dataset_root=%s, scratch_root=%s, "
        "tasks_per_job=%d, coalesce_window=%.2fs, sbatch=%s",
        len(app.state.records), dataset_root, scratch_root,
        tasks_per_job, coalesce_window_s, sbatch_script,
    )
    try:
        yield
    finally:
        gc_task.cancel()
        try:
            await gc_task
        except asyncio.CancelledError:
            pass
        # Cancel still-running rollout tasks so the process can exit cleanly.
        for t in app.state.tasks.values():
            t.cancel()
        await app.state.dispatcher.stop()


app = FastAPI(
    title="OmniCoding rollout coordinator",
    lifespan=_lifespan,
    dependencies=[Depends(require_coordinator_token)],
)


# ─── shared trajectory dispatch ──────────────────────────────────────────────


async def _run_trajectories(app: FastAPI, req: RolloutRequest) -> RolloutResponse:
    """Submit ``req.n_samples`` kira trajectories as sbatch jobs (chunked
    ``tasks_per_job`` per job); poll until all results land or the deadline
    elapses. Raises HTTPException(404) if the task_id is unknown."""
    record: Record | None = app.state.records.get(req.task_id)
    if record is None:
        raise HTTPException(status_code=404, detail=f"task_id {req.task_id!r} not found")

    started = time.time()
    # Per-rollout deadline: same shape as the prior in-process version
    # (request_timeout_s × (max_turns + 4)) plus slack for sbatch queue
    # latency in the cluster scheduler. Coordinator scancels any sbatch
    # still alive at the deadline.
    deadline_s = req.request_timeout_s * (req.max_turns + 4) + 600

    enforce_request_policy(req)
    async with app.state.capacity_lock:
        if app.state.in_flight + req.n_samples > app.state.max_in_flight:
            raise HTTPException(status_code=429, detail="rollout capacity exhausted")
        app.state.in_flight += req.n_samples
    try:
        trajectories = await app.state.dispatcher.submit_and_collect(
            record=record,
            req=req,
            dataset_root=app.state.dataset_root,
            deadline_s=deadline_s,
        )
    finally:
        async with app.state.capacity_lock:
            app.state.in_flight -= req.n_samples

    elapsed = time.time() - started
    LOGGER.info(
        "rollout done id=%s n=%d elapsed=%.1fs rewards=%s",
        req.task_id, req.n_samples, elapsed, [t.reward for t in trajectories],
    )
    return RolloutResponse(
        task_id=req.task_id,
        n_samples=req.n_samples,
        trajectories=trajectories,
        elapsed_s=elapsed,
    )


# ─── endpoints ───────────────────────────────────────────────────────────────


@app.get("/health")
async def health() -> dict:
    return {
        "status": "ok",
        "uptime_s": time.time() - app.state.startup_ts,
        "n_records": len(app.state.records),
        "in_flight": app.state.in_flight,
        "n_jobs_pending": sum(1 for jid in app.state.tasks if jid not in app.state.results),
        "n_jobs_completed_cached": len(app.state.results),
    }


@app.post("/rollout/submit", response_model=RolloutSubmitResponse)
async def rollout_submit(req: RolloutRequest) -> RolloutSubmitResponse:
    """Schedule trajectories in the background. Returns a job_id immediately
    (no Cloudflare 524 risk because the response is sub-second). Caller polls
    ``GET /rollout/result/{job_id}``."""
    # Validate task_id up front so the caller gets 404 synchronously rather
    # than a "completed with error" surprise on the first poll.
    if req.task_id not in app.state.records:
        raise HTTPException(status_code=404, detail=f"task_id {req.task_id!r} not found")
    enforce_request_policy(req)
    pending = sum(1 for jid in app.state.tasks if jid not in app.state.results)
    if pending >= app.state.max_queued_jobs:
        raise HTTPException(status_code=429, detail="rollout queue is full")

    job_id = uuid.uuid4().hex[:16]
    app.state.submitted_at[job_id] = time.time()

    async def _job() -> None:
        try:
            response = await _run_trajectories(app, req)
            app.state.results[job_id] = {
                "response": response,
                "completed_at": time.time(),
                "error": None,
            }
        except asyncio.CancelledError:
            app.state.results[job_id] = {
                "response": None,
                "completed_at": time.time(),
                "error": "cancelled",
            }
            raise
        except Exception as exc:  # noqa: BLE001
            LOGGER.exception("job %s failed", job_id)
            app.state.results[job_id] = {
                "response": None,
                "completed_at": time.time(),
                "error": f"{type(exc).__name__}: {exc}",
            }

    app.state.tasks[job_id] = asyncio.create_task(_job(), name=f"job-{job_id}")
    LOGGER.info("submitted job=%s task=%s n=%d", job_id, req.task_id, req.n_samples)
    return RolloutSubmitResponse(job_id=job_id, task_id=req.task_id, n_samples=req.n_samples)


@app.get("/rollout/result/{job_id}")
async def rollout_result(job_id: str):
    """Return the trajectory result for ``job_id`` if available, else 202
    pending. Cleaned up by the GC loop after ``ROLLOUT_RESULT_TTL_S``."""
    if job_id not in app.state.submitted_at:
        raise HTTPException(status_code=404, detail=f"job_id {job_id!r} not found")
    elapsed = time.time() - app.state.submitted_at[job_id]
    entry = app.state.results.get(job_id)
    if entry is None:
        return JSONResponse(
            status_code=202,
            content=RolloutResultResponse(
                job_id=job_id, status="pending", elapsed_s=elapsed,
            ).model_dump(),
        )
    if entry["error"]:
        return JSONResponse(
            status_code=200,
            content=RolloutResultResponse(
                job_id=job_id, status="error", elapsed_s=elapsed, error=entry["error"],
            ).model_dump(),
        )
    return JSONResponse(
        status_code=200,
        content=RolloutResultResponse(
            job_id=job_id,
            status="completed",
            elapsed_s=elapsed,
            response=entry["response"],
        ).model_dump(),
    )


@app.post("/rollout/run", response_model=RolloutResponse)
async def rollout_run(req: RolloutRequest) -> RolloutResponse:
    """Legacy synchronous endpoint: blocks until all trajectories complete.
    Kept for direct curl / smoke testing; the production Modal-side rollout
    fn uses ``submit + poll`` to dodge the Cloudflare quick-tunnel 100s
    timeout."""
    return await _run_trajectories(app, req)
