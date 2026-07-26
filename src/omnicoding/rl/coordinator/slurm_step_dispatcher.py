"""Run coding-agent trajectories as isolated job steps in one Slurm allocation.

Unlike :mod:`dispatcher`, this backend never submits child ``sbatch`` jobs.
The coordinator stays beside Relax/SGLang in one allocation and launches a
bounded number of short ``srun`` job steps. Pyxis gives every trajectory its
own read-only container root and only five explicit mounts:

* its private workspace (rw);
* its private temporary directory (rw);
* its gold-free request JSON (ro);
* its result JSON (rw).
* the shared harness dependency environment (ro).

The user's home, the dataset root, coordinator secrets, model weights, and
sibling trajectories are not mounted.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import signal
import time
import uuid
from pathlib import Path
from typing import Any

from omnicoding.rl.schemas import RolloutRequest, Trajectory

from .dispatcher import SlurmDispatcher

LOG = logging.getLogger("coordinator.slurm_step_dispatcher")


class SlurmStepDispatcher(SlurmDispatcher):
    """Bounded, container-isolated rollout execution within one allocation."""

    def __init__(
        self,
        *,
        scratch_root: Path,
        container_image: str,
        harness_venv: str = "",
        max_concurrency: int = 12,
        cpus_per_rollout: int = 2,
        memory_per_rollout: str = "8G",
        gpu_devices: tuple[str, ...] = (),
        rollouts_per_gpu: int = 1,
    ) -> None:
        # Deliberately do not call SlurmDispatcher.__init__: this backend has no
        # sbatch script, coalescer, or scheduler-polling loop. It inherits only
        # the audited payload, grading, result parsing, and GC helpers.
        self.scratch_root = scratch_root.resolve()
        self.container_image = container_image.strip()
        self.harness_venv = (
            Path(harness_venv).expanduser().resolve() if harness_venv.strip() else None
        )
        self.gpu_devices = tuple(
            value.strip() for value in gpu_devices if value.strip()
        )
        self.rollouts_per_gpu = max(1, rollouts_per_gpu)
        requested_concurrency = max(1, max_concurrency)
        if self.gpu_devices:
            requested_concurrency = min(
                requested_concurrency,
                len(self.gpu_devices) * self.rollouts_per_gpu,
            )
        self.max_concurrency = requested_concurrency
        self.cpus_per_rollout = max(1, cpus_per_rollout)
        self.memory_per_rollout = memory_per_rollout.strip()
        self._step_sem: asyncio.Semaphore | None = None
        self._running: set[asyncio.subprocess.Process] = set()
        self.scratch_root.mkdir(parents=True, exist_ok=True)

        if not self.container_image:
            raise ValueError("ROLLOUT_CONTAINER_IMAGE is required for slurm_step backend")
        if self.harness_venv is None:
            raise ValueError("ROLLOUT_HARNESS_VENV is required for slurm_step backend")
        if not (self.harness_venv / "bin" / "python").is_file():
            raise ValueError(
                f"ROLLOUT_HARNESS_VENV has no bin/python: {self.harness_venv}"
            )
        if not self.memory_per_rollout:
            raise ValueError("ROLLOUT_MEMORY_PER_TASK cannot be empty")

    def start(self) -> None:
        if self._step_sem is not None:
            return
        if not os.environ.get("SLURM_JOB_ID"):
            raise RuntimeError(
                "slurm_step rollout backend must run inside an existing Slurm allocation"
            )
        self._step_sem = asyncio.Semaphore(self.max_concurrency)
        LOG.info(
            "Slurm step dispatcher ready: allocation=%s concurrency=%d "
            "cpus_per_rollout=%d memory_per_rollout=%s gpu_devices=%s "
            "rollouts_per_gpu=%d image=%s harness_venv=%s",
            os.environ["SLURM_JOB_ID"],
            self.max_concurrency,
            self.cpus_per_rollout,
            self.memory_per_rollout,
            ",".join(self.gpu_devices) or "none",
            self.rollouts_per_gpu,
            self.container_image,
            self.harness_venv,
        )

    async def stop(self) -> None:
        running = list(self._running)
        for proc in running:
            if proc.returncode is None:
                proc.send_signal(signal.SIGTERM)
        if running:
            await asyncio.gather(*(proc.wait() for proc in running), return_exceptions=True)
        self._running.clear()
        self._step_sem = None

    async def submit_and_collect(
        self,
        record: Any,
        req: RolloutRequest,
        dataset_root: Path,
        deadline_s: float,
    ) -> list[Trajectory]:
        tasks = [
            asyncio.create_task(
                self._run_one(
                    record=record,
                    req=req,
                    dataset_root=dataset_root,
                    sample_idx=sample_idx,
                    deadline_s=deadline_s,
                ),
                name=f"rollout-step-{req.task_id}-{sample_idx}",
            )
            for sample_idx in range(req.n_samples)
        ]
        return list(await asyncio.gather(*tasks))

    async def _run_one(
        self,
        *,
        record: Any,
        req: RolloutRequest,
        dataset_root: Path,
        sample_idx: int,
        deadline_s: float,
    ) -> Trajectory:
        from .worker import _failed_trajectory  # noqa: PLC0415

        assert self._step_sem is not None
        async with self._step_sem:
            job_uuid = uuid.uuid4().hex[:12]
            job_dir = self.scratch_root / job_uuid
            workspace = job_dir / "workspace"
            tmp_dir = job_dir / "tmp"
            req_path = job_dir / "request.json"
            res_path = job_dir / "result.json"
            stdout_path = job_dir / "worker.stdout"
            stderr_path = job_dir / "worker.stderr"

            try:
                job_dir.mkdir(parents=True)
                tmp_dir.mkdir()
                payload = await asyncio.to_thread(
                    self._worker_payload,
                    job_id=job_uuid,
                    sample_index=sample_idx,
                    record=record,
                    request=req,
                    dataset_root=dataset_root,
                    workspace=workspace,
                )
                # The worker sees the bind-mounted container path only. Do not
                # disclose the host scratch layout in its request payload.
                payload["workspace"] = "/workspace"
                req_path.write_text(json.dumps(payload))
                # Pyxis bind mounts require the source file to exist.
                res_path.touch()
            except Exception as exc:  # noqa: BLE001
                LOG.exception("step: task staging failed for %s", req.task_id)
                return _failed_trajectory(
                    sample_idx, "error", f"task staging failed: {exc}"
                )

            command = self._srun_command(
                workspace=workspace,
                tmp_dir=tmp_dir,
                req_path=req_path,
                res_path=res_path,
                gpu_device=(
                    self.gpu_devices[sample_idx % len(self.gpu_devices)]
                    if self.gpu_devices
                    else None
                ),
            )
            started = time.monotonic()
            with stdout_path.open("wb") as stdout, stderr_path.open("wb") as stderr:
                try:
                    proc = await asyncio.create_subprocess_exec(
                        *command,
                        stdin=asyncio.subprocess.DEVNULL,
                        stdout=stdout,
                        stderr=stderr,
                        env=self._worker_control_env(
                            (
                                self.gpu_devices[sample_idx % len(self.gpu_devices)]
                                if self.gpu_devices
                                else None
                            )
                        ),
                    )
                except Exception as exc:  # noqa: BLE001
                    LOG.exception("step: srun launch failed for %s", req.task_id)
                    self._mark_completed(job_dir)
                    return _failed_trajectory(
                        sample_idx, "error", f"srun launch failed: {exc}"
                    )

                self._running.add(proc)
                try:
                    await asyncio.wait_for(proc.wait(), timeout=deadline_s)
                except asyncio.TimeoutError:
                    proc.send_signal(signal.SIGTERM)
                    try:
                        await asyncio.wait_for(proc.wait(), timeout=15)
                    except asyncio.TimeoutError:
                        proc.kill()
                        await proc.wait()
                    self._mark_completed(job_dir)
                    return _failed_trajectory(
                        sample_idx,
                        "timeout",
                        f"Slurm job step exceeded {deadline_s:.0f}s",
                    )
                finally:
                    self._running.discard(proc)

            if proc.returncode != 0:
                detail = stderr_path.read_text(errors="replace")[-2000:]
                self._mark_completed(job_dir)
                return _failed_trajectory(
                    sample_idx,
                    "error",
                    f"srun rc={proc.returncode}: {detail}",
                )

            trajectory = self._read_result(
                res_path,
                sample_idx=sample_idx,
                record=record,
                workspace=workspace,
            )
            self._mark_completed(job_dir)
            LOG.info(
                "step: completed task=%s sample=%d elapsed=%.1fs exit=%s",
                req.task_id,
                sample_idx,
                time.monotonic() - started,
                trajectory.exit_reason,
            )
            return trajectory

    def _srun_command(
        self,
        *,
        workspace: Path,
        tmp_dir: Path,
        req_path: Path,
        res_path: Path,
        gpu_device: str | None = None,
    ) -> list[str]:
        mounts = ",".join(
            (
                f"{workspace.resolve()}:/workspace:rw+rprivate",
                f"{tmp_dir.resolve()}:/tmp:rw+rprivate",
                f"{req_path.resolve()}:/run/request.json:ro+rprivate",
                f"{res_path.resolve()}:/run/result.json:rw+rprivate",
                f"{self.harness_venv}:{self.harness_venv}:ro+rprivate",
            )
        )
        command = [
            "srun",
            "--jobid",
            os.environ["SLURM_JOB_ID"],
            "--overlap",
            "--exact",
            "--nodes=1",
            "--ntasks=1",
            f"--cpus-per-task={self.cpus_per_rollout}",
            f"--mem={self.memory_per_rollout}",
            "--no-kill",
            f"--container-image={self.container_image}",
            "--container-remap-root",
            "--no-container-mount-home",
            "--container-readonly",
            f"--container-mounts={mounts}",
            "--container-workdir=/workspace",
            (
                "--container-env=PYTHONUNBUFFERED,PYTHONDONTWRITEBYTECODE,"
                "CUDA_VISIBLE_DEVICES,NVIDIA_VISIBLE_DEVICES,HF_HOME,"
                "MPLCONFIGDIR,NUMBA_CACHE_DIR,XDG_CACHE_HOME,"
                "OMNICODING_HARNESS_VENV,"
                "OMNICODING_WEB_SEARCH_PROXY_URL,"
                "OMNICODING_WEB_SEARCH_PROXY_TOKEN"
            ),
        ]
        # Slurm rewrites CUDA_VISIBLE_DEVICES to the allocation-wide physical
        # list after applying --export.  Pyxis still restricts /dev via
        # NVIDIA_VISIBLE_DEVICES, but CUDA clients inside the one-device
        # namespace must address that mounted device as logical index 0.
        if gpu_device is not None:
            command.extend(["env", "CUDA_VISIBLE_DEVICES=0"])
        command.extend([
            "python3",
            "-m",
            "omnicoding.rl.run_trajectory",
            "--request",
            "/run/request.json",
            "--result",
            "/run/result.json",
            "--workspace",
            "/workspace",
        ])
        return command

    def _worker_control_env(self, gpu_device: str | None = None) -> dict[str, str]:
        """Minimal environment for ``srun`` itself; no upstream secrets.

        A configured GPU identifier is a deliberately shared device: the
        allocation owns it, while several isolated trajectory containers may
        use it for ASR/OCR.  No ``--gres`` is requested per job step because
        Slurm cannot express fractional-GPU sharing for overlapping steps.
        """
        keep = (
            "PATH",
            "SLURM_CONF",
            "SLURM_JOB_ID",
            "SLURM_JOB_NODELIST",
            "SLURM_NODELIST",
            "SLURM_SUBMIT_DIR",
        )
        env = {key: os.environ[key] for key in keep if os.environ.get(key)}
        env.update(
            {
                "PYTHONUNBUFFERED": "1",
                "PYTHONDONTWRITEBYTECODE": "1",
                "CUDA_VISIBLE_DEVICES": gpu_device or "",
                "NVIDIA_VISIBLE_DEVICES": gpu_device or "none",
                "MELLANOX_VISIBLE_DEVICES": "none",
                # The cluster default exposes the host /dev tree. Ask Enroot
                # to rebuild a minimal device namespace for every trajectory.
                "ENROOT_RESTRICT_DEV": "y",
                # Every container has a private writable /tmp mount.
                "HF_HOME": "/tmp/huggingface",
                "MPLCONFIGDIR": "/tmp/matplotlib",
                "NUMBA_CACHE_DIR": "/tmp/numba",
                "XDG_CACHE_HOME": "/tmp/cache",
                "OMNICODING_HARNESS_VENV": str(self.harness_venv),
            }
        )
        # The raw Tavily key never enters a model-visible container.  These
        # two values grant only access to the search-only local proxy.
        for key in (
            "OMNICODING_WEB_SEARCH_PROXY_URL",
            "OMNICODING_WEB_SEARCH_PROXY_TOKEN",
        ):
            if os.environ.get(key):
                env[key] = os.environ[key]
        return env
