from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path

import pytest

from omnicoding.rl.coordinator.slurm_step_dispatcher import SlurmStepDispatcher
from omnicoding.rl.schemas import RolloutRequest


@dataclass(frozen=True)
class _Record:
    id: str = "fixture:1"
    question: str = "Return answer."
    answer_type: str = "open"
    ground_truth: tuple[str, ...] = ("answer",)
    options: tuple[str, ...] | None = None
    media: dict | None = None
    source_dataset: str = "fixture"
    category: str = "fixture"

    def __post_init__(self) -> None:
        if self.media is None:
            object.__setattr__(
                self,
                "media",
                {"videos": [], "audios": [], "images": []},
            )


def _request(n_samples: int = 1) -> RolloutRequest:
    return RolloutRequest(
        task_id="fixture:1",
        n_samples=n_samples,
        sglang_base_url="http://127.0.0.1:30000/v1",
        sglang_model_name="openai/test-model",
        max_turns=2,
        request_timeout_s=30,
        block_timeout_s=30,
    )


def _harness_venv(tmp_path: Path) -> str:
    harness_venv = tmp_path / "venv-harness"
    (harness_venv / "bin").mkdir(parents=True)
    (harness_venv / "bin" / "python").touch()
    return str(harness_venv)


def _trajectory(sample_index: int) -> dict:
    return {
        "sample_index": sample_index,
        "messages": [],
        "final_text": "",
        "extracted_answer": None,
        "prediction_normalized": None,
        "reward": 0.0,
        "outcome_reward": 0.0,
        "raw_outcome_reward": 0.0,
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


def test_backend_requires_existing_allocation(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    dispatcher = SlurmStepDispatcher(
        scratch_root=tmp_path / "scratch",
        container_image="/images/worker.sqsh",
        harness_venv=_harness_venv(tmp_path),
    )
    with pytest.raises(RuntimeError, match="existing Slurm allocation"):
        dispatcher.start()


def test_srun_mounts_only_private_inputs(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("SLURM_JOB_ID", "123")
    dispatcher = SlurmStepDispatcher(
        scratch_root=tmp_path / "scratch",
        container_image="/images/worker.sqsh",
        harness_venv=_harness_venv(tmp_path),
        cpus_per_rollout=3,
        memory_per_rollout="12G",
    )
    workspace = tmp_path / "private" / "workspace"
    tmp_dir = tmp_path / "private" / "tmp"
    req_path = tmp_path / "private" / "request.json"
    res_path = tmp_path / "private" / "result.json"
    workspace.mkdir(parents=True)
    tmp_dir.mkdir()
    req_path.write_text("{}")
    res_path.touch()

    command = dispatcher._srun_command(
        workspace=workspace,
        tmp_dir=tmp_dir,
        req_path=req_path,
        res_path=res_path,
        gpu_device=None,
    )
    joined = " ".join(command)

    assert "--jobid 123" in joined
    assert "--cpus-per-task=3" in command
    assert "--mem=12G" in command
    assert "--no-container-mount-home" in command
    assert "--container-readonly" in command
    assert f"{workspace.resolve()}:/workspace:rw+rprivate" in joined
    assert f"{req_path.resolve()}:/run/request.json:ro+rprivate" in joined
    assert f"{dispatcher.harness_venv}:{dispatcher.harness_venv}:ro+rprivate" in joined
    assert "/dataset" not in joined
    assert str(Path.home()) not in joined


def test_gpu_container_maps_one_physical_device_to_logical_zero(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setenv("SLURM_JOB_ID", "123")
    dispatcher = SlurmStepDispatcher(
        scratch_root=tmp_path / "scratch",
        container_image="/images/worker.sqsh",
        harness_venv=_harness_venv(tmp_path),
        gpu_devices=("7",),
        rollouts_per_gpu=8,
        max_concurrency=16,
    )
    workspace = tmp_path / "private" / "workspace"
    tmp_dir = tmp_path / "private" / "tmp"
    req_path = tmp_path / "private" / "request.json"
    res_path = tmp_path / "private" / "result.json"
    workspace.mkdir(parents=True)
    tmp_dir.mkdir()
    req_path.write_text("{}")
    res_path.touch()

    command = dispatcher._srun_command(
        workspace=workspace,
        tmp_dir=tmp_dir,
        req_path=req_path,
        res_path=res_path,
        gpu_device="7",
    )

    assert dispatcher.max_concurrency == 8
    env_index = command.index("env")
    assert command[env_index : env_index + 3] == [
        "env",
        "CUDA_VISIBLE_DEVICES=0",
        "python3",
    ]


def test_worker_control_env_drops_secrets(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("SLURM_JOB_ID", "123")
    monkeypatch.setenv("ROLLOUT_COORDINATOR_TOKEN", "secret")
    monkeypatch.setenv("HF_TOKEN", "secret")
    monkeypatch.setenv("TAVILY_API_KEY", "secret")

    dispatcher = SlurmStepDispatcher(
        scratch_root=tmp_path / "scratch",
        container_image="/images/worker.sqsh",
        harness_venv=_harness_venv(tmp_path),
    )
    env = dispatcher._worker_control_env()

    assert env["SLURM_JOB_ID"] == "123"
    assert env["CUDA_VISIBLE_DEVICES"] == ""
    assert env["NVIDIA_VISIBLE_DEVICES"] == "none"
    assert env["MELLANOX_VISIBLE_DEVICES"] == "none"
    assert env["ENROOT_RESTRICT_DEV"] == "y"
    assert env["MPLCONFIGDIR"] == "/tmp/matplotlib"
    assert env["NUMBA_CACHE_DIR"] == "/tmp/numba"
    assert env["OMNICODING_HARNESS_VENV"] == str(dispatcher.harness_venv)
    assert "ROLLOUT_COORDINATOR_TOKEN" not in env
    assert "HF_TOKEN" not in env
    assert "TAVILY_API_KEY" not in env


def test_worker_control_env_exposes_only_scoped_search_and_shared_gpu(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setenv("SLURM_JOB_ID", "123")
    monkeypatch.setenv(
        "OMNICODING_WEB_SEARCH_PROXY_URL", "http://10.0.0.1:19090"
    )
    monkeypatch.setenv("OMNICODING_WEB_SEARCH_PROXY_TOKEN", "scoped-token")
    monkeypatch.setenv("TAVILY_API_KEY", "upstream-secret")

    dispatcher = SlurmStepDispatcher(
        scratch_root=tmp_path / "scratch",
        container_image="/images/worker.sqsh",
        harness_venv=_harness_venv(tmp_path),
    )
    env = dispatcher._worker_control_env("7")

    assert env["CUDA_VISIBLE_DEVICES"] == "7"
    assert env["NVIDIA_VISIBLE_DEVICES"] == "7"
    assert env["OMNICODING_WEB_SEARCH_PROXY_URL"] == "http://10.0.0.1:19090"
    assert env["OMNICODING_WEB_SEARCH_PROXY_TOKEN"] == "scoped-token"
    assert "TAVILY_API_KEY" not in env


@pytest.mark.asyncio
async def test_in_allocation_backend_runs_samples_without_sbatch(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setenv("SLURM_JOB_ID", "123")
    dataset_root = tmp_path / "dataset"
    dataset_root.mkdir()
    dispatcher = SlurmStepDispatcher(
        scratch_root=tmp_path / "scratch",
        container_image="/images/worker.sqsh",
        harness_venv=_harness_venv(tmp_path),
        max_concurrency=2,
    )

    def fake_command(*, workspace, tmp_dir, req_path, res_path, gpu_device):
        del workspace, tmp_dir
        assert gpu_device is None
        request = json.loads(req_path.read_text())
        assert request["workspace"] == "/workspace"
        sample_index = request["sample_index"]
        payload = json.dumps(_trajectory(sample_index))
        return [
            sys.executable,
            "-c",
            "from pathlib import Path; "
            f"Path({str(res_path)!r}).write_text({payload!r})",
        ]

    monkeypatch.setattr(dispatcher, "_srun_command", fake_command)
    dispatcher.start()
    try:
        results = await dispatcher.submit_and_collect(
            record=_Record(),
            req=_request(n_samples=2),
            dataset_root=dataset_root,
            deadline_s=10,
        )
    finally:
        await dispatcher.stop()

    assert len(results) == 2
    assert [result.sample_index for result in results] == [0, 1]
    assert all(result.exit_reason == "task_complete" for result in results)
    completed = list((tmp_path / "scratch").glob("*/.completed"))
    assert len(completed) == 2
