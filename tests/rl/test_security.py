from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi import HTTPException
from fastapi.security import HTTPAuthorizationCredentials
from pydantic import ValidationError

from omnicoding.rl.coordinator.dataset import Record
from omnicoding.rl.coordinator.media import stage_media
from omnicoding.rl.coordinator.dispatcher import SlurmDispatcher
from omnicoding.rl.coordinator.worker import _build_agent
from omnicoding.rl.schemas import RolloutRequest
from omnicoding.rl.security import enforce_request_policy, require_coordinator_token
from omnicoding.rl.secrets import load_coordinator_token


def _request(**overrides) -> RolloutRequest:
    values = {
        "task_id": "task:1",
        "n_samples": 1,
        "sglang_base_url": "https://inference.example/v1",
        "sglang_model_name": "openai/test-model",
    }
    values.update(overrides)
    return RolloutRequest(**values)


def _record(media_path: str) -> Record:
    return Record(
        id="task:1",
        question="question",
        answer_type="open",
        ground_truth=["answer"],
        options=None,
        media={"videos": [media_path], "audios": [], "images": []},
        source_dataset="fixture",
        category="fixture",
    )


def test_rollout_request_has_resource_bounds() -> None:
    with pytest.raises(ValidationError):
        _request(n_samples=17)
    with pytest.raises(ValidationError):
        _request(max_turns=129)
    with pytest.raises(ValidationError):
        _request(task_id="../../escape")
    with pytest.raises(ValidationError):
        _request(sampling_params={"max_tokens": 32769})


def test_rollout_sampling_limit_reaches_kira_agent(tmp_path: Path) -> None:
    request = _request(sampling_params={"max_tokens": 4096})
    agent = _build_agent(tmp_path / "workspace", request)

    assert agent.max_tokens == 4096


def test_request_policy_requires_exact_origin_and_model(monkeypatch) -> None:
    monkeypatch.setenv("ROLLOUT_ALLOWED_SGLANG_ORIGINS", "https://inference.example")
    monkeypatch.setenv("ROLLOUT_ALLOWED_MODELS", "openai/test-model")
    enforce_request_policy(_request())

    with pytest.raises(HTTPException) as url_error:
        enforce_request_policy(_request(sglang_base_url="https://attacker.example/v1"))
    assert url_error.value.status_code == 403

    with pytest.raises(HTTPException) as model_error:
        enforce_request_policy(_request(sglang_model_name="openai/other"))
    assert model_error.value.status_code == 403


def test_coordinator_uses_bearer_token(monkeypatch) -> None:
    monkeypatch.delenv("ROLLOUT_COORDINATOR_TOKEN_FILE", raising=False)
    monkeypatch.setenv("ROLLOUT_COORDINATOR_TOKEN", "expected-token")
    require_coordinator_token(HTTPAuthorizationCredentials(scheme="Bearer", credentials="expected-token"))

    with pytest.raises(HTTPException) as error:
        require_coordinator_token(HTTPAuthorizationCredentials(scheme="Bearer", credentials="wrong"))
    assert error.value.status_code == 401


def test_coordinator_reads_restricted_token_file(tmp_path: Path, monkeypatch) -> None:
    token_file = tmp_path / "coordinator-token"
    token_file.write_text("file-token\n", encoding="utf-8")
    token_file.chmod(0o600)
    monkeypatch.setenv("ROLLOUT_COORDINATOR_TOKEN_FILE", str(token_file))
    monkeypatch.delenv("ROLLOUT_COORDINATOR_TOKEN", raising=False)

    assert load_coordinator_token() == "file-token"
    require_coordinator_token(
        HTTPAuthorizationCredentials(scheme="Bearer", credentials="file-token")
    )


def test_coordinator_rejects_exposed_or_symlinked_token_file(
    tmp_path: Path, monkeypatch
) -> None:
    token_file = tmp_path / "coordinator-token"
    token_file.write_text("file-token\n", encoding="utf-8")
    token_file.chmod(0o644)
    monkeypatch.setenv("ROLLOUT_COORDINATOR_TOKEN_FILE", str(token_file))

    with pytest.raises(RuntimeError, match="group or other"):
        load_coordinator_token()

    token_file.chmod(0o600)
    token_link = tmp_path / "token-link"
    token_link.symlink_to(token_file)
    monkeypatch.setenv("ROLLOUT_COORDINATOR_TOKEN_FILE", str(token_link))
    with pytest.raises(RuntimeError, match="cannot open"):
        load_coordinator_token()


def test_media_path_cannot_escape_dataset_root(tmp_path: Path) -> None:
    dataset_root = tmp_path / "dataset"
    workspace = tmp_path / "workspace"
    dataset_root.mkdir()
    workspace.mkdir()
    outside = tmp_path / "outside.mp4"
    outside.write_bytes(b"fixture")

    with pytest.raises(ValueError, match="unsafe media path"):
        stage_media(_record("../outside.mp4"), workspace, dataset_root)

    (dataset_root / "link.mp4").symlink_to(outside)
    with pytest.raises(ValueError, match="escapes dataset root"):
        stage_media(_record("link.mp4"), workspace, dataset_root)


def test_media_is_copied_without_exposing_dataset_root(tmp_path: Path) -> None:
    dataset_root = tmp_path / "dataset"
    workspace = tmp_path / "workspace"
    source = dataset_root / "media" / "videos" / "clip.mp4"
    source.parent.mkdir(parents=True)
    source.write_bytes(b"fixture")
    workspace.mkdir()

    staged = stage_media(_record("media/videos/clip.mp4"), workspace, dataset_root)
    destination = workspace / "media" / "videos" / "clip.mp4"

    assert staged == ["media/videos/clip.mp4"]
    assert destination.read_bytes() == b"fixture"
    assert not destination.is_symlink()


def test_slurm_export_does_not_inherit_coordinator_secrets(
    tmp_path: Path, monkeypatch
) -> None:
    script = tmp_path / "worker.sbatch"
    script.write_text("#!/usr/bin/env bash\n")
    captured: dict = {}

    def fake_run(command, **kwargs):
        captured["command"] = command
        captured["env"] = kwargs["env"]
        return SimpleNamespace(returncode=0, stdout="Submitted batch job 123\n", stderr="")

    monkeypatch.setenv("ROLLOUT_COORDINATOR_TOKEN", "must-not-reach-worker")
    monkeypatch.setenv("MODAL_TOKEN_SECRET", "must-not-reach-worker")
    monkeypatch.setattr("omnicoding.rl.coordinator.dispatcher.subprocess.run", fake_run)

    dispatcher = SlurmDispatcher(
        scratch_root=tmp_path / "scratch",
        sbatch_script=script,
    )
    request = tmp_path / "request.json"
    result = tmp_path / "result.json"
    assert dispatcher._sbatch([request], [result], tmp_path, 0) == 123
    assert "ROLLOUT_COORDINATOR_TOKEN" not in captured["env"]
    assert "MODAL_TOKEN_SECRET" not in captured["env"]
    assert "ALL" not in next(arg for arg in captured["command"] if arg.startswith("--export="))
