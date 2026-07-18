from __future__ import annotations

from types import SimpleNamespace

import pytest

from omnicoding.rl.rollout import _build_payload, _sglang_model_name


def _sample():
    return SimpleNamespace(metadata={"task_id": "fixture:1"})


def _args():
    return SimpleNamespace(
        max_turns=30,
        sglang_router_ip="127.0.0.1",
        sglang_router_port=8000,
        hf_checkpoint="fixture-model",
    )


def test_payload_uses_bounded_default_per_turn_output(monkeypatch) -> None:
    monkeypatch.delenv("KIRA_MAX_TOKENS_PER_TURN", raising=False)

    payload = _build_payload(_sample(), {"max_new_tokens": 200_000}, _args())

    assert payload["sampling_params"]["max_tokens"] == 8192


@pytest.mark.parametrize("value", ["0", "32769"])
def test_payload_rejects_invalid_per_turn_output_limit(monkeypatch, value: str) -> None:
    monkeypatch.setenv("KIRA_MAX_TOKENS_PER_TURN", value)

    with pytest.raises(ValueError, match="between 1 and 32768"):
        _build_payload(_sample(), {}, _args())


def test_sglang_model_adds_litellm_prefix_to_hugging_face_id(monkeypatch) -> None:
    monkeypatch.setenv("ROLLOUT_SGLANG_MODEL", "shuaishuaicdp/Code-X-SFT-27B")

    assert _sglang_model_name(_args()) == (
        "openai/shuaishuaicdp/Code-X-SFT-27B"
    )


def test_sglang_model_keeps_existing_litellm_prefix(monkeypatch) -> None:
    monkeypatch.setenv(
        "ROLLOUT_SGLANG_MODEL",
        "openai/shuaishuaicdp/Code-X-SFT-27B",
    )

    assert _sglang_model_name(_args()) == (
        "openai/shuaishuaicdp/Code-X-SFT-27B"
    )
