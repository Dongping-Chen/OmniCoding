from __future__ import annotations

import sys
from types import SimpleNamespace

import pytest

from omnicoding.rl import rollout
from omnicoding.rl.rollout import (
    _build_payload,
    _extract_rollout_log_probs,
    _failed_sample,
    _max_trajectory_tokens,
    _score_final_trajectory,
    _sglang_model_name,
)


def _sample():
    return SimpleNamespace(metadata={"task_id": "fixture:1"})


def _args():
    return SimpleNamespace(
        max_turns=30,
        sglang_router_ip="127.0.0.1",
        sglang_router_port=8000,
        hf_checkpoint="fixture-model",
    )


@pytest.fixture
def relax_types_stub(monkeypatch):
    sample_type = SimpleNamespace(
        Status=SimpleNamespace(
            COMPLETED="completed",
            FAILED="failed",
            TRUNCATED="truncated",
        )
    )
    monkeypatch.setitem(
        sys.modules,
        "relax.utils.types",
        SimpleNamespace(Sample=sample_type),
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


def test_failed_sample_marks_rollout_log_probs_ready(relax_types_stub) -> None:
    sample = SimpleNamespace(metadata={"task_id": "fixture:1"})

    result = _failed_sample(sample, "fixture failure")

    assert result.rollout_log_probs == []
    assert result.remove_sample is True
    assert result.metadata["rollout_router_error"] == "fixture failure"


@pytest.mark.parametrize("value", ["0", "-1", "not-an-int"])
def test_trajectory_limit_rejects_invalid_values(monkeypatch, value: str) -> None:
    monkeypatch.setenv("KIRA_MAX_TRAJECTORY_TOKENS", value)

    with pytest.raises(ValueError, match="positive integer"):
        _max_trajectory_tokens()


@pytest.mark.asyncio
async def test_overlong_kira_rollout_is_removed_before_logprob_scoring(
    monkeypatch,
    relax_types_stub,
) -> None:
    monkeypatch.setenv("KIRA_MAX_TRAJECTORY_TOKENS", "2")
    monkeypatch.setattr(
        rollout,
        "_build_payload",
        lambda sample, sampling_params, args: {"task_id": "fixture:1"},
    )

    async def fake_post_rollout(payload):
        return {
            "trajectories": [
                {
                    "messages": [],
                    "final_text": "done",
                    "reward": 1.0,
                    "exit_reason": "task_complete",
                    "removed": False,
                    "n_steps": 3,
                }
            ]
        }

    async def fail_if_scored(*args, **kwargs):
        pytest.fail("overlong trajectory should not be scored")

    monkeypatch.setattr(rollout, "_post_rollout", fake_post_rollout)
    monkeypatch.setattr(rollout, "_score_final_trajectory", fail_if_scored)
    monkeypatch.setattr(
        rollout,
        "tokenize_trajectory",
        lambda messages, tokenizer, **kwargs: ([1, 2, 3], [0, 1], 2),
    )
    monkeypatch.setitem(
        sys.modules,
        "relax.engine.rollout.sglang_rollout",
        SimpleNamespace(
            GenerateState=lambda args: SimpleNamespace(
                processor=None,
                tokenizer=object(),
            )
        ),
    )
    sample = SimpleNamespace(metadata={"task_id": "fixture:1"})
    args = SimpleNamespace(apply_chat_template_kwargs={}, use_rollout_logprobs=True)

    result = await rollout.generate(args, sample, {})

    assert result.tokens == []
    assert result.response_length == 0
    assert result.remove_sample is True
    assert result.metadata["rollout_router_error"] == (
        "trajectory token limit exceeded: 3 > 2"
    )


@pytest.mark.parametrize(
    ("use_rollout_logprobs", "removed"),
    [(False, False), (True, False), (True, True)],
)
@pytest.mark.asyncio
async def test_successful_kira_rollout_marks_log_probs_ready(
    monkeypatch,
    use_rollout_logprobs: bool,
    removed: bool,
) -> None:
    monkeypatch.setattr(
        rollout,
        "_build_payload",
        lambda sample, sampling_params, args: {"task_id": "fixture:1"},
    )

    async def fake_post_rollout(payload):
        assert payload["task_id"] == "fixture:1"
        return {
            "trajectories": [
                {
                    "messages": [],
                    "final_text": "done",
                    "reward": 1.0,
                    "exit_reason": "task_complete",
                    "removed": removed,
                }
            ]
        }

    monkeypatch.setattr(rollout, "_post_rollout", fake_post_rollout)
    score_calls = []

    async def fake_score(args, sample, *, rollout_tokens):
        score_calls.append((sample.response_length, rollout_tokens))
        return [-0.25] * sample.response_length

    monkeypatch.setattr(rollout, "_score_final_trajectory", fake_score)
    monkeypatch.setattr(
        rollout,
        "tokenize_trajectory",
        lambda messages, tokenizer, **kwargs: ([1, 2], [0, 1], 1),
    )
    monkeypatch.setattr(rollout, "_status_for", lambda exit_reason: "completed")

    monkeypatch.setitem(
        sys.modules,
        "relax.engine.rollout.sglang_rollout",
        SimpleNamespace(
            GenerateState=lambda args: SimpleNamespace(
                processor=None,
                tokenizer=object(),
            )
        ),
    )
    sample = SimpleNamespace(metadata={"task_id": "fixture:1"})
    args = SimpleNamespace(
        apply_chat_template_kwargs={},
        use_rollout_logprobs=use_rollout_logprobs,
    )

    result = await rollout.generate(args, sample, {})

    if use_rollout_logprobs and removed:
        expected = [0.0]
        expected_source = "removed_sample_zeros"
    elif use_rollout_logprobs:
        expected = [-0.25]
        expected_source = "sglang_final_trajectory"
    else:
        expected = []
        expected_source = "disabled"
    assert result.rollout_log_probs == expected
    assert bool(score_calls) is (use_rollout_logprobs and not removed)
    assert result.metadata["rollout_logprob_source"] == expected_source
    assert result.response == "done"
    assert result.status == "completed"


def test_extract_rollout_log_probs_drops_sentinel_and_checks_trainable_tokens() -> None:
    tokens = [10, 11, 20, 21, 22, 23]
    output = {
        "meta_info": {
            "prompt_tokens": 6,
            "input_token_logprobs": [
                [None, 11],
                [-0.1, 20],
                [-9.9, 999_999],  # masked multimodal observation
                [-0.3, 22],
                [-0.4, 23],
            ],
        }
    }

    result = _extract_rollout_log_probs(
        output,
        tokens=tokens,
        response_length=4,
        loss_mask=[1, 0, 1, 1],
        logprob_start_len=1,
    )

    assert result == pytest.approx([-0.1, 0.0, -0.3, -0.4])


def test_extract_rollout_log_probs_rejects_trainable_token_mismatch() -> None:
    output = {
        "meta_info": {
            "prompt_tokens": 3,
            "input_token_logprobs": [[None, 10], [-0.1, 999], [-0.2, 12]],
        }
    }

    with pytest.raises(ValueError, match="trainable response offset 0"):
        _extract_rollout_log_probs(
            output,
            tokens=[10, 11, 12],
            response_length=2,
            loss_mask=[1, 1],
            logprob_start_len=0,
        )


@pytest.mark.asyncio
async def test_score_final_trajectory_requests_response_start_minus_one(monkeypatch) -> None:
    seen = {}

    async def fake_post(args, payload):
        seen.update(payload)
        return {
            "meta_info": {
                "prompt_tokens": 5,
                "input_token_logprobs": [
                    [None, 2],
                    [-0.3, 3],
                    [-0.4, 4],
                    [-0.5, 5],
                ],
            }
        }

    monkeypatch.setattr(rollout, "_post_sglang_score", fake_post)
    sample = SimpleNamespace(
        tokens=[1, 2, 3, 4, 5],
        response_length=3,
        loss_mask=[1, 1, 1],
        multimodal_inputs=None,
    )

    result = await _score_final_trajectory(
        SimpleNamespace(sglang_router_ip="127.0.0.1", sglang_router_port=8000),
        sample,
        rollout_tokens=[1, 2, 3, 4, 5],
    )

    assert result == pytest.approx([-0.3, -0.4, -0.5])
    assert seen["logprob_start_len"] == 1
    assert seen["sampling_params"]["max_new_tokens"] == 0
    assert seen["return_logprob"] is True
