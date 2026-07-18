"""Wire types shared by the Modal-side custom rollout fn and the local
coordinator.

Kept dependency-light (pydantic only) so this module can be imported on either
side without dragging FastAPI / kira / Relax in.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


# ─── inbound: what Modal sends to coordinator ────────────────────────────────


class SamplingParams(BaseModel):
    """Subset of OpenAI / SGLang sampling knobs we propagate to kira's LLM
    client. Anything kira doesn't understand is ignored upstream."""

    temperature: float | None = Field(default=None, ge=0.0, le=2.0)
    top_p: float | None = Field(default=None, gt=0.0, le=1.0)
    max_tokens: int | None = Field(default=None, ge=1, le=32768)
    seed: int | None = None
    enable_thinking: bool | None = True  # Qwen-only; kira reads via extra_body


class RolloutRequest(BaseModel):
    """One coordinator job = N parallel trajectories for ONE prompt."""

    task_id: str = Field(min_length=1, max_length=256, pattern=r"^[A-Za-z0-9_:-]+$")
    n_samples: int = Field(ge=1, le=16)
    sglang_base_url: str = Field(min_length=8, max_length=2048)
    sglang_model_name: str = Field(min_length=1, max_length=256)
    sampling_params: SamplingParams = Field(default_factory=SamplingParams)
    max_turns: int = Field(default=30, ge=1, le=128)
    request_timeout_s: int = Field(default=900, ge=5, le=1800)
    # MEMORY: 600s default fires on QUEUED (not stuck) calls when sglang MRR ~16
    # so we run kira sharded — bump to 1200 to absorb queue tail.
    block_timeout_s: int = Field(default=1200, ge=5, le=1800)


# ─── outbound: what coordinator returns ──────────────────────────────────────

ExitReason = Literal[
    "task_complete",
    "step_limit",
    "no_tool_calls",
    "context_overflow",
    "error",
    "timeout",
]


class Trajectory(BaseModel):
    """One completed kira run."""

    sample_index: int  # 0..n_samples-1
    messages: list[dict[str, Any]]  # full OpenAI-shape chat history
    final_text: str  # kira's _walk_final_text concat of assistant + non-checklist tool replies
    extracted_answer: str | None  # what we pulled out of <answer>...</answer>, may be None
    prediction_normalized: str | None  # what we matched against ground_truth
    reward: float  # final per-trajectory score before group length shaping
    outcome_reward: float = 0.0  # gated correctness used by training
    raw_outcome_reward: float = 0.0  # raw exact-match result before tool/modality gating
    format_reward: float = 0.0   # 0.0 if format is clean, -0.2 otherwise
    modality_reward: float = 0.0
    bad_tool_reward: float = 0.0
    modality_match: float = 1.0
    p_bad_tool: float = 0.0
    n_unparseable: int = 0
    n_disallowed: int = 0
    n_escape: int = 0
    n_syntax_fail: int = 0
    removed: bool = False
    reward_details: dict[str, Any] = Field(default_factory=dict)
    exit_reason: ExitReason
    n_steps: int
    n_tool_calls: int
    cumulative_prompt_tokens: int
    cumulative_completion_tokens: int
    cumulative_reasoning_tokens: int
    error: str | None = None  # populated when exit_reason == "error" or "timeout"


class RolloutResponse(BaseModel):
    task_id: str
    n_samples: int
    trajectories: list[Trajectory]
    elapsed_s: float


# ─── async submit / poll (bypasses Cloudflare quick-tunnel 100s timeout) ──────


class RolloutSubmitResponse(BaseModel):
    """Returned from ``POST /rollout/submit``. Caller polls
    ``GET /rollout/result/{job_id}`` until the trajectories complete."""

    job_id: str
    task_id: str
    n_samples: int
    status: Literal["pending"] = "pending"


JobStatus = Literal["pending", "completed", "error"]


class RolloutResultResponse(BaseModel):
    """Returned from ``GET /rollout/result/{job_id}``.

    - ``status="pending"``: still running; ``response`` and ``error`` are None.
    - ``status="completed"``: ``response`` is the full RolloutResponse.
    - ``status="error"``: ``error`` carries the exception message.
    """

    job_id: str
    status: JobStatus
    elapsed_s: float
    response: RolloutResponse | None = None
    error: str | None = None
