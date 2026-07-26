#!/usr/bin/env bash
# Code-X-SFT-9B + the matching Kira harness on the OmniCoding RL split.
#
# Default full-training topology:
#   actor:   4 H100, TP=4 / CP=1 / DP=1
#   rollout: 4 independent TP=1 SGLang engines
#   data:    2 RL prompts x 4 samples = 8 concurrent Kira trajectories
#
# TP=4 halves the local vocab and logprob cross-entropy workspace versus TP=2.
# CP must remain 1 because Qwen3.5 gated-delta-net layers do not support CP in
# the current Megatron implementation. DP=1 avoids long/short group imbalance.
set -euo pipefail

repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"

export MODEL_PATH="${MODEL_PATH:-${repo_root}/models/Code-X-SFT-9B}"
export PROMPT_DATA="${PROMPT_DATA:-${repo_root}/data/omnicoding/processed/rl_smoke_small_media.parquet}"
export NUM_GPUS="${NUM_GPUS:-4}"
export ACTOR_TP="${ACTOR_TP:-4}"
export ACTOR_CP="${ACTOR_CP:-1}"
export LOG_PROBS_CHUNK_SIZE="${LOG_PROBS_CHUNK_SIZE:-4096}"
export ROLLOUT_GPUS_PER_ENGINE="${ROLLOUT_GPUS_PER_ENGINE:-1}"
export NUM_ROLLOUT="${NUM_ROLLOUT:-1}"
export ROLLOUT_BATCH_SIZE="${ROLLOUT_BATCH_SIZE:-2}"
export N_SAMPLES_PER_PROMPT="${N_SAMPLES_PER_PROMPT:-4}"
export KIRA_MAX_TURNS="${KIRA_MAX_TURNS:-50}"
export KIRA_MAX_TOKENS_PER_TURN="${KIRA_MAX_TOKENS_PER_TURN:-2048}"
export ROLLOUT_MAX_RESPONSE_LEN="${ROLLOUT_MAX_RESPONSE_LEN:-65536}"
export SGLANG_CONTEXT_LENGTH="${SGLANG_CONTEXT_LENGTH:-131072}"
export SGLANG_SERVER_CONCURRENCY="${SGLANG_SERVER_CONCURRENCY:-8}"
export SGLANG_MAX_RUNNING_REQUESTS="${SGLANG_MAX_RUNNING_REQUESTS:-8}"
export ROLLOUT_STEP_CONCURRENCY="${ROLLOUT_STEP_CONCURRENCY:-8}"
export ROLLOUTS_PER_SANDBOX_GPU="${ROLLOUTS_PER_SANDBOX_GPU:-8}"
export ROLLOUT_CPUS_PER_TASK="${ROLLOUT_CPUS_PER_TASK:-4}"
export ROLLOUT_MEMORY_PER_TASK="${ROLLOUT_MEMORY_PER_TASK:-40G}"
export SGLANG_MEM_FRACTION_STATIC="${SGLANG_MEM_FRACTION_STATIC:-0.52}"
export OUTPUT_ROOT="${OUTPUT_ROOT:-${repo_root}/outputs/relax-code-x-sft-kira-gspo/${SLURM_JOB_ID:-manual}-smoke}"

exec bash "${repo_root}/recipes/rl_qwen35_9b_4gpu_agent_rollout_smoke.sh"
