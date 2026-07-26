#!/usr/bin/env bash
# Default Code-X-SFT-9B Kira GSPO training entry point.
#
# Topology:
#   actor:   8 H100, TP=8 / CP=1 / DP=1
#   rollout: 8 independent TP1 SGLang engines
#   batch:   2 RL prompts x 4 trajectories = 8 concurrent trajectories
#
# Qwen3.5 GatedDeltaNet does not support context parallel in the installed
# Megatron build. TP8 gives every long sequence the full 8-GPU model shard and
# avoids DP straggler imbalance between intact four-sample GSPO groups.
set -euo pipefail

repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"

export MODEL_PATH="${MODEL_PATH:-${repo_root}/models/Code-X-SFT-9B}"
export PROMPT_DATA="${PROMPT_DATA:-${repo_root}/data/omnicoding/processed/rl_smoke_small_media.parquet}"
export NUM_GPUS="${NUM_GPUS:-8}"
export ACTOR_TP="${ACTOR_TP:-8}"
export ACTOR_CP="${ACTOR_CP:-1}"
export LOG_PROBS_CHUNK_SIZE="${LOG_PROBS_CHUNK_SIZE:-4096}"
export ROLLOUT_GPUS_PER_ENGINE="${ROLLOUT_GPUS_PER_ENGINE:-1}"
export NUM_ROLLOUT="${NUM_ROLLOUT:-10}"
export ROLLOUT_BATCH_SIZE="${ROLLOUT_BATCH_SIZE:-2}"
export N_SAMPLES_PER_PROMPT="${N_SAMPLES_PER_PROMPT:-4}"
export KIRA_MAX_TURNS="${KIRA_MAX_TURNS:-50}"
export KIRA_MAX_TOKENS_PER_TURN="${KIRA_MAX_TOKENS_PER_TURN:-2048}"
export KIRA_MAX_TRAJECTORY_TOKENS="${KIRA_MAX_TRAJECTORY_TOKENS:-48000}"
export MAX_TOKENS_PER_GPU="${MAX_TOKENS_PER_GPU:-48000}"
export ROLLOUT_MAX_RESPONSE_LEN="${ROLLOUT_MAX_RESPONSE_LEN:-65536}"
export SGLANG_CONTEXT_LENGTH="${SGLANG_CONTEXT_LENGTH:-131072}"
export SGLANG_SERVER_CONCURRENCY="${SGLANG_SERVER_CONCURRENCY:-8}"
export SGLANG_MAX_RUNNING_REQUESTS="${SGLANG_MAX_RUNNING_REQUESTS:-8}"
export ROLLOUT_STEP_CONCURRENCY="${ROLLOUT_STEP_CONCURRENCY:-8}"
export ROLLOUTS_PER_SANDBOX_GPU="${ROLLOUTS_PER_SANDBOX_GPU:-8}"
export ROLLOUT_CPUS_PER_TASK="${ROLLOUT_CPUS_PER_TASK:-4}"
export ROLLOUT_MEMORY_PER_TASK="${ROLLOUT_MEMORY_PER_TASK:-40G}"
export SGLANG_MEM_FRACTION_STATIC="${SGLANG_MEM_FRACTION_STATIC:-0.52}"
export USE_ROLLOUT_LOGPROBS="${USE_ROLLOUT_LOGPROBS:-1}"
export DISABLE_JIT_FUSER="${DISABLE_JIT_FUSER:-1}"
export RELAX_PROFILE_TRAIN_STAGES="${RELAX_PROFILE_TRAIN_STAGES:-1}"
export OUTPUT_ROOT="${OUTPUT_ROOT:-${repo_root}/outputs/relax-code-x-sft-kira-gspo/${SLURM_JOB_ID:-manual}-8gpu}"
export STAGE_TO_LOCAL_NVME="${STAGE_TO_LOCAL_NVME:-1}"
export PRECONVERT_ACTOR_DCP="${PRECONVERT_ACTOR_DCP:-1}"
export ACTOR_DCP_CACHE="${ACTOR_DCP_CACHE:-${repo_root}/models/Code-X-SFT-9B-mcore-tp${ACTOR_TP}}"
export COLD_START_TIMING_FILE="${COLD_START_TIMING_FILE:-${OUTPUT_ROOT}/cold-start-stages.tsv}"

# shellcheck disable=SC1091
source "${repo_root}/recipes/lib/prepare_code_x_sft_9b_runtime.sh"
prepare_code_x_sft_9b_runtime "${repo_root}"

exec bash "${repo_root}/recipes/rl_qwen35_9b_4gpu_agent_rollout_smoke.sh"
