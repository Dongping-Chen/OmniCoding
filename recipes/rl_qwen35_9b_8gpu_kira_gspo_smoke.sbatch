#!/usr/bin/env bash
#SBATCH --job-name=omnicoding-kira-8gpu
#SBATCH --partition=h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=96
#SBATCH --mem=850G
#SBATCH --gres=gpu:h100:8
#SBATCH --time=1-00:00:00
#SBATCH --output=outputs/slurm/%x-%j.out

# Eight-GPU official-Qwen smoke only; W&B is intentionally disabled.  It first
# compares 8xTP1 with 1xTP8 on the same 16 Kira trajectories, then runs ten
# colocated GSPO optimizer steps with 8 trajectories per prompt.
set -euo pipefail

repo_root="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "${repo_root}"

base_output="${repo_root}/outputs/relax-qwen35-kira-gspo/${SLURM_JOB_ID}-8gpu"
common_env=(
  NUM_GPUS=8
  PROMPT_DATA="${repo_root}/data/omnicoding/processed/rl_smoke_small_media.parquet"
  KIRA_MAX_TURNS=50
  KIRA_MAX_TOKENS_PER_TURN=1024
  ROLLOUT_MAX_RESPONSE_LEN=16384
  SGLANG_CONTEXT_LENGTH=32768
  SGLANG_SERVER_CONCURRENCY=16
  ROLLOUT_STEP_CONCURRENCY=16
  ROLLOUT_CPUS_PER_TASK=4
  ROLLOUT_MEMORY_PER_TASK=40G
  ROLLOUTS_PER_SANDBOX_GPU=16
  SGLANG_MEM_FRACTION_STATIC=0.48
)

env "${common_env[@]}" \
  DEBUG_ROLLOUT_ONLY=1 \
  NUM_ROLLOUT=1 \
  ROLLOUT_BATCH_SIZE=1 \
  N_SAMPLES_PER_PROMPT=16 \
  ROLLOUT_GPUS_PER_ENGINE=1 \
  OUTPUT_ROOT="${base_output}/tp1-16" \
  bash recipes/rl_qwen35_9b_4gpu_agent_rollout_smoke.sh

env "${common_env[@]}" \
  DEBUG_ROLLOUT_ONLY=1 \
  NUM_ROLLOUT=1 \
  ROLLOUT_BATCH_SIZE=1 \
  N_SAMPLES_PER_PROMPT=16 \
  ROLLOUT_GPUS_PER_ENGINE=8 \
  OUTPUT_ROOT="${base_output}/tp8-16" \
  bash recipes/rl_qwen35_9b_4gpu_agent_rollout_smoke.sh

env "${common_env[@]}" \
  DEBUG_ROLLOUT_ONLY=0 \
  ACTOR_TP=4 \
  NUM_ROLLOUT=10 \
  ROLLOUT_BATCH_SIZE=2 \
  N_SAMPLES_PER_PROMPT=4 \
  ROLLOUT_STEP_CONCURRENCY=8 \
  ROLLOUTS_PER_SANDBOX_GPU=8 \
  ROLLOUT_GPUS_PER_ENGINE=1 \
  OUTPUT_ROOT="${base_output}/gspo-10step-tp1" \
  bash recipes/rl_qwen35_9b_4gpu_agent_rollout_smoke.sh
