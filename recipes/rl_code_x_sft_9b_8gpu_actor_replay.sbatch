#!/usr/bin/env bash
#SBATCH --job-name=code-x-tp8-actor-replay
#SBATCH --partition=h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=1T
#SBATCH --gres=gpu:h100:8
#SBATCH --time=02:00:00
#SBATCH --output=slurm-%x-%j.out

# TP8 actor-update stress test using a previously generated Kira rollout.
#
# The default replay contains four trajectories with 79,786, 15,797, 22,049,
# and 31,608 total tokens. MAX_TOKENS_PER_GPU=100000 deliberately admits the
# 79.8k singleton that OOMed with TP4. Production training remains capped at
# 48k so a pathological trajectory cannot dominate an optimizer step.
set -euo pipefail

repo_root="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "${repo_root}"

replay_data="${REPLAY_DATA:-${repo_root}/outputs/profiles/job1006-9b/live-kira-gspo-dynamic-b4/rollout_0.pt}"
if [[ ! -f "${replay_data}" ]]; then
  echo "missing replay data: ${replay_data}" >&2
  exit 2
fi

export NUM_GPUS=8
export ACTOR_TP=8
export ACTOR_CP=1
export MODEL_PATH="${MODEL_PATH:-${repo_root}/models/Code-X-SFT-9B}"
export LOAD_DEBUG_ROLLOUT_DATA="${replay_data}"
export N_SAMPLES_PER_PROMPT=4
export GLOBAL_BATCH_SIZE=4
export NUM_ROLLOUT=1
export GRPO_ITERATIONS=1
export MAX_TOKENS_PER_GPU="${MAX_TOKENS_PER_GPU:-100000}"
export KIRA_MAX_TRAJECTORY_TOKENS="${KIRA_MAX_TRAJECTORY_TOKENS:-100000}"
export USE_ROLLOUT_LOGPROBS=1
export DISABLE_JIT_FUSER=1
export RELAX_PROFILE_TRAIN_STAGES=1
export RAY_NUM_CPUS="${RAY_NUM_CPUS:-${SLURM_CPUS_PER_TASK:-64}}"
export OUTPUT_ROOT="${OUTPUT_ROOT:-${repo_root}/outputs/profiles/${SLURM_JOB_ID}-actor-replay-tp8-full}"

mkdir -p "${OUTPUT_ROOT}"
nvidia-smi \
  --query-gpu=timestamp,index,memory.used,memory.free,utilization.gpu,power.draw \
  --format=csv \
  --loop=5 \
  >"${OUTPUT_ROOT}/gpu-memory.csv" &
monitor_pid=$!
cleanup_monitor() {
  kill "${monitor_pid}" >/dev/null 2>&1 || true
  wait "${monitor_pid}" >/dev/null 2>&1 || true
}
trap cleanup_monitor EXIT INT TERM

started_at="$(date +%s)"
set +e
bash recipes/rl_qwen35_9b_4gpu_agent_rollout_smoke.sh
status=$?
set -e
finished_at="$(date +%s)"
printf 'exit_status=%d wall_time_s=%d\n' \
  "${status}" "$((finished_at - started_at))" \
  | tee "${OUTPUT_ROOT}/tp8-replay-summary.log"
exit "${status}"
