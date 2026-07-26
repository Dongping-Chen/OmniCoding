#!/usr/bin/env bash
# Use one 8-H100 allocation to benchmark all 4/8-GPU serving topologies and
# actor replay micro-batches. No Kira harness or sandbox trajectory is run.
set -uo pipefail

repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
job_id="${SLURM_JOB_ID:-manual}"
model_path="${MODEL_PATH:-${repo_root}/models/Code-X-SFT-9B}"
replay_path="${REPLAY_PATH:-${repo_root}/outputs/training-capacity/rollout-real16-balanced.pt}"
matrix_root="${OUTPUT_ROOT:-${repo_root}/outputs/code-x-capacity-matrix/${job_id}}"
status_file="${matrix_root}/status.tsv"

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
  echo "run this script inside an 8-GPU Slurm allocation" >&2
  exit 2
fi
for path in "${model_path}/config.json" "${replay_path}"; do
  if [[ ! -e "${path}" ]]; then
    echo "missing required path: ${path}" >&2
    exit 2
  fi
done

mkdir -p "${matrix_root}"
printf 'kind\tlabel\treturncode\tstarted_utc\tended_utc\n' >"${status_file}"
original_gpus="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
IFS=',' read -r -a gpu_ids <<<"${original_gpus}"
if (( ${#gpu_ids[@]} < 8 )); then
  echo "the matrix requires 8 Slurm-visible GPUs; got ${original_gpus}" >&2
  exit 2
fi
gpus4="$(IFS=,; echo "${gpu_ids[*]:0:4}")"
gpus8="$(IFS=,; echo "${gpu_ids[*]:0:8}")"

record_run() {
  local kind="$1"
  local label="$2"
  local started="$3"
  local rc="$4"
  printf '%s\t%s\t%d\t%s\t%s\n' \
    "${kind}" "${label}" "${rc}" "${started}" "$(date -u +%FT%TZ)" \
    >>"${status_file}"
}

run_serving() {
  local num_gpus="$1"
  local tp_size="$2"
  local visible_gpus="$3"
  local label="g${num_gpus}-tp${tp_size}"
  local started
  started="$(date -u +%FT%TZ)"
  echo "=== serving ${label} ==="
  CUDA_VISIBLE_DEVICES="${visible_gpus}" \
  NUM_GPUS="${num_gpus}" \
  TP_SIZE="${tp_size}" \
  MODEL_PATH="${model_path}" \
  OUTPUT_ROOT="${matrix_root}/serving-${label}" \
    bash "${repo_root}/recipes/sglang_code_x_capacity.sh"
  local rc=$?
  record_run serving "${label}" "${started}" "${rc}"
  return 0
}

run_training() {
  local num_gpus="$1"
  local actor_tp="$2"
  local micro_batch="$3"
  local visible_gpus="$4"
  local label="g${num_gpus}-tp${actor_tp}-mbs${micro_batch}"
  local started
  started="$(date -u +%FT%TZ)"
  echo "=== actor replay ${label} ==="
  timeout --signal=TERM --kill-after=2m 2h \
    env \
      CUDA_VISIBLE_DEVICES="${visible_gpus}" \
      NUM_GPUS="${num_gpus}" \
      ACTOR_TENSOR_PARALLEL_SIZE="${actor_tp}" \
      ACTOR_CONTEXT_PARALLEL_SIZE=1 \
      MICRO_BATCH_SIZE="${micro_batch}" \
      MODEL_PATH="${model_path}" \
      LOAD_DEBUG_ROLLOUT_DATA="${replay_path}" \
      ROLLOUT_BATCH_SIZE=4 \
      N_SAMPLES_PER_PROMPT=4 \
      NUM_ROLLOUT=1 \
      LOG_PROBS_CHUNK_SIZE=4096 \
      OUTPUT_ROOT="${matrix_root}/training-${label}" \
      WANDB_MODE=disabled \
      RAY_TMPDIR="/tmp/cx${job_id}-${label}" \
      bash "${repo_root}/recipes/rl_qwen35_9b_4gpu_agent_rollout_smoke.sh"
  local rc=$?
  record_run training "${label}" "${started}" "${rc}"
  return 0
}

# Long-context serving is the priority. Each nested sweep starts at 100k.
run_serving 8 1 "${gpus8}"
run_serving 8 4 "${gpus8}"
run_serving 8 8 "${gpus8}"
run_serving 4 1 "${gpus4}"
run_serving 4 4 "${gpus4}"

# MBS=1 establishes the baseline. Higher values measure actual samples resident
# per forward/backward pass.
run_training 8 4 1 "${gpus8}"
run_training 8 4 2 "${gpus8}"
run_training 8 4 4 "${gpus8}"
run_training 8 8 1 "${gpus8}"
run_training 8 8 2 "${gpus8}"
run_training 4 4 1 "${gpus4}"
run_training 4 4 2 "${gpus4}"

# Data-parallel actor variants use the same four intact GSPO groups and expose
# four samples per micro-step at MBS=1 (eight at MBS=2).
run_training 8 2 1 "${gpus8}"
run_training 8 2 2 "${gpus8}"
run_training 4 1 1 "${gpus4}"
run_training 4 1 2 "${gpus4}"

run_training_dp8() {
  local micro_batch="$1"
  local label="g8-tp1-mbs${micro_batch}"
  local started
  started="$(date -u +%FT%TZ)"
  timeout --signal=TERM --kill-after=2m 2h \
    env \
      CUDA_VISIBLE_DEVICES="${gpus8}" \
      NUM_GPUS=8 \
      ACTOR_TENSOR_PARALLEL_SIZE=1 \
      ACTOR_CONTEXT_PARALLEL_SIZE=1 \
      MICRO_BATCH_SIZE="${micro_batch}" \
      MODEL_PATH="${model_path}" \
      LOAD_DEBUG_ROLLOUT_DATA="${repo_root}/outputs/training-capacity/rollout-real32-balanced.pt" \
      ROLLOUT_BATCH_SIZE=8 \
      N_SAMPLES_PER_PROMPT=4 \
      NUM_ROLLOUT=1 \
      LOG_PROBS_CHUNK_SIZE=4096 \
      OUTPUT_ROOT="${matrix_root}/training-${label}" \
      WANDB_MODE=disabled \
      RAY_TMPDIR="/tmp/cx${job_id}-${label}" \
      bash "${repo_root}/recipes/rl_qwen35_9b_4gpu_agent_rollout_smoke.sh"
  local rc=$?
  record_run training "${label}" "${started}" "${rc}"
  return 0
}
run_training_dp8 1
run_training_dp8 2

echo "Capacity matrix completed: ${status_file}"
