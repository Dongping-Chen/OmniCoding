#!/usr/bin/env bash
#SBATCH --job-name=code-x-8gpu-actor-replay
#SBATCH --partition=h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=1T
#SBATCH --gres=gpu:h100:8
#SBATCH --time=04:00:00
#SBATCH --output=slurm-%x-%j.out

# Eight-GPU TP4 x PP2 actor-update stress test using a previously generated
# Kira rollout. Pure TP8 is invalid for Qwen3.5-9B because its four KV groups
# are not divisible by eight.
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
export ACTOR_TP=4
export ACTOR_PP=2
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
export STAGE_TO_LOCAL_NVME="${STAGE_TO_LOCAL_NVME:-1}"
export PRECONVERT_ACTOR_DCP="${PRECONVERT_ACTOR_DCP:-1}"
export ACTOR_DCP_CACHE="${ACTOR_DCP_CACHE:-${repo_root}/models/Code-X-SFT-9B-mcore-tp4-pp2}"
export RAY_NUM_CPUS="${RAY_NUM_CPUS:-${SLURM_CPUS_PER_TASK:-64}}"
export OUTPUT_ROOT="${OUTPUT_ROOT:-${repo_root}/outputs/profiles/${SLURM_JOB_ID}-actor-replay-8gpu-tp4-pp2-full}"
export COLD_START_TIMING_FILE="${COLD_START_TIMING_FILE:-${OUTPUT_ROOT}/cold-start-stages.tsv}"

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
if [[ "${STAGE_TO_LOCAL_NVME}" == 1 ]]; then
  local_replay_dir="${LOCAL_RUNTIME_ROOT:-/tmp/omnicoding-${SLURM_JOB_ID}}/replay"
  local_replay_data="${local_replay_dir}/$(basename -- "${replay_data}")"
  replay_stage_started_at="$(date +%s)"
  mkdir -p "${local_replay_dir}"
  rsync -a "${replay_data}" "${local_replay_data}"
  export LOAD_DEBUG_ROLLOUT_DATA="${local_replay_data}"
  if [[ ! -s "${COLD_START_TIMING_FILE}" ]]; then
    printf 'stage\telapsed_s\tdetail\n' >"${COLD_START_TIMING_FILE}"
  fi
  printf 'stage_replay_data\t%d\t%s\n' \
    "$(( $(date +%s) - replay_stage_started_at ))" \
    "${local_replay_data}" \
    >>"${COLD_START_TIMING_FILE}"
fi
set +e
bash recipes/rl_code_x_sft_9b_kira_gspo.sh
status=$?
set -e
finished_at="$(date +%s)"
cleanup_monitor
trap - EXIT INT TERM

"${repo_root}/.venv/bin/python" - \
  "${OUTPUT_ROOT}/gpu-memory.csv" \
  "${OUTPUT_ROOT}/gpu-peak-memory.csv" <<'PY'
import csv
import re
import sys
from collections import defaultdict
from pathlib import Path

source = Path(sys.argv[1])
destination = Path(sys.argv[2])
peaks = defaultdict(int)
with source.open(newline="") as handle:
    for row in csv.reader(handle):
        if len(row) < 3 or not row[1].strip().isdigit():
            continue
        match = re.search(r"\d+", row[2])
        if match:
            index = int(row[1].strip())
            peaks[index] = max(peaks[index], int(match.group()))

with destination.open("w", newline="") as handle:
    writer = csv.writer(handle)
    writer.writerow(("gpu_index", "peak_memory_mib"))
    writer.writerows(sorted(peaks.items()))

print(
    "gpu_peak_memory_mib="
    + ",".join(f"{index}:{memory}" for index, memory in sorted(peaks.items()))
)
PY

oom_detected=0
if [[ -f "${OUTPUT_ROOT}/kira-gspo.log" ]] && \
   grep -Eiq \
     "CUDA out of memory|torch\\.OutOfMemoryError|OutOfMemoryError|CUDA calloc.*bytes|Cuda failure 'out of memory'" \
     "${OUTPUT_ROOT}/kira-gspo.log"; then
  oom_detected=1
fi
printf 'exit_status=%d wall_time_s=%d cuda_oom_detected=%d\n' \
  "${status}" "$((finished_at - started_at))" "${oom_detected}" \
  | tee "${OUTPUT_ROOT}/actor-replay-summary.log"
exit "${status}"
