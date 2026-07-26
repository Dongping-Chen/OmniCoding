#!/usr/bin/env bash
# Run inside a single-node Slurm allocation with four H100 GPUs.
# This isolates stock Relax + SGLang batching from the coding-agent rollout.
set -euo pipefail

repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
venv_root="${VENV_ROOT:-${repo_root}/.venv}"
relax_root="${RELAX_ROOT:-${venv_root}/src/Relax}"
megatron_root="${MEGATRON_ROOT:-${venv_root}/src/Megatron-LM}"
model_path="${MODEL_PATH:-${repo_root}/models/Qwen3.5-9B}"
prompt_data="${PROMPT_DATA:-${repo_root}/data/dapo-math-17k/dapo-math-17k.jsonl}"
output_root="${OUTPUT_ROOT:-${repo_root}/outputs/relax-qwen35-rollout-smoke/${SLURM_JOB_ID:-manual}}"

for path in \
  "${venv_root}/bin/python" \
  "${relax_root}/scripts/models/qwen35-9B.sh" \
  "${model_path}/config.json" \
  "${prompt_data}"; do
  if [[ ! -e "${path}" ]]; then
    echo "missing required path: ${path}" >&2
    exit 2
  fi
done

num_gpus="${NUM_GPUS:-4}"
gpus_per_engine="${ROLLOUT_GPUS_PER_ENGINE:-1}"
rollout_batch_size="${ROLLOUT_BATCH_SIZE:-4}"
n_samples_per_prompt="${N_SAMPLES_PER_PROMPT:-2}"
num_rollout="${NUM_ROLLOUT:-2}"
response_len="${ROLLOUT_MAX_RESPONSE_LEN:-256}"
context_len="${SGLANG_CONTEXT_LENGTH:-4096}"
server_concurrency="${SGLANG_SERVER_CONCURRENCY:-32}"
max_running_requests="${SGLANG_MAX_RUNNING_REQUESTS:-${server_concurrency}}"
disable_radix_cache="${SGLANG_DISABLE_RADIX_CACHE:-0}"
ray_num_cpus="${RAY_NUM_CPUS:-${SLURM_CPUS_PER_TASK:-$(nproc)}}"

if (( num_gpus % gpus_per_engine != 0 )); then
  echo "NUM_GPUS must be divisible by ROLLOUT_GPUS_PER_ENGINE" >&2
  exit 2
fi

mkdir -p "${output_root}"
export PYTHONUNBUFFERED=1
export CUDA_DEVICE_MAX_CONNECTIONS=1
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-12.9}"
export FLASHINFER_DISABLE_VERSION_CHECK=1
export TOKENIZERS_PARALLELISM=true
# Source installs repeatedly import large GPU stacks from shared storage.
# Avoid killing an otherwise healthy Serve proxy during that cold start.
export RAY_SERVE_PROXY_HEALTH_CHECK_TIMEOUT_S="${RAY_SERVE_PROXY_HEALTH_CHECK_TIMEOUT_S:-120}"
export RELAX="${relax_root}"
export MEGATRON="${megatron_root}"
export PYTHONPATH="${relax_root}:${megatron_root}:${repo_root}/src:${PYTHONPATH:-}"

if [[ -n "${SLURM_JOB_ID:-}" ]]; then
  port_offset=$((SLURM_JOB_ID % 10000))
else
  port_offset=$$
fi
ray_port="${RAY_PORT:-$((20000 + port_offset % 10000))}"
dashboard_port="${RAY_DASHBOARD_PORT:-$((30000 + port_offset % 10000))}"
detected_node_ip="$(ip route get 1.1.1.1 2>/dev/null | awk '{
  for (i = 1; i <= NF; i++) {
    if ($i == "src") {
      print $(i + 1)
      exit
    }
  }
}')"
detected_node_ip="${detected_node_ip:-$(hostname -I | awk '{print $1}')}"
node_ip="${MASTER_ADDR:-${detected_node_ip}}"
# Ray dashboard subprocesses use AF_UNIX sockets (107-byte path limit).
# Keep this deliberately short; email-style usernames plus Ray's long session
# suffix otherwise make every dashboard module fail before job submission.
ray_tmp="${RAY_TMPDIR:-/tmp/r${SLURM_JOB_ID:-$$}ro}"
network_interface="${RELAX_NETWORK_INTERFACE:-$(ip route show default | awk 'NR == 1 {print $5}')}"
network_interface="${network_interface:-lo}"
export MASTER_ADDR="${node_ip}"
export GLOO_SOCKET_IFNAME="${network_interface}"
export NCCL_SOCKET_IFNAME="${network_interface}"
export TP_SOCKET_IFNAME="${network_interface}"
export RAY_OVERRIDE_JOB_RUNTIME_ENV=1

cleanup() {
  "${venv_root}/bin/ray" stop --force >/dev/null 2>&1 || true
}
trap cleanup EXIT INT TERM

"${venv_root}/bin/ray" start --head \
  --node-ip-address "${node_ip}" \
  --port "${ray_port}" \
  --num-cpus "${ray_num_cpus}" \
  --num-gpus "${num_gpus}" \
  --temp-dir "${ray_tmp}" \
  --disable-usage-stats \
  --dashboard-host 127.0.0.1 \
  --dashboard-port "${dashboard_port}"

runtime_env_json="$("${venv_root}/bin/python" - <<'PY'
import json
import os

names = (
    "PYTHONPATH",
    "PYTHONUNBUFFERED",
    "CUDA_DEVICE_MAX_CONNECTIONS",
    "CUDA_HOME",
    "FLASHINFER_DISABLE_VERSION_CHECK",
    "TOKENIZERS_PARALLELISM",
    "RAY_SERVE_PROXY_HEALTH_CHECK_TIMEOUT_S",
    "MASTER_ADDR",
    "GLOO_SOCKET_IFNAME",
    "NCCL_SOCKET_IFNAME",
    "TP_SOCKET_IFNAME",
    "RAY_OVERRIDE_JOB_RUNTIME_ENV",
)
print(json.dumps({"env_vars": {name: os.environ[name] for name in names}}))
PY
)"

# shellcheck disable=SC1090
source "${relax_root}/scripts/models/qwen35-9B.sh"

global_batch_size=$((rollout_batch_size * n_samples_per_prompt))
log_path="${output_root}/rollout-tp${gpus_per_engine}.log"
sglang_extra_args=(
  --sglang-max-running-requests "${max_running_requests}"
)
if [[ "${disable_radix_cache}" == "1" ]]; then
  sglang_extra_args+=(--sglang-disable-radix-cache)
fi

"${venv_root}/bin/ray" job submit \
  --address "http://127.0.0.1:${dashboard_port}" \
  --runtime-env-json "${runtime_env_json}" \
  -- "${venv_root}/bin/python" -m relax.entrypoints.train \
  --debug-rollout-only \
  --resource "{\"rollout\": [1, ${num_gpus}]}" \
  --max-staleness 1 \
  --num-data-storage-units 1 \
  "${MODEL_ARGS[@]}" \
  --hf-checkpoint "${model_path}" \
  --megatron-to-hf-mode bridge \
  --prompt-data "${prompt_data}" \
  --input-key prompt \
  --label-key label \
  --apply-chat-template \
  --rollout-shuffle \
  --rm-type dapo \
  --reward-key score \
  --num-rollout "${num_rollout}" \
  --rollout-batch-size "${rollout_batch_size}" \
  --n-samples-per-prompt "${n_samples_per_prompt}" \
  --rollout-max-response-len "${response_len}" \
  --rollout-temperature 0.8 \
  --global-batch-size "${global_batch_size}" \
  --save-debug-rollout-data "${output_root}/rollout_{rollout_id}.pt" \
  --advantage-estimator gspo \
  --tensor-model-parallel-size 1 \
  --pipeline-model-parallel-size 1 \
  --context-parallel-size 1 \
  --expert-model-parallel-size 1 \
  --expert-tensor-parallel-size 1 \
  --micro-batch-size 1 \
  --no-rope-fusion \
  --rollout-num-gpus-per-engine "${gpus_per_engine}" \
  --sglang-server-concurrency "${server_concurrency}" \
  "${sglang_extra_args[@]}" \
  --sglang-router-policy round_robin \
  --sglang-mem-fraction-static 0.6 \
  --sglang-context-length "${context_len}" \
  --sglang-attention-backend flashinfer \
  --sglang-sampling-backend flashinfer \
  --sglang-enable-metrics \
  --attention-dropout 0.0 \
  --hidden-dropout 0.0 \
  --attention-backend flash \
  2>&1 | tee "${log_path}"
