#!/usr/bin/env bash
# Complete OmniCoding Kira + GSPO training/smoke launcher.
#
# Default: 8 H100s, an 8-way actor TP group, 8 TP1 SGLang engines, 8 Kira
# trajectories per optimizer step, max_turns=50, and 10 optimizer steps over
# the RL split. Set ROLLOUT_COORDINATOR_PUBLIC_URL and its token file when the
# Kira sandbox service is remote; otherwise workers run in this allocation.
# Set DEBUG_ROLLOUT_ONLY=1 for rollout-only diagnosis.
set -euo pipefail

repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
venv_root="${VENV_ROOT:-${repo_root}/.venv}"
relax_root="${RELAX_ROOT:-${venv_root}/src/Relax}"
megatron_root="${MEGATRON_ROOT:-${venv_root}/src/Megatron-LM}"
model_path="${MODEL_PATH:-${repo_root}/models/Qwen3.5-9B}"
prompt_data="${PROMPT_DATA:-${repo_root}/data/omnicoding/processed/rl_mcq_prompts.parquet}"
rl_train_jsonl="${RL_TRAIN_JSONL:-${repo_root}/data/omnicoding/processed/rl_train.jsonl}"
dataset_root="${DATASET_ROOT:-${repo_root}/data/omnicoding}"
container_image="${ROLLOUT_CONTAINER_IMAGE:-${repo_root}/runtime/omnicoding-rl-worker-20260724-v5.sqsh}"
harness_venv="${ROLLOUT_HARNESS_VENV:-${repo_root}/.venv-harness}"
output_root="${OUTPUT_ROOT:-${repo_root}/outputs/relax-qwen35-kira-gspo/${SLURM_JOB_ID:-manual}}"
tavily_keys_file="${TAVILY_KEYS_FILE:-${repo_root}/tavily.txt}"
external_coordinator_url="${ROLLOUT_COORDINATOR_PUBLIC_URL:-}"
external_coordinator_token_file="${ROLLOUT_COORDINATOR_TOKEN_FILE:-}"
external_coordinator=0
if [[ -n "${external_coordinator_url}" ]]; then
  external_coordinator=1
fi

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
if [[ "${external_coordinator}" == 0 ]]; then
  for path in \
    "${rl_train_jsonl}" \
    "${container_image}" \
    "${harness_venv}/bin/python" \
    "${tavily_keys_file}"; do
    if [[ ! -e "${path}" ]]; then
      echo "missing required local-sandbox path: ${path}" >&2
      exit 2
    fi
  done
elif [[ -z "${external_coordinator_token_file}" || \
        ! -f "${external_coordinator_token_file}" ]]; then
  echo "remote sandbox mode requires ROLLOUT_COORDINATOR_TOKEN_FILE" >&2
  exit 2
fi
if [[ -z "${SLURM_JOB_ID:-}" ]]; then
  echo "run this script inside an existing single-node Slurm allocation" >&2
  exit 2
fi
# Relax currently starts Ray Serve on the node-wide default port 8000. Two
# independent Relax roots on one physical node corrupt each other's runtime
# state even when their GCS/dashboard ports differ. Scale trajectories inside
# this job; do not launch another training job on the spare GPUs.
if [[ -n "$(ss -H -ltn 'sport = :8000' 2>/dev/null)" ]]; then
  echo "port 8000 is already in use; another Relax/Ray Serve root is running on this node" >&2
  exit 2
fi

num_gpus="${NUM_GPUS:-8}"
actor_tp="${ACTOR_TENSOR_PARALLEL_SIZE:-${ACTOR_TP:-${num_gpus}}}"
actor_cp="${ACTOR_CONTEXT_PARALLEL_SIZE:-${ACTOR_CP:-1}}"
rollout_gpus_per_engine="${ROLLOUT_GPUS_PER_ENGINE:-1}"
rollout_batch_size="${ROLLOUT_BATCH_SIZE:-2}"
n_samples_per_prompt="${N_SAMPLES_PER_PROMPT:-4}"
num_rollout="${NUM_ROLLOUT:-10}"
response_len="${ROLLOUT_MAX_RESPONSE_LEN:-65536}"
context_len="${SGLANG_CONTEXT_LENGTH:-131072}"
server_concurrency="${SGLANG_SERVER_CONCURRENCY:-16}"
max_running_requests="${SGLANG_MAX_RUNNING_REQUESTS:-${server_concurrency}}"
sglang_chunked_prefill_size="${SGLANG_CHUNKED_PREFILL_SIZE:-2048}"
sglang_max_prefill_tokens="${SGLANG_MAX_PREFILL_TOKENS:-4096}"
log_probs_chunk_size="${LOG_PROBS_CHUNK_SIZE:-4096}"
micro_batch_size="${MICRO_BATCH_SIZE:-1}"
# Relax packs THD sequences without padding between samples. The 48k budget
# allows compatible shorter trajectories to share a micro-batch. The matching
# Kira trajectory cap below rejects a singleton above that budget before actor
# training, because dynamic batching cannot split one sequence across batches.
max_tokens_per_gpu="${MAX_TOKENS_PER_GPU:-48000}"
kira_max_trajectory_tokens="${KIRA_MAX_TRAJECTORY_TOKENS:-${max_tokens_per_gpu}}"
disable_jit_fuser="${DISABLE_JIT_FUSER:-1}"
profile_train_stages="${RELAX_PROFILE_TRAIN_STAGES:-1}"
step_concurrency="${ROLLOUT_STEP_CONCURRENCY:-$((rollout_batch_size * n_samples_per_prompt))}"
debug_rollout_only="${DEBUG_ROLLOUT_ONLY:-0}"
load_debug_rollout_data="${LOAD_DEBUG_ROLLOUT_DATA:-}"
use_rollout_logprobs="${USE_ROLLOUT_LOGPROBS:-}"
if [[ -z "${use_rollout_logprobs}" ]]; then
  # Live Kira rollouts are scored on the final canonical trajectory by the
  # SGLang inference engines. Legacy debug .pt files predate that field, so
  # replay stays compatible unless explicitly opted in.
  if [[ -z "${load_debug_rollout_data}" ]]; then
    use_rollout_logprobs=1
  else
    use_rollout_logprobs=0
  fi
fi
if [[ -n "${ROLLOUT_GPU_DEVICES:-}" ]]; then
  sandbox_gpu_device="${ROLLOUT_GPU_DEVICES}"
elif [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  sandbox_gpu_device="${CUDA_VISIBLE_DEVICES##*,}"
else
  sandbox_gpu_device="$((num_gpus - 1))"
fi
slurm_cpus="${SLURM_CPUS_PER_TASK:-$(nproc)}"
rollout_cpu_reserve=$((step_concurrency * ${ROLLOUT_CPUS_PER_TASK:-4}))
if [[ -n "${load_debug_rollout_data}" || "${external_coordinator}" == 1 ]]; then
  # Actor-only replay does not launch local rollout/sandbox workers. Give Ray
  # the full allocation. The same applies when sandbox workers live on a
  # remote coordinator: this node then owns only inference and actor training.
  default_ray_cpus="${slurm_cpus}"
else
  default_ray_cpus=$((slurm_cpus - rollout_cpu_reserve))
fi
if (( default_ray_cpus < num_gpus * 2 )); then
  default_ray_cpus=$((num_gpus * 2))
fi
ray_num_cpus="${RAY_NUM_CPUS:-${default_ray_cpus}}"

if (( num_gpus < 1 )); then
  echo "NUM_GPUS must be positive" >&2
  exit 2
fi
if (( actor_tp < 1 || actor_cp < 1 || micro_batch_size < 1 || max_tokens_per_gpu < 1 )); then
  echo "actor TP, actor CP, micro batch size, and MAX_TOKENS_PER_GPU must be positive" >&2
  exit 2
fi
if [[ ! "${kira_max_trajectory_tokens}" =~ ^[1-9][0-9]*$ ]]; then
  echo "KIRA_MAX_TRAJECTORY_TOKENS must be a positive integer" >&2
  exit 2
fi
if (( sglang_chunked_prefill_size < 1 || sglang_max_prefill_tokens < sglang_chunked_prefill_size )); then
  echo "SGLANG_MAX_PREFILL_TOKENS must be >= positive SGLANG_CHUNKED_PREFILL_SIZE" >&2
  exit 2
fi
if (( num_gpus % (actor_tp * actor_cp) != 0 || num_gpus % rollout_gpus_per_engine != 0 )); then
  echo "NUM_GPUS must be divisible by actor TP*CP and rollout TP sizes" >&2
  exit 2
fi
actor_dp=$((num_gpus / (actor_tp * actor_cp)))
if [[ "${debug_rollout_only}" == 0 ]] && (( rollout_batch_size % actor_dp != 0 )); then
  cat >&2 <<EOF
invalid GSPO group placement: ROLLOUT_BATCH_SIZE=${rollout_batch_size} prompts
cannot be split across actor DP=${actor_dp}. Relax's group sampler must give
each DP rank whole ${n_samples_per_prompt}-sample groups. Choose a rollout
batch size divisible by ${actor_dp} (for example ${actor_dp}).
EOF
  exit 2
fi
if [[ "${debug_rollout_only}" != 0 && "${debug_rollout_only}" != 1 ]]; then
  echo "DEBUG_ROLLOUT_ONLY must be 0 or 1" >&2
  exit 2
fi
if [[ "${use_rollout_logprobs}" != 0 && "${use_rollout_logprobs}" != 1 ]]; then
  echo "USE_ROLLOUT_LOGPROBS must be 0 or 1" >&2
  exit 2
fi
if [[ "${disable_jit_fuser}" != 0 && "${disable_jit_fuser}" != 1 ]]; then
  echo "DISABLE_JIT_FUSER must be 0 or 1" >&2
  exit 2
fi
if [[ "${debug_rollout_only}" == 1 && -n "${load_debug_rollout_data}" ]]; then
  echo "DEBUG_ROLLOUT_ONLY and LOAD_DEBUG_ROLLOUT_DATA are mutually exclusive" >&2
  exit 2
fi
if [[ -n "${load_debug_rollout_data}" && ! -f "${load_debug_rollout_data}" ]]; then
  echo "LOAD_DEBUG_ROLLOUT_DATA does not exist: ${load_debug_rollout_data}" >&2
  exit 2
fi

mkdir -p "${output_root}/coordinator-runtime"
umask 077
if [[ "${external_coordinator}" == 1 ]]; then
  token_file="${external_coordinator_token_file}"
else
  token_file="${output_root}/coordinator-token"
fi
web_search_token_file="${output_root}/web-search-token"
if [[ "${external_coordinator}" == 0 ]]; then
  "${venv_root}/bin/python" - "${token_file}" <<'PY'
import secrets
import sys
from pathlib import Path

path = Path(sys.argv[1])
path.write_text(secrets.token_hex(32))
path.chmod(0o600)
PY
  "${venv_root}/bin/python" - "${web_search_token_file}" <<'PY'
import secrets
import sys
from pathlib import Path

path = Path(sys.argv[1])
path.write_text(secrets.token_hex(32))
path.chmod(0o600)
PY
fi

export PYTHONUNBUFFERED=1
export PYTHONDONTWRITEBYTECODE=1
export CUDA_DEVICE_MAX_CONNECTIONS=1
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-12.9}"
export FLASHINFER_DISABLE_VERSION_CHECK=1
export TOKENIZERS_PARALLELISM=true
export PATH="${venv_root}/bin:${PATH}"
export RAY_SERVE_PROXY_HEALTH_CHECK_TIMEOUT_S="${RAY_SERVE_PROXY_HEALTH_CHECK_TIMEOUT_S:-120}"
export RELAX="${relax_root}"
export MEGATRON="${megatron_root}"
export PYTHONPATH="${relax_root}:${megatron_root}:${repo_root}/src:${PYTHONPATH:-}"

port_offset=$((SLURM_JOB_ID % 10000))
ray_port="${RAY_PORT:-${RAY_HEAD_PORT:-$((20000 + port_offset % 10000))}}"
dashboard_port="${RAY_DASHBOARD_PORT:-$((30000 + port_offset % 10000))}"
coordinator_port="${ROLLOUT_COORDINATOR_PORT:-${COORDINATOR_PORT:-$((18000 + port_offset % 10000))}}"
web_search_port="${WEB_SEARCH_PROXY_PORT:-${SEARCH_PROXY_PORT:-$((19000 + port_offset % 10000))}}"
router_port="${SGLANG_ROUTER_PORT:-$((40000 + port_offset % 10000))}"
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
ray_tmp="${RAY_TMPDIR:-/tmp/r${SLURM_JOB_ID}ag}"
network_interface="${RELAX_NETWORK_INTERFACE:-$(ip route show default | awk 'NR == 1 {print $5}')}"
network_interface="${network_interface:-lo}"

export MASTER_ADDR="${node_ip}"
export GLOO_SOCKET_IFNAME="${network_interface}"
export NCCL_SOCKET_IFNAME="${network_interface}"
export TP_SOCKET_IFNAME="${network_interface}"
export RAY_OVERRIDE_JOB_RUNTIME_ENV=1

export ROLLOUT_COORDINATOR_TOKEN_FILE="${token_file}"
export ROLLOUT_SGLANG_MODEL="${model_path}"
export KIRA_MAX_TURNS="${KIRA_MAX_TURNS:-50}"
export KIRA_MAX_TOKENS_PER_TURN="${KIRA_MAX_TOKENS_PER_TURN:-2048}"
export KIRA_MAX_TRAJECTORY_TOKENS="${kira_max_trajectory_tokens}"
export RELAX_PROFILE_TRAIN_STAGES="${profile_train_stages}"
export KIRA_REQUEST_TIMEOUT="${KIRA_REQUEST_TIMEOUT:-180}"
export KIRA_BLOCK_TIMEOUT="${KIRA_BLOCK_TIMEOUT:-600}"
export ROLLOUT_POLL_INTERVAL_S="${ROLLOUT_POLL_INTERVAL_S:-2}"
if [[ "${external_coordinator}" == 1 ]]; then
  export ROLLOUT_COORDINATOR_PUBLIC_URL="${external_coordinator_url%/}"
  # The remote sandbox coordinator must be able to reach the rollout router.
  # Override this when the two servers use a different routed address.
  export ROLLOUT_SGLANG_PUBLIC_URL="${ROLLOUT_SGLANG_PUBLIC_URL:-http://${node_ip}:${router_port}}"
else
  export RL_TRAIN_JSONL="${rl_train_jsonl}"
  export DATASET_ROOT="${dataset_root}"
  export OMNICODING_RUNTIME_ROOT="${output_root}/coordinator-runtime"
  export ROLLOUT_EXECUTION_BACKEND=slurm_step
  export ROLLOUT_CONTAINER_IMAGE="${container_image}"
  export ROLLOUT_HARNESS_VENV="${harness_venv}"
  export ROLLOUT_STEP_CONCURRENCY="${step_concurrency}"
  export ROLLOUT_CPUS_PER_TASK="${ROLLOUT_CPUS_PER_TASK:-4}"
  export ROLLOUT_MEMORY_PER_TASK="${ROLLOUT_MEMORY_PER_TASK:-40G}"
  export ROLLOUT_GPU_DEVICES="${sandbox_gpu_device}"
  export ROLLOUTS_PER_SANDBOX_GPU="${ROLLOUTS_PER_SANDBOX_GPU:-${step_concurrency}}"
  export ROLLOUT_MAX_IN_FLIGHT="${ROLLOUT_MAX_IN_FLIGHT:-128}"
  export ROLLOUT_MAX_QUEUED_JOBS="${ROLLOUT_MAX_QUEUED_JOBS:-256}"
  export ROLLOUT_RESULT_TTL_S="${ROLLOUT_RESULT_TTL_S:-3600}"
  export ROLLOUT_JOB_DIR_TTL_S="${ROLLOUT_JOB_DIR_TTL_S:-7200}"
  export ROLLOUT_COORDINATOR_PUBLIC_URL="http://127.0.0.1:${coordinator_port}"
  export ROLLOUT_ALLOWED_SGLANG_ORIGINS="http://${node_ip}:${router_port}"
  export ROLLOUT_ALLOWED_MODELS="openai/${model_path}"
  export TAVILY_KEYS_FILE="${tavily_keys_file}"
  export OMNICODING_WEB_SEARCH_PROXY_TOKEN_FILE="${web_search_token_file}"
  export OMNICODING_WEB_SEARCH_PROXY_URL="http://${node_ip}:${web_search_port}"
  export OMNICODING_WEB_SEARCH_PROXY_TOKEN="$(
    <"${web_search_token_file}"
  )"
fi

coordinator_pid=""
web_search_pid=""
cleanup() {
  if [[ -n "${coordinator_pid}" ]]; then
    kill "${coordinator_pid}" >/dev/null 2>&1 || true
    wait "${coordinator_pid}" >/dev/null 2>&1 || true
  fi
  if [[ -n "${web_search_pid}" ]]; then
    kill "${web_search_pid}" >/dev/null 2>&1 || true
    wait "${web_search_pid}" >/dev/null 2>&1 || true
  fi
  "${venv_root}/bin/ray" stop --force >/dev/null 2>&1 || true
}
trap cleanup EXIT INT TERM

if [[ -z "${load_debug_rollout_data}" && "${external_coordinator}" == 0 ]]; then
  "${venv_root}/bin/uvicorn" omnicoding.tools.tavily_proxy:app \
    --host 0.0.0.0 \
    --port "${web_search_port}" \
    >"${output_root}/web-search-proxy.log" 2>&1 &
  web_search_pid=$!

  web_search_ready=0
  for _ in $(seq 1 30); do
    if WEB_SEARCH_URL="${OMNICODING_WEB_SEARCH_PROXY_URL}" \
       WEB_SEARCH_TOKEN="${OMNICODING_WEB_SEARCH_PROXY_TOKEN}" \
       "${venv_root}/bin/python" - 2>/dev/null <<'PY'
import os
import urllib.request

request = urllib.request.Request(
    os.environ["WEB_SEARCH_URL"] + "/health",
    headers={"Authorization": "Bearer " + os.environ["WEB_SEARCH_TOKEN"]},
)
with urllib.request.urlopen(request, timeout=2) as response:
    raise SystemExit(0 if response.status == 200 else 1)
PY
    then
      web_search_ready=1
      break
    fi
    if ! kill -0 "${web_search_pid}" 2>/dev/null; then
      echo "web-search proxy exited during startup" >&2
      tail -100 "${output_root}/web-search-proxy.log" >&2
      exit 1
    fi
    sleep 1
  done
  if [[ "${web_search_ready}" != 1 ]]; then
    echo "web-search proxy did not become healthy" >&2
    exit 1
  fi

  "${venv_root}/bin/uvicorn" omnicoding.rl.coordinator.app:app \
    --host 127.0.0.1 \
    --port "${coordinator_port}" \
    >"${output_root}/coordinator.log" 2>&1 &
  coordinator_pid=$!

  coordinator_ready=0
  for _ in $(seq 1 60); do
    if COORDINATOR_URL="${ROLLOUT_COORDINATOR_PUBLIC_URL}" \
       COORDINATOR_TOKEN_FILE="${token_file}" \
       "${venv_root}/bin/python" - 2>/dev/null <<'PY'
import os
import urllib.request

token = open(os.environ["COORDINATOR_TOKEN_FILE"]).read().strip()
request = urllib.request.Request(
    os.environ["COORDINATOR_URL"] + "/health",
    headers={"Authorization": "Bearer " + token},
)
with urllib.request.urlopen(request, timeout=2) as response:
    raise SystemExit(0 if response.status == 200 else 1)
PY
    then
      coordinator_ready=1
      break
    fi
    if ! kill -0 "${coordinator_pid}" 2>/dev/null; then
      echo "coordinator exited during startup" >&2
      tail -100 "${output_root}/coordinator.log" >&2
      exit 1
    fi
    sleep 1
  done
  if [[ "${coordinator_ready}" != 1 ]]; then
    echo "coordinator did not become healthy" >&2
    exit 1
  fi
elif [[ -z "${load_debug_rollout_data}" ]]; then
  remote_coordinator_ready=0
  for _ in $(seq 1 60); do
    if COORDINATOR_URL="${ROLLOUT_COORDINATOR_PUBLIC_URL}" \
       COORDINATOR_TOKEN_FILE="${token_file}" \
       "${venv_root}/bin/python" - 2>/dev/null <<'PY'
import os
import urllib.request

token = open(os.environ["COORDINATOR_TOKEN_FILE"]).read().strip()
request = urllib.request.Request(
    os.environ["COORDINATOR_URL"] + "/health",
    headers={"Authorization": "Bearer " + token},
)
with urllib.request.urlopen(request, timeout=2) as response:
    raise SystemExit(0 if response.status == 200 else 1)
PY
    then
      remote_coordinator_ready=1
      break
    fi
    sleep 1
  done
  if [[ "${remote_coordinator_ready}" != 1 ]]; then
    echo "remote rollout coordinator did not become healthy" >&2
    exit 1
  fi
fi

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
    "PATH",
    "PYTHONPATH",
    "PYTHONUNBUFFERED",
    "PYTHONDONTWRITEBYTECODE",
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
    "ROLLOUT_COORDINATOR_PUBLIC_URL",
    "ROLLOUT_COORDINATOR_TOKEN_FILE",
    "ROLLOUT_SGLANG_PUBLIC_URL",
    "ROLLOUT_SGLANG_MODEL",
    "KIRA_MAX_TURNS",
    "KIRA_MAX_TOKENS_PER_TURN",
    "KIRA_MAX_TRAJECTORY_TOKENS",
    "RELAX_PROFILE_TRAIN_STAGES",
    "KIRA_REQUEST_TIMEOUT",
    "KIRA_BLOCK_TIMEOUT",
    "ROLLOUT_POLL_INTERVAL_S",
)
print(json.dumps({"env_vars": {name: os.environ[name] for name in names if name in os.environ}}))
PY
)"

# shellcheck disable=SC1090
source "${relax_root}/scripts/models/qwen35-9B.sh"

if [[ -n "${load_debug_rollout_data}" ]]; then
  # A capacity replay only needs one intact GSPO group on each actor-DP rank.
  # Keeping all rollout groups here multiplies the token work without changing
  # the number of samples resident in a micro-step.
  global_batch_size="${GLOBAL_BATCH_SIZE:-$((actor_dp * n_samples_per_prompt))}"
  grpo_iterations="${GRPO_ITERATIONS:-1}"
else
  global_batch_size="${GLOBAL_BATCH_SIZE:-$((rollout_batch_size * n_samples_per_prompt))}"
  grpo_iterations="${GRPO_ITERATIONS:-2}"
fi
log_path="${output_root}/kira-gspo.log"

mode_args=()
rollout_logprob_args=()
jit_fuser_args=()
if [[ "${use_rollout_logprobs}" == 1 ]]; then
  rollout_logprob_args+=(--use-rollout-logprobs)
fi
if [[ "${disable_jit_fuser}" == 1 ]]; then
  # Qwen3.5 GatedDeltaNet decorates shape-dependent helpers with torch.compile.
  # RL alternates no-grad/train modes and highly variable trajectory lengths,
  # which otherwise repeatedly recompiles these helpers on the first step.
  jit_fuser_args+=(--disable-jit-fuser)
fi
placement_args=(--colocate)
resource_json="{\"actor\": [1, ${num_gpus}], \"rollout\": [1, ${num_gpus}]}"
if [[ -n "${load_debug_rollout_data}" ]]; then
  mode_args+=(--debug-train-only --load-debug-rollout-data "${load_debug_rollout_data}")
  # Relax's Actor waits for a TransferQueue train partition whenever
  # colocate=True, even in debug_train_only where data comes directly from
  # the saved .pt. No rollout service exists to create that partition.
  placement_args=()
  resource_json="{\"actor\": [1, ${num_gpus}]}"
elif [[ "${debug_rollout_only}" == 1 ]]; then
  mode_args+=(--debug-rollout-only)
  resource_json="{\"rollout\": [1, ${num_gpus}]}"
fi

"${venv_root}/bin/ray" job submit \
  --address "http://127.0.0.1:${dashboard_port}" \
  --runtime-env-json "${runtime_env_json}" \
  -- "${venv_root}/bin/python" -m relax.entrypoints.train \
  "${mode_args[@]}" \
  --resource "${resource_json}" \
  "${placement_args[@]}" \
  --num-gpus-per-node "${num_gpus}" \
  --max-staleness 0 \
  --num-data-storage-units 1 \
  "${MODEL_ARGS[@]}" \
  --hf-checkpoint "${model_path}" \
  --megatron-to-hf-mode bridge \
  --prompt-data "${prompt_data}" \
  --input-key prompt \
  --label-key label \
  --metadata-key metadata \
  --apply-chat-template \
  --rm-type none \
  --group-rm \
  --custom-rm-path omnicoding.rl.reward.reward_func_group \
  --reward-key score \
  --eval-reward-key correctness \
  --dynamic-sampling-filter-path omnicoding.rl.reward.check_active_reward_nonzero_std \
  --custom-reward-post-process-path omnicoding.rl.reward.reward_post_process \
  --custom-generate-function-path omnicoding.rl.rollout.generate \
  "${rollout_logprob_args[@]}" \
  --num-rollout "${num_rollout}" \
  --rollout-batch-size "${rollout_batch_size}" \
  --n-samples-per-prompt "${n_samples_per_prompt}" \
  --rollout-max-response-len "${response_len}" \
  --rollout-temperature 0.8 \
  --global-batch-size "${global_batch_size}" \
  --grpo-iterations "${grpo_iterations}" \
  --save-debug-rollout-data "${output_root}/rollout_{rollout_id}.pt" \
  --advantage-estimator gspo \
  --disable-grpo-std-normalization \
  --kl-coef 0 \
  --entropy-coef 0.0 \
  --log-probs-chunk-size "${log_probs_chunk_size}" \
  --eps-clip 3.0 \
  --eps-clip-high 0.28 \
  --optimizer adam \
  --lr 1e-6 \
  --lr-decay-style constant \
  --weight-decay 0.1 \
  --adam-beta1 0.9 \
  --adam-beta2 0.98 \
  --optimizer-cpu-offload \
  --overlap-cpu-optimizer-d2h-h2d \
  --use-precision-aware-optimizer \
  --tensor-model-parallel-size "${actor_tp}" \
  --sequence-parallel \
  --pipeline-model-parallel-size 1 \
  --context-parallel-size "${actor_cp}" \
  --expert-model-parallel-size 1 \
  --expert-tensor-parallel-size 1 \
  --micro-batch-size "${micro_batch_size}" \
  --use-dynamic-batch-size \
  --max-tokens-per-gpu "${max_tokens_per_gpu}" \
  "${jit_fuser_args[@]}" \
  --recompute-granularity full \
  --recompute-method uniform \
  --recompute-num-layers 1 \
  --recompute-loss-function \
  --no-rope-fusion \
  --rollout-num-gpus-per-engine "${rollout_gpus_per_engine}" \
  --sglang-server-concurrency "${server_concurrency}" \
  --sglang-max-running-requests "${max_running_requests}" \
  --sglang-chunked-prefill-size "${sglang_chunked_prefill_size}" \
  --sglang-max-prefill-tokens "${sglang_max_prefill_tokens}" \
  --sglang-router-port "${router_port}" \
  --sglang-router-policy round_robin \
  --sglang-mem-fraction-static "${SGLANG_MEM_FRACTION_STATIC:-0.52}" \
  --sglang-context-length "${context_len}" \
  --sglang-tool-call-parser qwen3_coder \
  --sglang-reasoning-parser qwen3 \
  --sglang-enable-multimodal \
  --sglang-attention-backend flashinfer \
  --sglang-sampling-backend flashinfer \
  --sglang-enable-metrics \
  --attention-dropout 0.0 \
  --hidden-dropout 0.0 \
  --attention-backend flash \
  2>&1 | tee "${log_path}"
