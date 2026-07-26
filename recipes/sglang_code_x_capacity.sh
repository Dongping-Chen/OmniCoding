#!/usr/bin/env bash
# Pure model-serving capacity sweep for Code-X-SFT-9B.
#
# Run this inside a single-node Slurm allocation.  No Kira harness, sandbox,
# Ray, or Relax service is started: the purpose is to measure how many model
# requests can be active concurrently for a given TP/DP topology and context
# length.  A separate actor replay benchmark measures training capacity.
set -euo pipefail

repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
venv_root="${VENV_ROOT:-${repo_root}/.venv}"
model_path="${MODEL_PATH:-${repo_root}/models/Code-X-SFT-9B}"
num_gpus="${NUM_GPUS:-4}"
tp_size="${TP_SIZE:-1}"
context_length="${SGLANG_CONTEXT_LENGTH:-131072}"
output_length="${OUTPUT_LENGTH:-256}"
mem_fraction="${SGLANG_MEM_FRACTION_STATIC:-0.80}"
max_running_requests="${SGLANG_MAX_RUNNING_REQUESTS:-256}"
contexts="${BENCH_CONTEXTS:-100000}"
concurrencies="${BENCH_CONCURRENCIES:-1 4 8 16 32 64}"
output_root="${OUTPUT_ROOT:-${repo_root}/outputs/sglang-code-x-capacity/${SLURM_JOB_ID:-manual}}"

for path in \
  "${venv_root}/bin/sglang" \
  "${venv_root}/bin/python" \
  "${model_path}/config.json" \
  "${repo_root}/integrations/relax/probe_sglang_generate_capacity.py" \
  "${repo_root}/integrations/relax/poll_sglang_metrics.py"; do
  if [[ ! -e "${path}" ]]; then
    echo "missing required path: ${path}" >&2
    exit 2
  fi
done
if (( num_gpus < 1 || tp_size < 1 || num_gpus % tp_size != 0 )); then
  echo "NUM_GPUS must be positive and divisible by TP_SIZE" >&2
  exit 2
fi
if (( context_length < 1 || output_length < 1 || max_running_requests < 1 )); then
  echo "context, output length, and max running requests must be positive" >&2
  exit 2
fi
if [[ -z "${SLURM_JOB_ID:-}" ]]; then
  echo "run this script inside a Slurm allocation" >&2
  exit 2
fi

dp_size=$((num_gpus / tp_size))
topology="g${num_gpus}-tp${tp_size}-dp${dp_size}"
server_port="${SGLANG_PORT:-$((30000 + SLURM_JOB_ID % 10000))}"
server_url="http://127.0.0.1:${server_port}"
served_model_name="code-x-sft-9b"
server_log="${output_root}/server.log"
summary_file="${output_root}/summary.jsonl"
server_info_file="${output_root}/server-info.json"

mkdir -p "${output_root}/benchmarks" "${output_root}/metrics"
: >"${server_log}"
: >"${summary_file}"

gpu_list="${CUDA_VISIBLE_DEVICES:-}"
if [[ -z "${gpu_list}" ]]; then
  for ((gpu = 0; gpu < num_gpus; gpu++)); do
    if [[ -n "${gpu_list}" ]]; then
      gpu_list+=","
    fi
    gpu_list+="${gpu}"
  done
fi

export PYTHONUNBUFFERED=1
export CUDA_DEVICE_MAX_CONNECTIONS=1
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-12.9}"
export FLASHINFER_DISABLE_VERSION_CHECK=1
export TOKENIZERS_PARALLELISM=true
export PATH="${venv_root}/bin:${PATH}"

server_pid=""
metrics_pid=""
cleanup() {
  if [[ -n "${metrics_pid}" ]]; then
    kill "${metrics_pid}" >/dev/null 2>&1 || true
    wait "${metrics_pid}" >/dev/null 2>&1 || true
  fi
  if [[ -n "${server_pid}" ]]; then
    kill "${server_pid}" >/dev/null 2>&1 || true
    wait "${server_pid}" >/dev/null 2>&1 || true
  fi
}
trap cleanup EXIT INT TERM

echo "Starting ${topology} on Slurm-visible GPUs ${gpu_list}" | tee -a "${server_log}"
CUDA_VISIBLE_DEVICES="${gpu_list}" \
  "${venv_root}/bin/sglang" serve \
  --model-path "${model_path}" \
  --served-model-name "${served_model_name}" \
  --host 127.0.0.1 \
  --port "${server_port}" \
  --tensor-parallel-size "${tp_size}" \
  --data-parallel-size "${dp_size}" \
  --load-balance-method round_robin \
  --context-length "${context_length}" \
  --mem-fraction-static "${mem_fraction}" \
  --max-running-requests "${max_running_requests}" \
  --max-queued-requests 512 \
  --chunked-prefill-size 8192 \
  --max-prefill-tokens 16384 \
  --trust-remote-code \
  --enable-multimodal \
  --reasoning-parser qwen3 \
  --tool-call-parser qwen3_coder \
  --attention-backend flashinfer \
  --sampling-backend flashinfer \
  --disable-overlap-schedule \
  --enable-metrics \
  --enable-metrics-for-all-schedulers \
  --decode-log-interval 10 \
  >>"${server_log}" 2>&1 &
server_pid=$!

ready=0
for _ in $(seq 1 1200); do
  if curl --fail --silent --max-time 2 "${server_url}/health" >/dev/null; then
    ready=1
    break
  fi
  if ! kill -0 "${server_pid}" 2>/dev/null; then
    echo "SGLang exited during startup" >&2
    tail -200 "${server_log}" >&2
    exit 1
  fi
  sleep 1
done
if [[ "${ready}" != 1 ]]; then
  echo "SGLang did not become healthy within 20 minutes" >&2
  tail -200 "${server_log}" >&2
  exit 1
fi
curl --fail --silent --max-time 30 "${server_url}/server_info" \
  >"${server_info_file}"

for input_length in ${contexts}; do
  if (( input_length + output_length > context_length )); then
    echo "skipping input=${input_length}: exceeds context length" >&2
    continue
  fi
  for concurrency in ${concurrencies}; do
    tag="${topology}-in${input_length}-out${output_length}-c${concurrency}"
    result_file="${output_root}/benchmarks/${tag}.jsonl"
    client_log="${output_root}/benchmarks/${tag}.log"
    metrics_file="${output_root}/metrics/${tag}.jsonl"

    printf '@@ BENCH_START %s %(%Y-%m-%dT%H:%M:%SZ)T\n' "${tag}" -1 \
      >>"${server_log}"
    "${venv_root}/bin/python" \
      "${repo_root}/integrations/relax/poll_sglang_metrics.py" \
      --base-url "${server_url}" \
      --output "${metrics_file}" \
      --interval 0.1 &
    metrics_pid=$!

    set +e
    "${venv_root}/bin/python" \
      "${repo_root}/integrations/relax/probe_sglang_generate_capacity.py" \
      --base-url "${server_url}" \
      --input-length "${input_length}" \
      --output-length "${output_length}" \
      --concurrency "${concurrency}" \
      --seed-offset "$((input_length + concurrency))" \
      --output "${result_file}" \
      >"${client_log}" 2>&1
    benchmark_rc=$?
    set -e

    kill "${metrics_pid}" >/dev/null 2>&1 || true
    wait "${metrics_pid}" >/dev/null 2>&1 || true
    metrics_pid=""
    printf '@@ BENCH_END %s rc=%d %(%Y-%m-%dT%H:%M:%SZ)T\n' \
      "${tag}" "${benchmark_rc}" -1 >>"${server_log}"

    "${venv_root}/bin/python" - \
      "${tag}" "${benchmark_rc}" "${result_file}" "${metrics_file}" \
      "${server_info_file}" \
      >>"${summary_file}" <<'PY'
import json
import sys
from pathlib import Path

tag, raw_rc, result_path, metrics_path, server_info_path = sys.argv[1:]
record = {"tag": tag, "returncode": int(raw_rc)}

results = Path(result_path)
if results.exists() and results.stat().st_size:
    raw_benchmark = json.loads(results.read_text().splitlines()[-1])
    benchmark_keys = (
        "completed",
        "duration",
        "total_input_tokens",
        "total_output_tokens",
        "request_throughput",
        "input_throughput",
        "output_throughput",
        "total_throughput",
        "max_concurrent_requests",
        "concurrency",
        "mean_e2e_latency_ms",
        "p90_e2e_latency_ms",
        "p99_e2e_latency_ms",
        "mean_ttft_ms",
        "p99_ttft_ms",
        "mean_tpot_ms",
        "p99_tpot_ms",
    )
    record["benchmark"] = {
        key: raw_benchmark.get(key)
        for key in benchmark_keys
    }
    errors = raw_benchmark.get("errors") or []
    record["benchmark"]["failed"] = raw_benchmark.get(
        "failed", sum(bool(error) for error in errors)
    )
    server_info_file = Path(server_info_path)
    server_info = (
        json.loads(server_info_file.read_text())
        if server_info_file.exists() and server_info_file.stat().st_size
        else {}
    )
    internal_states = server_info.get("internal_states") or []
    record["server_capacity"] = {
        "max_total_num_tokens": server_info.get("max_total_num_tokens"),
        "max_req_input_len": server_info.get("max_req_input_len"),
        "per_dp": [
            {
                "effective_max_running_requests": state.get(
                    "effective_max_running_requests_per_dp"
                ),
                "memory_usage": state.get("memory_usage"),
            }
            for state in internal_states
        ],
    }

peak = {
    "running_requests": 0.0,
    "queued_requests": 0.0,
    "token_usage": 0.0,
    "full_token_usage": 0.0,
}
metrics = Path(metrics_path)
if metrics.exists():
    for line in metrics.read_text().splitlines():
        sample = json.loads(line)
        for name in peak:
            peak[name] = max(peak[name], float(sample.get(name, 0.0)))
record["server_peak"] = peak
print(json.dumps(record, sort_keys=True))
PY

    if (( benchmark_rc != 0 )); then
      echo "${tag} failed with rc=${benchmark_rc}; continuing the sweep" >&2
      tail -80 "${client_log}" >&2
    fi
  done
done

echo "Capacity sweep completed: ${summary_file}"
