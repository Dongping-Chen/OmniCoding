#!/usr/bin/env bash
# Sanitized single-node 27B GSPO recipe for the pinned Relax integration.
# Apply integrations/relax/patches/*.patch to RELAX_ROOT in order and install
# this OmniCoding package into the same environment before launching.
set -euo pipefail

: "${RELAX_ROOT:?Set RELAX_ROOT to the patched Relax checkout}"
: "${MEGATRON_ROOT:?Set MEGATRON_ROOT to the compatible Megatron-LM checkout}"
: "${PROMPT_PARQUET:?Set PROMPT_PARQUET to the output of omnicoding-rl-build-prompts}"
: "${SAVE_DIR:?Set SAVE_DIR to a writable checkpoint directory}"
: "${ROLLOUT_COORDINATOR_PUBLIC_URL:?Set the authenticated coordinator URL}"
: "${ROLLOUT_COORDINATOR_TOKEN_FILE:?Set an absolute path to the mode-0600 coordinator token file}"

if [[ -n "${ROLLOUT_COORDINATOR_TOKEN:-}" ]]; then
  echo "use ROLLOUT_COORDINATOR_TOKEN_FILE; do not export the token into the trainer environment" >&2
  exit 2
fi
if [[ "$ROLLOUT_COORDINATOR_TOKEN_FILE" != /* ]]; then
  echo "ROLLOUT_COORDINATOR_TOKEN_FILE must be an absolute path" >&2
  exit 2
fi
if [[ ! -f "$ROLLOUT_COORDINATOR_TOKEN_FILE" ]]; then
  echo "coordinator token file does not exist" >&2
  exit 2
fi
token_file_mode="$(stat -c '%a' "$ROLLOUT_COORDINATOR_TOKEN_FILE")"
if [[ "$token_file_mode" != "600" && "$token_file_mode" != "400" ]]; then
  echo "coordinator token file mode must be 600 or 400" >&2
  exit 2
fi

MODEL_PATH="${MODEL_PATH:-shuaishuaicdp/Code-X-SFT-27B}"
ROLLOUT_SGLANG_MODEL="${ROLLOUT_SGLANG_MODEL:-shuaishuaicdp/Code-X-SFT-27B}"
NUM_GPUS="${NUM_GPUS:-8}"
SAVE_INTERVAL="${SAVE_INTERVAL:-50}"
NUM_ROLLOUT="${NUM_ROLLOUT:-10}"
ROLLOUT_BATCH_SIZE="${ROLLOUT_BATCH_SIZE:-16}"
N_SAMPLES_PER_PROMPT="${N_SAMPLES_PER_PROMPT:-8}"
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-128}"
ROLLOUT_MAX_RESPONSE_LEN="${ROLLOUT_MAX_RESPONSE_LEN:-192000}"
KIRA_MAX_TURNS="${KIRA_MAX_TURNS:-100}"
KIRA_MAX_TOKENS_PER_TURN="${KIRA_MAX_TOKENS_PER_TURN:-8192}"
TENSOR_MODEL_PARALLEL_SIZE="${TENSOR_MODEL_PARALLEL_SIZE:-4}"
MAX_TOKENS_PER_GPU="${MAX_TOKENS_PER_GPU:-9216}"
SGLANG_MEM_FRACTION_STATIC="${SGLANG_MEM_FRACTION_STATIC:-0.7}"
SGLANG_CONTEXT_LENGTH="${SGLANG_CONTEXT_LENGTH:-204800}"
MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
RAY_BOOTSTRAP_ADDRESS="${RAY_BOOTSTRAP_ADDRESS:-auto}"
RAY_DASHBOARD_ADDRESS="${RAY_DASHBOARD_ADDRESS:-http://127.0.0.1:8265}"

if ! [[ "$SAVE_INTERVAL" =~ ^[1-9][0-9]*$ ]]; then
  echo "SAVE_INTERVAL must be a positive integer" >&2
  exit 2
fi
mkdir -p "$SAVE_DIR"
if [[ ! -d "$SAVE_DIR" || ! -w "$SAVE_DIR" ]]; then
  echo "SAVE_DIR must be a writable directory" >&2
  exit 2
fi

export PYTHONUNBUFFERED=1
export CUDA_DEVICE_MAX_CONNECTIONS=1
export MASTER_ADDR
export PYTHONPATH="${RELAX_ROOT}:${MEGATRON_ROOT}:${PYTHONPATH:-}"
export ROLLOUT_COORDINATOR_PUBLIC_URL ROLLOUT_COORDINATOR_TOKEN_FILE
export ROLLOUT_SGLANG_MODEL KIRA_MAX_TURNS KIRA_MAX_TOKENS_PER_TURN
if [[ -n "${ROLLOUT_SGLANG_PUBLIC_URL:-}" ]]; then
  export ROLLOUT_SGLANG_PUBLIC_URL
fi

model_config="${RELAX_ROOT}/scripts/models/qwen36-27B.sh"
if [[ ! -f "$model_config" ]]; then
  echo "missing pinned Relax model config: $model_config" >&2
  exit 2
fi
# shellcheck disable=SC1090
source "$model_config"

if ! ray status --address="$RAY_BOOTSTRAP_ADDRESS" >/dev/null 2>&1; then
  ray start --head \
    --node-ip-address "$MASTER_ADDR" \
    --num-gpus "$NUM_GPUS" \
    --disable-usage-stats \
    --dashboard-host 127.0.0.1 \
    --dashboard-port 8265
fi

runtime_env_json="$(python - <<'PY'
import json
import os

names = (
    "PYTHONPATH",
    "PYTHONUNBUFFERED",
    "CUDA_DEVICE_MAX_CONNECTIONS",
    "ROLLOUT_COORDINATOR_PUBLIC_URL",
    "ROLLOUT_COORDINATOR_TOKEN_FILE",
    "ROLLOUT_SGLANG_MODEL",
    "ROLLOUT_SGLANG_PUBLIC_URL",
    "KIRA_MAX_TURNS",
    "KIRA_MAX_TOKENS_PER_TURN",
)
print(json.dumps({"env_vars": {name: os.environ[name] for name in names if os.environ.get(name)}}))
PY
)"

system_prompt="You are a multimodal coding agent solving a benchmark task. Explore staged media with shell and image tools, wrap the final response in <answer>...</answer>, then call task_complete."

ray job submit \
  --address "$RAY_DASHBOARD_ADDRESS" \
  --runtime-env-json "$runtime_env_json" \
  -- python -m relax.entrypoints.train \
  --resource "{\"actor\": [1, ${NUM_GPUS}], \"rollout\": [1, ${NUM_GPUS}]}" \
  --max-staleness 0 \
  --num-data-storage-units 1 \
  --colocate \
  "${MODEL_ARGS[@]}" \
  --hf-checkpoint "$MODEL_PATH" \
  --ref-load "$MODEL_PATH" \
  --megatron-to-hf-mode bridge \
  --save "$SAVE_DIR" \
  --save-interval "$SAVE_INTERVAL" \
  --prompt-data "$PROMPT_PARQUET" \
  --input-key prompt \
  --label-key label \
  --metadata-key metadata \
  --apply-chat-template \
  --rollout-shuffle \
  --rm-type none \
  --group-rm \
  --custom-rm-path omnicoding.rl.reward.reward_func_group \
  --reward-key score \
  --eval-reward-key correctness \
  --dynamic-sampling-filter-path omnicoding.rl.reward.check_active_reward_nonzero_std \
  --custom-reward-post-process-path omnicoding.rl.reward.reward_post_process \
  --custom-generate-function-path omnicoding.rl.rollout.generate \
  --num-rollout "$NUM_ROLLOUT" \
  --rollout-batch-size "$ROLLOUT_BATCH_SIZE" \
  --n-samples-per-prompt "$N_SAMPLES_PER_PROMPT" \
  --rollout-max-response-len "$ROLLOUT_MAX_RESPONSE_LEN" \
  --rollout-temperature 0.8 \
  --global-batch-size "$GLOBAL_BATCH_SIZE" \
  --balance-data \
  --use-fault-tolerance \
  --system-prompt "$system_prompt" \
  --train-backend megatron \
  --tensor-model-parallel-size "$TENSOR_MODEL_PARALLEL_SIZE" \
  --sequence-parallel \
  --pipeline-model-parallel-size 1 \
  --context-parallel-size 1 \
  --expert-model-parallel-size 1 \
  --expert-tensor-parallel-size 1 \
  --recompute-granularity full \
  --recompute-method uniform \
  --recompute-num-layers 1 \
  --use-dynamic-batch-size \
  --max-tokens-per-gpu "$MAX_TOKENS_PER_GPU" \
  --no-rope-fusion \
  --advantage-estimator gspo \
  --disable-grpo-std-normalization \
  --kl-loss-type low_var_kl \
  --entropy-coef 0.0 \
  --eps-clip 0.2 \
  --eps-clip-high 0.28 \
  --eps-clip-c 3 \
  --optimizer adam \
  --lr 1e-6 \
  --lr-decay-style constant \
  --weight-decay 0.1 \
  --adam-beta1 0.9 \
  --adam-beta2 0.98 \
  --rollout-num-gpus-per-engine "$TENSOR_MODEL_PARALLEL_SIZE" \
  --sglang-mem-fraction-static "$SGLANG_MEM_FRACTION_STATIC" \
  --sglang-tool-call-parser qwen3_coder \
  --sglang-reasoning-parser qwen3 \
  --sglang-enable-multimodal \
  --sglang-attention-backend flashinfer \
  --sglang-sampling-backend flashinfer \
  --sglang-context-length "$SGLANG_CONTEXT_LENGTH" \
  --attention-dropout 0.0 \
  --hidden-dropout 0.0 \
  --accumulate-allreduce-grads-in-fp32 \
  --attention-softmax-in-fp32 \
  --attention-backend flash \
  --optimizer-cpu-offload \
  --overlap-cpu-optimizer-d2h-h2d \
  --use-precision-aware-optimizer \
  "$@"
