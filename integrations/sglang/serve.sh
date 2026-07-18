#!/usr/bin/env bash
set -euo pipefail

: "${MODEL_PATH:?Set MODEL_PATH to a model id or checkpoint path}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-$MODEL_PATH}"

python -m sglang.launch_server \
  --model-path "$MODEL_PATH" \
  --served-model-name "$SERVED_MODEL_NAME" \
  --host "${HOST:-127.0.0.1}" \
  --port "${PORT:-8080}" \
  --tp-size "${TP_SIZE:-1}" \
  --trust-remote-code \
  "$@"
