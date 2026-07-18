#!/usr/bin/env bash
#SBATCH --job-name=omnicoding-item
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=slurm-%x-%j.out

set -euo pipefail

: "${INPUT:?INPUT is required}"
: "${DATASET_ROOT:?DATASET_ROOT is required}"
: "${OUT:?OUT is required}"
: "${SOURCE_INDEX:?SOURCE_INDEX is required}"
: "${BENCH:?BENCH is required}"

mkdir -p logs "$OUT"

omnicoding-bench-kira \
  --bench "$BENCH" \
  --input_file "$INPUT" \
  --dataset_root "$DATASET_ROOT" \
  --output_dir "$OUT" \
  --source_indices "$SOURCE_INDEX" \
  --model_name "${KIRA_MODEL_NAME:-openai/local-model}" \
  --reasoning_effort "${KIRA_REASONING_EFFORT:-high}" \
  --effort_strategy "${KIRA_EFFORT_STRATEGY:-fixed}" \
  --attempt "${KIRA_ATTEMPT:-1}" \
  --prev_effort "${KIRA_PREV_EFFORT:-}" \
  --step_limit "${KIRA_STEP_LIMIT:-120}" \
  --request_timeout "${KIRA_REQUEST_TIMEOUT:-1500}" \
  --block_timeout "${KIRA_BLOCK_TIMEOUT:-1200}" \
  --results_filename "results.${SOURCE_INDEX}.json"
