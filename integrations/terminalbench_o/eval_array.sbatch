#!/usr/bin/env bash
#SBATCH --job-name=terminalbench-o-eval
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64g
#SBATCH --time=4:00:00
#SBATCH --array=0-49%20

set -euo pipefail

if [[ -n "${TERMINALBENCH_O_DIR:-}" ]]; then
  SCRIPT_DIR="$(cd -- "$TERMINALBENCH_O_DIR" && pwd)"
elif [[
  -n "${SLURM_SUBMIT_DIR:-}" &&
  -f "${SLURM_SUBMIT_DIR}/integrations/terminalbench_o/task_ids.txt"
]]; then
  SCRIPT_DIR="${SLURM_SUBMIT_DIR}/integrations/terminalbench_o"
else
  SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
fi
TASK_LIST="${TASK_LIST:-${SCRIPT_DIR}/task_ids.txt}"
BENCHMARK_ROOT="${BENCHMARK_ROOT:?set BENCHMARK_ROOT to the private extracted benchmark}"
RUNS_DIR="${RUNS_DIR:?set RUNS_DIR to the existing case-run directory}"
GRADER_PYTHON="${GRADER_PYTHON:-python}"
GRADE_NAME="${GRADE_NAME:-terminalbench_o_v1}"
TIMEOUT="${TIMEOUT:-900}"
ALLOW_PROXY="${ALLOW_PROXY:-0}"

if [[ -z "${SLURM_ARRAY_TASK_ID:-}" ]]; then
  echo "SLURM_ARRAY_TASK_ID is required; submit this script with sbatch." >&2
  exit 2
fi

TASK="$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$TASK_LIST")"
if [[ -z "$TASK" ]]; then
  echo "No task for array index ${SLURM_ARRAY_TASK_ID} in ${TASK_LIST}" >&2
  exit 2
fi

command=(
  "$GRADER_PYTHON"
  "${SCRIPT_DIR}/evaluate.py"
  --task "$TASK"
  --benchmark-root "$BENCHMARK_ROOT"
  --runs-dir "$RUNS_DIR"
  --grade-name "$GRADE_NAME"
  --grader-python "$GRADER_PYTHON"
  --timeout "$TIMEOUT"
)
if [[ "$ALLOW_PROXY" == "1" ]]; then
  command+=(--allow-proxy)
fi

echo "[terminalbench-o] host=$(hostname) array_task_id=${SLURM_ARRAY_TASK_ID} task=${TASK}"
printf '+'
printf ' %q' "${command[@]}"
printf '\n'
exec "${command[@]}"
