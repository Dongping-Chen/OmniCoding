#!/usr/bin/env bash
#SBATCH --job-name=omnicoding-rl-worker
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=12:00:00

set -uo pipefail

: "${OMNICODING_PYTHON:?OMNICODING_PYTHON is required}"
: "${REQUEST_FILES:?REQUEST_FILES is required}"
: "${RESULT_FILES:?RESULT_FILES is required}"

IFS=';' read -ra requests <<< "$REQUEST_FILES"
IFS=';' read -ra results <<< "$RESULT_FILES"
if [[ "${#requests[@]}" -ne "${#results[@]}" ]]; then
  echo "request/result file counts differ" >&2
  exit 2
fi

pids=()
for index in "${!requests[@]}"; do
  "$OMNICODING_PYTHON" -m omnicoding.rl.run_trajectory \
    --request "${requests[$index]}" \
    --result "${results[$index]}" \
    >"${results[$index]}.worker.log" 2>&1 &
  pids+=("$!")
done

for pid in "${pids[@]}"; do
  wait "$pid" || true
done
