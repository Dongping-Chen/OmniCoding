#!/usr/bin/env bash
#SBATCH --job-name=omnicoding-serve
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --output=slurm-%x-%j.out

set -euo pipefail

: "${MODEL_PATH:?Set MODEL_PATH to a model id or checkpoint path}"
: "${OMNICODING_ROOT:?Set OMNICODING_ROOT to this repository checkout}"

mkdir -p logs
cd "$OMNICODING_ROOT"
bash integrations/sglang/serve.sh "$@"
