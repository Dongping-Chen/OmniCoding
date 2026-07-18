#!/usr/bin/env bash
#SBATCH --job-name=omnicoding-sft
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --output=slurm-%x-%j.out

set -euo pipefail

: "${OMNICODING_ROOT:?Set OMNICODING_ROOT to this repository checkout}"
mkdir -p "$OMNICODING_ROOT/logs"
cd "$OMNICODING_ROOT"

bash integrations/ms_swift/train_sft.sh "$@"
