#!/usr/bin/env bash
#SBATCH --job-name=omnicoding-rl-train
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=64
#SBATCH --mem=512G
#SBATCH --time=48:00:00
#SBATCH --output=slurm-%x-%j.out

set -euo pipefail

: "${OMNICODING_ROOT:?Set OMNICODING_ROOT to this repository checkout}"
: "${RELAX_ROOT:?Set RELAX_ROOT to the patched Relax checkout}"
: "${MEGATRON_ROOT:?Set MEGATRON_ROOT to the compatible Megatron-LM checkout}"
: "${PROMPT_PARQUET:?Set PROMPT_PARQUET to the RL prompt parquet}"
: "${SAVE_DIR:?Set SAVE_DIR to a writable checkpoint directory}"

cd "$OMNICODING_ROOT"
bash recipes/rl_27b_gspo.sh "$@"
