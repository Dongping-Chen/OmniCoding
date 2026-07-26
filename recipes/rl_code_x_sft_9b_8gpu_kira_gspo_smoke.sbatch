#!/usr/bin/env bash
#SBATCH --job-name=code-x-sft-kira-gspo
#SBATCH --partition=h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=96
#SBATCH --mem=850G
#SBATCH --gres=gpu:h100:8
#SBATCH --time=1-00:00:00
#SBATCH --output=outputs/slurm/%x-%j.out

# Default eight-GPU Code-X-SFT Kira GSPO run. W&B remains disabled until the
# smoke metrics and reward distribution have been reviewed.
set -euo pipefail

repo_root="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "${repo_root}"

NUM_ROLLOUT="${NUM_ROLLOUT:-10}" \
OUTPUT_ROOT="${OUTPUT_ROOT:-${repo_root}/outputs/relax-code-x-sft-kira-gspo/${SLURM_JOB_ID}-8gpu}" \
  bash recipes/rl_code_x_sft_9b_kira_gspo.sh
