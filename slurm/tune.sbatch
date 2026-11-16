#!/bin/bash
#SBATCH --job-name=nv-tune
#SBATCH --account=ostadabbas
#SBATCH --partition=sharing
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=slurm/logs/%x_%j.out
#SBATCH --error=slurm/logs/%x_%j.err
# One 10-epoch tuning run on an L40S: bash slurm/submit.sh slurm/tune.sbatch configs/tune/t_r0.yaml 0
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"
exec bash slurm/train.sbatch "$@"
