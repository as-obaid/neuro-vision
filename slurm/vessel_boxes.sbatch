#!/bin/bash
#SBATCH --job-name=nv-vbox-pred
#SBATCH --account=ostadabbas
#SBATCH --partition=sharing
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=slurm/logs/%x_%j.out
#SBATCH --error=slurm/logs/%x_%j.err
# Full version, stage 1: the step-1 U-Net's vessel box for every series, one fold's model:
# bash slurm/submit.sh slurm/vessel_boxes.sbatch 0
# A rerun resumes (labels/vessel-boxes-fold<k>.csv keeps the finished series).
set -euo pipefail

fold=${1:-}
case $fold in
  ''|*[!0-9]*) echo "refused: fold '$fold' is not a non-negative integer" >&2; exit 1 ;;
esac
cd "${SLURM_SUBMIT_DIR:-.}"
if [ -f .env ]; then
  set -a
  # shellcheck disable=SC1091
  . ./.env
  set +a
fi
: "${NV_STORE:?NV_STORE is not set}"
export PYTHONPATH="$PWD/src"
nvidia-smi -L
exec "$NV_STORE/envs/neuro-vision/bin/python" scripts/predict_vessel_boxes.py --fold "$fold"
