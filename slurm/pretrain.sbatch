#!/bin/bash
#SBATCH --job-name=nv-pretrain
#SBATCH --account=ostadabbas
#SBATCH --partition=sharing
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=slurm/logs/%x_%j.out
#SBATCH --error=slurm/logs/%x_%j.err
# Vessel pretraining for one fold: bash slurm/submit.sh slurm/pretrain.sbatch 0 [RUN]
# RUN (default vessel_r3d18_160) is the folder under runs/; a rerun resumes from
# runs/<RUN>/fold<k>/last.pt.
set -euo pipefail

fold=${1:-}
run=${2:-vessel_r3d18_160}
case $fold in
  ''|*[!0-9]*) echo "refused: fold '$fold' is not a non-negative integer" >&2; exit 1 ;;
esac
if [[ ! $run =~ ^[A-Za-z0-9_][A-Za-z0-9_.-]*$ ]]; then
  echo "refused: run '$run' is not a plain folder name" >&2
  exit 1
fi
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
exec "$NV_STORE/envs/neuro-vision/bin/python" scripts/pretrain_vessels.py --fold "$fold" --run "$run"
