#!/bin/bash
#SBATCH --job-name=nv-train
#SBATCH --account=ostadabbas
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=slurm/logs/%x_%j.out
#SBATCH --error=slurm/logs/%x_%j.err
# Train one CNN fold: bash slurm/submit.sh slurm/train.sbatch configs/base/b2.yaml 0
# A rerun resumes from runs/<config>/fold<k>/last.pt.
set -euo pipefail

config=${1:-}
fold=${2:-}
case $config in
  *..*) echo "refused: '$config' is not configs/**/*.yaml" >&2; exit 1 ;;
  configs/*.yaml) ;;
  *) echo "refused: '$config' is not configs/**/*.yaml" >&2; exit 1 ;;
esac
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
exec "$NV_STORE/envs/neuro-vision/bin/python" scripts/train.py "$config" --fold "$fold"
