#!/bin/bash
#SBATCH --job-name=nv-preprocess
#SBATCH --account=ostadabbas
#SBATCH --partition=short
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=10:00:00
#SBATCH --output=slurm/logs/%x_%j.out
#SBATCH --error=slurm/logs/%x_%j.err
# Cache all series as cubes (128 by default): bash slurm/submit.sh slurm/preprocess.sbatch [--size N]
# 16 workers, 8 GB each.
set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-.}"
if [ -f .env ]; then
  set -a
  # shellcheck disable=SC1091
  . ./.env
  set +a
fi
: "${NV_STORE:?NV_STORE is not set}"
export PYTHONPATH="$PWD/src"
exec "$NV_STORE/envs/neuro-vision/bin/python" scripts/preprocess.py "$@"
