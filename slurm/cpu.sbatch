#!/bin/bash
#SBATCH --job-name=nv-cpu
#SBATCH --account=ostadabbas
#SBATCH --partition=short
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=slurm/logs/%x_%j.out
#SBATCH --error=slurm/logs/%x_%j.err
# Run one scripts/*.py on a CPU node: bash slurm/submit.sh slurm/cpu.sbatch scripts/NAME.py [ARGS...]
set -euo pipefail

case ${1:-} in
  *..*) echo "refused: ${1:-} is not scripts/*.py" >&2; exit 1 ;;
  scripts/*.py) ;;
  *) echo "refused: '${1:-}' is not scripts/*.py" >&2; exit 1 ;;
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
exec "$NV_STORE/envs/neuro-vision/bin/python" "$@"
