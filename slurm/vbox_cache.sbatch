#!/bin/bash
#SBATCH --job-name=nv-vbox-cache
#SBATCH --account=ostadabbas
#SBATCH --partition=short
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=01:00:00
#SBATCH --output=slurm/logs/%x_%j.out
#SBATCH --error=slurm/logs/%x_%j.err
# Full version, stage 2: the 0.5 mm vessel-box cache from the DICOM, centred by one fold's boxes:
# bash slurm/submit.sh slurm/vbox_cache.sbatch 0
# The re-centred 1 mm head crop (cache vc1mm-160) instead:
# bash slurm/submit.sh slurm/vbox_cache.sbatch 0 --spacing 1.0 --size 160
# A rerun resumes (the cache's index.csv keeps the finished series).
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
shift
exec "$NV_STORE/envs/neuro-vision/bin/python" scripts/build_vessel_box_cache.py --fold "$fold" "$@"
