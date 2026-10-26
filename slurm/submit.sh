#!/bin/bash
# Submit a job: bash slurm/submit.sh slurm/NAME.sbatch [ARGS...]   (from the repo root)
# Refuses a dirty tree and a job whose time limit reaches a /scratch purge window
# (slurm/purge_dates). Passes --deadline = the next window's start, so a job that waits in the
# queue cannot start into a window either.
set -euo pipefail

refuse() {
  echo "refused: $*" >&2
  exit 1
}

[ "$#" -ge 1 ] || refuse "usage: bash slurm/submit.sh slurm/NAME.sbatch [ARGS...]"
batch=$1
shift

[ -z "$(git rev-parse --show-prefix 2>/dev/null || echo x)" ] || refuse "run from the repo root"
case $batch in
  *..*) refuse "$batch is not slurm/*.sbatch" ;;
  slurm/*.sbatch) ;;
  *) refuse "$batch is not slurm/*.sbatch" ;;
esac
[ -f "$batch" ] || refuse "$batch is not slurm/*.sbatch (no such file)"

dirty=$(git status --porcelain -- src scripts configs slurm pyproject.toml uv.lock data/manifests)
[ -z "$dirty" ] || refuse "uncommitted changes in the code paths:
$dirty"

limit=$(sed -n 's/^#SBATCH --time=\([^[:space:]]*\).*/\1/p' "$batch" | head -n 1)
[ -n "$limit" ] || refuse "$batch has no '#SBATCH --time=' line"

deadline=$(python3 slurm/purge_check.py --now "$(date +%s)" --limit "$limit") || exit 1

mkdir -p slurm/logs
if [ -n "$deadline" ]; then
  sbatch --parsable "--deadline=$deadline" "$batch" "$@"
else
  sbatch --parsable "$batch" "$@"
fi
