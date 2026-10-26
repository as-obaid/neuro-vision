#!/usr/bin/env bash
# Download the RSNA Intracranial Aneurysm Detection data into $NV_STORE/raw and unzip it.
# Run on the transfer node (compute nodes have no internet), inside tmux:
#   NV_STORE=/scratch/abdulsalam.m/neuro-vision-store bash scripts/download.sh
# Needs the competition rules accepted on Kaggle and a token in ~/.kaggle/kaggle.json.
# Expect ~1.00M DICOM files, 311 GB extracted. The zip is kept until the audit passes.
set -euo pipefail

COMP=rsna-intracranial-aneurysm-detection
: "${NV_STORE:?set NV_STORE to the data store}"
RAW="$NV_STORE/raw"
umask 077
mkdir -p "$RAW"
cd "$RAW"

if [[ -f .complete ]]; then
  echo "already complete: $(cat .complete)"
  exit 0
fi

echo "$(date -Is) start on $(hostname)"
kaggle competitions download -c "$COMP" -p "$RAW"
echo "$(date -Is) downloaded $(du -h "$COMP.zip" | cut -f1)"

unzip -q -o "$COMP.zip"
n=$(find "$RAW" -name '*.dcm' | wc -l)
echo "$(date -Is) extracted $n DICOM files, $(du -sh --exclude="$COMP.zip" "$RAW" | cut -f1)"

printf '%s dcm_files=%s\n' "$(date -Is)" "$n" > .complete
echo "done; zip kept at $RAW/$COMP.zip"
