"""Derive the 14 corrected targets from the Shi et al. 26-class masks (14..26 = aneurysms)."""

import argparse
import os
from collections.abc import Sequence
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd

from neurovision.data.labels import TARGETS
from neurovision.paths import store

FOLDERS = ("labelsTr_26classes_in_orig_space", "labelsTr_multiframe_cor_sag")
FIRST_ANEURYSM_CLASS = 14
SUFFIX = ".nii.gz"


def locations_present(path: Path) -> list[int]:
    counts = np.bincount(np.asanyarray(nib.load(path).dataobj).ravel().astype(np.intp))
    counts = np.pad(counts, (0, max(0, FIRST_ANEURYSM_CLASS + 13 - len(counts))))
    return [int(counts[FIRST_ANEURYSM_CLASS + k] > 0) for k in range(13)]


def main(argv: Sequence[str] | None = None) -> None:
    argparse.ArgumentParser(description=__doc__).parse_args(argv)
    root = store() / "labels"
    files: dict[str, Path] = {}
    for folder in FOLDERS:
        found = sorted((root / "shi2026-masks" / folder).glob(f"*{SUFFIX}"))
        if not found:
            raise SystemExit(f"no masks in {root / 'shi2026-masks' / folder}")
        print(f"{folder}: {len(found)} masks")
        for path in found:
            uid = path.name.removesuffix(SUFFIX)
            if uid in files:
                raise SystemExit("a series has masks in both folders")
            files[uid] = path
    ids = sorted(files)
    with ProcessPoolExecutor(int(os.environ.get("SLURM_CPUS_PER_TASK", "1"))) as pool:
        rows = list(pool.map(locations_present, [files[i] for i in ids], chunksize=8))
    out = pd.DataFrame(rows, columns=list(TARGETS[:13]), index=pd.Index(ids))
    out[TARGETS[13]] = out.max(axis=1)
    out.index.name = "SeriesInstanceUID"
    out.to_csv(root / "corrected.csv")
    print(f"wrote {len(out)} series, {int(out[TARGETS[13]].sum())} with an aneurysm")


if __name__ == "__main__":
    main()
