"""Derive the 14 corrected targets from the Shi et al. 26-class masks (14..26 = aneurysms)."""

import argparse
import os
from collections.abc import Sequence
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import cast

import nibabel as nib
import numpy as np
import pandas as pd

from neurovision.data.labels import ANEURYSM_PRESENT, LOCATIONS
from neurovision.paths import store

FOLDERS = ("labelsTr_26classes_in_orig_space", "labelsTr_multiframe_cor_sag")
FIRST_ANEURYSM_CLASS = 14
SUFFIX = ".nii.gz"
# Mask counts recorded in data/manifests/labels.yaml.
EXPECTED_COUNTS = {FOLDERS[0]: 4317, FOLDERS[1]: 27}


def locations_present(path: Path) -> list[int]:
    data = np.asanyarray(cast(nib.Nifti1Image, nib.load(path)).dataobj)
    if data.dtype.kind not in "iu":
        raise ValueError(f"{path.name}: mask is not integer-valued")
    classes = set(np.unique(data).tolist())
    return [int(FIRST_ANEURYSM_CLASS + k in classes) for k in range(len(LOCATIONS))]


def main(argv: Sequence[str] | None = None) -> None:
    argparse.ArgumentParser(description=__doc__).parse_args(argv)
    root = store() / "labels"
    files: dict[str, Path] = {}
    for folder in FOLDERS:
        found = sorted((root / "shi2026-masks" / folder).glob(f"*{SUFFIX}"))
        if not found:
            raise SystemExit(f"no masks in {root / 'shi2026-masks' / folder}")
        print(f"{folder}: {len(found)} masks")
        if len(found) != EXPECTED_COUNTS[folder]:
            raise SystemExit(f"{folder}: expected {EXPECTED_COUNTS[folder]} masks")
        for path in found:
            uid = path.name.removesuffix(SUFFIX)
            if uid in files:
                raise SystemExit("a series has masks in both folders")
            files[uid] = path
    ids = sorted(files)
    with ProcessPoolExecutor(int(os.environ.get("SLURM_CPUS_PER_TASK", "1"))) as pool:
        rows = list(pool.map(locations_present, [files[i] for i in ids], chunksize=8))
    out = pd.DataFrame(rows, columns=list(LOCATIONS), index=pd.Index(ids))
    out[ANEURYSM_PRESENT] = out.max(axis=1)
    out.index.name = "SeriesInstanceUID"
    out.to_csv(root / "corrected.csv")
    print(f"wrote {len(out)} series, {int(out[ANEURYSM_PRESENT].sum())} with an aneurysm")


if __name__ == "__main__":
    main()
