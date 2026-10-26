# Neuro-Vision

A 3D convolutional network that reads one brain scan and answers two questions: is an intracranial
aneurysm present, and on which of 13 arteries? AAI 6640 Applied Deep Learning term project,
Northeastern University.

| | |
|:--|:--|
| Input | One 3D DICOM series: CTA, MRA, T1 post-contrast or T2 MRI |
| Output | 14 probabilities: Aneurysm Present + 13 locations |
| Data | [RSNA Intracranial Aneurysm Detection](https://www.kaggle.com/competitions/rsna-intracranial-aneurysm-detection) (Kaggle, 2025) |
| Score | ½ · (Aneurysm Present AUC + mean AUC over the 13 locations) |
| Targets | 5-fold cross-validation ≥0.80; Kaggle private score ≥0.78 |
| Status | Stage 1 setup |

## Setup

```bash
uv sync
uv run pytest
```

The data is not in this repo: the Kaggle rules forbid sharing it. `scripts/download.sh` fetches it
into `$NV_STORE` after you accept the rules on Kaggle.

Author: Muhammad Obaidullah.
