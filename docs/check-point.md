# Neuro-Vision: midterm checkpoint

A 3D CNN reads one brain scan and flags an intracranial aneurysm and its artery. Splits are fixed and leakage-checked; metadata alone scores 0.640, which the image model (B2, not yet run) must beat.

Full report: https://forge.coreweave.com/muhammadobaidullah98-northeastern-university/neuro-vision/reports/Neuro-Vision:-midterm-checkpoint--VmlldzoxODIzOTI4Nw==

## 1 Problem and data

Task: from one scan, return 14 probabilities. One is "aneurysm present"; 13 are arteries. Score is the mean of the Aneurysm Present AUC and the mean of the 13 location AUCs.

| Item | Value | Source |
|:--|:--|:--|
| Training series | 4,348 | `results/audit.json` |
| CTA / MRA / MRI T2 / MRI T1post | 1,808 / 1,252 / 983 / 305 | `results/audit.json` |
| Positive series | 1,863 (42.8%) | `results/audit.json` |
| Age | 18-89, median 60 | `results/audit.json` |
| Sex | 3,005 female, 1,343 male | `results/audit.json` |
| Missing values | none | `results/audit.json` |
| Kaggle hidden test | nearly 2,500 series | `results/audit.json` |

Labels:

- Corrected labels are derived from the 26-class masks of Shi et al. (CC BY-NC 4.0).
- 4 series changed: 7 cells, 2 left-right swaps, no Aneurysm Present change.
- Source: `data/manifests/labels.yaml`.

## 2 Splits and leakage

Headers (second header scan; step 5):

| Header | Finding |
|:--|:--|
| PatientID, StudyInstanceUID | Unique per series. Kaggle's data page says they "do not identify anything". |
| FrameOfReferenceUID | Links 1,511 series into 672 same-session groups (largest 5; sex and age agree in every group). Empty for 532. |
| StudyDate | Too coarse: one date covers 316 series. |

Folds (`data/manifests/folds.yaml`):

| Item | Value |
|:--|:--|
| Method | 5 folds, StratifiedGroupKFold by Aneurysm Present x modality |
| Group key | FrameOfReferenceUID (3,509 groups) |
| Seed, labels | 0, corrected |
| Fold size | 869-870 series |
| Positive rate per fold | 42.8-42.9% |
| Thinnest cells (positives per fold) | Left ACA 8-11; Right ACA 6-16; Left infraclinoid ICA 8-28 |

## 3 Stage 1: baselines

| Baseline | Inputs | Score | Aneurysm Present AUC | Mean location AUC |
|:--|:--|:--|:--|:--|
| B0 | Constant prior | 0.500 | 0.500 | 0.500 |
| B1 | Gradient boosting on age, sex, scan type | 0.640 | 0.672 | 0.609 |
| B2 | 3D ResNet-18 | not run | not run | not run |

Source: `results/b0.json`, `results/b1.json`.

- Metadata alone beats chance.
- B2 must beat 0.640 to show image signal.

## 4 Stage 2: focused experiments

## 5 Behaviour analysis

## 6 Stage 3 direction

## 7 Limitations

| Limit | Effect |
|:--|:--|
| No patient key | The same patient across different visits cannot be linked. |
| 532 series lack FrameOfReferenceUID | Each is its own group, so leakage is possible among them. |
| Rare locations | Per-location AUCs are noisy (see thin cells, section 2). |
| Scan type confounds the label | B1 gains from scan type, so a high score does not prove image signal. |
