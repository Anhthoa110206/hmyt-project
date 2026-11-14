# Data

The raw STEW recordings are **not** stored in this repository.

- Dataset: STEW — Simultaneous Task EEG Workload (Lim et al., 2018)
- Source: IEEE DataPort, https://ieee-dataport.org/open-access/stew-simultaneous-task-eeg-workload-dataset (DOI 10.21227/44r8-ya50), licence CC BY 4.0, free IEEE account login required
- Downloaded: 2026-11-09 as `STEW Dataset.zip` (96 files `subXX_lo.txt` / `subXX_hi.txt` + `ratings.txt`)
- Put the zip here as `data/STEW Dataset.zip`, then run `python src/stew_pipeline.py --data "data/STEW Dataset.zip" --out results/pipeline`
- Participants are identified only by numbers (sub01–sub48); no personal identifiers.
