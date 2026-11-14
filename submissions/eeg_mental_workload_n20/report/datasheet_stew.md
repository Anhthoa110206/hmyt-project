# Datasheet — STEW (Simultaneous Task EEG Workload) as used by team N20

Source of facts: [IEEE DataPort dataset page](https://ieee-dataport.org/open-access/stew-simultaneous-task-eeg-workload-dataset) and the dataset article (Lim et al., 2018, DOI 10.1109/TNSRE.2018.2872924). Items marked *TODO* must be checked against the full article.

## Motivation
- Created to study EEG correlates of mental workload during multitasking (SIMKAP test) versus rest.
- Used here for model development and internal validation of a low/high workload classifier (Phase 1 of the project).

## Composition
- 48 participants; 2 recordings each: `subXX_lo.txt` (rest) and `subXX_hi.txt` (SIMKAP multitasking); 96 recordings in total.
- Each recording: 2.5 min, 14 EEG channels (AF3, F7, F3, FC5, T7, P7, O1, O2, P8, T8, FC6, F4, F8, AF4), 128 Hz; plain text, rows = samples, columns = channels.
- `ratings.txt`: subject number, rest rating, test rating (self-rated workload, 1–9). Ratings missing for subjects 5, 24, 42.
- Verified by the team: 96 files × 19,200 samples × 14 channels, 0 NaN (`results/pipeline/01_recordings_inspect.csv`).
- Population characteristics (age, sex, inclusion/exclusion): *TODO from Lim et al. 2018*.

## Collection and preprocessing
- Device: Emotiv EPOC (14-channel wireless headset), laboratory setting.
- Team preprocessing: mean removal; zero-phase Butterworth band-pass 1–40 Hz (order 4); 4-s epochs, 2-s step; rejection of epochs whose peak-to-peak amplitude exceeds 5× the channel median of that recording (`src/stew_pipeline.py`).
- Units: as distributed (assumed µV, Emotiv raw values with DC offset removed by filtering) — *TODO confirm in Lim et al. 2018*.

## Recommended uses
- Research and teaching on EEG workload classification; subject-independent evaluation is recommended because epochs from one person are strongly correlated.

## Limitations
- Not driving data; rest vs SIMKAP differ in eye/hand movement as well as workload.
- Small sample (48), single device, single site; no external validation set.
- Rest condition always vs one multitasking task — label reflects condition, not graded workload.

## Licence and identifiers
- Licence: Creative Commons Attribution 4.0 (CC BY 4.0). Dataset DOI: 10.21227/44r8-ya50. Access requires a (free) IEEE account login.
- Participants are identified only by numbers (sub01…sub48); no names or personal identifiers. The raw data are **not** committed to this repository (download from the link above).
