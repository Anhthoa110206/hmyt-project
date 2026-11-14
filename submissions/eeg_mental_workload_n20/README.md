# EEG-based classification of low vs high mental workload (team N20)

- **Topic:** EEG-Based Classification of Low vs. High Mental Workload Using Welch PSD Band-Power Features and Machine Learning with Subject-Independent Evaluation (model development + internal validation, healthy adults, STEW dataset)
- **Team:** N20 — members 202414870 (leader), 202414866, 202414852
- **Main model:** Logistic regression on 84 log absolute + relative band-power features (theta 4–8, alpha 8–12, beta 12–30 Hz; 14 channels)
- **Key result (GroupKFold by subject, 48 subjects, 6,198 epochs):** AUC 0.888 (95% CI 0.850–0.922), balanced accuracy 0.801, recall (High) 0.822, Brier 0.143, calibration slope 0.56 — see `results/b1_results_with_ci.csv`

## Folder structure

- `report/kiem_tra_lan_2.md` — Progress Test 2
- `report/datasheet_stew.md` — datasheet of the dataset
- `report/tripod_ai_checklist.md` — TRIPOD+AI checklist
- `src/stew_pipeline.py` — raw EEG → features → models (also packaged as `src/STEW_pipeline_N20.ipynb` for Colab)
- `src/evaluate_extended.py` — AUC, Brier, calibration, bootstrap 95% CI, ablation
- `results/pipeline/` — outputs of `stew_pipeline.py`
- `results/` — outputs of `evaluate_extended.py`

## Reproduce

1. Download STEW from IEEE DataPort (free IEEE account): https://ieee-dataport.org/open-access/stew-simultaneous-task-eeg-workload-dataset (DOI 10.21227/44r8-ya50). The data are not committed here.
2. Python ≥ 3.10 with `numpy scipy pandas scikit-learn matplotlib`.
3. `python src/stew_pipeline.py --data "STEW Dataset.zip" --out results/pipeline`
   (or open `src/STEW_pipeline_N20.ipynb` in Google Colab and Run all).
4. `python src/evaluate_extended.py --features results/pipeline/02_features_all_epochs.csv --out results`
   (`02_features_all_epochs.csv`, 10 MB, is regenerated in step 3 and not committed; SHA-256 of the team's copy: `8eaeecc5674a4997f8c56f268b7a12c215d48e8bc22a86b46816570ce24b5314`).
