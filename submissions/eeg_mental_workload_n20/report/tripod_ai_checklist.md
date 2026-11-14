# TRIPOD+AI checklist — N20 (eeg_mental_workload)

Self-assessment of the current report (`report/kiem_tra_lan_2.md`, slides, code in `src/`).
Status: **Met** · **Partly** · **Not yet** · **NA** (not applicable, with reason).
Item list follows TRIPOD+AI (Collins et al., BMJ 2024;385:e078378).

| Item | Topic | Status | Where / note |
|---|---|---|---|
| 1 | Title | Met | Title in `README.md` names model development, population (healthy adults, STEW) and outcome (low vs high workload) |
| 2 | Abstract | Not yet | Abstract to be written for the final report (week 15) |
| 3a | Background | Met | Slides 2–8; report §A3 |
| 3b | Target population and purpose | Partly | Intended use (driver workload warning) stated on slide 10; intended users not yet described |
| 3c | Health inequalities | Not yet | Not discussed |
| 4 | Objectives | Met | Development + internal validation (report §C2) |
| 5a | Data sources | Met | STEW, IEEE DataPort, DOI 10.21227/44r8-ya50 (`report/datasheet_stew.md`) |
| 5b | Data dates | Partly | Original collection dates not reported by the dataset page; download date recorded in §C1 |
| 6a | Study setting | Met | Laboratory, single centre, Emotiv EPOC (datasheet) |
| 6b | Eligibility criteria | Partly | Only "48 participants" known from the dataset page; detailed criteria to be taken from Lim et al. 2018 |
| 6c | Treatments | NA | No treatment; healthy volunteers |
| 7 | Data preparation | Met | Filtering, epoching, artefact rejection (`src/stew_pipeline.py` l.48–54, l.111–124) |
| 8a | Outcome definition | Met | Experimental condition rest vs SIMKAP per 4-s epoch (§C2) |
| 8b | Outcome assessors | NA | Outcome is the experimental condition, not a subjective rating |
| 8c | Outcome blinding | NA | Condition set by protocol |
| 9a | Initial predictors | Met | Band powers chosen a priori from Chikhi et al. 2022 (theta/alpha/beta) |
| 9b | Predictor definitions | Met | `src/stew_pipeline.py` `epoch_features()` (l.131–152) |
| 9c | Predictor assessors | NA | Computed automatically |
| 10 | Sample size | Partly | n = 48 subjects / 6,198 epochs reported; no formal sample-size calculation (stated as limitation) |
| 11 | Missing data | Met | No NaN in 96 files (`results/pipeline/01_recordings_inspect.csv`); 906 epochs rejected as artefacts, counts per recording reported |
| 12a | Data use and partitioning | Met | GroupKFold 8 folds by subject |
| 12b | Predictor handling | Met | log10 absolute power, relative power, StandardScaler inside Pipeline |
| 12c | Model type and building | Met | Logistic regression (default C = 1, no tuning); comparison RF/SVM/KNN |
| 12d | Cluster heterogeneity | Partly | Fold-to-fold SD reported; per-subject performance not yet reported |
| 12e | Performance measures | Met | AUC, balanced accuracy, recall/precision High, Brier, calibration intercept/slope, 95% CI |
| 12f | Model updating | Not yet | No recalibration yet (planned, slope 0.56) |
| 12g | Prediction calculation | Met | Code in `src/` |
| 13 | Class imbalance | Met | Near-balanced (3,205 vs 2,993); no resampling used |
| 14 | Fairness | Not yet | Age/sex subgroups not analysed |
| 15 | Model output | Met | Probability of High; threshold 0.5 (not tuned) |
| 16 | Development vs evaluation differences | NA | Internal validation only (same dataset) |
| 17 | Ethical approval | Partly | Public de-identified data, CC BY 4.0; original study ethics to be cited from Lim et al. 2018 |
| 18a | Funding | Met | None (student project) |
| 18b | Conflicts of interest | Met | None |
| 18c | Protocol | Met | No separate protocol; methods fixed in slides week 7 |
| 18d | Registration | Met | Not registered |
| 18e | Data sharing | Met | Public dataset link in README |
| 18f | Code sharing | Met | `src/` in this repository |
| 19 | Patient and public involvement | Met | None |
| 20a | Participant flow | Partly | Epoch flow reported (7,104 → 6,198); participant flow diagram not yet drawn |
| 20b | Participant characteristics | Not yet | Age/sex not yet extracted from Lim et al. 2018 |
| 20c | Evaluation data comparison | NA | No separate evaluation dataset |
| 21 | Participants and events per analysis | Met | 48 subjects, 3,205 Low / 2,993 High epochs |
| 22 | Full model specification | Partly | Code available; coefficients not yet exported |
| 23a | Performance estimates | Met | `results/b1_results_with_ci.csv` |
| 23b | Cluster performance heterogeneity | Partly | Fold SD only |
| 24 | Model updating results | Not yet | — |
| 25 | Interpretation | Partly | Report §B3; to be expanded in final report |
| 26 | Limitations | Met | Report §B5, §C3 |
| 27a | Poor-quality input data | Partly | Artefact rejection rule described; guidance for deployment not yet written |
| 27b | User interaction and expertise | Not yet | — |
| 27c | Next steps | Met | Phase 2–3 plan (slide 15) |

**Count (52 items incl. sub-items):** Met 28 · Partly 11 · Not yet 7 · NA 6.
