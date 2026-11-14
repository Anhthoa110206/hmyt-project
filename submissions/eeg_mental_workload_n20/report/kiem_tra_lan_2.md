# Progress Test 2 — n20 — eeg_mental_workload

- Fork: https://github.com/Anhthoa110206/hmyt-project · Branch: gd5-proposed-model
- Members (student ID or alias) and roles:
  - 202414870 — team leader: pipeline, experiments, PROBAST+AI, FAIR, TRIPOD+AI
  - 202414866 — literature search, literature matrix, citation check
  - 202414852 — timeline, risk register, AI-use declaration, literature summaries

Topic: EEG-based classification of low vs high mental workload (rest vs SIMKAP multitasking) from Welch band-power features, with subject-independent evaluation. Dataset: STEW (48 healthy adults, 14-channel EEG). Input X = band-power features of one 4-s EEG epoch; output Y = 0 (low, rest) / 1 (high, SIMKAP). Error costs are asymmetric: a false negative (missed high-workload episode) is worse than a false positive (unnecessary warning), so recall for the High class is the primary threshold-dependent metric, always reported with precision and balanced accuracy.

## A. Literature review and research gaps

### A1. Search strategy
| Source (PubMed, Scopus, IEEE Xplore, Google Scholar…) | Keywords / query | Year range | No. of hits | No. kept |
|---|---|---|---|---|
| Web search (title queries, TV-A) | STEW Lim 2018; Safari 2024 brain connectivity | 2018–2024 | 20 (10 per query) | 2 (#4, #10) |
| Web search (title queries, TV-A) | Barua 2020; Di Flumeri 2018; Zhou 2022 review; Demirezen 2024 | 2018–2024 | 40 (10 per query) | 6 (#6, #7, #8, #11, #12, #13) |
| Backward citation search | Reference lists of Chikhi et al. 2022 and Atici-Ulusu et al. 2024 | 2009–2024 | Not applicable — hand search of 2 reference lists | 5 (#5–#9 in A2) |
| IEEE DataPort | `STEW simultaneous task EEG workload` | 2018 | 1 | 1 (#4, dataset article) |

Full search log (date, query, hits, kept, reasons for exclusion): team document "hmtyt w10", section 2 (exported as `report/literature_review_TV-A.md`). #1–#3 were provided by the instructor/team at the start; #5, #9 from the reference lists below.

Inclusion / exclusion criteria:
- Include: EEG recorded; contrast between low and high workload (or workload levels); spectral/band-power features or a machine-learning classifier; peer-reviewed journal or conference paper; English full text.
- Exclude: no EEG; clinical/patient population; no workload manipulation; abstract only.

### A2. Literature matrix (≥ 10 papers, preferably from the last 3–5 years)
| # | Paper (authors, year) | DOI / link | Dataset (n, source) | Method | Main result (metric, value) | External validation? (yes / no) | Main limitation |
|---|---|---|---|---|---|---|---|
| 1 | Chikhi, Matton, Blanchet, 2022 | 10.1111/psyp.14009 | 24 studies, 723 participants (meta-analysis) | Random-effects meta-analysis of theta/alpha/beta power, high vs low workload | Theta g = 0.68 (95% CI 0.41–0.95); alpha g = −0.25; beta g = 0.50 | Not applicable — meta-analysis, no model | Heterogeneity (I² 64% for theta) and publication bias; after trim-and-fill alpha and beta effects not significant |
| 2 | Atici-Ulusu, Taskapilioglu, Gunduz, 2024 | 10.1016/j.trf.2024.04.004 | 39 drivers, real traffic, Emotiv Epoc X 14 ch, 128 Hz | Band-power indexes (frontal theta / parietal alpha, frontal midline theta, (α+θ)/β); Kruskal-Wallis tests | Indexes differ between road sections (p < 0.001); correlation Index 1–Index 3 r = 0.678 | No (no predictive model) | Statistical comparison only, no classifier; data on request only |
| 3 | Walther, Viehweg, Haueisen, Mäder, 2023 | 10.3389/fninf.2023.1067095 | DEAP (32 subjects), seizure dataset, in-house entrainment dataset | Comparison of RNN/LSTM/GRU/TCN/Transformer/ELM on raw EEG | Best DEAP accuracy cited 72.9% (TCN); models overfit and stay below hand-crafted-feature methods on DEAP | No (intra-subject evaluation) | Intra-subject splits only; not a workload task |
| 4 | Lim, Sourina, Wang, 2018 | 10.1109/TNSRE.2018.2872924 | STEW: 48 subjects, Emotiv EPOC 14 ch, 128 Hz, rest vs SIMKAP | PSD features (delta–beta, 14 ch); support vector regression, 3 workload levels | Accuracy 69% on held-out subjects (from abstract; full text not accessed) | No (held-out subjects of the same dataset) | Single task (SIMKAP) and single device; full text not yet checked by the team |
| 5 | Holm et al., 2009 | 10.1100/tsw.2009.83 | Laboratory cognitive tasks of varying difficulty (abstract only) | Spectral EEG; frontal theta / parietal alpha brain-load index | Theta and alpha power change with task difficulty (abstract) | No | Older study; abstract only, sample size not checked |
| 6 | Di Flumeri et al., 2018 | 10.3389/fnhum.2018.00509 | 20 male students, real driving (Bologna), 12 EEG channels, 256 Hz | Frontal theta + parietal alpha (IAF-based), asSWLDA classifier, Easy vs Hard road | AUC 0.744 ± 0.13 (normal hour), 0.727 ± 0.06 (rush hour) | No | Small all-male sample; classifier calibrated per subject |
| 7 | Barua, Ahmed, Begum, 2020 | 10.3390/brainsci10080526 | 66 male drivers, VTI driving simulator, n-back secondary task; EEG + EOG/ECG/GSR + vehicle signals | 323 features, SFFS/RF feature selection → 42; k-NN, SVM, RF; 70/30 split, 5-fold CV on training set | RF F1 = 0.75 and 0.80 | No | Split of observations, not explicitly by subject; contextual features not fully understood (authors) |
| 8 | Zhou, Huang, Xu, Wang, Wu, Zhang, 2022 | 10.1109/TCDS.2021.3090217 | Review of EEG workload studies | Survey of preprocessing, features, classifiers, evaluation, deep learning | Summarises methods and open problems (cross-subject/cross-task generalisation) | Not applicable — review | Narrative review, no quantitative synthesis |
| 9 | Aricò et al., 2016 | 10.3389/fnhum.2016.00539 | 12 air-traffic-controller students (ENAC), realistic ATC simulation | Theta/alpha EEG workload index (passive BCI) triggering adaptive automation online | EEG workload index used online to trigger adaptive automation in a realistic ATC simulation (feasibility of passive BCI) | No | Very small sample; trained ATC students, not drivers |
| 10 | Safari, Shalbaf, Bagherzadeh, Shalbaf, 2024 | 10.1038/s41598-024-59652-w | STEW, 48 subjects | Effective connectivity (direct directed transfer function), hierarchical feature selection; SVM, LDA, RF, DT | SVM + forward selection: accuracy 89.53% ± 1.36% (7-fold CV) | No | Validation unit of the 7-fold CV and timing of feature selection to be checked (A4) |
| 11 | Islam et al., 2020 | 10.3390/brainsci10080551 | 20 male drivers, real driving (same study as #6); EEG + vehicle signals | Mutual-information features; LR, MLP, SVM, RF; 10-fold CV and leave-one-subject-out | Event classification accuracy 94% (MI features); for workload classification EEG features beat MI features | Partly (leave-one-subject-out reported) | Small all-male sample; MI features weaker for workload |
| 12 | Zhou et al., 2022 | 10.1109/TNSRE.2022.3140456 | Private EEG dataset: working-memory and mental-addition tasks | Four domain-adaptation methods + SVM, leave-one-task-out | +3% to 8% mean accuracy vs no transfer; TJM best | Yes (other task, same subjects) | Private data; two lab tasks only |
| 13 | Demirezen, Taşkaya Temizel, Brouwer, 2024 | 10.3389/fnrgo.2024.1346794 | Systematic review of EEG workload ML studies (Scopus, WoS, PubMed, ACM DL) | Reproducibility guideline (CRISP-DM based) applied to the literature | Only 3 studies with own recordings shared raw data on request and only 2 shared processing code | Not applicable — review | Review of reporting, not of model performance |

### A3. Research gaps
| ID | Gap (1–2 sentences) | Evidence (# in A2) | How your project addresses it |
|---|---|---|---|
| G1 | High STEW accuracies are reported with k-fold CV where the split unit is not clearly the subject; generalisation to unseen people is unknown, and code/data are rarely shared. | #10 (89.53%, 7-fold CV); #7 (70/30 split of observations); #13 (only 2 studies shared code) | GroupKFold by subject; every fold asserts no subject overlap (`src/stew_pipeline.py` l.181–184) |
| G2 | Results are rarely compared with a chance-level model and rarely report recall for the high-workload class or calibration. | #10 (accuracy only in abstract) | Dummy baseline; recall (High) as primary metric; Brier and calibration slope with 95% CI (`results/b1_results_with_ci.csv`) |
| G3 | The most sensitive physiological marker (frontal theta) is established statistically but seldom tested as a classifier on unseen subjects. | #1 (theta g = 0.68); #2 (statistics only, ML suggested as future work) | Ablation by band: theta-only vs without-theta (`results/b2_ablation.csv`) |
| G4 | Real-driving EEG datasets are small and not openly available. | #2 (data on request); #6 | Use public STEW for Phase 1; state limited transfer to driving; plan Phase 2 recording |

### A4. PROBAST+AI appraisal of the most important paper in A2
Paper #10 (Safari et al., 2024) · Why this paper: it uses the same dataset (STEW) and is the main performance benchmark on the slides.
| Domain | Risk of bias (low / high / unclear) | Justification (quote or page number in the paper) |
|---|---|---|
| Participants and data sources | Low | Public STEW dataset, 48 healthy subjects (abstract). Page not verified: full text not yet accessed by the team |
| Predictors | Low | Connectivity features (dDTF) computed with one fixed procedure for all subjects (abstract). Page not verified: full text not yet accessed by the team |
| Outcome | Unclear | Abstract reports "workload levels"; whether labels are rest/SIMKAP or derived from self-ratings must be checked. Page not verified: full text not yet accessed by the team |
| Analysis | High | 7-fold CV with hierarchical feature selection (abstract); no external validation; n = 48. If segments of one subject fall in both train and test, or feature selection used all data, performance is optimistic. Page not verified: full text not yet accessed by the team |
Applicability to your team's problem: same dataset and task, so directly comparable on data, but not on evaluation; neither study uses driving data.

> Note: the full text could not be opened by the team's AI tool; page numbers and the CV unit must be filled from the PDF before the oral check.

## B. Results and timeline

### B1. Current results
| Model | Evaluation set (split method, n) | Main metric (value, 95% CI if available) | Secondary metrics | Source of numbers (file / notebook / W&B) |
|---|---|---|---|---|
| Baseline 1 (Milestone 2) — logistic regression on 4 regional features (frontal theta, parietal alpha, mean beta, frontal-theta/parietal-alpha index) | Out-of-fold, GroupKFold 8 folds by subject; 48 subjects, 6,198 epochs (3,205 Low / 2,993 High) | AUC 0.649 (0.585–0.716) | Balanced acc 0.603 (0.557–0.650); recall High 0.562 (0.460–0.667); Brier 0.236 (0.219–0.253); calibration-in-the-large −0.01 (−0.16 to 0.13); slope 0.76 (0.39–1.33) | `results/b1_results_with_ci.csv`, `src/evaluate_extended.py` |
| Proposed model (Milestone 3) — logistic regression on 84 features (log absolute + relative power, 3 bands × 14 channels) | Same as above | AUC 0.888 (0.850–0.922) | Balanced acc 0.801 (0.761–0.840); recall High 0.822 (0.765–0.871); precision High 0.778 (0.723–0.838); Brier 0.143 (0.115–0.175); calibration-in-the-large −0.27 (−0.79 to 0.26); slope 0.56 (0.46–0.70); recording-level AUC 0.951 | `results/b1_results_with_ci.csv`; fold metrics `results/pipeline/09_folds_Logistic_Regression.csv` |

Other models on the same 84 features (same split): Random Forest AUC 0.882 (0.848–0.916), balanced acc 0.804; SVM-RBF AUC 0.885 (0.847–0.921), balanced acc 0.794; KNN k=15 AUC 0.817 (0.771–0.859), balanced acc 0.747; Dummy balanced acc 0.500. 95% CIs: cluster bootstrap over subjects, 2,000 resamples. Threshold 0.5, no hyperparameter tuning.

### B2. Ablation (Milestone 3)
Full model = proposed model in B1 (AUC 0.888, balanced acc 0.801). Source for all rows: `results/b2_ablation.csv`.

| Configuration (component removed / changed) | Main metric | Difference vs full model | Source |
|---|---|---|---|
| Absolute power only (relative removed), 42 features | AUC 0.859 | −0.028 | `results/b2_ablation.csv` |
| Relative power only (absolute removed), 42 features | AUC 0.850 | −0.037 | `results/b2_ablation.csv` |
| Band ratios added (TBR, ABR, (θ+α)/β), 87 features | AUC 0.885 | −0.003 | `results/b2_ablation.csv` |
| Theta removed, 56 features | AUC 0.837 | −0.051 | `results/b2_ablation.csv` |
| Alpha removed, 56 features | AUC 0.876 | −0.011 | `results/b2_ablation.csv` |
| Beta removed, 56 features | AUC 0.893 | +0.005 | `results/b2_ablation.csv` |
| Theta only, 28 features | AUC 0.889 | +0.002 | `results/b2_ablation.csv` |
| Artefact-prone channels removed (T7, T8, AF3, AF4, F7, F8), 48 features | AUC 0.858 | −0.030 | `results/b2_ablation.csv` |
| 4 regional features only (= Baseline 1) | AUC 0.649 | −0.238 | `results/b2_ablation.csv` |

Reading: theta carries most of the information (removing it costs the most; theta alone matches the full model), consistent with paper #1. Ratios add nothing because log band power already contains them. Removing artefact-prone channels costs only 0.03 AUC.

### B3. Comparison with the literature
- **#10 Safari et al. 2024 (STEW): accuracy 89.53% ± 1.36% vs our balanced accuracy 0.801.** Not a fair comparison. Same dataset, but (a) different split: 7-fold CV whose unit (segments or subjects) is not confirmed, vs our subject-level GroupKFold; (b) different features (connectivity + feature selection vs band power, no selection); (c) different metric (accuracy vs balanced accuracy; our accuracy is 0.802 in `results/pipeline/10_model_comparison.csv`).
- **#1 Chikhi et al. 2022:** not a classifier, but our subject-level paired tests agree in direction: frontal theta increases in 35/48 subjects (Wilcoxon p = 0.002), parietal alpha decreases in 30/48 (p = 0.014), beta unchanged (p = 0.42). Source: `results/paired_subject_tests.csv` (subject means of the epoch features; Wilcoxon signed-rank; same step is in `src/stew_pipeline.py`). Non-equivalence: meta-analysis effect sizes (Hedges g) vs our within-subject medians.
- **#3 Walther et al. 2023:** best DEAP accuracy 72.9% with deep learning on raw EEG. Different dataset and task (emotion), intra-subject split; cited only to justify hand-crafted features + simple models for small data.

### B4. Timeline to completion
| Task | Owner (student ID / role) | Milestone (M3 week 10 · M4 week 12 · Final week 15) | Planned date | Status (Not started / In progress / Done + evidence) |
|---|---|---|---|---|
| Download and inspect STEW | 202414870 | M3 | 2026-11-09 | Done — `results/pipeline/01_recordings_inspect.csv` |
| Preprocessing + Welch features + EDA | 202414870 | M3 | 2026-11-10 | Done — `src/stew_pipeline.py`, `results/pipeline/03_descriptive_stats.csv` |
| Baseline + proposed model, subject-level CV | 202414870 | M3 | 2026-11-10 | Done — `results/b1_results_with_ci.csv` |
| Ablation | 202414870 | M3 | 2026-11-11 | Done — `results/b2_ablation.csv` |
| Literature matrix ≥ 10 papers + search log | 202414866 | M3 | 2026-11-12 | In progress — 13 papers listed (A2), summaries being merged |
| Citation and DOI check | 202414866 | M3 | 2026-11-13 | In progress |
| Timeline, risk and AI-use slides | 202414852 | M3 | 2026-11-12 | Done — team slides (Drive) |
| PROBAST+AI of #10 with page numbers | 202414870 | M3 | 2026-11-13 | In progress — A4 |
| Recalibration (calibration slope 0.56) and per-subject performance | 202414870 | M4 | 2026-11-24 | Not started |
| Leave-one-subject-out check; threshold choice for recall High | 202414870 | M4 | 2026-11-24 | Not started |
| Full TRIPOD+AI checklist (Not yet / Partly items) | 202414852 | M4 | 2026-11-24 | In progress — `report/tripod_ai_checklist.md` |
| Phase 2 recording protocol (if approved) | All | Final | 2026-12-08 | Not started |
| Final report, slides, PR | All | Final | 2026-12-15 | Not started |

### B5. Risks / blockers / questions for the instructor
- Risk — label confound: rest vs SIMKAP also differ in eye/hand movement. Evidence: more epochs rejected in SIMKAP (559 vs 347, `results/pipeline/01_recordings_inspect.csv`); F8 (near the eye) power higher in 42/48 subjects. Mitigation: artefact-channel ablation (−0.03 AUC); ICA/EOG regression planned.
- Risk — overconfident probabilities: calibration slope 0.56 (0.46–0.70). Mitigation: stronger regularisation and recalibration inside CV (M4).
- Risk — small sample: 48 subjects, fold SD of balanced accuracy 0.053; no external validation set.
- Risk — not driving data: results do not transfer directly to drivers.
- Question 1: Is STEW acceptable as the Phase 1 dataset for a driver-workload motivation?
- Question 2: Is Phase 2 (own EEG recording) required this term?
- Question 3: Should the primary estimate be GroupKFold (8 folds) or leave-one-subject-out?
- Question 4: Should performance be reported per epoch or per recording (recording-level AUC 0.951)?

## C. Self-assessment against the FAIR, TRIPOD+AI, PROBAST+AI appendix

### C1. FAIR — data
| Item | Answer | Evidence |
|---|---|---|
| Source, version, download date; checksum of the training file | STEW, IEEE DataPort, DOI 10.21227/44r8-ya50 (no version number given on the page); downloaded 2026-11-09; SHA-256 of `STEW Dataset.zip`: not yet recorded (to add, see C4); SHA-256 of derived feature table `02_features_all_epochs.csv`: 8eaeecc5674a4997f8c56f268b7a12c215d48e8bc22a86b46816570ce24b5314 | `README.md`; https://ieee-dataport.org/open-access/stew-simultaneous-task-eeg-workload-dataset |
| Licence / DUA; basis for being allowed to use the data | CC BY 4.0; free IEEE account login required; no DUA | Dataset page |
| Datasheet: population, inclusion/exclusion criteria, variable definitions and units, limitations | Written; population details still to be taken from Lim et al. 2018 | `report/datasheet_stew.md` |
| No patient identifiers | Yes — subjects are numbered (sub01–sub48); raw data not committed | `report/datasheet_stew.md` |

### C2. Reporting guideline
- Your research question: In healthy adults, can a model using band power (theta, alpha, beta) from a 4-s, 14-channel EEG epoch predict whether the person is in a high-workload (SIMKAP multitasking) or low-workload (rest) condition, when tested on people not used for training?
- Matching reporting guideline (TRIPOD+AI / STARD-AI / DECIDE-AI) and why (Appendix §3.2): TRIPOD+AI — the study develops and internally validates a multivariable prediction model with ML. Not STARD-AI (no index test vs clinical reference standard) and not DECIDE-AI (no live use in a real setting).
- Checklist items met / total; items still missing: 28 Met, 11 Partly, 7 Not yet, 6 NA out of 52 items (`report/tripod_ai_checklist.md`). Missing: abstract (2), health inequalities (3c), fairness (14), participant characteristics (20b), model updating (12f, 24), user interaction (27b).
- Outcome, label, prediction time T0, variables available at T0: Outcome/label = experimental condition of the epoch (0 = rest, 1 = SIMKAP), fixed by protocol and not derived from EEG (no incorporation bias). T0 = end of each 4-s epoch. Variables available at T0 = band powers computed only from that epoch's EEG. Self-ratings are collected after the task, so they are not used as predictors (only as a manipulation check: rest 1.80 ± 1.01 vs SIMKAP 6.58 ± 1.42, all 45 rated subjects higher in SIMKAP, `results/pipeline/08_ratings.csv`).

### C3. PROBAST+AI — self-appraisal of your project
| Domain | Risk of bias (low / high / unclear) | Evidence in the repo (file, line) | What you will do to reduce the risk |
|---|---|---|---|
| Participants and data sources | Low (risk of bias); high concern for applicability | `report/datasheet_stew.md`; all 48 subjects used, `results/pipeline/01_recordings_inspect.csv` | Report population details from Lim et al. 2018; Phase 2 data closer to the target setting |
| Predictors | Unclear | Same processing for all recordings: `src/stew_pipeline.py` l.48–54, l.111–152; but conditions differ in eye/hand movement (559 vs 347 rejected epochs) | ICA/EOG-based artefact removal; keep artefact-channel ablation in the report |
| Outcome (watch for incorporation bias) | Low | Label from file name `subXX_lo/hi` (`src/stew_pipeline.py`, `main()`), not from EEG features; ratings used only as manipulation check | — |
| Analysis (patient-level split, fit on train only, sample size, missing data) | High | Subject-level GroupKFold with overlap assertion (`src/stew_pipeline.py` l.181–184; `src/evaluate_extended.py` l.66–67); scaler inside Pipeline (`src/evaluate_extended.py` l.46); no tuning; but n = 48, no external validation, calibration slope 0.56 | Recalibration and regularisation inside CV; leave-one-subject-out; external dataset in Phase 2–3 |
Applicability (population, setting where the model would be used): developed on healthy adults in a laboratory rest vs multitasking task; intended future use is drivers in real traffic, so applicability to the target setting is low until validated on driving data.

### C4. Quick checklist (Appendix §7)
| Item | Met / Not yet | Evidence |
|---|---|---|
| Datasheet | Met (population details pending) | `report/datasheet_stew.md` |
| Data version + checksum | Not yet — zip checksum and download date to add; derived-table checksum given | `README.md`, C1 |
| No identifiers; basis for data use | Met | CC BY 4.0; numbered subjects |
| Correct reporting guideline + completed checklist | Met (checklist partly complete) | `report/tripod_ai_checklist.md` |
| Outcome, label, T0 defined | Met | C2 |
| Patient-level split; preprocessing fit on train only | Met | `src/evaluate_extended.py` l.46, l.66–67 |
| Logistic regression baseline (or a suitable baseline, with reason) | Met | B1 (Baseline 1 and proposed are both logistic regression; Dummy as chance level) |
| AUC + calibration (curve, in-the-large, slope) + Brier, with 95% CI | Met | `results/b1_results_with_ci.csv`, `results/calibration_curve.png` |
| Is the sample size sufficient; applicability limits | Not yet — no formal sample-size calculation; limits stated (B5, C3) | B5, C3 |

## D. AI use declaration
| Tool | Used for which part | How the team verified the output |
|---|---|---|
| Claude (Anthropic) | Reading and summarising papers #1–#3; drafting this file, the datasheet and the TRIPOD+AI checklist; writing `src/stew_pipeline.py` and `src/evaluate_extended.py`; planning tasks | Numbers in A2 checked against the PDFs; code run by the team on the real STEW data in Colab and results files committed; every result number in B traced to a file in `results/` |
| Declaration of member 202414866 | Pending — member to confirm | — |
| Declaration of member 202414852 | Pending — member to confirm | — |
