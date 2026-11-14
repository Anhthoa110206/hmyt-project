"""
STEW mental-workload pipeline (Nhom N20)
=======================================
Raw EEG -> loc 1-40 Hz -> epoch 4 s (50% overlap) -> loai epoch nhieu
-> Welch PSD (cua so Hann 1 s, overlap 50%) -> band power theta/alpha/beta
-> EDA -> Dummy / Logistic Regression / KNN / SVM / Random Forest
-> danh gia chia theo NGUOI (GroupKFold) + ablation theo bo dac trung.

Chay:
    python stew_pipeline.py --data "duong/dan/STEW Dataset.zip" --out outputs

--data co the la file .zip hoac thu muc da giai nen (chua subXX_lo.txt, subXX_hi.txt, ratings.txt).
Ket qua (bang .csv, hinh .png, summary.md) nam trong thu muc --out.
"""
import argparse
import os
import re
import sys
import tempfile
import zipfile
from glob import glob

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.signal import butter, sosfiltfilt, welch
from scipy.stats import wilcoxon
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (balanced_accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score, accuracy_score)
from sklearn.model_selection import GroupKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

# ----------------------------------------------------------------------------
# THAM SO (chot theo B1 - sua o day neu nhom doi y)
# ----------------------------------------------------------------------------
FS = 128                      # Hz, Emotiv EPOC (theo trang STEW tren IEEE DataPort)
CHANNELS = ["AF3", "F7", "F3", "FC5", "T7", "P7", "O1",
            "O2", "P8", "T8", "FC6", "F4", "F8", "AF4"]
BANDS = {"theta": (4, 8), "alpha": (8, 12), "beta": (12, 30)}
TOTAL_BAND = (1, 30)          # mau so cua relative power
FILTER_BAND = (1, 40)         # loc thong dai truoc khi chia epoch
EPOCH_SEC = 4.0
EPOCH_STEP_SEC = 2.0          # 50% overlap
WELCH_SEC = 1.0               # nperseg = 128 mau -> do phan giai 1 Hz
REJECT_FACTOR = 5.0           # loai epoch neu bien do dinh-dinh > 5 x trung vi cua kenh do
N_FOLDS = 8                   # GroupKFold: 48 nguoi -> 6 nguoi / fold
SEED = 42

FRONTAL = ["AF3", "AF4", "F3", "F4", "F7", "F8", "FC5", "FC6"]
PARIETAL = ["P7", "P8"]
LABEL_NAMES = {0: "Low (rest)", 1: "High (SIMKAP)"}


# ----------------------------------------------------------------------------
# 1. LOAD
# ----------------------------------------------------------------------------
def resolve_data_dir(path):
    if os.path.isfile(path) and path.lower().endswith(".zip"):
        tmp = tempfile.mkdtemp(prefix="stew_")
        with zipfile.ZipFile(path) as z:
            z.extractall(tmp)
        path = tmp
    files = glob(os.path.join(path, "**", "sub*_*.txt"), recursive=True)
    if not files:
        sys.exit(f"Khong tim thay file subXX_lo.txt / subXX_hi.txt trong: {path}")
    return path, sorted(files)


def load_txt(fp):
    with open(fp) as f:
        head = f.readline()
    delim = "," if "," in head else None
    x = np.loadtxt(fp, delimiter=delim)
    if x.ndim != 2:
        raise ValueError(f"{fp}: khong doc duoc thanh bang 2 chieu")
    if x.shape[1] != len(CHANNELS) and x.shape[0] == len(CHANNELS):
        x = x.T
    return x  # (samples, channels)


def load_ratings(root):
    cand = glob(os.path.join(root, "**", "ratings*.txt"), recursive=True)
    if not cand:
        return None
    rows = []
    with open(cand[0]) as f:
        for line in f:
            parts = [p.strip() for p in re.split(r"[,\s]+", line.strip()) if p.strip()]
            if len(parts) >= 3 and parts[0].isdigit():
                rows.append({"subject": int(parts[0]),
                             "rating_rest": pd.to_numeric(parts[1], errors="coerce"),
                             "rating_test": pd.to_numeric(parts[2], errors="coerce")})
    return pd.DataFrame(rows) if rows else None


# ----------------------------------------------------------------------------
# 2. CLEAN + EPOCH
# ----------------------------------------------------------------------------
SOS = butter(4, FILTER_BAND, btype="bandpass", fs=FS, output="sos")


def epoch_recording(x):
    x = x - x.mean(axis=0, keepdims=True)
    x = sosfiltfilt(SOS, x, axis=0)
    win, step = int(EPOCH_SEC * FS), int(EPOCH_STEP_SEC * FS)
    starts = np.arange(0, x.shape[0] - win + 1, step)
    ep = np.stack([x[s:s + win] for s in starts])          # (n_ep, win, ch)
    ptp = ep.max(axis=1) - ep.min(axis=1)                   # (n_ep, ch)
    thr = REJECT_FACTOR * np.median(ptp, axis=0, keepdims=True)
    keep = ~(ptp > thr).any(axis=1)
    return ep[keep], int((~keep).sum()), x


# ----------------------------------------------------------------------------
# 3. WELCH PSD + FEATURES
# ----------------------------------------------------------------------------
def band_power(f, pxx, lo, hi):
    m = (f >= lo) & (f < hi)
    return np.trapezoid(pxx[..., m], f[m], axis=-1)


def epoch_features(ep):
    nper = int(WELCH_SEC * FS)
    f, pxx = welch(ep, fs=FS, window="hann", nperseg=nper, noverlap=nper // 2, axis=1)
    pxx = np.moveaxis(pxx, 1, -1)                           # (n_ep, ch, freq)
    total = band_power(f, pxx, *TOTAL_BAND)
    feats = {}
    bp = {b: band_power(f, pxx, lo, hi) for b, (lo, hi) in BANDS.items()}
    for b in BANDS:
        for ci, ch in enumerate(CHANNELS):
            feats[f"abs_{b}_{ch}"] = np.log10(bp[b][:, ci])
            feats[f"rel_{b}_{ch}"] = bp[b][:, ci] / total[:, ci]
    mean_bp = {b: bp[b].mean(axis=1) for b in BANDS}
    feats["ratio_TBR"] = np.log10(mean_bp["theta"] / mean_bp["beta"])
    feats["ratio_ABR"] = np.log10(mean_bp["alpha"] / mean_bp["beta"])
    feats["ratio_TAB"] = np.log10((mean_bp["theta"] + mean_bp["alpha"]) / mean_bp["beta"])
    fi = [CHANNELS.index(c) for c in FRONTAL]
    pi = [CHANNELS.index(c) for c in PARIETAL]
    feats["reg_frontal_theta"] = np.log10(bp["theta"][:, fi].mean(axis=1))
    feats["reg_parietal_alpha"] = np.log10(bp["alpha"][:, pi].mean(axis=1))
    feats["reg_all_beta"] = np.log10(bp["beta"].mean(axis=1))
    feats["reg_FT_PA_index"] = feats["reg_frontal_theta"] - feats["reg_parietal_alpha"]
    return pd.DataFrame(feats), f, pxx


FEATURE_SETS = {
    "absolute (log)": lambda c: [x for x in c if x.startswith("abs_")],
    "relative": lambda c: [x for x in c if x.startswith("rel_")],
    "absolute + relative": lambda c: [x for x in c if x.startswith(("abs_", "rel_"))],
    "abs + rel + ratios": lambda c: [x for x in c if x.startswith(("abs_", "rel_", "ratio_"))],
    "regional (4 features)": lambda c: [x for x in c if x.startswith("reg_")],
    "theta only (abs + rel)": lambda c: [x for x in c if x.startswith(("abs_theta", "rel_theta"))],
    "alpha only (abs + rel)": lambda c: [x for x in c if x.startswith(("abs_alpha", "rel_alpha"))],
    "beta only (abs + rel)": lambda c: [x for x in c if x.startswith(("abs_beta", "rel_beta"))],
    "abs + rel, no T7/T8/AF3/AF4/F7/F8": lambda c: [x for x in c if x.startswith(("abs_", "rel_"))
                                                     and not x.endswith(("_T7", "_T8", "_AF3", "_AF4", "_F7", "_F8"))],
}

MODELS = {
    "Dummy (most frequent)": lambda: DummyClassifier(strategy="most_frequent"),
    "Logistic Regression": lambda: make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=1.0)),
    "KNN (k=15)": lambda: make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=15)),
    "SVM (RBF)": lambda: make_pipeline(StandardScaler(), SVC(kernel="rbf", C=1.0, gamma="scale")),
    "Random Forest": lambda: RandomForestClassifier(n_estimators=300, random_state=SEED, n_jobs=-1),
}


# ----------------------------------------------------------------------------
# 4. EVALUATION (chia theo nguoi, scaler fit chi tren train trong Pipeline)
# ----------------------------------------------------------------------------
def evaluate(X, y, groups, model_fn):
    gkf = GroupKFold(n_splits=N_FOLDS)
    rows, y_true_all, y_pred_all = [], [], []
    for k, (tr, te) in enumerate(gkf.split(X, y, groups)):
        assert set(groups[tr]).isdisjoint(groups[te]), "Ro ri: cung mot nguoi o ca train va test"
        m = model_fn()
        m.fit(X[tr], y[tr])
        p = m.predict(X[te])
        rows.append({
            "fold": k,
            "accuracy": accuracy_score(y[te], p),
            "balanced_accuracy": balanced_accuracy_score(y[te], p),
            "recall_high": recall_score(y[te], p, pos_label=1, zero_division=0),
            "precision_high": precision_score(y[te], p, pos_label=1, zero_division=0),
            "f1_macro": f1_score(y[te], p, average="macro", zero_division=0),
        })
        y_true_all.append(y[te]); y_pred_all.append(p)
    return pd.DataFrame(rows), np.concatenate(y_true_all), np.concatenate(y_pred_all)


# ----------------------------------------------------------------------------
# MAIN
# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", default="outputs")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    root, files = resolve_data_dir(args.data)

    feat_frames, psd_rows, info_rows = [], [], []
    raw_example = {}
    for fp in files:
        m = re.search(r"sub(\d+)_(lo|hi)", os.path.basename(fp), re.I)
        if not m:
            continue
        subj, cond = int(m.group(1)), m.group(2).lower()
        label = 1 if cond == "hi" else 0
        x = load_txt(fp)
        ep, n_rej, xf = epoch_recording(x)
        if subj == 1 or not raw_example.get(cond):
            raw_example.setdefault(cond, (subj, xf[: int(10 * FS)]))
        df, f, pxx = epoch_features(ep)
        df.insert(0, "label", label); df.insert(0, "subject", subj)
        feat_frames.append(df)
        psd_rows.append((label, np.log10(pxx).mean(axis=(0, 1))))
        info_rows.append({"subject": subj, "condition": cond, "samples": x.shape[0],
                          "duration_s": x.shape[0] / FS, "channels": x.shape[1],
                          "nan_values": int(np.isnan(x).sum()),
                          "epochs_kept": len(ep), "epochs_rejected": n_rej})

    data = pd.concat(feat_frames, ignore_index=True)
    info = pd.DataFrame(info_rows).sort_values(["subject", "condition"])
    info.to_csv(os.path.join(args.out, "01_recordings_inspect.csv"), index=False)
    data.to_csv(os.path.join(args.out, "02_features_all_epochs.csv"), index=False)

    # ---------------- EDA ----------------
    key = ["reg_frontal_theta", "reg_parietal_alpha", "reg_all_beta", "ratio_TBR", "ratio_ABR", "reg_FT_PA_index"]
    desc = []
    for c in key:
        for lab, g in data.groupby("label"):
            v = g[c]
            q1, q3 = v.quantile(.25), v.quantile(.75)
            iqr = q3 - q1
            desc.append({"feature": c, "class": LABEL_NAMES[lab], "n": len(v), "mean": v.mean(),
                         "median": v.median(), "sd": v.std(), "min": v.min(), "max": v.max(),
                         "Q1": q1, "Q3": q3, "IQR": iqr,
                         "outliers_1.5IQR": int(((v < q1 - 1.5 * iqr) | (v > q3 + 1.5 * iqr)).sum())})
    pd.DataFrame(desc).to_csv(os.path.join(args.out, "03_descriptive_stats.csv"), index=False)

    # subject-level mean per class (tranh dem trung epoch khi ve)
    subj_mean = data.groupby(["subject", "label"])[key].mean().reset_index()

    fig, axes = plt.subplots(2, 3, figsize=(13, 7))
    for ax, c in zip(axes.ravel(), key):
        ax.boxplot([subj_mean.loc[subj_mean.label == l, c] for l in (0, 1)],
                   tick_labels=[LABEL_NAMES[0], LABEL_NAMES[1]])
        for s, g in subj_mean.groupby("subject"):
            if len(g) == 2:
                ax.plot([1, 2], g.sort_values("label")[c].values, color="0.7", lw=.6, zorder=0)
        ax.set_title(c, fontsize=10)
    fig.suptitle("Features by class (one point = one subject mean; grey line = same subject)")
    fig.tight_layout(); fig.savefig(os.path.join(args.out, "04_features_by_class.png"), dpi=150); plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    vis = (f >= 1) & (f <= 40)
    curves = []
    for lab in (0, 1):
        P = 10 ** np.stack([p for l, p in psd_rows if l == lab]).mean(axis=0)  # geometric mean
        curves.append(P[vis])
        ax.semilogy(f, P, label=LABEL_NAMES[lab])
    lo_y, hi_y = min(c.min() for c in curves), max(c.max() for c in curves)
    ax.set_ylim(lo_y / 2, hi_y * 3)
    for b, (lo, hi) in BANDS.items():
        ax.axvspan(lo, hi, alpha=.08, color={"theta": "C2", "alpha": "C3", "beta": "C4"}[b])
        ax.text((lo + hi) / 2, ax.get_ylim()[1] * 0.6, b, ha="center", fontsize=9)
    ax.set_xlim(1, 40); ax.set_xlabel("Frequency (Hz)"); ax.set_ylabel("PSD (uV^2/Hz, log scale)")
    ax.set_title("Welch PSD by class (geometric mean over recordings and channels)"); ax.legend()
    fig.tight_layout(); fig.savefig(os.path.join(args.out, "05_psd_by_class.png"), dpi=150); plt.close(fig)

    fig, axes = plt.subplots(2, 1, figsize=(11, 6), sharex=True)
    for ax, cond in zip(axes, ("lo", "hi")):
        if cond in raw_example:
            s, seg = raw_example[cond]
            t = np.arange(seg.shape[0]) / FS
            for i, ch in enumerate(CHANNELS):
                sd = np.std(seg[:, i]) or 1.0
                ax.plot(t, seg[:, i] / (sd * 6) + i, lw=.5)
            ax.set_yticks(range(len(CHANNELS))); ax.set_yticklabels(CHANNELS, fontsize=7)
            ax.set_title(f"Filtered EEG (1-40 Hz), first 10 s - sub{s:02d}_{cond} (each channel scaled separately)", fontsize=10)
    axes[-1].set_xlabel("Time (s)")
    fig.tight_layout(); fig.savefig(os.path.join(args.out, "06_raw_eeg_example.png"), dpi=150); plt.close(fig)

    corr = data[key].corr()
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(key))); ax.set_xticklabels(key, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(key))); ax.set_yticklabels(key, fontsize=8)
    for i in range(len(key)):
        for j in range(len(key)):
            ax.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center", fontsize=7)
    fig.colorbar(im); ax.set_title("Correlation between key features")
    fig.tight_layout(); fig.savefig(os.path.join(args.out, "07_feature_correlation.png"), dpi=150); plt.close(fig)

    # paired subject-level tests (High - Low)
    w = subj_mean.pivot(index="subject", columns="label")
    prow = []
    for c in key:
        if (c, 0) in w and (c, 1) in w:
            diff = (w[(c, 1)] - w[(c, 0)]).dropna()
            prow.append({"feature": c, "n_subjects": len(diff), "median_diff_high_minus_low": diff.median(),
                         "subjects_increase": int((diff > 0).sum()), "wilcoxon_p": wilcoxon(diff).pvalue})
    paired = pd.DataFrame(prow)
    paired.to_csv(os.path.join(args.out, "13_paired_subject_tests.csv"), index=False)

    k3 = ["reg_frontal_theta", "reg_parietal_alpha", "reg_FT_PA_index"]
    t3 = ["Frontal theta (log power)", "Parietal alpha (log power)", "Frontal theta / parietal alpha (log ratio)"]
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for ax, c, tt in zip(axes, k3, t3):
        ax.boxplot([subj_mean.loc[subj_mean.label == l, c] for l in (0, 1)], tick_labels=["Low (rest)", "High (SIMKAP)"])
        for s_, g_ in subj_mean.groupby("subject"):
            if len(g_) == 2:
                ax.plot([1, 2], g_.sort_values("label")[c].values, color="0.7", lw=.6, zorder=0)
        ax.set_title(tt, fontsize=10)
    fig.suptitle("Subject-level features (n = %d; grey line = same subject)" % subj_mean.subject.nunique())
    fig.tight_layout(); fig.savefig(os.path.join(args.out, "04b_key_features_3panel.png"), dpi=200); plt.close(fig)

    ratings = load_ratings(root)
    if ratings is not None:
        ratings.to_csv(os.path.join(args.out, "08_ratings.csv"), index=False)

    # ---------------- MODELS ----------------
    y = data["label"].values
    groups = data["subject"].values
    cols = [c for c in data.columns if c not in ("subject", "label")]

    # (a) so sanh mo hinh tren bo dac trung chinh
    main_set = "absolute + relative"
    X = data[FEATURE_SETS[main_set](cols)].values
    model_rows, cm_store = [], {}
    for name, fn in MODELS.items():
        folds, yt, yp = evaluate(X, y, groups, fn)
        folds.to_csv(os.path.join(args.out, f"09_folds_{re.sub(r'[^A-Za-z]+', '_', name)}.csv"), index=False)
        r = {"model": name, "feature_set": main_set, "n_features": X.shape[1]}
        for mcol in ["accuracy", "balanced_accuracy", "recall_high", "precision_high", "f1_macro"]:
            r[mcol + "_mean"] = folds[mcol].mean(); r[mcol + "_sd"] = folds[mcol].std()
        model_rows.append(r); cm_store[name] = confusion_matrix(yt, yp, labels=[0, 1])
        print(f"{name:24s} bal_acc={r['balanced_accuracy_mean']:.3f}±{r['balanced_accuracy_sd']:.3f} "
              f"recall_high={r['recall_high_mean']:.3f}")
    res = pd.DataFrame(model_rows)
    res.to_csv(os.path.join(args.out, "10_model_comparison.csv"), index=False)

    fig, axes = plt.subplots(1, len(cm_store), figsize=(3.2 * len(cm_store), 3.2))
    for ax, (name, cm) in zip(axes, cm_store.items()):
        ax.imshow(cm, cmap="Blues")
        for i in range(2):
            for j in range(2):
                ax.text(j, i, cm[i, j], ha="center", va="center")
        ax.set_xticks([0, 1]); ax.set_xticklabels(["Low", "High"]); ax.set_yticks([0, 1]); ax.set_yticklabels(["Low", "High"])
        ax.set_xlabel("Predicted"); ax.set_ylabel("True"); ax.set_title(name, fontsize=9)
    fig.tight_layout(); fig.savefig(os.path.join(args.out, "11_confusion_matrices.png"), dpi=150); plt.close(fig)

    cm = cm_store["Logistic Regression"]
    fig, ax = plt.subplots(figsize=(3.8, 3.4))
    ax.imshow(cm, cmap="Blues")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, cm[i, j], ha="center", va="center", color="white" if cm[i, j] > cm.max() / 2 else "black")
    ax.set_xticks([0, 1]); ax.set_xticklabels(["Low", "High"]); ax.set_yticks([0, 1]); ax.set_yticklabels(["Low", "High"])
    ax.set_xlabel("Predicted"); ax.set_ylabel("True"); ax.set_title("Logistic Regression (8 folds pooled)", fontsize=10)
    fig.tight_layout(); fig.savefig(os.path.join(args.out, "11b_confusion_logreg.png"), dpi=200); plt.close(fig)

    # (b) ablation: Logistic Regression tren tung bo dac trung
    abl_rows = []
    for fs_name, sel in FEATURE_SETS.items():
        Xs = data[sel(cols)].values
        folds, _, _ = evaluate(Xs, y, groups, MODELS["Logistic Regression"])
        abl_rows.append({"feature_set": fs_name, "n_features": Xs.shape[1],
                         "balanced_accuracy_mean": folds.balanced_accuracy.mean(),
                         "balanced_accuracy_sd": folds.balanced_accuracy.std(),
                         "recall_high_mean": folds.recall_high.mean(),
                         "recall_high_sd": folds.recall_high.std()})
    abl = pd.DataFrame(abl_rows)
    abl.to_csv(os.path.join(args.out, "12_ablation_logreg.csv"), index=False)

    # ---------------- SUMMARY ----------------
    n_subj = data.subject.nunique()
    lines = ["# Ket qua pipeline STEW", "",
             f"- So nguoi: {n_subj}; so file: {len(info)}; so epoch giu lai: {len(data)} "
             f"(Low {int((y == 0).sum())}, High {int((y == 1).sum())}); epoch bi loai: {int(info.epochs_rejected.sum())}",
             f"- Thoi luong moi file: {info.duration_s.min():.1f}-{info.duration_s.max():.1f} s; so kenh: {info.channels.unique().tolist()}; gia tri NaN: {int(info.nan_values.sum())}",
             f"- Tham so: loc {FILTER_BAND[0]}-{FILTER_BAND[1]} Hz; epoch {EPOCH_SEC:.0f} s buoc {EPOCH_STEP_SEC:.0f} s; "
             f"Welch Hann {WELCH_SEC:.0f} s overlap 50%; bands {BANDS}; relative / {TOTAL_BAND} Hz",
             f"- Danh gia: GroupKFold {N_FOLDS} fold theo nguoi (khong nguoi nao o ca train va test); scaler nam trong Pipeline (fit tren train).",
             ""]
    if ratings is not None:
        r = ratings.dropna()
        lines.append(f"- Diem workload tu danh gia (1-9, n={len(r)}): nghi {r.rating_rest.mean():.2f} ± {r.rating_rest.std():.2f}, "
                     f"SIMKAP {r.rating_test.mean():.2f} ± {r.rating_test.std():.2f}")
        lines.append("")
    lines += ["## So sanh mo hinh (bo dac trung: absolute + relative)", "",
              "| Mo hinh | Balanced acc | Recall High | Precision High | F1 macro |", "|---|---|---|---|---|"]
    for _, r in res.iterrows():
        lines.append(f"| {r.model} | {r.balanced_accuracy_mean:.3f} ± {r.balanced_accuracy_sd:.3f} | "
                     f"{r.recall_high_mean:.3f} ± {r.recall_high_sd:.3f} | {r.precision_high_mean:.3f} ± {r.precision_high_sd:.3f} | "
                     f"{r.f1_macro_mean:.3f} ± {r.f1_macro_sd:.3f} |")
    lines += ["", "## Ablation (Logistic Regression)", "", "| Bo dac trung | So dac trung | Balanced acc | Recall High |", "|---|---|---|---|"]
    for _, r in abl.iterrows():
        lines.append(f"| {r.feature_set} | {r.n_features} | {r.balanced_accuracy_mean:.3f} ± {r.balanced_accuracy_sd:.3f} | "
                     f"{r.recall_high_mean:.3f} ± {r.recall_high_sd:.3f} |")
    lines += ["", "## Paired subject-level tests (High - Low, Wilcoxon)", "",
              "| Feature | Median diff | Subjects increasing | p |", "|---|---|---|---|"]
    for _, r in paired.iterrows():
        lines.append(f"| {r.feature} | {r.median_diff_high_minus_low:+.3f} | {r.subjects_increase}/{r.n_subjects} | {r.wilcoxon_p:.2g} |")
    with open(os.path.join(args.out, "summary.md"), "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\nDa luu ket qua vao: {os.path.abspath(args.out)}")


if __name__ == "__main__":
    main()
