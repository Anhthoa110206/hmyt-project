"""
Extended evaluation for Progress Test 2 (team N20).

Reads the epoch-level feature table produced by stew_pipeline.py
(outputs/02_features_all_epochs.csv) and reports, for the baseline and the
proposed model, under the same subject-level GroupKFold (8 folds):
  - out-of-fold (OOF) ROC-AUC, balanced accuracy, recall/precision for High, F1 macro
  - Brier score, calibration-in-the-large, calibration slope, calibration curve
  - 95% CIs by cluster bootstrap over subjects (2000 resamples)
  - recording-level AUC (mean probability per recording, 96 recordings)
and an ablation table (proposed model with one component removed/changed).

Run:
    python evaluate_extended.py --features outputs/02_features_all_epochs.csv --out results
All preprocessing that learns parameters (StandardScaler) is inside the sklearn
Pipeline and is fitted on the training subjects of each fold only.
"""
import argparse
import json
import os

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar
from scipy.special import expit, logit
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (balanced_accuracy_score, brier_score_loss, f1_score,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.dummy import DummyClassifier

N_FOLDS, N_BOOT, SEED = 8, 2000, 42
ART = ("_T7", "_T8", "_AF3", "_AF4", "_F7", "_F8")


def lr():
    return make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, C=1.0))


SETS = {
    "abs+rel (84)": lambda c: [x for x in c if x.startswith(("abs_", "rel_"))],
    "regional (4)": lambda c: [x for x in c if x.startswith("reg_")],
    "absolute only (42)": lambda c: [x for x in c if x.startswith("abs_")],
    "relative only (42)": lambda c: [x for x in c if x.startswith("rel_")],
    "abs+rel+ratios (87)": lambda c: [x for x in c if x.startswith(("abs_", "rel_", "ratio_"))],
    "without theta (56)": lambda c: [x for x in c if x.startswith(("abs_", "rel_")) and "_theta_" not in x],
    "without alpha (56)": lambda c: [x for x in c if x.startswith(("abs_", "rel_")) and "_alpha_" not in x],
    "without beta (56)": lambda c: [x for x in c if x.startswith(("abs_", "rel_")) and "_beta_" not in x],
    "theta only (28)": lambda c: [x for x in c if x.startswith(("abs_theta", "rel_theta"))],
    "without T7/T8/AF3/AF4/F7/F8 (48)": lambda c: [x for x in c if x.startswith(("abs_", "rel_")) and not x.endswith(ART)],
}


def oof_proba(X, y, g, make_model):
    p = np.zeros(len(y))
    fold = np.zeros(len(y), dtype=int)
    for k, (tr, te) in enumerate(GroupKFold(N_FOLDS).split(X, y, g)):
        assert set(g[tr]).isdisjoint(g[te])
        m = make_model().fit(X[tr], y[tr])
        p[te] = m.predict_proba(X[te])[:, 1] if hasattr(m, "predict_proba") else m.decision_function(X[te])
        fold[te] = k
    return p, fold


def calib(y, p):
    eps = 1e-6
    lp = logit(np.clip(p, eps, 1 - eps))
    slope_m = LogisticRegression(C=1e9, max_iter=1000).fit(lp.reshape(-1, 1), y)
    slope = float(slope_m.coef_[0, 0])

    def nll(a):
        q = expit(a + lp)
        return -np.mean(y * np.log(q + eps) + (1 - y) * np.log(1 - q + eps))

    citl = float(minimize_scalar(nll, bounds=(-5, 5), method="bounded").x)
    return citl, slope


def metrics(y, p, thr=0.5):
    yhat = (p >= thr).astype(int)
    citl, slope = calib(y, p)
    return {"auc": roc_auc_score(y, p),
            "balanced_accuracy": balanced_accuracy_score(y, yhat),
            "recall_high": recall_score(y, yhat, zero_division=0),
            "precision_high": precision_score(y, yhat, zero_division=0),
            "f1_macro": f1_score(y, yhat, average="macro", zero_division=0),
            "brier": brier_score_loss(y, p),
            "calibration_in_the_large": citl,
            "calibration_slope": slope}


def boot_ci(y, p, g, rng, keys):
    subs = np.unique(g)
    idx_by = {s: np.where(g == s)[0] for s in subs}
    vals = {k: [] for k in keys}
    for _ in range(N_BOOT):
        pick = rng.choice(subs, size=len(subs), replace=True)
        idx = np.concatenate([idx_by[s] for s in pick])
        if len(np.unique(y[idx])) < 2:
            continue
        m = metrics(y[idx], p[idx])
        for k in keys:
            vals[k].append(m[k])
    return {k: (float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))) for k, v in vals.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", required=True)
    ap.add_argument("--out", default="results")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    d = pd.read_csv(a.features)
    y, g = d.label.values, d.subject.values
    cols = [c for c in d.columns if c not in ("subject", "label")]
    rng = np.random.default_rng(SEED)
    keys = ["auc", "balanced_accuracy", "recall_high", "precision_high", "brier",
            "calibration_in_the_large", "calibration_slope"]

    # ---- B1: baseline vs proposed (+ comparison models) ----
    runs = {
        "Dummy (prior)": (SETS["abs+rel (84)"], lambda: DummyClassifier(strategy="prior")),
        "Baseline 1: LogReg, 4 regional features": (SETS["regional (4)"], lr),
        "Proposed: LogReg, 84 abs+rel features": (SETS["abs+rel (84)"], lr),
        "Comparison: Random Forest, 84 features": (SETS["abs+rel (84)"], lambda: RandomForestClassifier(n_estimators=300, random_state=SEED, n_jobs=-1)),
        "Comparison: SVM-RBF (Platt), 84 features": (SETS["abs+rel (84)"], lambda: make_pipeline(StandardScaler(), SVC(probability=True, random_state=SEED))),
        "Comparison: KNN k=15, 84 features": (SETS["abs+rel (84)"], lambda: make_pipeline(StandardScaler(), KNeighborsClassifier(15))),
    }
    rows, store = [], {}
    for name, (sel, fn) in runs.items():
        X = d[sel(cols)].values
        p, fold = oof_proba(X, y, g, fn)
        m = metrics(y, p)
        ci = boot_ci(y, p, g, rng, keys) if not name.startswith("Dummy") else {}
        # recording-level: mean probability per (subject, condition)
        rec = pd.DataFrame({"s": g, "y": y, "p": p}).groupby(["s", "y"]).p.mean().reset_index()
        m["auc_recording_level"] = roc_auc_score(rec.y, rec.p) if not name.startswith("Dummy") else 0.5
        fm = [balanced_accuracy_score(y[fold == k], (p[fold == k] >= .5).astype(int)) for k in range(N_FOLDS)]
        row = {"model": name, "n_features": X.shape[1], "n_epochs": len(y), "n_subjects": len(np.unique(g)),
               **{k: round(v, 4) for k, v in m.items()},
               "balanced_accuracy_fold_mean": round(float(np.mean(fm)), 4),
               "balanced_accuracy_fold_sd": round(float(np.std(fm, ddof=1)), 4)}
        for k, (lo, hi) in ci.items():
            row[k + "_ci95"] = f"[{lo:.3f}, {hi:.3f}]"
        rows.append(row)
        store[name] = p
        print(f"{name:45s} AUC={m['auc']:.3f} balacc={m['balanced_accuracy']:.3f} brier={m['brier']:.3f} slope={m['calibration_slope']:.2f}")
    res = pd.DataFrame(rows)
    res.to_csv(os.path.join(a.out, "b1_results_with_ci.csv"), index=False)

    # calibration curve: baseline vs proposed
    fig, ax = plt.subplots(figsize=(4.8, 4.5))
    ax.plot([0, 1], [0, 1], "k--", lw=.8, label="Perfect")
    for name in ["Baseline 1: LogReg, 4 regional features", "Proposed: LogReg, 84 abs+rel features"]:
        p = store[name]
        bins = np.quantile(p, np.linspace(0, 1, 11))
        b = np.clip(np.digitize(p, bins[1:-1]), 0, 9)
        ax.plot([p[b == i].mean() for i in range(10)], [y[b == i].mean() for i in range(10)], "o-", ms=4,
                label=name.split(":")[0])
    ax.set_xlabel("Predicted probability (High)"); ax.set_ylabel("Observed fraction High")
    ax.set_title("Calibration (OOF, deciles)", fontsize=10); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(a.out, "calibration_curve.png"), dpi=200); plt.close(fig)

    # ---- B2: ablation of the proposed model ----
    full = res.set_index("model").loc["Proposed: LogReg, 84 abs+rel features"]
    ab = []
    for sname in ["absolute only (42)", "relative only (42)", "abs+rel+ratios (87)", "without theta (56)",
                  "without alpha (56)", "without beta (56)", "theta only (28)",
                  "without T7/T8/AF3/AF4/F7/F8 (48)", "regional (4)"]:
        p, _ = oof_proba(d[SETS[sname](cols)].values, y, g, lr)
        m = metrics(y, p)
        ab.append({"configuration": sname, "auc": round(m["auc"], 4),
                   "balanced_accuracy": round(m["balanced_accuracy"], 4),
                   "delta_auc_vs_full": round(m["auc"] - full["auc"], 4),
                   "delta_balacc_vs_full": round(m["balanced_accuracy"] - full["balanced_accuracy"], 4)})
    pd.DataFrame(ab).to_csv(os.path.join(a.out, "b2_ablation.csv"), index=False)
    with open(os.path.join(a.out, "run_info.json"), "w") as f:
        json.dump({"features_file": os.path.basename(a.features), "n_folds": N_FOLDS, "n_bootstrap": N_BOOT,
                   "seed": SEED, "threshold": 0.5}, f, indent=2)
    print(pd.DataFrame(ab).to_string(index=False))


if __name__ == "__main__":
    main()
