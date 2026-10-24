"""
Shared helpers for Experiment 2 (threshold) and Experiment 3 (feedback cost).
Place at: <project>/shared/exp_common.py

tuning_source:
  "val"        -> thresholds / feedback signals come from the 20% validation split
                  of the TRAIN file (strict, no test data touched). Final test = full test file.
  "test_calib" -> a stratified 30% of the TEST file is used for tuning, and all final
                  numbers are reported on the remaining 70%. This tells you whether
                  the high FPR is caused by train->test distribution shift.
"""
import sys
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "cost_sensitive"))   # so cost_module + its pickles import
sys.path.insert(0, str(ROOT / "shared"))

TRAIN_PATH = ROOT / "data" / "raw" / "UNSW_NB15_training-set.csv"
TEST_PATH = ROOT / "data" / "raw" / "UNSW_NB15_testing-set.csv"
BASE_DIR = ROOT / "baseline" / "models"
CS_DIR = ROOT / "cost_sensitive" / "models"
RESULTS_DIR = ROOT / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42
DROP_COLUMNS = ["id", "attack_cat"]
TARGET = "label"
GRID = np.round(np.arange(0.01, 1.0, 0.01), 2)

CLASSIFIERS = {"Random Forest": "random_forest",
               "Logistic Regression": "logistic_regression",
               "MLP": "mlp"}


def load_splits(tuning_source="val", calib_frac=0.30):
    df = pd.read_csv(TRAIN_PATH)
    y = df[TARGET]
    X = df.drop(columns=DROP_COLUMNS + [TARGET])
    X_tr, X_val, _, _ = train_test_split(
        X, y, test_size=0.20, random_state=RANDOM_STATE, stratify=y)   # identical to baseline
    raw_train = df.loc[X_tr.index].reset_index(drop=True)
    raw_val = df.loc[X_val.index].reset_index(drop=True)
    test = pd.read_csv(TEST_PATH)

    if tuning_source == "val":
        tune, final = raw_val, test
    elif tuning_source == "test_calib":
        ic, i_f = train_test_split(np.arange(len(test)), train_size=calib_frac,
                                   random_state=RANDOM_STATE, stratify=test[TARGET])
        tune = test.iloc[ic].reset_index(drop=True)
        final = test.iloc[i_f].reset_index(drop=True)
    else:
        raise ValueError(tuning_source)
    return dict(raw_train=raw_train, raw_val=raw_val, raw_test_full=test,
                raw_tune=tune, raw_final=final)


def featurize(preprocessor, raw):
    return preprocessor.transform(raw.drop(columns=["id", "attack_cat", "label"]))


def load_fixed_costs():
    """Evaluation costs (fitted on TRAIN in Experiment 1). Never changed by feedback."""
    return (joblib.load(CS_DIR / "cost_static.pkl"),
            joblib.load(CS_DIR / "cost_dynamic.pkl"))


# ---------------- thresholds ----------------
def cost_curve(y, prob, c_fn, c_fp, grid=GRID):
    pos = y == 1
    neg = ~pos
    return np.array([(c_fn[pos & (prob < t)].sum() + c_fp[neg & (prob >= t)].sum()) / len(y)
                     for t in grid])


def best_threshold(y, prob, c_fn, c_fp, grid=GRID):
    return float(grid[int(np.argmin(cost_curve(y, prob, c_fn, c_fp, grid)))])


def fpr_cap_threshold(y, prob, cap, grid=GRID):
    """Smallest threshold (=highest recall) whose FPR <= cap."""
    neg = prob[y == 0]
    for t in grid:
        if (neg >= t).mean() <= cap:
            return float(t)
    return float(grid[-1])


# ---------------- metrics ----------------
def evaluate(y, prob, thr, raw, static_cost, dynamic_cost):
    pred = (prob >= thr).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum()); tn = int(((pred == 0) & (y == 0)).sum())
    fp = int(((pred == 1) & (y == 0)).sum()); fn = int(((pred == 0) & (y == 1)).sum())
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return {"Threshold": thr, "Accuracy": (tp + tn) / len(y), "Precision": prec,
            "Recall": rec, "F1": f1, "FPR": fp / (fp + tn), "FNR": fn / (fn + tp),
            "FP": fp, "FN": fn,
            "Static_Avg_Cost": static_cost.total_cost(raw, pred)[1],
            "Dynamic_Avg_Cost": dynamic_cost.total_cost(raw, pred)[1]}


# ---------------- threshold-free ranking quality ----------------
FPR_CAPS = (0.05, 0.10, 0.20)
FUZZ_CAP = 0.10


def recall_at_fpr(y, prob, caps=FPR_CAPS):
    """TPR read off the ROC curve at fixed FPR values (threshold-free)."""
    fpr, tpr, _ = roc_curve(y, prob)
    return {c: float(np.interp(c, fpr, tpr)) for c in caps}


def mean_recall_at_fpr(y, prob, caps=FPR_CAPS):
    """Single ranking-quality score = mean recall at FPR 5%, 10%, 20%. Used to pick the best round."""
    return float(np.mean(list(recall_at_fpr(y, prob, caps).values())))


def group_recall_at_fpr(y, prob, mask, cap=FUZZ_CAP):
    """Recall of one attack type at the threshold that gives overall FPR = cap."""
    thr = np.quantile(prob[y == 0], 1 - cap)
    return float((prob[mask] > thr).mean())