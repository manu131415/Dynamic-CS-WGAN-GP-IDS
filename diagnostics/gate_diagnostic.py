"""
GATE DIAGNOSTIC - run BEFORE extending the WGAN-GP to a new attack class.
Place at: <project>/diagnostics/gate_diagnostic.py        Run: python gate_diagnostic.py

Question it answers, per attack type: is the weakness caused by
   SCARCE   too few training flows            -> oversampling / WGAN-GP can help
   SHIFT    train->test distribution shift    -> synthetic data from TRAIN will not fix it
   OVERLAP  attack looks like normal traffic  -> needs better features / models, not more samples

Four checks (no test labels are used to fit anything; the test file is only scored):
 1. class counts in train / val / test
 2. saved baseline RF: one-vs-Normal AUC per attack type on VAL (IID with train) vs TEST
 3. specialist RF (attack-type vs Normal) trained on train only: AUC on VAL vs TEST
 4. domain classifier: can a model tell TRAIN flows from TEST flows of the same class?
    AUC ~0.5 = no shift, AUC -> 1.0 = strong shift; top features show WHAT shifted.
The verdict thresholds (--scarce-below, --shift-gap, --overlap-auc) are my heuristic
choices, not established standards: read the numbers, not only the labels.
"""
import argparse, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "shared"))
import joblib, numpy as np, pandas as pd
import scipy.sparse as sp
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from exp_common import *

ap = argparse.ArgumentParser()
ap.add_argument("--specialists", nargs="+", default=["Fuzzers", "Shellcode"])
ap.add_argument("--max-per-group", type=int, default=20000)
ap.add_argument("--scarce-below", type=int, default=2000, help="train flows below this = scarce")
ap.add_argument("--shift-gap", type=float, default=0.03, help="AUC(val)-AUC(test) above this = shift")
ap.add_argument("--overlap-auc", type=float, default=0.95, help="test AUC below this = weak separation")
args = ap.parse_args()

S = load_splits("val")
raw_tr, raw_val, raw_te = S["raw_train"], S["raw_val"], S["raw_test_full"]
pre = joblib.load(BASE_DIR / "preprocessor.pkl")
X_tr, X_val, X_te = featurize(pre, raw_tr), featurize(pre, raw_val), featurize(pre, raw_te)
try:
    feat_names = list(pre.get_feature_names_out())
except Exception:
    feat_names = None
rng = np.random.default_rng(RANDOM_STATE)
cats = sorted(c for c in raw_tr["attack_cat"].unique() if c != "Normal")


def vstack(a, b):
    return sp.vstack([a, b]).tocsr() if sp.issparse(a) else np.vstack([a, b])


def vs_normal_auc(prob, raw, cat):
    m = (raw["attack_cat"].values == cat) | (raw["label"].values == 0)
    y = (raw["attack_cat"].values[m] == cat).astype(int)
    return float(roc_auc_score(y, prob[m])) if 0 < y.sum() < len(y) else np.nan


def subsample(idx, k):
    return idx if len(idx) <= k else rng.choice(idx, size=k, replace=False)


# ---- 1. counts ----
rows = []
for c in ["Normal"] + cats:
    rows.append({"category": c,
                 "n_train": int((raw_tr.attack_cat == c).sum()), "pct_train": (raw_tr.attack_cat == c).mean() * 100,
                 "n_val": int((raw_val.attack_cat == c).sum()),
                 "n_test": int((raw_te.attack_cat == c).sum()), "pct_test": (raw_te.attack_cat == c).mean() * 100})
counts = pd.DataFrame(rows).set_index("category")

# ---- 2. saved baseline: one-vs-Normal AUC on val vs test ----
base = joblib.load(BASE_DIR / "random_forest.pkl")
p_val, p_te = base.predict_proba(X_val)[:, 1], base.predict_proba(X_te)[:, 1]
for c in cats:
    counts.loc[c, "AUC_val(baseline RF)"] = vs_normal_auc(p_val, raw_val, c)
    counts.loc[c, "AUC_test(baseline RF)"] = vs_normal_auc(p_te, raw_te, c)

# ---- 3. specialist attack-vs-Normal RF ----
for c in args.specialists:
    if c not in cats:
        print(f"[skip] specialist '{c}' not in training data")
        continue
    sel = np.where((raw_tr.attack_cat.values == c) | (raw_tr.label.values == 0))[0]
    spec = RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE, n_jobs=-1,
                                  class_weight="balanced")
    spec.fit(X_tr[sel], (raw_tr.attack_cat.values[sel] == c).astype(int))
    counts.loc[c, "AUC_val(specialist)"] = vs_normal_auc(spec.predict_proba(X_val)[:, 1], raw_val, c)
    counts.loc[c, "AUC_test(specialist)"] = vs_normal_auc(spec.predict_proba(X_te)[:, 1], raw_te, c)

# ---- 4. domain classifier: train vs test, same class ----
dom_rows, dom_top = [], {}
for g in ["Normal"] + [c for c in args.specialists if c in cats]:
    ia = subsample(np.where(raw_tr.attack_cat.values == g)[0], args.max_per_group)
    ib = subsample(np.where(raw_te.attack_cat.values == g)[0], args.max_per_group)
    if len(ia) < 50 or len(ib) < 50:
        continue
    X = vstack(X_tr[ia], X_te[ib]); y = np.r_[np.zeros(len(ia)), np.ones(len(ib))]
    Xa, Xb, ya, yb = train_test_split(X, y, test_size=0.3, random_state=RANDOM_STATE, stratify=y)
    dc = RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE, n_jobs=-1).fit(Xa, ya)
    auc = float(roc_auc_score(yb, dc.predict_proba(Xb)[:, 1]))
    imp = dc.feature_importances_
    top = np.argsort(imp)[::-1][:8]
    names = [(feat_names[i] if feat_names else f"f{i}") for i in top]
    dom_rows.append({"group": g, "n_train_used": len(ia), "n_test_used": len(ib), "domain_AUC": auc,
                     "top_shift_features": ", ".join(f"{n}({imp[i]:.2f})" for n, i in zip(names, top))})
    counts.loc[g, "domain_AUC(train-vs-test)"] = auc
dom = pd.DataFrame(dom_rows)

# ---- verdict ----
def verdict(c):
    r = counts.loc[c]; tags = []
    if r["n_train"] < args.scarce_below:
        tags.append("SCARCE")
    gap = r["AUC_val(baseline RF)"] - r["AUC_test(baseline RF)"]
    dom_auc = r.get("domain_AUC(train-vs-test)", np.nan)
    if gap > args.shift_gap or (not np.isnan(dom_auc) and dom_auc > 0.85):
        tags.append("SHIFT")
    if r["AUC_test(baseline RF)"] < args.overlap_auc and "SHIFT" not in tags:
        tags.append("OVERLAP")
    return "+".join(tags) if tags else "ok"


counts["AUC_gap(val-test)"] = counts["AUC_val(baseline RF)"] - counts["AUC_test(baseline RF)"]
counts["verdict"] = [verdict(c) if c in cats else "" for c in counts.index]
counts.to_csv(RESULTS_DIR / "diag_gate_summary.csv")
dom.to_csv(RESULTS_DIR / "diag_gate_domain_shift.csv", index=False)

pd.set_option("display.width", 250); pd.set_option("display.max_columns", None)
pd.set_option("display.max_colwidth", 120)
print("\n=== Class counts + separability (AUC = attack type vs Normal) ===")
print(counts.round(4).to_string())
print("\n=== Domain classifier: train vs test flows of the same class ===")
print(dom.round(4).to_string(index=False))
print("\n=== How to read the verdicts ===")
print("SCARCE only          -> WGAN-GP / oversampling is a sensible tool for this class.")
print("SHIFT (+/- SCARCE)   -> synthetic data drawn from TRAIN reproduces the train distribution; "
      "do not expect it to fix test errors. Report it as a limitation, or adapt (e.g. unlabeled test-domain data).")
print("OVERLAP              -> class is not separable from Normal with these features; "
      "more samples will not help much; look at features / model capacity.")
print("ok                   -> nothing to fix.")
print("Saved diag_gate_summary.csv and diag_gate_domain_shift.csv to", RESULTS_DIR)