"""
Diagnostic before Experiment 3: WHERE do false positives / misses come from?
Place at: <project>/threshold_opt/group_diagnostic.py
Run:  python group_diagnostic.py --tuning-source test_calib

Output 1: FPR by service (normal flows) at threshold 0.5, A vs C  -> is FP concentrated?
Output 2: per-attack-type recall at a MATCHED operating point (FPR <= cap on tuning set),
          A vs B vs C -> does cost-weighting help any attack type once FPR is equal?
If FP is spread evenly over services and per-attack recall at matched FPR is the same for
A/B/C, group-level feedback has little leverage.
"""
import argparse, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "shared"))
import joblib, numpy as np, pandas as pd
from exp_common import *

ap = argparse.ArgumentParser()
ap.add_argument("--tuning-source", default="test_calib", choices=["val", "test_calib"])
ap.add_argument("--fpr-cap", type=float, default=0.10)
args = ap.parse_args()

S = load_splits(args.tuning_source)
pre = joblib.load(BASE_DIR / "preprocessor.pkl")
raw_t, raw_f = S["raw_tune"], S["raw_final"]
X_t, X_f = featurize(pre, raw_t), featurize(pre, raw_f)
y_t, y_f = raw_t["label"].values, raw_f["label"].values

CONFIGS = {"A": lambda f: BASE_DIR / f"{f}.pkl",
           "B": lambda f: CS_DIR / f"{f}_static.pkl",
           "C": lambda f: CS_DIR / f"{f}_dynamic.pkl"}
fp_rows, rec_rows = [], []
normal = raw_f["label"].values == 0
svc = raw_f["service"].values
for cfg, pf in CONFIGS.items():
    for clf, fname in CLASSIFIERS.items():
        m = joblib.load(pf(fname))
        p_t, p_f = m.predict_proba(X_t)[:, 1], m.predict_proba(X_f)[:, 1]
        for s in pd.unique(svc[normal]):
            k = normal & (svc == s)
            if k.sum() >= 50:
                fp_rows.append({"Config": cfg, "Classifier": clf, "service": s,
                                "n_normal": int(k.sum()), "FPR@0.5": float((p_f[k] >= .5).mean())})
        thr = fpr_cap_threshold(y_t, p_t, args.fpr_cap)
        pred = p_f >= thr
        atk = raw_f["label"].values == 1
        for cat in sorted(raw_f.loc[atk, "attack_cat"].unique()):
            k = atk & (raw_f["attack_cat"].values == cat)
            rec_rows.append({"Config": cfg, "Classifier": clf, "attack_cat": cat, "n": int(k.sum()),
                             "thr": thr, "recall@FPRcap": float(pred[k].mean()),
                             "overall_FPR": float(pred[normal].mean())})
pd.set_option("display.width", 250)
fp = pd.DataFrame(fp_rows)
print("\nFPR@0.5 by service (A vs C):")
print(fp[fp.Config.isin(["A", "C"])].pivot_table(index=["Classifier", "service", "n_normal"],
      columns="Config", values="FPR@0.5").round(3).to_string())
rc = pd.DataFrame(rec_rows)
print(f"\nRecall by attack type at FPR<={args.fpr_cap} (threshold from tuning set):")
print(rc.pivot_table(index=["Classifier", "attack_cat", "n"], columns="Config",
                     values="recall@FPRcap").round(3).to_string())
fp.to_csv(RESULTS_DIR / f"diag_fpr_by_service_{args.tuning_source}.csv", index=False)
rc.to_csv(RESULTS_DIR / f"diag_recall_matched_fpr_{args.tuning_source}.csv", index=False)