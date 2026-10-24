"""
EXPERIMENT 2 - cost-aware threshold selection.
Place at: <project>/threshold_opt/threshold_optimization.py
No retraining: re-uses the 9 models from baseline/ and cost_sensitive/.

Run:  python threshold_optimization.py --tuning-source val
      python threshold_optimization.py --tuning-source test_calib

Strategies (threshold always chosen on the TUNING set, reported on the FINAL set):
  default_0.50      : the usual 0.5
  static_opt        : minimises static avg cost
  dynamic_opt       : minimises dynamic avg cost
  fpr_cap_<x>       : highest recall with FPR <= x on the tuning set
  ORACLE_dynamic    : threshold picked ON the final set. DIAGNOSTIC ONLY, never report as a result.
"""
import argparse, json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "shared"))
import joblib, numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from exp_common import *

ap = argparse.ArgumentParser()
ap.add_argument("--tuning-source", default="val", choices=["val", "test_calib"])
ap.add_argument("--fpr-cap", type=float, default=0.10)
args = ap.parse_args()
tag = args.tuning_source

S = load_splits(args.tuning_source)
pre = joblib.load(BASE_DIR / "preprocessor.pkl")
static_cost, dynamic_cost = load_fixed_costs()

sets = {k: (featurize(pre, S[k]), S[k]["label"].values, S[k])
        for k in ["raw_val", "raw_tune", "raw_final", "raw_test_full"]}
y_t, raw_t = sets["raw_tune"][1], sets["raw_tune"][2]
y_f, raw_f = sets["raw_final"][1], sets["raw_final"][2]
cfn_t, cfp_t = dynamic_cost.costs(raw_t)
sfn_t, sfp_t = static_cost.costs(raw_t)
cfn_f, cfp_f = dynamic_cost.costs(raw_f)

CONFIGS = {"A. Baseline": lambda f: BASE_DIR / f"{f}.pkl",
           "B. Static-cost": lambda f: CS_DIR / f"{f}_static.pkl",
           "C. Dynamic-cost": lambda f: CS_DIR / f"{f}_dynamic.pkl"}

rows, shift_rows, thr_log, curves = [], [], {}, {}
for cfg, path_fn in CONFIGS.items():
    for clf, fname in CLASSIFIERS.items():
        model = joblib.load(path_fn(fname))
        P = {k: model.predict_proba(v[0])[:, 1] for k, v in sets.items()}
        p_t, p_f = P["raw_tune"], P["raw_final"]

        # --- shift diagnostic at the default threshold ---
        for nm, key in [("val (IID with train)", "raw_val"), ("full test file", "raw_test_full")]:
            m = evaluate(sets[key][1], P[key], 0.5, sets[key][2], static_cost, dynamic_cost)
            shift_rows.append({"Configuration": cfg, "Classifier": clf, "Set": nm,
                               "FPR@0.5": m["FPR"], "Recall@0.5": m["Recall"],
                               "Dynamic_Avg_Cost@0.5": m["Dynamic_Avg_Cost"]})

        thr = {"default_0.50": 0.5,
               "static_opt": best_threshold(y_t, p_t, sfn_t, sfp_t),
               "dynamic_opt": best_threshold(y_t, p_t, cfn_t, cfp_t),
               f"fpr_cap_{args.fpr_cap:.2f}": fpr_cap_threshold(y_t, p_t, args.fpr_cap),
               "ORACLE_dynamic (diagnostic)": best_threshold(y_f, p_f, cfn_f, cfp_f)}
        thr_log[f"{cfg} | {clf}"] = thr
        for strat, t in thr.items():
            rows.append({"Configuration": cfg, "Classifier": clf, "Strategy": strat,
                         **evaluate(y_f, p_f, t, raw_f, static_cost, dynamic_cost)})
        curves[(cfg, clf)] = (cost_curve(y_f, p_f, cfn_f, cfp_f), cost_curve(y_t, p_t, cfn_t, cfp_t),
                              thr["dynamic_opt"])

res = pd.DataFrame(rows)
res.to_csv(RESULTS_DIR / f"exp2_threshold_results_{tag}.csv", index=False)
pd.DataFrame(shift_rows).to_csv(RESULTS_DIR / f"exp2_shift_diagnostic_{tag}.csv", index=False)
json.dump(thr_log, open(RESULTS_DIR / f"exp2_thresholds_{tag}.json", "w"), indent=2)

pd.set_option("display.width", 250); pd.set_option("display.max_columns", None)
cols = ["Configuration", "Classifier", "Strategy", "Threshold", "Recall", "FPR", "FNR",
        "F1", "Static_Avg_Cost", "Dynamic_Avg_Cost"]
print(f"\n=== Exp 2 | tuning={tag} | final set n={len(raw_f)} ===")
print(res[cols].round(4).to_string(index=False))
print("\nShift diagnostic (same model, same threshold 0.5):")
print(pd.DataFrame(shift_rows).round(4).to_string(index=False))

# ---- plot: dynamic cost vs threshold ----
colors = {"A. Baseline": "#8c8c8c", "B. Static-cost": "#4c78a8", "C. Dynamic-cost": "#e45756"}
fig, axes = plt.subplots(1, 3, figsize=(18, 4.5), sharey=True)
for ax, clf in zip(axes, CLASSIFIERS):
    for cfg in CONFIGS:
        cf, ct, tstar = curves[(cfg, clf)]
        ax.plot(GRID, cf, color=colors[cfg], label=f"{cfg} (final)")
        ax.plot(GRID, ct, color=colors[cfg], ls="--", alpha=.5)
        ax.axvline(tstar, color=colors[cfg], ls=":", alpha=.6)
    ax.axvline(0.5, color="k", lw=.8)
    ax.set_title(clf); ax.set_xlabel("threshold"); ax.grid(alpha=.3)
axes[0].set_ylabel("Dynamic avg cost / flow")
axes[0].legend(fontsize=8, title="solid=final, dashed=tuning, dotted=chosen thr")
plt.tight_layout(); plt.savefig(RESULTS_DIR / f"exp2_cost_vs_threshold_{tag}.png", dpi=150)
print("\nSaved to", RESULTS_DIR)