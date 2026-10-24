"""
MULTI-SEED ROBUSTNESS: A (baseline) vs C (dynamic cost, round 0) vs D (feedback, best round).
Place at: <project>/feedback_cost/multi_seed_comparison.py

Only MODEL randomness varies across seeds (RF trees, MLP init/shuffling/resampling).
The data splits and cost objects stay fixed (same as all earlier experiments).
Logistic regression is deterministic, so it is excluded by default (seeds would be identical).

Run:   python multi_seed_comparison.py --tuning-source test_calib --seeds 42 43 44 45 46
Quick: python multi_seed_comparison.py --seeds 42 43 --rounds 4 --classifiers random_forest

Rough laptop time: each (seed, classifier) = 1 baseline fit + up to rounds+1 weighted fits.
RF ~0.5 min per fit, MLP ~1 min per fit (estimates; depends on CPU). Partial results are
written after every (seed, classifier), so you can stop and still use what finished.
"""
import argparse, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "shared"))
import joblib, numpy as np, pandas as pd
from feedback_loop import *

ap = argparse.ArgumentParser()
ap.add_argument("--tuning-source", default="test_calib", choices=["val", "test_calib"])
ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44, 45, 46])
ap.add_argument("--classifiers", nargs="+", default=["random_forest", "mlp"])
ap.add_argument("--rounds", type=int, default=6)
ap.add_argument("--rf-trees", type=int, default=100)
ap.add_argument("--eta", type=float, default=0.35)
ap.add_argument("--patience", type=int, default=2)
ap.add_argument("--target-slack", type=float, default=1.0)
ap.add_argument("--fpr-cap", type=float, default=0.10)
args = ap.parse_args()
tag = args.tuning_source
pretty = {v: k for k, v in CLASSIFIERS.items()}
CFG_A, CFG_C, CFG_D = "A. Baseline", "C. Dynamic (round 0)", "D. Feedback (best)"
FZ = f"Fuzzers_Recall@FPR{FUZZ_CAP}"

ctx = make_context(tag)
rows, hist = [], []
out_raw = RESULTS_DIR / f"multiseed_raw_{tag}.csv"

for seed in args.seeds:
    for name in args.classifiers:
        print(f"\n##### seed {seed} | {name} | tuning={tag}")
        A = fit_model(ctx, name, seed, rf_trees=args.rf_trees)
        pA_t, pA_f = A.predict_proba(ctx.X_t)[:, 1], A.predict_proba(ctx.X_f)[:, 1]

        if seed == args.seeds[0] and (BASE_DIR / f"{name}.pkl").exists():     # sanity check
            saved = joblib.load(BASE_DIR / f"{name}.pkl")
            ps = eval_row(ctx, saved.predict_proba(ctx.X_f)[:, 1], 0.5)
            pa = eval_row(ctx, pA_f, 0.5)
            print(f"  [check] saved baseline: DynCost@0.5={ps['Dynamic_Avg_Cost']:.4f} "
                  f"MeanRecall@FPR={ps['MeanRecall@FPR']:.4f} | retrained A: "
                  f"{pa['Dynamic_Avg_Cost']:.4f} / {pa['MeanRecall@FPR']:.4f}  "
                  f"(large gaps => your baseline used different hyper-parameters, e.g. class_weight)")

        res = run_feedback_loop(ctx, name, seed, rounds=args.rounds, eta=args.eta,
                                target_mode="baseline", p_base=pA_t, slack=args.target_slack,
                                patience=args.patience, rf_trees=args.rf_trees, verbose=True)
        hist += res["history"]
        r0, best = res["round0"], res["best"]
        for cfg, p_t, p_f in [(CFG_A, pA_t, pA_f),
                              (CFG_C, r0["p_t"], r0["model"].predict_proba(ctx.X_f)[:, 1]),
                              (CFG_D, best["p_t"], best["model"].predict_proba(ctx.X_f)[:, 1])]:
            for op, thr in operating_points(ctx, p_t, args.fpr_cap).items():
                rows.append({"Seed": seed, "Classifier": pretty[name], "Config": cfg,
                             "OperatingPoint": op, "BestRound": best["round"] if cfg == CFG_D else 0,
                             **eval_row(ctx, p_f, thr)})
        pd.DataFrame(rows).to_csv(out_raw, index=False)                       # partial save

df = pd.DataFrame(rows)
pd.DataFrame(hist).to_csv(RESULTS_DIR / f"multiseed_history_{tag}.csv", index=False)

# ---------------- summary: mean +- std ----------------
METRICS = ["Dynamic_Avg_Cost", "Recall", "FPR", "MeanRecall@FPR", "ROC-AUC", FZ]
g = df.groupby(["Classifier", "OperatingPoint", "Config"])[METRICS].agg(["mean", "std"])
summary = pd.DataFrame(index=g.index)
for m in METRICS:
    summary[m] = [f"{a:.4f} ± {0 if np.isnan(b) else b:.4f}" for a, b in zip(g[(m, "mean")], g[(m, "std")])]
summary.to_csv(RESULTS_DIR / f"multiseed_summary_{tag}.csv")

# ---------------- paired differences ----------------
try:
    from scipy.stats import ttest_rel
except Exception:
    ttest_rel = None
HIGHER = {"Dynamic_Avg_Cost": False, "MeanRecall@FPR": True, "ROC-AUC": True, FZ: True}
paired_rows = []
for clf in df.Classifier.unique():
    for other_label, other in [("D vs A", CFG_A), ("D vs C", CFG_C)]:
        for metric, hb in HIGHER.items():
            ops = ["default", "cost", "cap"] if metric == "Dynamic_Avg_Cost" else ["default"]
            for op in ops:
                sel = df[(df.Classifier == clf) & (df.OperatingPoint == op)]
                d = sel[sel.Config == CFG_D].set_index("Seed")[metric]
                o = sel[sel.Config == other].set_index("Seed")[metric]
                diff = (d - o).dropna()
                wins = int(((diff > 0) if hb else (diff < 0)).sum())
                p = (ttest_rel(d.loc[diff.index], o.loc[diff.index]).pvalue
                     if ttest_rel is not None and len(diff) >= 3 and diff.std() > 0 else np.nan)
                paired_rows.append({"Classifier": clf, "Comparison": other_label, "Metric": metric,
                                    "OperatingPoint": op, "Mean_diff(D-other)": diff.mean(),
                                    "Std_diff": diff.std(), "D_better_in": f"{wins}/{len(diff)} seeds",
                                    "paired_t_p": p, "better_direction": "higher" if hb else "lower"})
paired = pd.DataFrame(paired_rows)
paired.to_csv(RESULTS_DIR / f"multiseed_paired_{tag}.csv", index=False)

pd.set_option("display.width", 250); pd.set_option("display.max_columns", None)
print(f"\n=== MULTI-SEED SUMMARY | tuning={tag} | seeds={args.seeds} | final n={len(ctx.raw_f)} ===")
print(summary.to_string())
print("\n=== PAIRED DIFFERENCES (D minus other) ===")
print(paired.round(4).to_string(index=False))
print("\nNote: with 3-5 seeds p-values are indicative only; look at 'D_better_in' and the std.")
print("Saved to", RESULTS_DIR)