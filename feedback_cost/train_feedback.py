"""
EXPERIMENT 3 (v2) - feedback-based dynamic cost, single seed.
Place at: <project>/feedback_cost/train_feedback.py

v2 changes:  (1) baseline-parity targets  (2) best round chosen by ranking score.
Per classifier:  A (retrained unweighted baseline) -> feedback loop (round 0 = C, best = D).
Rows are reported at three operating points chosen on the tuning set:
default 0.5 | cost-tuned | FPR-cap.

Run:  python train_feedback.py --tuning-source test_calib
      python train_feedback.py --tuning-source val
Smoke test:  python train_feedback.py --rounds 2 --classifiers logistic_regression
"""
import argparse, json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "shared"))
import joblib, numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from feedback_loop import *

ap = argparse.ArgumentParser()
ap.add_argument("--tuning-source", default="test_calib", choices=["val", "test_calib"])
ap.add_argument("--rounds", type=int, default=8)
ap.add_argument("--classifiers", nargs="+", default=["random_forest", "logistic_regression", "mlp"])
ap.add_argument("--rf-trees", type=int, default=100)
ap.add_argument("--seed", type=int, default=42)
ap.add_argument("--target-mode", default="baseline", choices=["baseline", "fixed"])
ap.add_argument("--target-slack", type=float, default=1.0,
                help="baseline targets x slack (1.0 = parity, 1.1 = allow 10%% worse)")
ap.add_argument("--fn-target", type=float, default=0.02, help="only for --target-mode fixed")
ap.add_argument("--fpr-target", type=float, default=0.15, help="only for --target-mode fixed")
ap.add_argument("--eta", type=float, default=0.35)
ap.add_argument("--patience", type=int, default=3)
ap.add_argument("--fpr-cap", type=float, default=0.10)
args = ap.parse_args()
tag = args.tuning_source

OUT = Path(__file__).resolve().parent
(OUT / "models").mkdir(exist_ok=True)
ctx = make_context(tag)
pretty = {v: k for k, v in CLASSIFIERS.items()}
hist_rows, final_rows = [], []

for name in args.classifiers:
    print(f"\n===== {name} | tuning={tag} | seed={args.seed} | targets={args.target_mode} =====")
    A = fit_model(ctx, name, args.seed, rf_trees=args.rf_trees)
    pA_t, pA_f = A.predict_proba(ctx.X_t)[:, 1], A.predict_proba(ctx.X_f)[:, 1]
    print(f"  baseline (unweighted): tuning ranking score = {mean_recall_at_fpr(ctx.y_t, pA_t):.4f}")

    res = run_feedback_loop(ctx, name, args.seed, rounds=args.rounds, eta=args.eta,
                            target_mode=args.target_mode, fn_target=args.fn_target,
                            fpr_target=args.fpr_target, p_base=pA_t, slack=args.target_slack,
                            patience=args.patience, rf_trees=args.rf_trees)
    hist_rows += res["history"]
    r0, best = res["round0"], res["best"]

    for cfg, p_t, p_f in [("A. Baseline (retrained, unweighted)", pA_t, pA_f),
                          ("C. Dynamic (round 0)", r0["p_t"], r0["model"].predict_proba(ctx.X_f)[:, 1]),
                          (f"D. Feedback (best round {best['round']})", best["p_t"],
                           best["model"].predict_proba(ctx.X_f)[:, 1])]:
        for op, thr in operating_points(ctx, p_t, args.fpr_cap).items():
            final_rows.append({"Classifier": pretty[name], "Configuration": cfg,
                               "OperatingPoint": op, **eval_row(ctx, p_f, thr)})
    joblib.dump(best["model"], OUT / "models" / f"{name}_feedback_{tag}.pkl")
    joblib.dump(best["cost"], OUT / "models" / f"cost_feedback_{name}_{tag}.pkl")
    json.dump({"round": best["round"], "seed": args.seed, "stop": res["stop"]},
              open(OUT / "models" / f"{name}_feedback_{tag}.json", "w"))

hist = pd.DataFrame(hist_rows)
hist.to_csv(RESULTS_DIR / f"exp3v2_feedback_history_{tag}.csv", index=False)
fin = pd.DataFrame(final_rows)
fin.to_csv(RESULTS_DIR / f"exp3v2_feedback_results_{tag}.csv", index=False)
pd.set_option("display.width", 250); pd.set_option("display.max_columns", None)
cols = ["Classifier", "Configuration", "OperatingPoint", "Threshold", "Recall", "FPR",
        "Dynamic_Avg_Cost", "MeanRecall@FPR", "ROC-AUC", f"Fuzzers_Recall@FPR{FUZZ_CAP}"]
print(f"\n=== Exp 3 v2 | tuning={tag} | final n={len(ctx.raw_f)} | single seed={args.seed} ===")
print(fin[cols].round(4).to_string(index=False))

n = len(args.classifiers)
fig, axes = plt.subplots(n, 4, figsize=(20, 3.6 * n), squeeze=False)
for i, name in enumerate(args.classifiers):
    h = hist[hist.Classifier == name]
    ax = axes[i, 0]; ax.plot(h["round"], h["FNR@0.5"], "o-", label="FNR"); ax.plot(h["round"], h["FPR@0.5"], "s-", label="FPR")
    ax.set_title(f"{pretty[name]}: error rates @0.5"); ax.legend(); ax.grid(alpha=.3)
    ax = axes[i, 1]; ax.plot(h["round"], h["score_MeanRecall@FPR"], "o-")
    ax.set_title("selection score: mean recall@FPR 5/10/20%"); ax.grid(alpha=.3)
    ax = axes[i, 2]
    for c in [c for c in h.columns if c.startswith("mfn_")]: ax.plot(h["round"], h[c], label=c[4:])
    ax.set_title("FN multipliers"); ax.legend(fontsize=6, ncol=2); ax.grid(alpha=.3)
    ax = axes[i, 3]
    for c in [c for c in h.columns if c.startswith("mfp_")]: ax.plot(h["round"], h[c], label=c[4:])
    ax.set_title("FP multipliers (service)"); ax.legend(fontsize=6, ncol=2); ax.grid(alpha=.3)
    axes[i, 0].set_xlabel("round")
plt.tight_layout(); plt.savefig(RESULTS_DIR / f"exp3v2_feedback_convergence_{tag}.png", dpi=150)
print("Saved to", RESULTS_DIR)