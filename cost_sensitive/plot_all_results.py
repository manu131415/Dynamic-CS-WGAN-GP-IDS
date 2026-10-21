"""
Saves figures for every result set found in results/:

  results/figures/baseline/   <- baseline_results.csv + <model>_attack_category_results.csv
  results/figures/cost/       <- cost_comparison_results.csv + cost_comparison_per_category.csv
  results/figures/wgan_gp/    <- wgan_gp_results.csv (only if the file exists)
  results/figures/overall/    <- every configuration found, side by side

Expected format of results/wgan_gp_results.csv (same columns as the cost CSV):
  Configuration, Classifier, Accuracy, Precision, Recall, F1, ROC-AUC, FPR, FNR,
  FP, FN, Static_Total_Cost, Static_Avg_Cost, Dynamic_Total_Cost, Dynamic_Avg_Cost
(Configuration e.g. "D. WGAN-GP only", "F. WGAN-GP + dynamic cost", ...)
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RES = PROJECT_ROOT / "results"
FIG = RES / "figures"

CLASSIFIERS = ["Random Forest", "Logistic Regression", "MLP"]
PALETTE = ["#8c8c8c", "#4c78a8", "#e45756", "#54a24b", "#f58518", "#b279a2", "#72b7b2"]


def out_dir(name):
    d = FIG / name
    d.mkdir(parents=True, exist_ok=True)
    return d


def bars(ax, xlabels, series, title, ylabel, fmt="{:.3f}"):
    """series: dict name -> list of values (one per xlabel)."""
    n = len(series)
    w = 0.8 / n
    x = np.arange(len(xlabels))
    for i, (name, vals) in enumerate(series.items()):
        b = ax.bar(x + (i - (n - 1) / 2) * w, vals, w, label=name,
                   color=PALETTE[i % len(PALETTE)])
        for r, v in zip(b, vals):
            ax.text(r.get_x() + r.get_width() / 2, r.get_height(), fmt.format(v),
                    ha="center", va="bottom", fontsize=6, rotation=90 if n > 4 else 0)
    ax.set_xticks(x)
    ax.set_xticklabels(xlabels)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", alpha=0.3)


def config_series(df, column, configs):
    return {c: [df[(df.Configuration == c) & (df.Classifier == k)][column].values[0]
                for k in CLASSIFIERS] for c in configs}


def short(c):
    return c.split(". ", 1)[-1]


def heatmap(ax, pv, title, vmin=0.7):
    im = ax.imshow(pv.values, vmin=vmin, vmax=1.0, cmap="YlGn", aspect="auto")
    ax.set_xticks(range(pv.shape[1]))
    ax.set_xticklabels([short(c) for c in pv.columns], rotation=25, ha="right", fontsize=8)
    ax.set_yticks(range(pv.shape[0]))
    ax.set_yticklabels(pv.index)
    ax.set_title(title)
    for i in range(pv.shape[0]):
        for j in range(pv.shape[1]):
            ax.text(j, i, f"{pv.values[i, j]:.3f}", ha="center", va="center", fontsize=7)
    return im


# =====================================================================
# 1. BASELINE
# =====================================================================
base_path = RES / "baseline_results.csv"
if base_path.exists():
    d = out_dir("baseline")
    base = pd.read_csv(base_path)

    metrics = ["Accuracy", "Precision", "Recall", "F1", "ROC-AUC"]
    fig, ax = plt.subplots(figsize=(10, 4.5))
    bars(ax, metrics, {m: base[base.Model == m][metrics].values[0].tolist()
                       for m in base.Model}, "Baseline IDS - performance metrics", "Score")
    ax.set_ylim(0.6, 1.02)
    ax.legend()
    plt.tight_layout()
    plt.savefig(d / "baseline_metrics.png", dpi=200)
    plt.close()

    errs = ["FPR", "FNR"]
    fig, ax = plt.subplots(figsize=(6, 4.5))
    bars(ax, errs, {m: base[base.Model == m][errs].values[0].tolist() for m in base.Model},
         "Baseline IDS - error rates", "Rate")
    ax.legend()
    plt.tight_layout()
    plt.savefig(d / "baseline_error_rates.png", dpi=200)
    plt.close()

    # per-category recall (one CSV per model, written by baseline/evaluate.py)
    frames = []
    for name in CLASSIFIERS:
        p = RES / f"{name.lower().replace(' ', '_')}_attack_category_results.csv"
        if p.exists():
            t = pd.read_csv(p)
            t["Model"] = name
            frames.append(t)
    if frames:
        pc = pd.concat(frames).pivot_table(index="attack_category", columns="Model", values="recall")
        pc = pc[[c for c in CLASSIFIERS if c in pc.columns]]
        fig, ax = plt.subplots(figsize=(7, 5))
        im = heatmap(ax, pc, "Baseline IDS - per-attack-category recall", vmin=0.6)
        fig.colorbar(im, ax=ax, label="Recall")
        plt.tight_layout()
        plt.savefig(d / "baseline_per_category_recall.png", dpi=200)
        plt.close()
    print("Baseline figures saved ->", d)
else:
    print("baseline_results.csv not found - skipped baseline plots")


# =====================================================================
# 2. COST (static / dynamic)  and 3. WGAN-GP  -> same plotting routine
# =====================================================================
def plot_config_set(csv_name, cat_csv_name, folder, label):
    p = RES / csv_name
    if not p.exists():
        print(f"{csv_name} not found - skipped {label} plots")
        return None
    d = out_dir(folder)
    df = pd.read_csv(p)
    configs = list(dict.fromkeys(df.Configuration))

    # cost
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    bars(axes[0], CLASSIFIERS, config_series(df, "Static_Avg_Cost", configs),
         f"{label} - avg cost per flow (static cost)", "Avg cost")
    bars(axes[1], CLASSIFIERS, config_series(df, "Dynamic_Avg_Cost", configs),
         f"{label} - avg cost per flow (dynamic cost)", "Avg cost")
    axes[0].legend(fontsize=7)
    plt.tight_layout()
    plt.savefig(d / f"{folder}_cost_comparison.png", dpi=200)
    plt.close()

    # detection metrics
    fig, axes = plt.subplots(1, 3, figsize=(17, 4.8))
    for ax, col, t in zip(axes, ["Recall", "FPR", "F1"],
                          ["Recall", "False positive rate", "F1 score"]):
        bars(ax, CLASSIFIERS, config_series(df, col, configs), f"{label} - {t}", col)
    axes[0].legend(fontsize=7)
    plt.tight_layout()
    plt.savefig(d / f"{folder}_detection_metrics.png", dpi=200)
    plt.close()

    # error counts
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    bars(axes[0], CLASSIFIERS, config_series(df, "FN", configs),
         f"{label} - missed attacks (FN)", "Count", fmt="{:.0f}")
    bars(axes[1], CLASSIFIERS, config_series(df, "FP", configs),
         f"{label} - false alarms (FP)", "Count", fmt="{:.0f}")
    axes[0].legend(fontsize=7)
    plt.tight_layout()
    plt.savefig(d / f"{folder}_error_counts.png", dpi=200)
    plt.close()

    # per-category heatmaps
    cp = RES / cat_csv_name
    if cp.exists():
        cat = pd.read_csv(cp)
        fig, axes = plt.subplots(1, 3, figsize=(18, 5.2), sharey=True)
        for ax, clf in zip(axes, CLASSIFIERS):
            pv = cat[cat.Classifier == clf].pivot_table(
                index="attack_cat", columns="Configuration", values="recall")
            pv = pv[[c for c in configs if c in pv.columns]]
            im = heatmap(ax, pv, clf)
        fig.colorbar(im, ax=axes, fraction=0.02, label="Recall")
        plt.savefig(d / f"{folder}_per_category_recall.png", dpi=200, bbox_inches="tight")
        plt.close()

    print(f"{label} figures saved ->", d)
    return df


cost_df = plot_config_set("cost_comparison_results.csv",
                          "cost_comparison_per_category.csv", "cost", "Cost-sensitive IDS")
wgan_df = plot_config_set("wgan_gp_results.csv",
                          "wgan_gp_per_category.csv", "wgan_gp", "WGAN-GP IDS")

# =====================================================================
# 4. OVERALL - every configuration found (A..I as they become available)
# =====================================================================
parts = [x for x in (cost_df, wgan_df) if x is not None]
if parts:
    allres = pd.concat(parts, ignore_index=True)
    allres = allres.drop_duplicates(["Configuration", "Classifier"])
    allres = allres.sort_values("Configuration")
    configs = list(dict.fromkeys(allres.Configuration))
    d = out_dir("overall")

    fig, axes = plt.subplots(1, 3, figsize=(19, 5))
    for ax, col, t in zip(axes, ["Dynamic_Avg_Cost", "Recall", "FPR"],
                          ["Avg dynamic cost per flow", "Recall", "False positive rate"]):
        bars(ax, CLASSIFIERS, config_series(allres, col, configs),
             f"All configurations - {t}", col)
    axes[0].legend(fontsize=6)
    plt.tight_layout()
    plt.savefig(d / "overall_all_configurations.png", dpi=200)
    plt.close()
    print("Overall figure saved ->", d)

print("\nDone. All figures are under", FIG)