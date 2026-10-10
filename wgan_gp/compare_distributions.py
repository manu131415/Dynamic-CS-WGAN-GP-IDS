
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.stats import ks_2samp
from sklearn.model_selection import train_test_split

from wgan_gp.config import (
    TRAIN_PATH,
    OUTPUT_DIR,
    TARGET_COLUMN,
    ATTACK_CATEGORY_COLUMN,
    TARGET_ATTACK_CATEGORY,
    RANDOM_STATE,
    VALIDATION_SIZE,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent

SYNTHETIC_PATH = (
    OUTPUT_DIR / "synthetic"
    / "shellcode_v2_synthetic_epoch_100.csv"
)

RESULT_DIR = OUTPUT_DIR / "results"
PLOT_DIR = RESULT_DIR / "distribution_plots"

CATEGORICAL_COLUMNS = ["proto", "service", "state"]
EXCLUDED_COLUMNS = {"id", "attack_cat", "label"}

# A small, representative set of numerical features to plot.
PLOT_FEATURES = ["dur", "spkts", "sbytes", "dbytes", "rate", "sload"]


def total_variation_distance(real, synthetic):
    """Compare categorical proportions; 0 means identical proportions."""
    real_counts = real.astype(str).value_counts(normalize=True)
    synthetic_counts = synthetic.astype(str).value_counts(normalize=True)

    categories = real_counts.index.union(synthetic_counts.index)

    p = real_counts.reindex(categories, fill_value=0)
    q = synthetic_counts.reindex(categories, fill_value=0)

    return 0.5 * np.abs(p - q).sum()


def main():
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    PLOT_DIR.mkdir(parents=True, exist_ok=True)

    if not SYNTHETIC_PATH.exists():
        raise FileNotFoundError(f"Missing synthetic file: {SYNTHETIC_PATH}")

    # Reproduce the same training partition used by the baseline.
    df = pd.read_csv(TRAIN_PATH)

    train_df, _ = train_test_split(
        df,
        test_size=VALIDATION_SIZE,
        random_state=RANDOM_STATE,
        stratify=df[TARGET_COLUMN],
    )

    real = train_df[
        train_df[ATTACK_CATEGORY_COLUMN] == TARGET_ATTACK_CATEGORY
    ].copy()

    synthetic_df = pd.read_csv(SYNTHETIC_PATH)

    feature_columns = [
        c for c in real.columns if c not in EXCLUDED_COLUMNS
    ]

    missing = [c for c in feature_columns if c not in synthetic_df.columns]
    if missing:
        raise ValueError(f"Missing synthetic feature columns: {missing}")

    synthetic = synthetic_df[feature_columns].copy()
    real = real[feature_columns].copy()

    print(f"Real Shellcode training samples: {len(real)}")
    print(f"Synthetic Shellcode samples: {len(synthetic)}")

    results = []

    # Numerical feature comparison using the KS statistic.
    numerical_columns = [
        c for c in feature_columns
        if c not in CATEGORICAL_COLUMNS
        and pd.api.types.is_numeric_dtype(real[c])
    ]

    for col in numerical_columns:
        real_values = pd.to_numeric(real[col], errors="coerce").dropna()
        fake_values = pd.to_numeric(synthetic[col], errors="coerce").dropna()

        if real_values.empty or fake_values.empty:
            continue

        ks = ks_2samp(real_values, fake_values)

        results.append({
            "feature": col,
            "feature_type": "numerical",
            "metric": "KS_statistic",
            "distance": float(ks.statistic),
            "real_samples": len(real_values),
            "synthetic_samples": len(fake_values),
        })

    # Categorical comparison using total variation distance.
    for col in CATEGORICAL_COLUMNS:
        if col not in feature_columns:
            continue

        distance = total_variation_distance(real[col], synthetic[col])

        results.append({
            "feature": col,
            "feature_type": "categorical",
            "metric": "Total_variation_distance",
            "distance": float(distance),
            "real_samples": real[col].notna().sum(),
            "synthetic_samples": synthetic[col].notna().sum(),
        })

    results_df = pd.DataFrame(results)
    results_path = RESULT_DIR / "distribution_comparison.csv"
    results_df.to_csv(results_path, index=False)

    print("\nSmallest numerical KS distances:")
    print(
        results_df[results_df["feature_type"] == "numerical"]
        .sort_values("distance")
        .head(10)
        .to_string(index=False)
    )

    print("\nLargest numerical KS distances:")
    print(
        results_df[results_df["feature_type"] == "numerical"]
        .sort_values("distance", ascending=False)
        .head(10)
        .to_string(index=False)
    )

    print("\nCategorical distances:")
    print(
        results_df[results_df["feature_type"] == "categorical"]
        .to_string(index=False)
    )

    # Save representative real-vs-synthetic histograms.
    for col in PLOT_FEATURES:
        if col not in numerical_columns:
            continue

        real_values = pd.to_numeric(real[col], errors="coerce").dropna()
        fake_values = pd.to_numeric(synthetic[col], errors="coerce").dropna()

        if real_values.empty or fake_values.empty:
            continue

        plt.figure(figsize=(8, 5))
        plt.hist(
            real_values,
            bins=40,
            density=True,
            alpha=0.5,
            label="Real Shellcode",
        )
        plt.hist(
            fake_values,
            bins=40,
            density=True,
            alpha=0.5,
            label="Synthetic Shellcode",
        )
        plt.xlabel(col)
        plt.ylabel("Density")
        plt.title(f"Real vs Synthetic Shellcode: {col}")
        plt.legend()
        plt.tight_layout()
        plt.savefig(PLOT_DIR / f"{col}_distribution.png", dpi=150)
        plt.close()

    print(f"\nStatistics saved to: {results_path}")
    print(f"Plots saved in: {PLOT_DIR}")


if __name__ == "__main__":
    main()
