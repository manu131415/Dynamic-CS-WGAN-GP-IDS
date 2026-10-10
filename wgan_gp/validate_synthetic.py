
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from wgan_gp.config import (
    TRAIN_PATH,
    TARGET_COLUMN,
    ATTACK_CATEGORY_COLUMN,
    TARGET_ATTACK_CATEGORY,
    RANDOM_STATE,
    VALIDATION_SIZE,
)

SYNTHETIC_PATH = (
    Path(__file__).resolve().parent.parent
    / "outputs"
    / "synthetic"
    / "shellcode_v2_synthetic_epoch_100.csv"
)

# These columns represent metadata or labels, not model input features.
EXCLUDED_COLUMNS = {"id", "attack_cat", "label"}

INTEGER_COLUMNS = {
    "spkts", "dpkts", "sbytes", "dbytes", "sloss", "dloss",
    "sttl", "dttl", "swin", "stcpb", "dtcpb", "dwin",
    "trans_depth", "response_body_len", "ct_srv_src",
    "ct_state_ttl", "ct_dst_ltm", "ct_src_dport_ltm",
    "ct_dst_sport_ltm", "ct_dst_src_ltm", "ct_ftp_cmd",
    "ct_flw_http_mthd", "ct_src_ltm", "ct_srv_dst",
}


def main():
    print("=" * 60)
    print("SYNTHETIC SHELLCODE DATA VALIDATION")
    print("=" * 60)

    # 1. Load the original training dataset and reproduce its split.
    df = pd.read_csv(TRAIN_PATH)

    train_partition, _ = train_test_split(
        df,
        test_size=VALIDATION_SIZE,
        random_state=RANDOM_STATE,
        stratify=df[TARGET_COLUMN],
    )

    real = train_partition[
        train_partition[ATTACK_CATEGORY_COLUMN]
        == TARGET_ATTACK_CATEGORY
    ].copy()

    synthetic = pd.read_csv(SYNTHETIC_PATH)

    print(f"\nReal Shellcode training samples: {len(real)}")
    print(f"Synthetic samples: {len(synthetic)}")

    # 2. Check that all expected feature columns exist.
    feature_cols = [
        col for col in real.columns
        if col not in EXCLUDED_COLUMNS
    ]

    missing_columns = [
        col for col in feature_cols
        if col not in synthetic.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Synthetic CSV is missing feature columns: {missing_columns}"
        )

    synthetic_features = synthetic[feature_cols].copy()
    real_features = real[feature_cols].copy()

    print(f"Expected feature count: {len(feature_cols)}")
    print("Feature columns: PASS")

    # 3. Check missing and infinite values.
    missing_count = synthetic_features.isna().sum().sum()
    numeric_synthetic = synthetic_features.select_dtypes(
        include=[np.number]
    )

    infinite_count = np.isinf(
        numeric_synthetic.to_numpy(dtype=float)
    ).sum()

    print(f"\nMissing feature values: {missing_count}")
    print(f"Infinite numeric values: {infinite_count}")

    # 4. Check labels and attack category if included in the CSV.
    if TARGET_COLUMN in synthetic.columns:
        print(
            "Binary labels:",
            synthetic[TARGET_COLUMN].value_counts().to_dict(),
        )
        assert (synthetic[TARGET_COLUMN] == 1).all(), (
            "Synthetic samples must have attack label 1."
        )

    if ATTACK_CATEGORY_COLUMN in synthetic.columns:
        print(
            "Attack categories:",
            synthetic[ATTACK_CATEGORY_COLUMN].value_counts().to_dict(),
        )
        assert (
            synthetic[ATTACK_CATEGORY_COLUMN]
            == TARGET_ATTACK_CATEGORY
        ).all(), "Unexpected synthetic attack category."

    # 5. Check numeric feature ranges against real training Shellcode.
    range_violations = []

    for col in numeric_synthetic.columns:
        real_values = pd.to_numeric(
            real_features[col], errors="coerce"
        ).dropna()

        generated_values = pd.to_numeric(
            synthetic_features[col], errors="coerce"
        ).dropna()

        if real_values.empty or generated_values.empty:
            continue

        real_min = real_values.min()
        real_max = real_values.max()

        outside = (
            (generated_values < real_min - 1e-6)
            | (generated_values > real_max + 1e-6)
        ).sum()

        if outside:
            range_violations.append((col, int(outside)))

    print("\nNumeric range violations:", range_violations)

    # 6. Check integer-valued features.
    integer_violations = []

    for col in INTEGER_COLUMNS.intersection(feature_cols):
        values = pd.to_numeric(
            synthetic_features[col], errors="coerce"
        ).dropna().to_numpy(dtype=float)

        invalid = np.abs(values - np.round(values)) > 1e-6

        if invalid.any():
            integer_violations.append((col, int(invalid.sum())))

    print("Integer feature violations:", integer_violations)

    # 7. Check binary-valued features based on the real subset.
    binary_violations = []

    for col in feature_cols:
        real_values = real_features[col].dropna()

        if pd.api.types.is_numeric_dtype(real_values):
            unique_values = set(real_values.unique())

            if unique_values.issubset({0, 1, 0.0, 1.0}):
                generated_values = pd.to_numeric(
                    synthetic_features[col], errors="coerce"
                ).dropna()

                invalid = ~generated_values.isin([0, 1])

                if invalid.any():
                    binary_violations.append(
                        (col, int(invalid.sum()))
                    )

    print("Binary feature violations:", binary_violations)

    # 8. Check duplicates and diversity.
    duplicates = synthetic_features.duplicated().sum()
    unique_rows = len(synthetic_features.drop_duplicates())

    print(f"\nDuplicate synthetic rows: {duplicates}")
    print(f"Unique synthetic rows: {unique_rows}/{len(synthetic_features)}")

    # 9. Compare categorical values against real training samples.
    for col in ["proto", "service", "state"]:
        if col in feature_cols:
            allowed = set(real_features[col].dropna().unique())
            generated = set(synthetic_features[col].dropna().unique())
            unexpected = generated - allowed

            print(f"Unexpected {col} values: {unexpected}")

    # 10. Overall summary.
    print("\n" + "=" * 60)

    basic_checks_pass = (
        missing_count == 0
        and infinite_count == 0
        and not missing_columns
    )

    print("Basic data integrity:", "PASS" if basic_checks_pass else "FAIL")
    print("Review all violations above before using augmentation.")
    print("=" * 60)


if __name__ == "__main__":
    main()
