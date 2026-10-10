
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split

from wgan_gp.config import (
    TRAIN_PATH,
    TEST_PATH,
    BASELINE_MODEL_DIR,
    OUTPUT_DIR,
    RANDOM_STATE,
    VALIDATION_SIZE,
    TARGET_ATTACK_CATEGORY,
)

# Keep all new experiment artifacts outside baseline/.
EXPERIMENT_DIR = OUTPUT_DIR / "results"
EXPERIMENT_DIR.mkdir(parents=True, exist_ok=True)

SYNTHETIC_PATH = (
    OUTPUT_DIR
    / "synthetic"
    / "shellcode_v2_synthetic_epoch_100.csv"
)

BASELINE_RF_PATH = BASELINE_MODEL_DIR / "random_forest.pkl"
PREPROCESSOR_PATH = BASELINE_MODEL_DIR / "preprocessor.pkl"

TARGET = "label"
ATTACK_CATEGORY = "attack_cat"
CATEGORICAL_FEATURES = ["proto", "service", "state"]
EXCLUDED_COLUMNS = ["id", ATTACK_CATEGORY, TARGET]


def calculate_metrics(name, model, X_test, y_test, attack_categories):
    predictions = model.predict(X_test)

    probabilities = None
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(X_test)[:, 1]

    tn, fp, fn, tp = confusion_matrix(
        y_test, predictions, labels=[0, 1]
    ).ravel()

    shellcode_mask = (
        attack_categories.astype(str) == TARGET_ATTACK_CATEGORY
    )

    shellcode_recall = np.nan
    if shellcode_mask.any():
        shellcode_recall = np.mean(
            predictions[shellcode_mask.to_numpy()] == 1
        )

    metrics = {
        "Experiment": name,
        "Accuracy": accuracy_score(y_test, predictions),
        "Precision": precision_score(
            y_test, predictions, zero_division=0
        ),
        "Recall": recall_score(
            y_test, predictions, zero_division=0
        ),
        "F1": f1_score(y_test, predictions, zero_division=0),
        "ROC_AUC": (
            roc_auc_score(y_test, probabilities)
            if probabilities is not None else np.nan
        ),
        "FPR": fp / (fp + tn) if (fp + tn) else 0.0,
        "FNR": fn / (fn + tp) if (fn + tp) else 0.0,
        "Shellcode_Recall": shellcode_recall,
        "Test_Shellcode_Count": int(shellcode_mask.sum()),
    }

    print("\n" + "=" * 65)
    print(name)
    print("=" * 65)

    for key, value in metrics.items():
        if isinstance(value, (float, np.floating)):
            print(f"{key:24s}: {value:.4f}")
        else:
            print(f"{key:24s}: {value}")

    print("\nConfusion matrix (rows=actual, columns=predicted):")
    print(confusion_matrix(y_test, predictions, labels=[0, 1]))

    return metrics


def main():
    print("Loading datasets and saved baseline artifacts...")

    for path in [
        TRAIN_PATH,
        TEST_PATH,
        SYNTHETIC_PATH,
        BASELINE_RF_PATH,
        PREPROCESSOR_PATH,
    ]:
        if not path.exists():
            raise FileNotFoundError(f"Required file not found: {path}")

    full_train = pd.read_csv(TRAIN_PATH)
    test_df = pd.read_csv(TEST_PATH)
    synthetic_df = pd.read_csv(SYNTHETIC_PATH)

    # Reproduce the exact stratified split used by baseline/train.py.
    train_df, _ = train_test_split(
        full_train,
        test_size=VALIDATION_SIZE,
        random_state=RANDOM_STATE,
        stratify=full_train[TARGET],
    )

    feature_columns = [
        column for column in train_df.columns
        if column not in EXCLUDED_COLUMNS
    ]

    missing_columns = [
        column for column in feature_columns
        if column not in synthetic_df.columns
    ]
    if missing_columns:
        raise ValueError(
            f"Synthetic CSV is missing features: {missing_columns}"
        )

    X_train = train_df[feature_columns].copy()
    y_train = train_df[TARGET].astype(int)

    X_synthetic = synthetic_df[feature_columns].copy()

    X_test = test_df[feature_columns].copy()
    y_test = test_df[TARGET].astype(int)
    attack_categories = test_df[ATTACK_CATEGORY]

    # Ensure synthetic numeric features have numeric data types.
    for column in feature_columns:
        if column not in CATEGORICAL_FEATURES:
            X_synthetic[column] = pd.to_numeric(
                X_synthetic[column], errors="coerce"
            )

    if X_synthetic.isna().any().any():
        raise ValueError(
            "Synthetic feature data contains missing/non-numeric values."
        )

    # Synthetic samples must be attack-labelled.
    if TARGET in synthetic_df.columns:
        if not (synthetic_df[TARGET] == 1).all():
            raise ValueError("Not all synthetic labels are 1.")

    if ATTACK_CATEGORY in synthetic_df.columns:
        if not (
            synthetic_df[ATTACK_CATEGORY].astype(str)
            == TARGET_ATTACK_CATEGORY
        ).all():
            raise ValueError("Unexpected synthetic attack category.")

    print(f"Original training rows: {len(X_train)}")
    print(f"Synthetic rows:         {len(X_synthetic)}")
    print(f"Test rows:              {len(X_test)}")
    print(f"Feature count:          {len(feature_columns)}")

    # Reuse the original baseline preprocessing; do not fit on test data.
    preprocessor = joblib.load(PREPROCESSOR_PATH)

    X_train_processed = preprocessor.transform(X_train)
    X_synthetic_processed = preprocessor.transform(X_synthetic)
    X_test_processed = preprocessor.transform(X_test)

    y_synthetic = np.ones(len(X_synthetic), dtype=int)

    X_augmented = np.concatenate(
        [X_train_processed, X_synthetic_processed], axis=0
    )
    y_augmented = np.concatenate(
        [y_train.to_numpy(), y_synthetic], axis=0
    )

    print("\nProcessed training shape:", X_train_processed.shape)
    print("Processed augmented shape:", X_augmented.shape)
    print("Processed test shape:", X_test_processed.shape)


    # Load all three existing baseline models.
    baseline_models = {
        "Random Forest": joblib.load(
            BASELINE_MODEL_DIR / "random_forest.pkl"
        ),
        "Logistic Regression": joblib.load(
            BASELINE_MODEL_DIR / "logistic_regression.pkl"
        ),
        "MLP": joblib.load(
            BASELINE_MODEL_DIR / "mlp.pkl"
        ),
    }

    # Match the baseline model configurations as closely as possible.
    augmented_models = {
        "Random Forest": RandomForestClassifier(
            n_estimators=100,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            class_weight="balanced",
        ),
        "Logistic Regression": LogisticRegression(
            max_iter=1000,
            random_state=RANDOM_STATE,
            class_weight="balanced",
        ),
        "MLP": MLPClassifier(
            hidden_layer_sizes=(128, 64),
            activation="relu",
            solver="adam",
            batch_size=256,
            learning_rate_init=0.001,
            max_iter=30,
            early_stopping=True,
            validation_fraction=0.1,
            random_state=RANDOM_STATE,
            verbose=True,
        ),
    }

    results = []

    for model_name in baseline_models:
        print("\n" + "#" * 65)
        print(f"PROCESSING: {model_name}")
        print("#" * 65)

        # Evaluate the existing model.
        baseline_result = calculate_metrics(
            f"Baseline {model_name}",
            baseline_models[model_name],
            X_test_processed,
            y_test,
            attack_categories,
        )
        results.append(baseline_result)

        # Train a fresh model on augmented training data only.
        print(f"\nTraining augmented {model_name}...")

        augmented_model = augmented_models[model_name]
        augmented_model.fit(X_augmented, y_augmented)

        safe_name = model_name.lower().replace(" ", "_")
        model_path = (
            EXPERIMENT_DIR / f"{safe_name}_augmented.pkl"
        )
        joblib.dump(augmented_model, model_path)
        print(f"Saved augmented model: {model_path}")

        # Evaluate on the exact same untouched test set.
        augmented_result = calculate_metrics(
            f"WGAN-GP Augmented {model_name}",
            augmented_model,
            X_test_processed,
            y_test,
            attack_categories,
        )
        results.append(augmented_result)

    # Save one consolidated comparison table.
    results_df = pd.DataFrame(results)

    results_path = (
        EXPERIMENT_DIR / "all_models_augmentation_comparison.csv"
    )
    results_df.to_csv(results_path, index=False)

    print("\n" + "=" * 75)
    print("BASELINE VS. WGAN-GP AUGMENTATION — ALL THREE CLASSIFIERS")
    print("=" * 75)
    print(results_df.to_string(index=False))
    print(f"\nResults saved to: {results_path}")


if __name__ == "__main__":
    main()
