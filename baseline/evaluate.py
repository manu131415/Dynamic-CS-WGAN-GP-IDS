from pathlib import Path
import joblib
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

TEST_PATH = PROJECT_ROOT / "data" / "raw" / "UNSW_NB15_testing-set.csv"

MODEL_DIR = PROJECT_ROOT / "baseline" / "models"

RESULTS_DIR = PROJECT_ROOT / "results"

RESULTS_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# LOAD TEST DATA
# ============================================================

print("Loading test dataset...")

test_df = pd.read_csv(TEST_PATH)

print("Test dataset shape:", test_df.shape)


# ============================================================
# SEPARATE FEATURES AND TARGET
# ============================================================

y_test = test_df["label"]

X_test = test_df.drop(
    columns=[
        "id",
        "attack_cat",
        "label"
    ]
)


# Keep attack categories for later failure analysis
attack_categories = test_df["attack_cat"]


# ============================================================
# LOAD PREPROCESSOR
# ============================================================

print("\nLoading preprocessor...")

preprocessor = joblib.load(
    MODEL_DIR / "preprocessor.pkl"
)


# ============================================================
# PREPROCESS TEST DATA
# ============================================================

print("Preprocessing test data...")

X_test_processed = preprocessor.transform(
    X_test
)

print(
    "Processed test shape:",
    X_test_processed.shape
)


# ============================================================
# MODELS
# ============================================================

models = {

    "Random Forest":
        MODEL_DIR / "random_forest.pkl",

    "Logistic Regression":
        MODEL_DIR / "logistic_regression.pkl",

    "MLP":
        MODEL_DIR / "mlp.pkl"
}


# ============================================================
# EVALUATION FUNCTION
# ============================================================

def evaluate_model(model_name, model):

    print("\n")
    print("=" * 60)
    print(model_name)
    print("=" * 60)

    # Predictions
    y_pred = model.predict(X_test_processed)

    # Probability for positive class
    if hasattr(model, "predict_proba"):

        y_prob = model.predict_proba(
            X_test_processed
        )[:, 1]

    else:

        y_prob = None


    # --------------------------------------------------------
    # BASIC METRICS
    # --------------------------------------------------------

    accuracy = accuracy_score(
        y_test,
        y_pred
    )

    precision = precision_score(
        y_test,
        y_pred,
        zero_division=0
    )

    recall = recall_score(
        y_test,
        y_pred,
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        y_pred,
        zero_division=0
    )


    # --------------------------------------------------------
    # ROC-AUC
    # --------------------------------------------------------

    if y_prob is not None:

        roc_auc = roc_auc_score(
            y_test,
            y_prob
        )

    else:

        roc_auc = None


    # --------------------------------------------------------
    # CONFUSION MATRIX
    # --------------------------------------------------------

    cm = confusion_matrix(
        y_test,
        y_pred
    )

    tn, fp, fn, tp = cm.ravel()


    # --------------------------------------------------------
    # FALSE POSITIVE RATE
    # --------------------------------------------------------

    if (fp + tn) > 0:

        fpr = fp / (fp + tn)

    else:

        fpr = 0


    # --------------------------------------------------------
    # FALSE NEGATIVE RATE
    # --------------------------------------------------------

    if (fn + tp) > 0:

        fnr = fn / (fn + tp)

    else:

        fnr = 0


    # --------------------------------------------------------
    # PRINT RESULTS
    # --------------------------------------------------------

    print("\nMetrics:")

    print(f"Accuracy      : {accuracy:.4f}")
    print(f"Precision     : {precision:.4f}")
    print(f"Recall        : {recall:.4f}")
    print(f"F1 Score      : {f1:.4f}")

    if roc_auc is not None:
        print(f"ROC-AUC       : {roc_auc:.4f}")

    print(f"False Positive Rate : {fpr:.4f}")
    print(f"False Negative Rate : {fnr:.4f}")


    # --------------------------------------------------------
    # CONFUSION MATRIX
    # --------------------------------------------------------

    print("\nConfusion Matrix:")

    print(cm)


    # --------------------------------------------------------
    # CLASSIFICATION REPORT
    # --------------------------------------------------------

    print("\nClassification Report:")

    print(
        classification_report(
            y_test,
            y_pred,
            target_names=[
                "Normal",
                "Attack"
            ],
            zero_division=0
        )
    )


    # --------------------------------------------------------
    # PER-ATTACK-CATEGORY ANALYSIS
    # --------------------------------------------------------

    analysis_df = pd.DataFrame({

        "attack_cat": attack_categories.values,

        "actual": y_test.values,

        "predicted": y_pred

    })


    attack_only = analysis_df[
        analysis_df["actual"] == 1
    ]


    print("\nPer-Attack-Category Recall:")

    category_results = []


    for category in sorted(
        attack_only["attack_cat"].unique()
    ):

        category_data = attack_only[
            attack_only["attack_cat"] == category
        ]

        total = len(category_data)

        detected = (
            category_data["predicted"] == 1
        ).sum()

        category_recall = detected / total

        category_results.append({

            "attack_category": category,

            "samples": total,

            "detected": detected,

            "recall": category_recall

        })

        print(
            f"{category:20s} "
            f"Samples: {total:6d} "
            f"Recall: {category_recall:.4f}"
        )


    # Save category results

    category_df = pd.DataFrame(
        category_results
    )

    safe_name = model_name.lower().replace(
        " ",
        "_"
    )

    category_df.to_csv(
        RESULTS_DIR /
        f"{safe_name}_attack_category_results.csv",
        index=False
    )


    # --------------------------------------------------------
    # RETURN RESULTS
    # --------------------------------------------------------

    return {

        "Model": model_name,

        "Accuracy": accuracy,

        "Precision": precision,

        "Recall": recall,

        "F1": f1,

        "ROC-AUC": roc_auc,

        "FPR": fpr,

        "FNR": fnr

    }


# ============================================================
# RUN ALL MODELS
# ============================================================

all_results = []


for model_name, model_path in models.items():

    print(
        f"\nLoading {model_name}..."
    )

    model = joblib.load(model_path)

    result = evaluate_model(
        model_name,
        model
    )

    all_results.append(result)


# ============================================================
# SAVE FINAL RESULTS
# ============================================================

results_df = pd.DataFrame(
    all_results
)

results_path = (
    RESULTS_DIR /
    "baseline_results.csv"
)

results_df.to_csv(
    results_path,
    index=False
)


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n")
print("=" * 60)
print("BASELINE EVALUATION COMPLETED")
print("=" * 60)

print("\nFinal Results:")

print(
    results_df.to_string(
        index=False
    )
)

print(
    f"\nResults saved to:\n{results_path}"
)