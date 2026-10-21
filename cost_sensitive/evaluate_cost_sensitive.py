"""
Compares A. Baseline, B. Static-cost IDS, C. Dynamic-cost IDS
for RF / LR / MLP on the UNSW-NB15 test set.

Every configuration (including the baseline) is scored with BOTH the static
and the dynamic cost, so the comparison is fair.
"""

from pathlib import Path
import joblib
import pandas as pd

from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, confusion_matrix)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
TEST_PATH = PROJECT_ROOT / "data" / "raw" / "UNSW_NB15_testing-set.csv"
BASE_DIR = PROJECT_ROOT / "baseline" / "models"
CS_DIR = PROJECT_ROOT / "cost_sensitive" / "models"
RESULTS_DIR = PROJECT_ROOT / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

test_df = pd.read_csv(TEST_PATH)
y_test = test_df["label"].values
X_test = test_df.drop(columns=["id", "attack_cat", "label"])

preprocessor = joblib.load(BASE_DIR / "preprocessor.pkl")
X_test_p = preprocessor.transform(X_test)

# cost objects fitted on TRAIN data, applied to TEST data
static_cost = joblib.load(CS_DIR / "cost_static.pkl")
dynamic_cost = joblib.load(CS_DIR / "cost_dynamic.pkl")

CLASSIFIERS = {
    "Random Forest": "random_forest",
    "Logistic Regression": "logistic_regression",
    "MLP": "mlp",
}

CONFIGS = {
    "A. Baseline IDS": lambda f: BASE_DIR / f"{f}.pkl",
    "B. Static cost-sensitive IDS": lambda f: CS_DIR / f"{f}_static.pkl",
    "C. Dynamic-cost IDS": lambda f: CS_DIR / f"{f}_dynamic.pkl",
}

rows = []
cat_rows = []

for config, path_fn in CONFIGS.items():
    for clf_name, fname in CLASSIFIERS.items():
        model = joblib.load(path_fn(fname))
        y_pred = model.predict(X_test_p)
        y_prob = model.predict_proba(X_test_p)[:, 1]

        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
        s_tot, s_avg = static_cost.total_cost(test_df, y_pred)
        d_tot, d_avg = dynamic_cost.total_cost(test_df, y_pred)

        rows.append({
            "Configuration": config,
            "Classifier": clf_name,
            "Accuracy": accuracy_score(y_test, y_pred),
            "Precision": precision_score(y_test, y_pred, zero_division=0),
            "Recall": recall_score(y_test, y_pred, zero_division=0),
            "F1": f1_score(y_test, y_pred, zero_division=0),
            "ROC-AUC": roc_auc_score(y_test, y_prob),
            "FPR": fp / (fp + tn),
            "FNR": fn / (fn + tp),
            "FP": fp,
            "FN": fn,
            "Static_Total_Cost": s_tot,
            "Static_Avg_Cost": s_avg,
            "Dynamic_Total_Cost": d_tot,
            "Dynamic_Avg_Cost": d_avg,
        })

        # per-attack-category recall (shows WHICH attacks the dynamic cost helps)
        attacks = test_df[test_df["label"] == 1].copy()
        attacks["pred"] = y_pred[test_df["label"].values == 1]
        for cat, g in attacks.groupby("attack_cat"):
            cat_rows.append({
                "Configuration": config, "Classifier": clf_name,
                "attack_cat": cat, "samples": len(g),
                "recall": (g["pred"] == 1).mean(),
            })

results = pd.DataFrame(rows)
cat_results = pd.DataFrame(cat_rows)

results.to_csv(RESULTS_DIR / "cost_comparison_results.csv", index=False)
cat_results.to_csv(RESULTS_DIR / "cost_comparison_per_category.csv", index=False)

pd.set_option("display.width", 250)
pd.set_option("display.max_columns", None)
print(results.round(4).to_string(index=False))

print("\nPer-category recall (pivot):")
pivot = cat_results.pivot_table(index=["Classifier", "attack_cat"],
                                columns="Configuration", values="recall")
print(pivot.round(4).to_string())

print("\nSaved to", RESULTS_DIR)