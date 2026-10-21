"""
Trains Static-cost IDS and Dynamic-cost IDS for RF, LR, MLP.
- Same train/val split (random_state=42, stratified) as baseline train.py
- Re-uses the baseline preprocessor.pkl (identical features, preprocess.py untouched)
- Cost enters via per-sample weights (RF, LR). MLPClassifier has no reliable
  sample_weight support, so for MLP the training set is resampled with
  probability proportional to the weights (equivalent in expectation).
- class_weight is NOT used here: the cost weights are the only imbalance handling.
"""

from pathlib import Path
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier

from cost_module import StaticCost, DynamicCost

PROJECT_ROOT = Path(__file__).resolve().parent.parent
TRAIN_PATH = PROJECT_ROOT / "data" / "raw" / "UNSW_NB15_training-set.csv"
BASE_DIR = PROJECT_ROOT / "baseline" / "models"
OUT_DIR = PROJECT_ROOT / "cost_sensitive" / "models"
OUT_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42
DROP_COLUMNS = ["id", "attack_cat"]
TARGET = "label"

# ---------------- data (same split as baseline) ----------------
df = pd.read_csv(TRAIN_PATH)
y = df[TARGET]
X = df.drop(columns=DROP_COLUMNS + [TARGET])

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.20, random_state=RANDOM_STATE, stratify=y
)

# raw rows (incl. attack_cat, sbytes, dbytes) aligned with X_train
raw_train = df.loc[X_train.index].reset_index(drop=True)
y_train = y_train.reset_index(drop=True)

preprocessor = joblib.load(BASE_DIR / "preprocessor.pkl")
X_train_p = preprocessor.transform(X_train)
print("Processed train shape:", X_train_p.shape)

# ---------------- cost objects (fit on training data only) ----------------
cost_schemes = {
    "static": StaticCost(c_fn=5.0, c_fp=1.0).fit(raw_train),
    "dynamic": DynamicCost(target_mean_fn=5.0, c_fp_base=1.0).fit(raw_train),
}
for name, c in cost_schemes.items():
    joblib.dump(c, OUT_DIR / f"cost_{name}.pkl")
    c_fn, c_fp = c.costs(raw_train)
    a = raw_train["label"].values == 1
    print(f"{name:8s} mean FN cost (attacks): {c_fn[a].mean():.3f} | "
          f"mean FP cost (normal): {c_fp[~a].mean():.3f}")


def build_models():
    return {
        "random_forest": RandomForestClassifier(
            n_estimators=100, random_state=RANDOM_STATE, n_jobs=-1),
        "logistic_regression": LogisticRegression(
            max_iter=1000, random_state=RANDOM_STATE),
        "mlp": MLPClassifier(
            hidden_layer_sizes=(128, 64), activation="relu", solver="adam",
            batch_size=256, learning_rate_init=0.001, max_iter=30,
            early_stopping=True, validation_fraction=0.1,
            random_state=RANDOM_STATE),
    }


rng = np.random.default_rng(RANDOM_STATE)

for scheme, cost in cost_schemes.items():
    w = cost.sample_weights(raw_train)

    for model_name, model in build_models().items():
        print(f"\nTraining {scheme} | {model_name} ...")

        if model_name == "mlp":
            idx = rng.choice(len(w), size=len(w), replace=True, p=w / w.sum())
            model.fit(X_train_p[idx], y_train.values[idx])
        else:
            model.fit(X_train_p, y_train, sample_weight=w)

        joblib.dump(model, OUT_DIR / f"{model_name}_{scheme}.pkl")
        print("saved.")

print("\nCOST-SENSITIVE TRAINING COMPLETED ->", OUT_DIR)