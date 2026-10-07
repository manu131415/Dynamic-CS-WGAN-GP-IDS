from pathlib import Path
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

TRAIN_PATH = PROJECT_ROOT / "data" / "raw" / "UNSW_NB15_training-set.csv"
MODEL_DIR = PROJECT_ROOT / "baseline" / "models"

MODEL_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# SETTINGS
# ============================================================

TARGET = "label"

DROP_COLUMNS = [
    "id",
    "attack_cat"
]

CATEGORICAL_FEATURES = [
    "proto",
    "service",
    "state"
]

RANDOM_STATE = 42


# ============================================================
# LOAD DATA
# ============================================================

print("Loading training dataset...")

df = pd.read_csv(TRAIN_PATH)

print("Dataset shape:", df.shape)


# ============================================================
# SEPARATE FEATURES AND TARGET
# ============================================================

y = df[TARGET]

X = df.drop(
    columns=DROP_COLUMNS + [TARGET]
)


# ============================================================
# TRAIN / VALIDATION SPLIT
# ============================================================

X_train, X_val, y_train, y_val = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=RANDOM_STATE,
    stratify=y
)

print("\nTrain samples:", len(X_train))
print("Validation samples:", len(X_val))

print("\nTraining class distribution:")
print(y_train.value_counts())

print("\nValidation class distribution:")
print(y_val.value_counts())


# ============================================================
# IDENTIFY NUMERICAL FEATURES
# ============================================================

NUMERICAL_FEATURES = [
    col
    for col in X_train.columns
    if col not in CATEGORICAL_FEATURES
]


# ============================================================
# PREPROCESSING
# ============================================================

preprocessor = ColumnTransformer(
    transformers=[
        (
            "num",
            StandardScaler(),
            NUMERICAL_FEATURES
        ),
        (
            "cat",
            OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=False
            ),
            CATEGORICAL_FEATURES
        )
    ]
)


# ============================================================
# TRANSFORM DATA
# ============================================================

print("\nFitting preprocessing...")

X_train_processed = preprocessor.fit_transform(X_train)

X_val_processed = preprocessor.transform(X_val)

print("Processed training shape:", X_train_processed.shape)
print("Processed validation shape:", X_val_processed.shape)


# Save preprocessing object
joblib.dump(
    preprocessor,
    MODEL_DIR / "preprocessor.pkl"
)

print("Saved preprocessor.")


# ============================================================
# RANDOM FOREST
# ============================================================

print("\nTraining Random Forest...")

rf_model = RandomForestClassifier(
    n_estimators=100,
    random_state=RANDOM_STATE,
    n_jobs=-1,
    class_weight="balanced"
)

rf_model.fit(
    X_train_processed,
    y_train
)

joblib.dump(
    rf_model,
    MODEL_DIR / "random_forest.pkl"
)

print("Random Forest saved.")


# ============================================================
# LOGISTIC REGRESSION
# ============================================================

print("\nTraining Logistic Regression...")

lr_model = LogisticRegression(
    max_iter=1000,
    random_state=RANDOM_STATE,
    class_weight="balanced",
    n_jobs=-1
)

lr_model.fit(
    X_train_processed,
    y_train
)

joblib.dump(
    lr_model,
    MODEL_DIR / "logistic_regression.pkl"
)

print("Logistic Regression saved.")


# ============================================================
# MLP
# ============================================================

print("\nTraining MLP...")

mlp_model = MLPClassifier(
    hidden_layer_sizes=(128, 64),
    activation="relu",
    solver="adam",
    batch_size=256,
    learning_rate_init=0.001,
    max_iter=30,
    early_stopping=True,
    validation_fraction=0.1,
    random_state=RANDOM_STATE,
    verbose=True
)

mlp_model.fit(
    X_train_processed,
    y_train
)

joblib.dump(
    mlp_model,
    MODEL_DIR / "mlp.pkl"
)

print("MLP saved.")


# ============================================================
# SAVE VALIDATION DATA
# ============================================================

joblib.dump(
    X_val_processed,
    MODEL_DIR / "X_val.pkl"
)

joblib.dump(
    y_val,
    MODEL_DIR / "y_val.pkl"
)


# ============================================================
# COMPLETE
# ============================================================

print("\n========================================")
print("BASELINE TRAINING COMPLETED")
print("========================================")

print("\nSaved files:")

print("baseline/models/preprocessor.pkl")
print("baseline/models/random_forest.pkl")
print("baseline/models/logistic_regression.pkl")
print("baseline/models/mlp.pkl")
print("baseline/models/X_val.pkl")
print("baseline/models/y_val.pkl")