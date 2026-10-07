from pathlib import Path
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder


# Find the project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Dataset paths
TRAIN_PATH = PROJECT_ROOT / "data" / "raw" / "UNSW_NB15_training-set.csv"
TEST_PATH = PROJECT_ROOT / "data" / "raw" / "UNSW_NB15_testing-set.csv"


TARGET = "label"

DROP_COLUMNS = [
    "id",
    "attack_cat"
]


def load_data():
    train_df = pd.read_csv(TRAIN_PATH)
    test_df = pd.read_csv(TEST_PATH)

    return train_df, test_df


def prepare_features(train_df, test_df):

    y_train = train_df[TARGET]
    y_test = test_df[TARGET]

    X_train = train_df.drop(
        columns=DROP_COLUMNS + [TARGET]
    )

    X_test = test_df.drop(
        columns=DROP_COLUMNS + [TARGET]
    )

    categorical_features = [
        "proto",
        "service",
        "state"
    ]

    numerical_features = [
        col for col in X_train.columns
        if col not in categorical_features
    ]

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                StandardScaler(),
                numerical_features
            ),
            (
                "cat",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False
                ),
                categorical_features
            )
        ]
    )

    X_train_processed = preprocessor.fit_transform(X_train)

    X_test_processed = preprocessor.transform(X_test)

    return (
        X_train_processed,
        X_test_processed,
        y_train,
        y_test,
        preprocessor
    )


if __name__ == "__main__":

    train_df, test_df = load_data()

    print("Training shape:", train_df.shape)
    print("Testing shape:", test_df.shape)

    X_train, X_test, y_train, y_test, preprocessor = prepare_features(
        train_df,
        test_df
    )

    print("Processed X_train:", X_train.shape)
    print("Processed X_test:", X_test.shape)

    print("y_train:", y_train.shape)
    print("y_test:", y_test.shape)