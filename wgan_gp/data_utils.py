import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split

from . import config


def load_training_csv():
    """Load the official UNSW-NB15 training CSV."""

    if not config.TRAIN_PATH.exists():
        raise FileNotFoundError(
            f"Training dataset not found: {config.TRAIN_PATH}\n"
            "Place UNSW_NB15_training-set.csv inside data/raw/."
        )

    df = pd.read_csv(config.TRAIN_PATH)

    required_columns = {
        config.TARGET_COLUMN,
        config.ATTACK_CATEGORY_COLUMN,
        *config.DROP_COLUMNS,
    }

    missing = required_columns.difference(df.columns)

    if missing:
        raise ValueError(
            f"Dataset is missing required columns: {sorted(missing)}"
        )

    return df


def load_preprocessor():
    """Load the fitted preprocessor from the baseline pipeline."""

    if not config.PREPROCESSOR_PATH.exists():
        raise FileNotFoundError(
            f"Preprocessor not found: {config.PREPROCESSOR_PATH}\n"
            "Run the existing baseline training script first."
        )

    return joblib.load(config.PREPROCESSOR_PATH)


def create_training_split(df):
    """
    Reproduce the baseline 80/20 split.

    attack_cat is retained as metadata but is not included
    in the model input features.
    """

    X = df.drop(columns=config.DROP_COLUMNS)
    y = df[config.TARGET_COLUMN]
    attack_categories = df[config.ATTACK_CATEGORY_COLUMN]

    (
        X_train,
        X_val,
        y_train,
        y_val,
        categories_train,
        categories_val,
    ) = train_test_split(
        X,
        y,
        attack_categories,
        test_size=config.VALIDATION_SIZE,
        random_state=config.RANDOM_STATE,
        stratify=y,
    )

    return (
        X_train,
        X_val,
        y_train,
        y_val,
        categories_train,
        categories_val,
    )


def load_target_attack_data():
    """
    Prepare real samples for WGAN-GP training.

    Only the selected attack category from the training
    partition is used to fit the GAN.

    Returns:
        X_attack: float32 NumPy feature matrix
        feature_names: names of transformed features
        metadata: information about the selected data
    """

    df = load_training_csv()
    preprocessor = load_preprocessor()

    (
        X_train,
        X_val,
        y_train,
        y_val,
        categories_train,
        categories_val,
    ) = create_training_split(df)

    target = config.TARGET_ATTACK_CATEGORY.strip().casefold()

    category_mask = (
        categories_train.astype(str)
        .str.strip()
        .str.casefold()
        .eq(target)
    )

    X_target_raw = X_train.loc[category_mask]

    if X_target_raw.empty:
        available = sorted(
            categories_train.astype(str).unique().tolist()
        )

        raise ValueError(
            f"No training samples found for "
            f"'{config.TARGET_ATTACK_CATEGORY}'.\n"
            f"Available categories: {available}"
        )

    # Confirm that the selected category represents attacks.
    target_labels = y_train.loc[X_target_raw.index]

    if not (target_labels == 1).all():
        raise ValueError(
            "The selected category contains samples whose "
            "binary label is not 1. Check the dataset labels."
        )

    # Use the existing baseline transformations.
    X_attack = preprocessor.transform(X_target_raw)

    X_attack = np.asarray(X_attack, dtype=np.float32)

    if not np.isfinite(X_attack).all():
        raise ValueError(
            "Processed features contain NaN or infinite values."
        )

    feature_names = preprocessor.get_feature_names_out().tolist()

    if X_attack.shape[1] != len(feature_names):
        raise ValueError(
            "Feature dimension does not match the preprocessor."
        )

    category_counts = (
        categories_train.value_counts()
        .sort_values(ascending=False)
        .to_dict()
    )

    metadata = {
        "target_attack_category": config.TARGET_ATTACK_CATEGORY,
        "selected_training_samples": len(X_attack),
        "number_of_features": X_attack.shape[1],
        "binary_training_distribution": (
            y_train.value_counts().sort_index().to_dict()
        ),
        "attack_category_training_distribution": category_counts,
        "validation_samples_reserved": len(X_val),
        "validation_labels_reserved": len(y_val),
        "feature_names": feature_names,
    }

    return X_attack, feature_names, metadata


if __name__ == "__main__":
    X_attack, feature_names, metadata = (
        load_target_attack_data()
    )

    print("\nWGAN-GP data preparation successful")
    print("-----------------------------------")
    print("Target category:", metadata["target_attack_category"])
    print("Selected training samples:", X_attack.shape[0])
    print("Processed feature count:", X_attack.shape[1])
    print("Feature matrix dtype:", X_attack.dtype)

    print("\nBinary training distribution:")
    print(metadata["binary_training_distribution"])

    print("\nAttack-category training distribution:")
    for category, count in (
        metadata["attack_category_training_distribution"].items()
    ):
        print(f"{category}: {count}")

    print("\nFirst 10 transformed feature names:")
    print(feature_names[:10])