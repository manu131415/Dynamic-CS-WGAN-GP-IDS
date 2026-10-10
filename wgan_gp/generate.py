
import argparse

import joblib
import numpy as np
import pandas as pd
import torch

from . import config
from .models import Generator


def load_generator(epoch, device):
    """Load the Generator weights from a saved checkpoint."""
    checkpoint_path = (
        config.CHECKPOINT_DIR / f"checkpoint_epoch_{epoch}.pt"
    )

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}"
        )

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=True,
    )

    generator = Generator(
        latent_dim=checkpoint["latent_dim"],
        output_dim=checkpoint["feature_dim"],
        hidden_dims=tuple(checkpoint["hidden_dims"]),
    ).to(device)

    generator.load_state_dict(checkpoint["generator_state_dict"])
    generator.eval()

    return generator, checkpoint


def decode_samples(X_generated, preprocessor, source_df):
    """
    Convert generated preprocessed vectors into a raw-style
    UNSW-NB15 DataFrame.

    Numerical columns are inverse-transformed.
    Categorical groups are decoded with argmax.
    """
    transformers = {
        name: (transformer, columns)
        for name, transformer, columns
        in preprocessor.transformers_
        if name != "remainder"
    }

    numerical_scaler, numerical_columns = transformers["num"]
    categorical_encoder, categorical_columns = transformers["cat"]

    numerical_columns = list(numerical_columns)
    categorical_columns = list(categorical_columns)

    n_numerical = len(numerical_columns)

    # The ColumnTransformer outputs numerical features first.
    X_numerical_scaled = X_generated[:, :n_numerical]
    X_numerical = numerical_scaler.inverse_transform(
        X_numerical_scaled
    )

    decoded = pd.DataFrame(
        X_numerical,
        columns=numerical_columns,
    )

    # Each original categorical feature has a one-hot group.
    offset = n_numerical

    for column, categories in zip(
        categorical_columns,
        categorical_encoder.categories_,
    ):
        group_width = len(categories)
        group = X_generated[:, offset:offset + group_width]

        if group.shape[1] != group_width:
            raise ValueError(
                f"Unexpected encoded width for category {column}."
            )

        selected_indices = np.argmax(group, axis=1)
        decoded[column] = categories[selected_indices]

        offset += group_width

    if offset != X_generated.shape[1]:
        raise ValueError(
            "Generated feature dimension does not match the "
            "preprocessor's numerical and categorical features."
        )

    # Match the original raw feature order.
    feature_columns = [
        column for column in source_df.columns
        if column not in config.DROP_COLUMNS
    ]
    decoded = decoded[feature_columns]

    # Add metadata expected by the original dataset schema.
    if "id" in source_df.columns:
        ids = pd.to_numeric(source_df["id"], errors="coerce")
        next_id = int(ids.max()) + 1
        decoded.insert(
            0,
            "id",
            np.arange(next_id, next_id + len(decoded)),
        )

    decoded[config.ATTACK_CATEGORY_COLUMN] = (
        config.TARGET_ATTACK_CATEGORY
    )
    decoded[config.TARGET_COLUMN] = 1

    # Preserve the original CSV column order.
    decoded = decoded[source_df.columns.tolist()]

    return decoded


def generate_samples(epoch, number_of_samples, batch_size):
    """Generate, decode, and save synthetic attack samples."""
    if number_of_samples < 1 or batch_size < 1:
        raise ValueError(
            "number_of_samples and batch_size must be positive."
        )

    if not config.TRAIN_PATH.exists():
        raise FileNotFoundError(
            f"Training CSV not found: {config.TRAIN_PATH}"
        )

    if not config.PREPROCESSOR_PATH.exists():
        raise FileNotFoundError(
            f"Preprocessor not found: {config.PREPROCESSOR_PATH}"
        )

    config.create_output_directories()

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    print(f"Using device: {device}")

    generator, checkpoint = load_generator(epoch, device)
    preprocessor = joblib.load(config.PREPROCESSOR_PATH)
    source_df = pd.read_csv(config.TRAIN_PATH)

    generated_batches = []
    remaining = number_of_samples

    with torch.no_grad():
        while remaining > 0:
            current_batch_size = min(batch_size, remaining)

            noise = torch.randn(
                current_batch_size,
                checkpoint["latent_dim"],
                device=device,
            )

            generated = generator(noise).cpu().numpy()
            generated_batches.append(generated)
            remaining -= current_batch_size

    X_generated = np.concatenate(generated_batches, axis=0)

    if not np.isfinite(X_generated).all():
        raise ValueError(
            "Generated samples contain NaN or infinite values."
        )

    synthetic_df = decode_samples(
        X_generated,
        preprocessor,
        source_df,
    )

    transformed_path = (
        config.SYNTHETIC_DIR
        / f"shellcode_transformed_epoch_{epoch}.npy"
    )
    csv_path = (
        config.SYNTHETIC_DIR
        / f"shellcode_synthetic_epoch_{epoch}.csv"
    )

    np.save(transformed_path, X_generated)
    synthetic_df.to_csv(csv_path, index=False)

    print("\nSynthetic generation completed.")
    print(f"Checkpoint epoch: {epoch}")
    print(f"Generated samples: {len(synthetic_df)}")
    print(f"Transformed feature shape: {X_generated.shape}")
    print(f"CSV shape: {synthetic_df.shape}")
    print(f"Attack category: {config.TARGET_ATTACK_CATEGORY}")
    print("Binary label counts:")
    print(synthetic_df[config.TARGET_COLUMN].value_counts())
    print(f"\nTransformed data saved to: {transformed_path}")
    print(f"Decoded CSV saved to: {csv_path}")
    print("\nFirst five decoded samples:")
    print(synthetic_df.head().to_string(index=False))


def main():
    parser = argparse.ArgumentParser(
        description="Generate synthetic UNSW-NB15 attack samples."
    )
    parser.add_argument(
        "--epoch",
        type=int,
        default=100,
        help="Checkpoint epoch to use.",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=config.NUM_SYNTHETIC_SAMPLES,
        help="Number of synthetic samples to generate.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=config.BATCH_SIZE,
        help="Generation batch size.",
    )

    args = parser.parse_args()

    if args.epoch < 1:
        parser.error("epoch must be >= 1.")

    generate_samples(
        epoch=args.epoch,
        number_of_samples=args.samples,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()
