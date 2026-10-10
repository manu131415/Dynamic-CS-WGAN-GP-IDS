
import argparse

import joblib
import numpy as np
import pandas as pd
import torch

from . import config
from .models import Generator


V2_CHECKPOINT_DIR = config.CHECKPOINT_DIR / "shellcode_v2"


def load_generator(epoch, device):
    checkpoint_path = (
        V2_CHECKPOINT_DIR / f"checkpoint_epoch_{epoch}.pt"
    )
    processor_path = V2_CHECKPOINT_DIR / "feature_processor.joblib"

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}"
        )

    if not processor_path.exists():
        raise FileNotFoundError(
            f"Feature processor not found: {processor_path}"
        )

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=True,
    )

    processor = joblib.load(processor_path)

    generator = Generator(
        latent_dim=checkpoint["latent_dim"],
        output_dim=checkpoint["feature_dim"],
        hidden_dims=tuple(checkpoint["hidden_dims"]),
    ).to(device)

    generator.load_state_dict(checkpoint["generator_state_dict"])
    generator.eval()

    if processor.feature_dim != checkpoint["feature_dim"]:
        raise ValueError(
            "Processor feature dimension does not match checkpoint."
        )

    return generator, processor, checkpoint


def generate_samples(epoch, number_of_samples, batch_size):
    if number_of_samples < 1 or batch_size < 1:
        raise ValueError("Sample count and batch size must be positive.")

    config.SYNTHETIC_DIR.mkdir(parents=True, exist_ok=True)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    print(f"Using device: {device}")

    generator, processor, checkpoint = load_generator(epoch, device)

    torch.manual_seed(config.RANDOM_STATE)

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
        raise ValueError("Generated features contain non-finite values.")

    # Decode into the original raw feature schema.
    decoded = processor.inverse_transform(X_generated)

    # Add the dataset metadata after decoding.
    source_df = pd.read_csv(config.TRAIN_PATH)

    if "id" in source_df.columns:
        source_ids = pd.to_numeric(
            source_df["id"], errors="coerce"
        ).dropna()

        next_id = int(source_ids.max()) + 1

        decoded.insert(
            0,
            "id",
            np.arange(next_id, next_id + len(decoded)),
        )

    decoded[config.ATTACK_CATEGORY_COLUMN] = (
        config.TARGET_ATTACK_CATEGORY
    )
    decoded[config.TARGET_COLUMN] = 1

    # Match the original dataset's column order.
    decoded = decoded[source_df.columns.tolist()]

    transformed_path = (
        config.SYNTHETIC_DIR
        / f"shellcode_v2_transformed_epoch_{epoch}.npy"
    )
    csv_path = (
        config.SYNTHETIC_DIR
        / f"shellcode_v2_synthetic_epoch_{epoch}.csv"
    )

    np.save(transformed_path, X_generated)
    decoded.to_csv(csv_path, index=False)

    print("\nSynthetic generation completed.")
    print(f"Checkpoint epoch: {epoch}")
    print(f"Generated samples: {len(decoded)}")
    print(f"Transformed shape: {X_generated.shape}")
    print(f"Decoded shape: {decoded.shape}")
    print(f"Processor feature dimension: {processor.feature_dim}")
    print(f"Constant features restored: {len(processor.constant_columns)}")
    print(f"Attack category: {config.TARGET_ATTACK_CATEGORY}")
    print(f"Transformed data: {transformed_path}")
    print(f"Decoded CSV: {csv_path}")
    print("\nFirst five samples:")
    print(decoded.head().to_string(index=False))


def main():
    parser = argparse.ArgumentParser(
        description="Generate samples using the improved Shellcode WGAN-GP."
    )
    parser.add_argument("--epoch", type=int, default=100)
    parser.add_argument(
        "--samples",
        type=int,
        default=config.NUM_SYNTHETIC_SAMPLES,
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=config.BATCH_SIZE,
    )

    args = parser.parse_args()

    if args.epoch < 1:
        parser.error("epoch must be >= 1.")

    generate_samples(
        args.epoch,
        args.samples,
        args.batch_size,
    )


if __name__ == "__main__":
    main()
