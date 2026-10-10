
import argparse
import csv
import time

import joblib
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from . import config
from .data_utils import load_training_csv, create_training_split
from .feature_preprocessing import WGANFeatureProcessor
from .models import Generator, Critic
from .losses import gradient_penalty, critic_loss, generator_loss


V2_CHECKPOINT_DIR = config.CHECKPOINT_DIR / "shellcode_v2"
V2_LOG_PATH = config.LOG_DIR / "shellcode_v2_training_log.csv"


def set_random_seeds(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)


def prepare_dataloader(X, batch_size):
    if len(X) < 2:
        raise ValueError("At least two samples are required.")

    dataset = TensorDataset(torch.from_numpy(X).float())

    loader = DataLoader(
        dataset,
        batch_size=min(batch_size, len(dataset)),
        shuffle=True,
        drop_last=True,
        num_workers=0,
    )

    if len(loader) == 0:
        raise ValueError("No complete batches are available.")

    return loader


def save_checkpoint(generator, critic, epoch, feature_dim):
    path = V2_CHECKPOINT_DIR / f"checkpoint_epoch_{epoch}.pt"

    torch.save(
        {
            "epoch": epoch,
            "feature_dim": feature_dim,
            "latent_dim": config.LATENT_DIM,
            "hidden_dims": config.HIDDEN_DIMS,
            "generator_state_dict": generator.state_dict(),
            "critic_state_dict": critic.state_dict(),
        },
        path,
    )

    return path


def train_wgan_gp(epochs, batch_size, max_batches=None):
    set_random_seeds(config.RANDOM_STATE)

    V2_CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    config.LOG_DIR.mkdir(parents=True, exist_ok=True)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    print(f"Using device: {device}")

    # Reproduce the existing training/validation split.
    df = load_training_csv()
    X_train, X_val, y_train, y_val, categories_train, categories_val = (
        create_training_split(df)
    )

    target = config.TARGET_ATTACK_CATEGORY.strip().casefold()
    mask = (
        categories_train.astype(str)
        .str.strip()
        .str.casefold()
        .eq(target)
    )

    X_target = X_train.loc[mask].copy()
    y_target = y_train.loc[X_target.index]

    if X_target.empty:
        raise ValueError(
            f"No training samples found for "
            f"{config.TARGET_ATTACK_CATEGORY}."
        )

    if not (y_target == 1).all():
        raise ValueError("Target attack samples must have label 1.")

    # Fit a separate processor only on target-category training samples.
    processor = WGANFeatureProcessor()
    processor.fit(X_target)

    X_processed = processor.transform(X_target).astype(np.float32)

    processor_path = V2_CHECKPOINT_DIR / "feature_processor.joblib"
    joblib.dump(processor, processor_path)

    feature_dim = X_processed.shape[1]
    loader = prepare_dataloader(X_processed, batch_size)

    print(f"Target category: {config.TARGET_ATTACK_CATEGORY}")
    print(f"Training samples: {len(X_processed)}")
    print(f"Original feature count: {X_target.shape[1]}")
    print(f"Processed feature count: {feature_dim}")
    print(f"Constant features preserved: {len(processor.constant_columns)}")
    print(f"Batches per epoch: {len(loader)}")
    print(f"Epochs: {epochs}")
    print(f"Processor saved: {processor_path}")

    generator = Generator(
        config.LATENT_DIM,
        feature_dim,
        config.HIDDEN_DIMS,
    ).to(device)

    critic = Critic(
        feature_dim,
        config.HIDDEN_DIMS,
    ).to(device)

    optimizer_G = torch.optim.Adam(
        generator.parameters(),
        lr=config.LEARNING_RATE,
        betas=(0.0, 0.9),
    )
    optimizer_D = torch.optim.Adam(
        critic.parameters(),
        lr=config.LEARNING_RATE,
        betas=(0.0, 0.9),
    )

    start_time = time.perf_counter()

    with open(V2_LOG_PATH, "w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow([
            "epoch",
            "critic_loss",
            "generator_loss",
            "gradient_penalty",
            "elapsed_seconds",
        ])

        for epoch in range(1, epochs + 1):
            epoch_start = time.perf_counter()

            total_d = 0.0
            total_g = 0.0
            total_gp = 0.0
            batches = 0

            generator.train()
            critic.train()

            for batch_index, (real_samples,) in enumerate(loader):
                if max_batches is not None and batch_index >= max_batches:
                    break

                real_samples = real_samples.to(device)
                current_batch_size = real_samples.size(0)

                # Critic updates.
                for _ in range(config.CRITIC_STEPS):
                    noise = torch.randn(
                        current_batch_size,
                        config.LATENT_DIM,
                        device=device,
                    )

                    with torch.no_grad():
                        fake_samples = generator(noise)

                    real_scores = critic(real_samples)
                    fake_scores = critic(fake_samples)

                    gp = gradient_penalty(
                        critic, real_samples, fake_samples
                    )

                    d_loss = critic_loss(
                        real_scores,
                        fake_scores,
                        gp,
                        lambda_gp=config.GRADIENT_PENALTY_WEIGHT,
                    )

                    optimizer_D.zero_grad(set_to_none=True)
                    d_loss.backward()
                    optimizer_D.step()

                # One Generator update.
                noise = torch.randn(
                    current_batch_size,
                    config.LATENT_DIM,
                    device=device,
                )
                fake_samples = generator(noise)
                fake_scores = critic(fake_samples)
                g_loss = generator_loss(fake_scores)

                optimizer_G.zero_grad(set_to_none=True)
                g_loss.backward()
                optimizer_G.step()

                total_d += d_loss.item()
                total_g += g_loss.item()
                total_gp += gp.item()
                batches += 1

            if batches == 0:
                raise RuntimeError("No batches were processed.")

            elapsed = time.perf_counter() - start_time
            epoch_seconds = time.perf_counter() - epoch_start

            average_d = total_d / batches
            average_g = total_g / batches
            average_gp = total_gp / batches

            writer.writerow([
                epoch, average_d, average_g, average_gp, elapsed
            ])
            file.flush()

            print(
                f"Epoch {epoch:03d}/{epochs} | "
                f"D Loss: {average_d:.4f} | "
                f"G Loss: {average_g:.4f} | "
                f"GP: {average_gp:.4f} | "
                f"Epoch time: {epoch_seconds:.3f}s | "
                f"Total: {elapsed / 60:.2f} min"
            )

            if epoch % 25 == 0 or epoch == epochs:
                path = save_checkpoint(
                    generator, critic, epoch, feature_dim
                )
                print(f"Checkpoint saved: {path}")

    torch.save(
        generator.state_dict(),
        V2_CHECKPOINT_DIR / "generator_final.pt",
    )
    torch.save(
        critic.state_dict(),
        V2_CHECKPOINT_DIR / "critic_final.pt",
    )

    print("\nTraining completed.")
    print(f"Log: {V2_LOG_PATH}")
    print(f"Checkpoints: {V2_CHECKPOINT_DIR}")
    print(f"Total time: {(time.perf_counter() - start_time) / 60:.2f} minutes")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=config.EPOCHS)
    parser.add_argument("--batch-size", type=int, default=config.BATCH_SIZE)
    parser.add_argument("--max-batches", type=int, default=None)
    args = parser.parse_args()

    if args.epochs < 1 or args.batch_size < 2:
        parser.error("epochs must be >= 1 and batch-size must be >= 2.")

    if args.max_batches is not None and args.max_batches < 1:
        parser.error("max-batches must be >= 1.")

    train_wgan_gp(
        args.epochs,
        args.batch_size,
        args.max_batches,
    )


if __name__ == "__main__":
    main()
