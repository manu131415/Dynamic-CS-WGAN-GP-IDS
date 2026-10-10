
import argparse
import csv
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from . import config
from .data_utils import load_target_attack_data
from .models import Generator, Critic
from .losses import gradient_penalty, critic_loss, generator_loss


def set_random_seeds(seed):
    """Set random seeds for more reproducible experiments."""
    np.random.seed(seed)
    torch.manual_seed(seed)


def prepare_dataloader(X, batch_size):
    """Convert NumPy features into batches of PyTorch tensors."""
    if len(X) < 2:
        raise ValueError(
            "At least two training samples are required for this model."
        )

    dataset = TensorDataset(torch.from_numpy(X).float())

    # Avoid a final batch of size 1 because the Generator uses BatchNorm.
    effective_batch_size = min(batch_size, len(dataset))

    loader = DataLoader(
        dataset,
        batch_size=effective_batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=0,
    )

    return loader


def save_checkpoint(generator, critic, epoch, feature_dim):
    """Save model weights and the information needed to reload them."""
    checkpoint_path = config.CHECKPOINT_DIR / f"checkpoint_epoch_{epoch}.pt"

    torch.save(
        {
            "epoch": epoch,
            "feature_dim": feature_dim,
            "latent_dim": config.LATENT_DIM,
            "hidden_dims": config.HIDDEN_DIMS,
            "generator_state_dict": generator.state_dict(),
            "critic_state_dict": critic.state_dict(),
        },
        checkpoint_path,
    )

    return checkpoint_path


def train_wgan_gp(epochs, batch_size, max_batches=None):
    """Train the Generator and Critic using the WGAN-GP objective."""
    set_random_seeds(config.RANDOM_STATE)
    config.create_output_directories()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load only the selected attack category from the training split.
    X_attack, feature_names, metadata = load_target_attack_data()

    feature_dim = X_attack.shape[1]
    loader = prepare_dataloader(X_attack, batch_size)

    print(f"Target category: {metadata['target_attack_category']}")
    print(f"Training samples: {len(X_attack)}")
    print(f"Processed feature count: {feature_dim}")
    print(f"Batches per epoch: {len(loader)}")
    print(f"Epochs: {epochs}")
    print(f"Critic updates per Generator update: {config.CRITIC_STEPS}")

    generator = Generator(
        latent_dim=config.LATENT_DIM,
        output_dim=feature_dim,
        hidden_dims=config.HIDDEN_DIMS,
    ).to(device)

    critic = Critic(
        input_dim=feature_dim,
        hidden_dims=config.HIDDEN_DIMS,
    ).to(device)

    # Adam settings commonly used for WGAN-GP.
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

    log_path = config.LOG_DIR / "training_log.csv"
    start_time = time.perf_counter()

    with open(log_path, "w", newline="", encoding="utf-8") as file:
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

            total_d_loss = 0.0
            total_g_loss = 0.0
            total_gp = 0.0
            batches_processed = 0

            generator.train()
            critic.train()

            for batch_index, (real_samples,) in enumerate(loader):
                if max_batches is not None and batch_index >= max_batches:
                    break

                real_samples = real_samples.to(device)
                current_batch_size = real_samples.size(0)

                # ------------------------------------------
                # 1. Update the Critic multiple times
                # ------------------------------------------
                for _ in range(config.CRITIC_STEPS):
                    noise = torch.randn(
                        current_batch_size,
                        config.LATENT_DIM,
                        device=device,
                    )

                    # Do not calculate Generator gradients
                    # during the Critic update.
                    with torch.no_grad():
                        fake_samples = generator(noise)

                    real_scores = critic(real_samples)
                    fake_scores = critic(fake_samples)

                    gp = gradient_penalty(
                        critic,
                        real_samples,
                        fake_samples,
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

                # ------------------------------------------
                # 2. Update the Generator once
                # ------------------------------------------
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

                total_d_loss += d_loss.item()
                total_g_loss += g_loss.item()
                total_gp += gp.item()
                batches_processed += 1

            if batches_processed == 0:
                raise RuntimeError(
                    "No batches were processed. Check the dataset and batch size."
                )

            average_d_loss = total_d_loss / batches_processed
            average_g_loss = total_g_loss / batches_processed
            average_gp = total_gp / batches_processed

            elapsed = time.perf_counter() - start_time
            epoch_seconds = time.perf_counter() - epoch_start

            writer.writerow([
                epoch,
                average_d_loss,
                average_g_loss,
                average_gp,
                elapsed,
            ])
            file.flush()

            print(
                f"Epoch {epoch:03d}/{epochs} | "
                f"D Loss: {average_d_loss:.4f} | "
                f"G Loss: {average_g_loss:.4f} | "
                f"GP: {average_gp:.4f} | "
                f"Epoch time: {epoch_seconds:.1f}s | "
                f"Total: {elapsed / 60:.1f} min"
            )

            # Save periodic checkpoints and the final epoch.
            if epoch % 25 == 0 or epoch == epochs:
                path = save_checkpoint(
                    generator, critic, epoch, feature_dim
                )
                print(f"Checkpoint saved: {path}")

    # Save convenient final model files as well.
    torch.save(
        generator.state_dict(),
        config.CHECKPOINT_DIR / "generator_final.pt",
    )
    torch.save(
        critic.state_dict(),
        config.CHECKPOINT_DIR / "critic_final.pt",
    )

    total_minutes = (time.perf_counter() - start_time) / 60

    print("\nTraining completed.")
    print(f"Total training time: {total_minutes:.2f} minutes")
    print(f"Training log: {log_path}")
    print(f"Checkpoints: {config.CHECKPOINT_DIR}")


def main():
    parser = argparse.ArgumentParser(
        description="Train a WGAN-GP on one UNSW-NB15 attack category."
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=config.EPOCHS,
        help="Number of training epochs.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=config.BATCH_SIZE,
        help="Training batch size.",
    )
    parser.add_argument(
        "--max-batches",
        type=int,
        default=None,
        help="Optional limit on batches processed per epoch for testing.",
    )

    args = parser.parse_args()

    if args.epochs < 1 or args.batch_size < 2:
        parser.error("epochs must be >= 1 and batch-size must be >= 2.")

    if args.max_batches is not None and args.max_batches < 1:
        parser.error("max-batches must be >= 1.")

    train_wgan_gp(
        epochs=args.epochs,
        batch_size=args.batch_size,
        max_batches=args.max_batches,
    )


if __name__ == "__main__":
    main()
