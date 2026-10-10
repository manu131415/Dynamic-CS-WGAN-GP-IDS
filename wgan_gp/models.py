
import torch
import torch.nn as nn


class Generator(nn.Module):
    """Generate synthetic samples in the processed feature space."""

    def __init__(self, latent_dim, output_dim, hidden_dims=(256, 128)):
        super().__init__()

        layers = []
        input_dim = latent_dim

        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(input_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
            ])
            input_dim = hidden_dim

        layers.append(nn.Linear(input_dim, output_dim))

        self.network = nn.Sequential(*layers)

    def forward(self, noise):
        return self.network(noise)


class Critic(nn.Module):
    """Assign a real-valued score to each input sample."""

    def __init__(self, input_dim, hidden_dims=(256, 128)):
        super().__init__()

        layers = []
        current_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(current_dim, hidden_dim),
                nn.LeakyReLU(0.2),
            ])
            current_dim = hidden_dim

        # One unrestricted score per sample; no sigmoid.
        layers.append(nn.Linear(current_dim, 1))

        self.network = nn.Sequential(*layers)

    def forward(self, samples):
        return self.network(samples).squeeze(1)
