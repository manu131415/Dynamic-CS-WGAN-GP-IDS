
import torch
from torch import autograd


def gradient_penalty(critic, real_samples, fake_samples):
    """
    Calculate the gradient penalty used in WGAN-GP.

    real_samples and fake_samples must have the same shape:
    (batch_size, number_of_features).
    """
    batch_size = real_samples.size(0)

    # Generate random interpolation weights for each sample.
    alpha = torch.rand(
        batch_size, 1, device=real_samples.device
    )

    # Interpolate between real and generated samples.
    interpolated = (
        alpha * real_samples
        + (1 - alpha) * fake_samples
    )
    interpolated.requires_grad_(True)

    # Obtain critic scores for interpolated samples.
    scores = critic(interpolated)

    # Calculate gradients of scores with respect to the inputs.
    gradients = autograd.grad(
        outputs=scores,
        inputs=interpolated,
        grad_outputs=torch.ones_like(scores),
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]

    # Flatten each sample's gradients and calculate their L2 norm.
    gradients = gradients.reshape(batch_size, -1)
    gradient_norm = gradients.norm(2, dim=1)

    # Penalize gradient norms that differ from 1.
    penalty = ((gradient_norm - 1) ** 2).mean()

    return penalty


def critic_loss(real_scores, fake_scores, gp, lambda_gp=10.0):
    """
    Calculate the WGAN-GP critic loss.

    Minimizing this loss encourages the critic to assign
    higher scores to real samples than to generated samples.
    """
    return fake_scores.mean() - real_scores.mean() + lambda_gp * gp


def generator_loss(fake_scores):
    """
    Calculate the Generator loss.

    The Generator tries to increase the critic's scores
    for generated samples.
    """
    return -fake_scores.mean()
