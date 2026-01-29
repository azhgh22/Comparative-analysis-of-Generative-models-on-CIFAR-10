"""
SDEdit Sampler: Guided Image Editing via Stochastic Differential Equations

This module provides sampling/editing functions for the SDEdit model.
Based on: https://arxiv.org/pdf/2108.01073
"""

import torch
import numpy as np


def sdedit_transform(score_net, guide_image, sigmas, t0=0.5, n_steps=100, eps=2e-5, device=None):
    """
    Transform a guide image into a realistic image using SDEdit.

    Args:
        score_net: Trained score network
        guide_image: Input guide/stroke image (B, C, H, W)
        sigmas: Noise level schedule (tensor, from max to min)
        t0: Noise level ratio (0 < t0 < 1)
            - Higher t0: More noise, more realistic but less faithful
            - Lower t0: Less noise, more faithful but potentially less realistic
        n_steps: Number of Langevin steps per noise level
        eps: Base step size
        device: Device to run on

    Returns:
        Edited/transformed image
    """
    if device is None:
        device = guide_image.DEVICE

    score_net.eval()
    batch_size = guide_image.shape[0]
    num_scales = len(sigmas)

    # Calculate starting index based on t0
    start_idx = int((1 - t0) * (num_scales - 1))
    start_idx = max(0, min(start_idx, num_scales - 1))

    # Get starting sigma and add noise
    sigma_start = sigmas[start_idx].item()
    noise = torch.randn_like(guide_image)
    x = guide_image + sigma_start * noise

    # Run reverse SDE (Annealed Langevin Dynamics)
    with torch.no_grad():
        for i in range(start_idx, num_scales):
            sigma = sigmas[i]
            sigma_val = sigma.item()

            # Step size: α_i = ε * (σ_i / σ_L)²
            alpha = eps * (sigma_val / sigmas[-1].item()) ** 2

            for t in range(n_steps):
                # No noise on final step
                if i == num_scales - 1 and t == n_steps - 1:
                    z = torch.zeros_like(x)
                else:
                    z = torch.randn_like(x)

                # Get score
                sigma_input = torch.ones(batch_size, device=device) * sigma_val
                score = score_net(x, sigma_input)

                # Langevin step
                x = x + (alpha / 2) * score + np.sqrt(alpha) * z

                # Clamp to valid range
                x = x.clamp(0.0, 1.0)

    return x


def sdedit_composite(score_net, foreground, background, mask, sigmas, t0=0.5, n_steps=100, eps=2e-5, device=None):
    """
    Composite foreground onto background and harmonize using SDEdit.

    Args:
        score_net: Trained score network
        foreground: Foreground image (B, C, H, W)
        background: Background image (B, C, H, W)
        mask: Binary mask where 1 indicates foreground (B, 1, H, W)
        sigmas: Noise level schedule
        t0: Noise level ratio
        n_steps: Number of Langevin steps per noise level
        eps: Base step size
        device: Device to run on

    Returns:
        Harmonized composite image
    """
    if device is None:
        device = foreground.DEVICE

    # Create composite
    composite = mask * foreground + (1 - mask) * background

    # Apply SDEdit
    return sdedit_transform(score_net, composite, sigmas, t0=t0, n_steps=n_steps, eps=eps, device=device)


def sdedit_inpaint(score_net, image, mask, sigmas, t0=0.5, n_steps=100, eps=2e-5, device=None):
    """
    Inpaint masked regions of an image using SDEdit.

    Args:
        score_net: Trained score network
        image: Input image with regions to inpaint (B, C, H, W)
        mask: Binary mask where 1 indicates regions to inpaint (B, 1, H, W)
        sigmas: Noise level schedule
        t0: Noise level ratio
        n_steps: Number of Langevin steps per noise level
        eps: Base step size
        device: Device to run on

    Returns:
        Inpainted image
    """
    if device is None:
        device = image.DEVICE

    score_net.eval()
    batch_size = image.shape[0]
    num_scales = len(sigmas)

    # Calculate starting index
    start_idx = int((1 - t0) * (num_scales - 1))
    start_idx = max(0, min(start_idx, num_scales - 1))

    # Get starting sigma and add noise
    sigma_start = sigmas[start_idx].item()
    noise = torch.randn_like(image)
    x = image + sigma_start * noise

    # Run reverse SDE with inpainting constraint
    with torch.no_grad():
        for i in range(start_idx, num_scales):
            sigma = sigmas[i]
            sigma_val = sigma.item()

            alpha = eps * (sigma_val / sigmas[-1].item()) ** 2

            for t in range(n_steps):
                if i == num_scales - 1 and t == n_steps - 1:
                    z = torch.zeros_like(x)
                else:
                    z = torch.randn_like(x)

                sigma_input = torch.ones(batch_size, device=device) * sigma_val
                score = score_net(x, sigma_input)

                # Langevin step
                x = x + (alpha / 2) * score + np.sqrt(alpha) * z

                # Apply inpainting constraint: keep unmasked regions from original
                # Add noise to original for consistency at current noise level
                noisy_original = image + sigma_val * torch.randn_like(image)
                x = mask * x + (1 - mask) * noisy_original

                x = x.clamp(0.0, 1.0)

    return x


def sdedit_with_trajectory(score_net, guide_image, sigmas, t0=0.5, n_steps=100, eps=2e-5, device=None, save_every=1):
    """
    SDEdit transformation that returns intermediate results for visualization.

    Args:
        score_net: Trained score network
        guide_image: Input guide image (B, C, H, W)
        sigmas: Noise level schedule
        t0: Noise level ratio
        n_steps: Number of Langevin steps per noise level
        eps: Base step size
        device: Device to run on
        save_every: Save trajectory every N sigma levels

    Returns:
        final_image: The final edited image
        trajectory: List of intermediate images
    """
    if device is None:
        device = guide_image.DEVICE

    score_net.eval()
    batch_size = guide_image.shape[0]
    num_scales = len(sigmas)

    start_idx = int((1 - t0) * (num_scales - 1))
    start_idx = max(0, min(start_idx, num_scales - 1))

    sigma_start = sigmas[start_idx].item()
    noise = torch.randn_like(guide_image)
    x = guide_image + sigma_start * noise

    trajectory = [x.clone().cpu()]

    with torch.no_grad():
        for i in range(start_idx, num_scales):
            sigma = sigmas[i]
            sigma_val = sigma.item()

            alpha = eps * (sigma_val / sigmas[-1].item()) ** 2

            for t in range(n_steps):
                if i == num_scales - 1 and t == n_steps - 1:
                    z = torch.zeros_like(x)
                else:
                    z = torch.randn_like(x)

                sigma_input = torch.ones(batch_size, device=device) * sigma_val
                score = score_net(x, sigma_input)

                x = x + (alpha / 2) * score + np.sqrt(alpha) * z
                x = x.clamp(0.0, 1.0)

            # Save intermediate result
            if (i - start_idx) % save_every == 0:
                trajectory.append(x.clone().cpu())

    # Ensure final result is in trajectory
    if len(trajectory) == 0 or not torch.allclose(trajectory[-1], x.cpu()):
        trajectory.append(x.clone().cpu())

    return x, trajectory
