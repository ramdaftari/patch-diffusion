"""
Patch-based diffusion training loss (Algorithm 2, arXiv 2512.18161).

Replaces the original epsilon_based_loss_fn which operates on full images.
This version:
  1. Noises the full 3D volume
  2. Extracts a random P x P x P patch from the noisy volume
  3. Builds the 5-channel input (patch + downsampled + positional)
  4. Predicts noise on the patch
  5. Computes MSE against the patch-extracted ground truth noise

Also includes a standalone DDPM noise schedule (same math as sde.py)
to avoid the omegaconf dependency.

Usage:
    from patch_loss import PatchDiffusionLoss
    loss_fn = PatchDiffusionLoss(patch_size=32, volume_shape=(Nz, Ny, Nx))
    loss = loss_fn(x_clean, model)
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Tuple

from patch_utils import (
    zero_pad_volume,
    extract_patch,
    build_positional_arrays,
    sample_random_patch_location,
    build_network_input,
)


class DDPMSchedule:
    """
    DDPM noise schedule. Same math as sde.py but without omegaconf.

    Paper uses: beta_min=0.0001, beta_max=0.02, num_steps=1000
    (matching Table 6 supplementary).
    """

    def __init__(
        self,
        beta_min: float = 0.0001,
        beta_max: float = 0.02,
        num_steps: int = 1000,
    ):
        self.num_steps = num_steps
        self.betas = torch.from_numpy(
            np.linspace(beta_min, beta_max, num_steps, dtype=np.float64)
        )

    def _compute_alpha_cumprod(self, t: torch.Tensor) -> torch.Tensor:
        """Compute alpha_bar_t for given timesteps."""
        betas = torch.cat([torch.zeros(1), self.betas], dim=0)
        return (
            (1 - betas.to(t.device))
            .cumprod(dim=0)
            .index_select(0, t.long() + 1)
            .to(torch.float32)
        )

    def marginal_prob(self, x: torch.Tensor, t: torch.Tensor):
        """
        Compute mean and std of q(x_t | x_0).

        Returns:
            mean: x_0 * sqrt(alpha_bar_t), same shape as x
            std:  sqrt(1 - alpha_bar_t), shape (B,)
        """
        alpha_bar = self._compute_alpha_cumprod(t)
        mean_coeff = alpha_bar.pow(0.5)
        std = (1.0 - alpha_bar).pow(0.5)

        # Reshape for broadcasting with 5D tensor (B, C, Z, Y, X)
        while mean_coeff.dim() < x.dim():
            mean_coeff = mean_coeff.unsqueeze(-1)

        return x * mean_coeff, std


class PatchDiffusionLoss(nn.Module):
    """
    Patch-based diffusion training loss (Algorithm 2).

    Precomputes positional arrays once for the given volume shape,
    then on each call:
      1. Samples random timestep
      2. Noises the full volume
      3. Samples a random patch location
      4. Extracts patch from noisy volume
      5. Builds 5-channel network input
      6. Forward through model
      7. Computes MSE against patch-extracted ground truth noise

    Args:
        patch_size: P, the cubic patch size (default: 32)
        volume_shape: (Nz, Ny, Nx) of the un-padded volume
        beta_min, beta_max, num_steps: noise schedule parameters
    """

    def __init__(
        self,
        patch_size: int = 32,
        volume_shape: Tuple[int, int, int] = (256, 256, 256),
        beta_min: float = 0.0001,
        beta_max: float = 0.02,
        num_steps: int = 1000,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.volume_shape = volume_shape
        self.schedule = DDPMSchedule(beta_min, beta_max, num_steps)

        # Positional arrays depend only on geometry, precompute once.
        # Not a parameter (no gradients), so use register_buffer.
        pos = build_positional_arrays(volume_shape, patch_size)
        self.register_buffer('positional_arrays', pos)

    def forward(self, x: torch.Tensor, model: nn.Module) -> torch.Tensor:
        """
        Compute the patch-based denoising loss.

        Args:
            x: Clean volume, shape (B, 1, Nz, Ny, Nx)
            model: UNet that takes (B, 5, P, P, P) -> (B, 1, P, P, P)

        Returns:
            Scalar loss
        """
        B = x.shape[0]
        P = self.patch_size
        device = x.device

        # Step 1: sample random timestep (Algorithm 2, line 2)
        t = torch.randint(1, self.schedule.num_steps, (B,), device=device)

        # Step 2: sample noise (Algorithm 2, line 4)
        epsilon = torch.randn_like(x)

        # Step 3: create noisy volume (Algorithm 2, line 5)
        mean, std = self.schedule.marginal_prob(x, t)
        # Reshape std for 5D broadcasting
        while std.dim() < x.dim():
            std = std.unsqueeze(-1)
        x_t = mean + epsilon * std

        # Step 4: pad noisy volume and noise
        x_t_padded = zero_pad_volume(x_t, P)
        epsilon_padded = zero_pad_volume(epsilon, P)

        # Step 5: sample random patch location (Algorithm 2, line 3)
        # For batch training, use the same patch location for all samples in batch
        offset, patch_idx = sample_random_patch_location(self.volume_shape, P)

        # Step 6: extract noisy patch (Algorithm 2, line 6: u = G_c * x_t)
        noisy_patch = extract_patch(x_t_padded, P, offset, patch_idx)

        # Step 7: build 5-channel input [patch, downsample, pos_z, pos_y, pos_x]
        net_input = build_network_input(
            noisy_patch, x_t, self.positional_arrays, P, offset, patch_idx
        )

        # Step 8: predict noise (Algorithm 2, line 8: D_theta(u, v))
        epsilon_hat = model(net_input, t)

        # Step 9: extract target noise from same patch location
        # (Algorithm 2, line 8: G_c * epsilon)
        target = extract_patch(epsilon_padded, P, offset, patch_idx)

        # Step 10: MSE loss (Algorithm 2, line 8)
        # Sum over (C, Z, Y, X) dims, mean over batch
        loss = torch.mean(
            torch.sum((epsilon_hat - target).pow(2), dim=(1, 2, 3, 4))
        )

        return loss