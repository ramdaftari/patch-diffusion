"""
Extended sanity checks for patch_unet + patch_utils integration.

Run from ~/patch_diffusion:
    conda activate res_rob
    python test_sanity.py
"""

import torch
from patch_unet import create_patch_unet
from patch_utils import (
    zero_pad_volume,
    extract_patch,
    build_positional_arrays,
    sample_random_patch_location,
    build_network_input,
)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}\n")


def test_backward_pass():
    """
    Verify gradients flow through the model.

    The UNet's final conv uses zero_module (weights/bias initialized to 0).
    This means at initialization, gradients for most earlier layers are
    numerically zero since they must flow through zero-weight conv.
    We check: (1) all grads are populated, (2) at least some are non-zero.
    The overfit tests below are the real proof the model can learn.
    """
    print("=== test_backward_pass ===")
    model = create_patch_unet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    x = torch.randn(1, 5, 32, 32, 32, device=device)
    t = torch.randint(0, 1000, (1,), device=device)
    target = torch.randn(1, 1, 32, 32, 32, device=device)

    # Step 1: first forward/backward populates grads,
    # but most are zero due to zero_module on last layer
    optimizer.zero_grad()
    y = model(x, t)
    loss = (y - target).pow(2).mean()
    loss.backward()

    num_params = sum(1 for _ in model.parameters())
    num_grad_populated = sum(1 for p in model.parameters() if p.grad is not None)
    print(f"  After first backward: {num_grad_populated}/{num_params} grads populated")
    assert num_grad_populated == num_params, "All parameters should have .grad after backward()"

    # Confirm gradients are flowing: some must be non-zero.
    # Not all will be, because zero_module on the last layer makes
    # gradients vanish for most earlier layers at initialization.
    # The overfit tests below are the real proof the model can learn.
    num_nonzero = sum(
        1 for p in model.parameters()
        if p.grad is not None and p.grad.abs().sum() > 0
    )
    print(f"  Non-zero gradients: {num_nonzero}/{num_params}")
    assert num_nonzero > 0, "At least some gradients should be non-zero"
    print(f"  PASSED\n")


def test_overfit_single_sample():
    """
    Classic ML sanity check: overfit on one fixed input.
    If loss doesn't decrease, something is wrong with the architecture.
    """
    print("=== test_overfit_single_sample ===")
    model = create_patch_unet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    x = torch.randn(1, 5, 32, 32, 32, device=device)
    t = torch.randint(0, 1000, (1,), device=device)
    target_noise = torch.randn(1, 1, 32, 32, 32, device=device)

    losses = []
    for step in range(50):
        optimizer.zero_grad()
        pred = model(x, t)
        loss = (pred - target_noise).pow(2).mean()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    print(f"  Loss step  0: {losses[0]:.4f}")
    print(f"  Loss step 24: {losses[24]:.4f}")
    print(f"  Loss step 49: {losses[49]:.4f}")

    assert losses[49] < losses[0], \
        f"Loss did not decrease: {losses[0]:.4f} -> {losses[49]:.4f}"
    print(f"  Loss decreased: {losses[0]:.4f} -> {losses[49]:.4f}")
    print(f"  PASSED\n")


def test_full_training_iteration():
    """
    Simulate one complete training iteration from Algorithm 2:
      1. Start with a clean volume
      2. Sample timestep, generate noise
      3. Create noisy volume
      4. Pad, extract patch from noisy volume
      5. Build 5-channel input (patch + downsample + positional)
      6. Forward through UNet -> predicted noise
      7. Extract same patch from ground truth noise
      8. Compute loss between predicted and target patch noise
      9. Backward pass
    """
    print("=== test_full_training_iteration ===")
    P = 32
    Nz, Ny, Nx = 64, 64, 64

    model = create_patch_unet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # Step 1: clean volume
    x_clean = torch.randn(1, 1, Nz, Ny, Nx, device=device)
    print(f"  Clean volume: {x_clean.shape}")

    # Step 2: sample timestep and noise
    t = torch.randint(1, 1000, (1,), device=device)
    epsilon = torch.randn_like(x_clean)
    alpha_bar_t = 1.0 - t.float() / 1000.0
    mean_coeff = alpha_bar_t.sqrt()
    std_coeff = (1.0 - alpha_bar_t).sqrt()

    # Step 3: noisy volume
    x_noisy = mean_coeff * x_clean + std_coeff * epsilon
    print(f"  Noisy volume: {x_noisy.shape}, t={t.item()}")

    # Step 4: pad and extract patch
    x_noisy_padded = zero_pad_volume(x_noisy, P)
    offset, patch_idx = sample_random_patch_location((Nz, Ny, Nx), P)
    noisy_patch = extract_patch(x_noisy_padded, P, offset, patch_idx)
    print(f"  Noisy patch: {noisy_patch.shape}, offset={offset}, idx={patch_idx}")

    # Step 5: build 5-channel input
    pos = build_positional_arrays((Nz, Ny, Nx), P, device=device)
    net_input = build_network_input(noisy_patch, x_noisy, pos, P, offset, patch_idx)
    print(f"  Network input: {net_input.shape}")
    assert net_input.shape == (1, 5, P, P, P)

    # Step 6: forward
    pred_noise = model(net_input, t)
    print(f"  Predicted noise: {pred_noise.shape}")
    assert pred_noise.shape == (1, 1, P, P, P)

    # Step 7: extract target noise from same patch location
    epsilon_padded = zero_pad_volume(epsilon, P)
    target_noise = extract_patch(epsilon_padded, P, offset, patch_idx)
    print(f"  Target noise: {target_noise.shape}")
    assert target_noise.shape == (1, 1, P, P, P)

    # Step 8: loss
    loss = (pred_noise - target_noise).pow(2).sum(dim=(1, 2, 3, 4)).mean()
    print(f"  Loss: {loss.item():.4f}")

    # Step 9: backward
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    has_grad = all(p.grad is not None for p in model.parameters())
    assert has_grad, "Some parameters missing gradients"
    print(f"  Backward pass successful, all gradients computed")
    print(f"  PASSED\n")


def test_overfit_full_pipeline():
    """
    Overfit the full Algorithm 2 pipeline on one fixed volume.
    Uses the same patch location each time so the model sees
    identical data every step. Loss must decrease.
    """
    print("=== test_overfit_full_pipeline ===")
    P = 32
    Nz, Ny, Nx = 64, 64, 64

    model = create_patch_unet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # Fixed volume, timestep, noise, and patch location
    x_clean = torch.randn(1, 1, Nz, Ny, Nx, device=device)
    t = torch.tensor([500], device=device)
    epsilon = torch.randn_like(x_clean)
    offset = (0, 0, 0)
    patch_idx = (1, 1, 1)

    # Precompute fixed quantities
    alpha_bar_t = 1.0 - t.float() / 1000.0
    x_noisy = alpha_bar_t.sqrt() * x_clean + (1 - alpha_bar_t).sqrt() * epsilon
    x_noisy_padded = zero_pad_volume(x_noisy, P)
    noisy_patch = extract_patch(x_noisy_padded, P, offset, patch_idx)
    pos = build_positional_arrays((Nz, Ny, Nx), P, device=device)
    net_input = build_network_input(noisy_patch, x_noisy, pos, P, offset, patch_idx)

    epsilon_padded = zero_pad_volume(epsilon, P)
    target_noise = extract_patch(epsilon_padded, P, offset, patch_idx)

    losses = []
    for step in range(50):
        optimizer.zero_grad()
        pred = model(net_input, t)
        loss = (pred - target_noise).pow(2).sum(dim=(1, 2, 3, 4)).mean()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    print(f"  Loss step  0: {losses[0]:.4f}")
    print(f"  Loss step 24: {losses[24]:.4f}")
    print(f"  Loss step 49: {losses[49]:.4f}")

    assert losses[49] < losses[0] * 0.9, \
        f"Loss didn't decrease enough: {losses[0]:.4f} -> {losses[49]:.4f}"
    print(f"  Full pipeline overfit successful")
    print(f"  PASSED\n")


if __name__ == "__main__":
    test_backward_pass()
    test_overfit_single_sample()
    test_full_training_iteration()
    test_overfit_full_pipeline()
    print("=" * 50)
    print("ALL SANITY CHECKS PASSED")