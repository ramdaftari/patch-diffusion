"""
Tests for patch_loss.py

Run from ~/patch_diffusion:
    conda activate res_rob
    python test_patch_loss.py
"""

import torch
from patch_unet import create_patch_unet
from patch_loss import DDPMSchedule, PatchDiffusionLoss

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}\n")


def test_ddpm_schedule():
    """Verify noise schedule matches the original sde.py behavior."""
    print("=== test_ddpm_schedule ===")
    schedule = DDPMSchedule()

    # At t=0: alpha_bar should be close to 1 (almost no noise)
    t0 = torch.tensor([0])
    _, std0 = schedule.marginal_prob(torch.ones(1, 1, 4, 4, 4), t0)
    print(f"  t=0:   std={std0.item():.6f} (should be ~0, almost no noise)")
    assert std0.item() < 0.02

    # At t=999: alpha_bar should be close to 0 (almost all noise)
    t999 = torch.tensor([999])
    _, std999 = schedule.marginal_prob(torch.ones(1, 1, 4, 4, 4), t999)
    print(f"  t=999: std={std999.item():.6f} (should be ~1, mostly noise)")
    assert std999.item() > 0.9

    # Check broadcasting: mean should have same shape as input
    x = torch.randn(2, 1, 32, 32, 32)
    t = torch.tensor([100, 500])
    mean, std = schedule.marginal_prob(x, t)
    assert mean.shape == x.shape, f"Mean shape {mean.shape} != input shape {x.shape}"
    print(f"  Broadcasting: input {x.shape} -> mean {mean.shape}, std {std.shape}")

    print(f"  PASSED\n")


def test_loss_output_shape():
    """Loss should return a scalar."""
    print("=== test_loss_output_shape ===")
    Nz, Ny, Nx = 64, 64, 64
    model = create_patch_unet().to(device)
    loss_fn = PatchDiffusionLoss(
        patch_size=32, volume_shape=(Nz, Ny, Nx)
    ).to(device)

    x = torch.randn(1, 1, Nz, Ny, Nx, device=device)
    loss = loss_fn(x, model)

    assert loss.dim() == 0, f"Loss should be scalar, got shape {loss.shape}"
    assert not torch.isnan(loss), "Loss is NaN"
    assert not torch.isinf(loss), "Loss is Inf"

    print(f"  Loss value: {loss.item():.4f}")
    print(f"  PASSED\n")


def test_loss_backward():
    """Verify loss.backward() populates all gradients."""
    print("=== test_loss_backward ===")
    Nz, Ny, Nx = 64, 64, 64
    model = create_patch_unet().to(device)
    loss_fn = PatchDiffusionLoss(
        patch_size=32, volume_shape=(Nz, Ny, Nx)
    ).to(device)

    x = torch.randn(1, 1, Nz, Ny, Nx, device=device)
    loss = loss_fn(x, model)
    loss.backward()

    num_params = sum(1 for _ in model.parameters())
    num_grad = sum(1 for p in model.parameters() if p.grad is not None)

    print(f"  Gradients populated: {num_grad}/{num_params}")
    assert num_grad == num_params, "All parameters should have gradients"
    print(f"  PASSED\n")


def test_loss_batched():
    """Loss should work with batch_size > 1."""
    print("=== test_loss_batched ===")
    Nz, Ny, Nx = 64, 64, 64
    model = create_patch_unet().to(device)
    loss_fn = PatchDiffusionLoss(
        patch_size=32, volume_shape=(Nz, Ny, Nx)
    ).to(device)

    x = torch.randn(2, 1, Nz, Ny, Nx, device=device)
    loss = loss_fn(x, model)

    assert loss.dim() == 0, f"Loss should be scalar, got shape {loss.shape}"
    assert not torch.isnan(loss), "Loss is NaN"

    print(f"  Batch size 2, loss: {loss.item():.4f}")
    print(f"  PASSED\n")


def test_overfit_one_volume():
    """
    The real test: can the model overfit a single volume using our loss?
    This exercises the full Algorithm 2 pipeline over many steps.
    We use more steps (200) and proper LR to account for zero_module init.
    """
    print("=== test_overfit_one_volume ===")
    Nz, Ny, Nx = 64, 64, 64
    model = create_patch_unet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-4)
    loss_fn = PatchDiffusionLoss(
        patch_size=32, volume_shape=(Nz, Ny, Nx)
    ).to(device)

    # Fixed volume to overfit
    x = torch.randn(1, 1, Nz, Ny, Nx, device=device)

    losses = []
    for step in range(200):
        optimizer.zero_grad()
        loss = loss_fn(x, model)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        if step % 50 == 0:
            print(f"  Step {step:3d}: loss = {loss.item():.4f}")

    print(f"  Step 199: loss = {losses[-1]:.4f}")

    # Loss should decrease from initial value
    # Use average of first 5 vs last 5 to smooth out randomness
    # (each call samples random t and patch location)
    avg_first = sum(losses[:5]) / 5
    avg_last = sum(losses[-5:]) / 5
    print(f"  Avg first 5: {avg_first:.4f}")
    print(f"  Avg last  5: {avg_last:.4f}")

    assert avg_last < avg_first, \
        f"Loss should decrease: {avg_first:.4f} -> {avg_last:.4f}"
    print(f"  Loss decreased: {avg_first:.4f} -> {avg_last:.4f}")
    print(f"  PASSED\n")


def test_matches_original_schedule():
    """
    Verify our DDPMSchedule produces identical alpha_bar values
    to the original sde.py DDPM class.
    """
    print("=== test_matches_original_schedule ===")
    schedule = DDPMSchedule(beta_min=0.0001, beta_max=0.02, num_steps=1000)

    # Manually compute alpha_bar the same way sde.py does
    import numpy as np
    betas = np.linspace(0.0001, 0.02, 1000, dtype=np.float64)
    alphas = 1.0 - betas
    alpha_bar = np.cumprod(alphas)

    # Check a few timesteps
    for t_val in [0, 100, 500, 999]:
        t = torch.tensor([t_val])
        our_alpha_bar = schedule._compute_alpha_cumprod(t).item()
        ref_alpha_bar = alpha_bar[t_val]
        diff = abs(our_alpha_bar - ref_alpha_bar)
        print(f"  t={t_val:3d}: ours={our_alpha_bar:.8f}, ref={ref_alpha_bar:.8f}, diff={diff:.2e}")
        assert diff < 1e-6, f"Alpha bar mismatch at t={t_val}"

    print(f"  PASSED\n")


if __name__ == "__main__":
    test_ddpm_schedule()
    test_matches_original_schedule()
    test_loss_output_shape()
    test_loss_backward()
    test_loss_batched()
    test_overfit_one_volume()
    print("=" * 50)
    print("ALL TESTS PASSED")