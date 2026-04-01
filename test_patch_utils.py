"""
Tests for patch_utils.py

Run: python test_patch_utils.py
"""

import torch
from patch_utils import (
    zero_pad_volume,
    compute_num_patches_per_dim,
    extract_patch,
    downsample_to_patch_size,
    build_positional_arrays,
    sample_random_patch_location,
    build_network_input,
)


def test_zero_pad_volume():
    print("=== test_zero_pad_volume ===")
    P = 32
    vol = torch.randn(1, 1, 64, 64, 64)  # (B, C, Nz, Ny, Nx)
    padded = zero_pad_volume(vol, P)

    expected = (1, 1, 64 + 2*P, 64 + 2*P, 64 + 2*P)
    assert padded.shape == expected, f"Expected {expected}, got {padded.shape}"

    # Check that the padding region is zero
    assert padded[0, 0, 0, 0, 0].item() == 0.0, "Corner of padding should be zero"
    assert padded[0, 0, -1, -1, -1].item() == 0.0, "Corner of padding should be zero"

    # Check that the original content is preserved in the interior
    interior = padded[:, :, P:P+64, P:P+64, P:P+64]
    assert torch.allclose(interior, vol), "Interior should match original volume"

    print(f"  Input:  {vol.shape}")
    print(f"  Output: {padded.shape}")
    print(f"  PASSED\n")


def test_compute_num_patches():
    print("=== test_compute_num_patches_per_dim ===")

    # 256 / 32 + 1 = 9 patches per dim
    n = compute_num_patches_per_dim(256, 32)
    assert n == 9, f"Expected 9, got {n}"
    print(f"  volume_size=256, patch_size=32 -> {n} patches")

    # 64 / 32 + 1 = 3
    n = compute_num_patches_per_dim(64, 32)
    assert n == 3, f"Expected 3, got {n}"
    print(f"  volume_size=64,  patch_size=32 -> {n} patches")

    # 512 / 64 + 1 = 9
    n = compute_num_patches_per_dim(512, 64)
    assert n == 9, f"Expected 9, got {n}"
    print(f"  volume_size=512, patch_size=64 -> {n} patches")

    print(f"  PASSED\n")


def test_extract_patch():
    print("=== test_extract_patch ===")
    P = 4  # small patch for easy verification
    Nz, Ny, Nx = 8, 8, 8

    # Create a volume with known values: each voxel = its flat index
    vol = torch.arange(Nz * Ny * Nx, dtype=torch.float32).reshape(1, Nz, Ny, Nx)
    padded = zero_pad_volume(vol, P)  # (1, 16, 16, 16)

    # Extract patch at offset (0,0,0), grid index (0,0,0)
    # This should grab padded[0, 0:4, 0:4, 0:4] which is all zeros (padding region)
    patch = extract_patch(padded, P, offset=(0, 0, 0), patch_idx=(0, 0, 0))
    assert patch.shape == (1, P, P, P), f"Expected (1,4,4,4), got {patch.shape}"
    assert torch.all(patch == 0), "Patch at (0,0,0) with offset (0,0,0) should be all zeros (padding)"

    # Extract patch at offset (0,0,0), grid index (1,1,1)
    # Starts at (4, 4, 4) in padded volume = (0, 0, 0) in original volume
    patch = extract_patch(padded, P, offset=(0, 0, 0), patch_idx=(1, 1, 1))
    expected = vol[0, 0:4, 0:4, 0:4]  # first 4x4x4 of original
    assert torch.allclose(patch[0], expected), "Patch (1,1,1) should match original volume corner"

    print(f"  Patch shape: {patch.shape}")
    print(f"  PASSED\n")


def test_downsample_to_patch_size():
    print("=== test_downsample_to_patch_size ===")
    P = 32

    # Test with a uniform volume: downsampled should also be uniform
    vol = torch.ones(1, 1, 256, 256, 256) * 3.14
    ds = downsample_to_patch_size(vol, P)
    assert ds.shape == (1, 1, P, P, P), f"Expected (1,1,32,32,32), got {ds.shape}"
    assert torch.allclose(ds, torch.tensor(3.14), atol=1e-5), "Uniform volume should downsample to same uniform value"

    # Test with non-cubic volume
    vol2 = torch.randn(2, 1, 128, 64, 64)
    ds2 = downsample_to_patch_size(vol2, P)
    assert ds2.shape == (2, 1, P, P, P), f"Expected (2,1,32,32,32), got {ds2.shape}"

    print(f"  Uniform input -> downsampled value: {ds[0,0,0,0,0].item():.4f} (expected 3.14)")
    print(f"  Non-cubic input {vol2.shape} -> {ds2.shape}")
    print(f"  PASSED\n")


def test_build_positional_arrays():
    print("=== test_build_positional_arrays ===")
    P = 4
    Nz, Ny, Nx = 8, 8, 8
    pos = build_positional_arrays((Nz, Ny, Nx), P)

    expected_shape = (3, Nz + 2*P, Ny + 2*P, Nx + 2*P)
    assert pos.shape == expected_shape, f"Expected {expected_shape}, got {pos.shape}"

    # Check that the center of the original volume maps to 0
    # Original volume center is at padded index P + (N-1)/2
    center_z = P + (Nz - 1) / 2  # = 4 + 3.5 = 7.5 (between indices 7 and 8)
    # At padded index P (= first original voxel): should be -1
    # At padded index P + N - 1 (= last original voxel): should be +1
    z_channel = pos[0]  # (Nz+2P, Ny+2P, Nx+2P)

    first_orig_z = z_channel[P, P, P].item()
    last_orig_z = z_channel[P + Nz - 1, P, P].item()
    assert abs(first_orig_z - (-1.0)) < 1e-5, f"First original z voxel should be -1, got {first_orig_z}"
    assert abs(last_orig_z - 1.0) < 1e-5, f"Last original z voxel should be +1, got {last_orig_z}"

    # Padding region should extend beyond [-1, 1]
    before_padding_z = z_channel[0, P, P].item()
    assert before_padding_z < -1.0, f"Padding before original should be < -1, got {before_padding_z}"

    print(f"  Shape: {pos.shape}")
    print(f"  Z range in original region: [{first_orig_z:.2f}, {last_orig_z:.2f}]")
    print(f"  Z in padding region: {before_padding_z:.2f} (should be < -1)")
    print(f"  PASSED\n")


def test_sample_random_patch_location():
    print("=== test_sample_random_patch_location ===")
    P = 32
    volume_shape = (256, 256, 256)

    # Sample many times and check bounds
    for _ in range(100):
        offset, patch_idx = sample_random_patch_location(volume_shape, P)
        o1, o2, o3 = offset
        iz, iy, ix = patch_idx

        assert 0 <= o1 < P, f"offset z out of range: {o1}"
        assert 0 <= o2 < P, f"offset y out of range: {o2}"
        assert 0 <= o3 < P, f"offset x out of range: {o3}"

        nz = compute_num_patches_per_dim(256, P)
        assert 0 <= iz < nz, f"patch_idx z out of range: {iz}"
        assert 0 <= iy < nz, f"patch_idx y out of range: {iy}"
        assert 0 <= ix < nz, f"patch_idx x out of range: {ix}"

    print(f"  100 random samples all within valid bounds")
    print(f"  Patches per dim for 256/32: {nz}")
    print(f"  PASSED\n")


def test_build_network_input():
    print("=== test_build_network_input ===")
    P = 32
    B = 2
    Nz, Ny, Nx = 64, 64, 64

    # Create fake data
    noisy_vol = torch.randn(B, 1, Nz, Ny, Nx)
    noisy_vol_padded = zero_pad_volume(noisy_vol, P)
    pos_arrays = build_positional_arrays((Nz, Ny, Nx), P)

    # Sample a patch location
    offset, patch_idx = sample_random_patch_location((Nz, Ny, Nx), P)

    # Extract noisy patch from padded volume
    noisy_patch = extract_patch(noisy_vol_padded, P, offset, patch_idx)  # (B, 1, P, P, P)

    # Build the 5-channel input
    net_input = build_network_input(
        noisy_patch=noisy_patch,
        noisy_volume_unpadded=noisy_vol,
        positional_arrays=pos_arrays,
        patch_size=P,
        offset=offset,
        patch_idx=patch_idx
    )

    expected_shape = (B, 5, P, P, P)
    assert net_input.shape == expected_shape, f"Expected {expected_shape}, got {net_input.shape}"

    # Verify channel 0 is the noisy patch
    assert torch.allclose(net_input[:, 0:1], noisy_patch), "Channel 0 should be the noisy patch"

    # Verify channel 1 is the downsampled volume
    ds = downsample_to_patch_size(noisy_vol, P)
    assert torch.allclose(net_input[:, 1:2], ds), "Channel 1 should be the downsampled volume"

    # Verify channels 2-4 are positional
    pos_patch = extract_patch(pos_arrays, P, offset, patch_idx)
    for b in range(B):
        assert torch.allclose(net_input[b, 2:5], pos_patch), "Channels 2-4 should be positional patches"

    print(f"  Input shape: {net_input.shape}")
    print(f"  Channel 0 (noisy patch): verified")
    print(f"  Channel 1 (downsampled): verified")
    print(f"  Channel 2-4 (positional): verified")
    print(f"  PASSED\n")


def test_end_to_end_training_step_shapes():
    """
    Simulate one training iteration from Algorithm 2 to verify
    all shapes are consistent end-to-end.
    """
    print("=== test_end_to_end_training_step_shapes ===")
    P = 32
    Nz, Ny, Nx = 64, 64, 64

    # Step 1: clean volume (single sample, as in Algorithm 2)
    x = torch.randn(1, 1, Nz, Ny, Nx)
    print(f"  Clean volume: {x.shape}")

    # Step 2: sample timestep and noise
    alpha_bar_t = 0.5  # pretend
    epsilon = torch.randn_like(x)
    x_t = (alpha_bar_t ** 0.5) * x + ((1 - alpha_bar_t) ** 0.5) * epsilon
    print(f"  Noisy volume: {x_t.shape}")

    # Step 3: pad and extract patch
    x_t_padded = zero_pad_volume(x_t, P)
    print(f"  Padded noisy volume: {x_t_padded.shape}")

    offset, patch_idx = sample_random_patch_location((Nz, Ny, Nx), P)
    noisy_patch = extract_patch(x_t_padded, P, offset, patch_idx)
    print(f"  Noisy patch (G_c * x_t): {noisy_patch.shape}")

    # Step 4: build positional arrays
    pos = build_positional_arrays((Nz, Ny, Nx), P, device=x.device)
    print(f"  Positional arrays: {pos.shape}")

    # Step 5: assemble 5-channel input
    net_input = build_network_input(noisy_patch, x_t, pos, P, offset, patch_idx)
    print(f"  Network input: {net_input.shape}")

    # Step 6: target is G_c(epsilon) - the patch-extracted noise
    epsilon_padded = zero_pad_volume(epsilon, P)
    target = extract_patch(epsilon_padded, P, offset, patch_idx)
    print(f"  Target (G_c * epsilon): {target.shape}")

    # Verify everything is P x P x P
    assert net_input.shape == (1, 5, P, P, P)
    assert target.shape == (1, 1, P, P, P)

    print(f"  All shapes consistent for training step")
    print(f"  PASSED\n")


if __name__ == "__main__":
    test_zero_pad_volume()
    test_compute_num_patches()
    test_extract_patch()
    test_downsample_to_patch_size()
    test_build_positional_arrays()
    test_sample_random_patch_location()
    test_build_network_input()
    test_end_to_end_training_step_shapes()
    print("=" * 50)
    print("ALL TESTS PASSED")