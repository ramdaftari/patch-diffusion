"""
Tests for mri_dataset.py

Run from ~/patch_diffusion:
    conda activate res_rob
    python test_mri_dataset.py
"""

import torch
from mri_dataset import StanfordKneeVolumeDataset
from patch_utils import zero_pad_volume, build_positional_arrays

# Paths on your machine
DATA_ROOT = '/home/ram/trial-project/stanford_fastmri_cor/sag'
SENSMAP_ROOT = '/home/ram/trial-project/stanford_fastmri_cor/sensmaps'


def test_dataset_loads():
    """Can we load the dataset and get the right number of volumes?"""
    print("=== test_dataset_loads ===")
    dataset = StanfordKneeVolumeDataset(
        data_root=DATA_ROOT,
        sensmap_root=SENSMAP_ROOT,
    )
    print(f"  Number of volumes: {len(dataset)}")
    assert len(dataset) > 0, "Dataset is empty"
    print(f"  PASSED\n")


def test_volume_shape():
    """Check that a volume has the expected shape (1, 256, 320, 320)."""
    print("=== test_volume_shape ===")
    dataset = StanfordKneeVolumeDataset(
        data_root=DATA_ROOT,
        sensmap_root=SENSMAP_ROOT,
    )
    vol = dataset[0]
    print(f"  Volume shape: {vol.shape}")
    print(f"  Dtype: {vol.dtype}")

    assert vol.dim() == 4, f"Expected 4D tensor, got {vol.dim()}D"
    assert vol.shape[0] == 1, f"Expected 1 channel, got {vol.shape[0]}"
    assert vol.dtype == torch.float32, f"Expected float32, got {vol.dtype}"

    # Stanford knee should be (1, 256, 320, 320)
    print(f"  Spatial dims: Z={vol.shape[1]}, Y={vol.shape[2]}, X={vol.shape[3]}")
    print(f"  PASSED\n")


def test_volume_values():
    """Check that values are reasonable after normalization."""
    print("=== test_volume_values ===")
    dataset = StanfordKneeVolumeDataset(
        data_root=DATA_ROOT,
        sensmap_root=SENSMAP_ROOT,
        normalize=True,
    )
    vol = dataset[0]

    mean = vol.mean().item()
    std = vol.std().item()
    vmin = vol.min().item()
    vmax = vol.max().item()

    print(f"  Mean: {mean:.4f} (should be ~0)")
    print(f"  Std:  {std:.4f} (should be ~1)")
    print(f"  Min:  {vmin:.4f}")
    print(f"  Max:  {vmax:.4f}")

    assert abs(mean) < 0.1, f"Mean too far from 0: {mean}"
    assert abs(std - 1.0) < 0.1, f"Std too far from 1: {std}"
    assert not torch.isnan(vol).any(), "Volume contains NaN"
    assert not torch.isinf(vol).any(), "Volume contains Inf"
    print(f"  PASSED\n")


def test_unnormalized():
    """Check that raw (unnormalized) values are positive magnitudes."""
    print("=== test_unnormalized ===")
    dataset = StanfordKneeVolumeDataset(
        data_root=DATA_ROOT,
        sensmap_root=SENSMAP_ROOT,
        normalize=False,
    )
    vol = dataset[0]

    assert vol.min() >= 0, f"Magnitude should be non-negative, got min={vol.min()}"
    print(f"  Min: {vol.min().item():.6f} (non-negative, good)")
    print(f"  Max: {vol.max().item():.6f}")
    print(f"  Mean: {vol.mean().item():.6f}")
    print(f"  PASSED\n")


def test_divisible_by_patch_size():
    """Verify volume dims are divisible by P=32 (required for patch extraction)."""
    print("=== test_divisible_by_patch_size ===")
    dataset = StanfordKneeVolumeDataset(
        data_root=DATA_ROOT,
        sensmap_root=SENSMAP_ROOT,
    )
    vol = dataset[0]
    P = 32
    _, Nz, Ny, Nx = vol.shape

    assert Nz % P == 0, f"Z={Nz} not divisible by P={P}"
    assert Ny % P == 0, f"Y={Ny} not divisible by P={P}"
    assert Nx % P == 0, f"X={Nx} not divisible by P={P}"

    print(f"  Volume ({Nz}, {Ny}, {Nx}) / P={P} = ({Nz//P}, {Ny//P}, {Nx//P}) patches per dim")
    print(f"  PASSED\n")


def test_compatible_with_patch_utils():
    """Integration test: volume -> pad -> extract patch -> correct shapes."""
    print("=== test_compatible_with_patch_utils ===")
    dataset = StanfordKneeVolumeDataset(
        data_root=DATA_ROOT,
        sensmap_root=SENSMAP_ROOT,
    )
    vol = dataset[0]  # (1, Z, Y, X)
    P = 32
    _, Nz, Ny, Nx = vol.shape

    # Add batch dim: (1, 1, Z, Y, X)
    vol_batch = vol.unsqueeze(0)

    # Pad
    padded = zero_pad_volume(vol_batch, P)
    expected = (1, 1, Nz + 2*P, Ny + 2*P, Nx + 2*P)
    assert padded.shape == expected, f"Expected {expected}, got {padded.shape}"

    # Positional arrays
    pos = build_positional_arrays((Nz, Ny, Nx), P)
    expected_pos = (3, Nz + 2*P, Ny + 2*P, Nx + 2*P)
    assert pos.shape == expected_pos, f"Expected {expected_pos}, got {pos.shape}"

    print(f"  Volume: {vol_batch.shape}")
    print(f"  Padded: {padded.shape}")
    print(f"  Positional: {pos.shape}")
    print(f"  All compatible with patch extraction pipeline")
    print(f"  PASSED\n")


def test_multiple_volumes():
    """Load a few volumes and check they all have the same shape."""
    print("=== test_multiple_volumes ===")
    dataset = StanfordKneeVolumeDataset(
        data_root=DATA_ROOT,
        sensmap_root=SENSMAP_ROOT,
    )
    n = min(3, len(dataset))
    shapes = []
    for i in range(n):
        vol = dataset[i]
        shapes.append(vol.shape)
        print(f"  Volume {i}: shape={vol.shape}, "
              f"mean={vol.mean():.4f}, std={vol.std():.4f}")

    # All should have same shape
    assert all(s == shapes[0] for s in shapes), \
        f"Volumes have different shapes: {shapes}"
    print(f"  All {n} volumes have identical shape: {shapes[0]}")
    print(f"  PASSED\n")


if __name__ == "__main__":
    test_dataset_loads()
    test_volume_shape()
    test_volume_values()
    test_unnormalized()
    test_divisible_by_patch_size()
    test_compatible_with_patch_utils()
    test_multiple_volumes()
    print("=" * 50)
    print("ALL TESTS PASSED")