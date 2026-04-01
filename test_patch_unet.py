"""
Tests for patch_unet.py

Run from ~/patch_diffusion:
    conda activate res_rob
    python test_patch_unet.py
"""

import torch
from patch_unet import (
    IsotropicUpsample3D,
    IsotropicDownsample3D,
    create_patch_unet,
)
from unet_arch.unet import Upsample as OrigUpsample, Downsample as OrigDownsample
from unet_arch.nn import timestep_embedding


def test_isotropic_upsample():
    print("=== test_isotropic_upsample ===")
    up = IsotropicUpsample3D(channels=64, use_conv=False, dims=3)

    x = torch.randn(1, 64, 4, 4, 4)
    y = up(x)
    expected = (1, 64, 8, 8, 8)
    assert y.shape == expected, f"Expected {expected}, got {y.shape}"
    print(f"  {x.shape} -> {y.shape}  (all dims doubled)")

    # Compare with original anisotropic behavior
    up_orig = OrigUpsample(channels=64, use_conv=False, dims=3)
    y_orig = up_orig(x)
    print(f"  Original would give: {y_orig.shape}  (Z stays fixed)")
    assert y_orig.shape[2] == 4, "Original should keep Z=4"
    assert y.shape[2] == 8, "Isotropic should double Z to 8"

    print(f"  PASSED\n")


def test_isotropic_downsample():
    print("=== test_isotropic_downsample ===")
    down = IsotropicDownsample3D(channels=64, use_conv=True, dims=3)

    x = torch.randn(1, 64, 8, 8, 8)
    y = down(x)
    expected = (1, 64, 4, 4, 4)
    assert y.shape == expected, f"Expected {expected}, got {y.shape}"
    print(f"  {x.shape} -> {y.shape}  (all dims halved)")

    # Compare with original
    down_orig = OrigDownsample(channels=64, use_conv=True, dims=3)
    y_orig = down_orig(x)
    print(f"  Original would give: {y_orig.shape}  (Z stays fixed)")
    assert y_orig.shape[2] == 8, "Original should keep Z=8"
    assert y.shape[2] == 4, "Isotropic should halve Z to 4"

    print(f"  PASSED\n")


def test_create_patch_unet_shapes():
    """
    Full forward pass.
    Input:  (B, 5, 32, 32, 32)
    Output: (B, 1, 32, 32, 32)
    """
    print("=== test_create_patch_unet_shapes ===")
    model = create_patch_unet()

    P = 32
    B = 1
    x = torch.randn(B, 5, P, P, P)
    t = torch.randint(0, 1000, (B,))

    with torch.no_grad():
        y = model(x, t)

    expected = (B, 1, P, P, P)
    assert y.shape == expected, f"Expected {expected}, got {y.shape}"

    print(f"  Input:  {x.shape}")
    print(f"  Output: {y.shape}")
    print(f"  PASSED\n")


def test_spatial_dims_through_levels():
    """
    Trace spatial dims through encoder. Should be:
      32 -> 16 -> 8 (attention) -> 4 (bottleneck)
    All three spatial dims equal at every stage.
    """
    print("=== test_spatial_dims_through_levels ===")
    model = create_patch_unet()

    P = 32
    x = torch.randn(1, 5, P, P, P)
    t = torch.randint(0, 1000, (1,))
    emb = model.time_embed(timestep_embedding(t, model.model_channels))

    h = x.float()
    print(f"  Input: {h.shape}")
    for i, module in enumerate(model.input_blocks):
        h = module(h, emb)
        # Check isotropy at every stage
        assert h.shape[2] == h.shape[3] == h.shape[4], \
            f"Block {i}: spatial dims not equal: {h.shape[2:]}"
        print(f"  After input_block[{i}]: {h.shape}")

    h = model.middle_block(h, emb)
    print(f"  After middle_block: {h.shape}")

    assert h.shape[2] == h.shape[3] == h.shape[4] == 4, \
        f"Bottleneck should be 4x4x4, got {h.shape[2:]}"

    print(f"  PASSED\n")


def test_param_count():
    """
    Paper reports 68.59M parameters.
    """
    print("=== test_param_count ===")
    model = create_patch_unet()

    total = sum(p.numel() for p in model.parameters())
    total_m = total / 1e6

    print(f"  Total parameters: {total_m:.2f}M")
    print(f"  Paper reports:    68.59M")

    assert 10 < total_m < 200, f"Parameter count {total_m:.2f}M seems off"
    print(f"  PASSED\n")


def test_original_unet_unaffected():
    """
    After create_patch_unet(), the original Upsample/Downsample
    in unet_arch should be restored.
    """
    print("=== test_original_unet_unaffected ===")

    # Create our model (temporarily patches the module)
    _ = create_patch_unet()

    # Check originals are restored
    from unet_arch import unet as unet_module
    assert unet_module.Upsample is OrigUpsample, "Original Upsample should be restored"
    assert unet_module.Downsample is OrigDownsample, "Original Downsample should be restored"

    # Verify original still has anisotropic behavior
    down_orig = OrigDownsample(channels=32, use_conv=True, dims=3)
    x = torch.randn(1, 32, 8, 8, 8)
    y = down_orig(x)
    assert y.shape == (1, 32, 8, 4, 4), f"Original should be anisotropic, got {y.shape}"

    print(f"  Original Upsample/Downsample restored")
    print(f"  Original Downsample still anisotropic: (8,8,8) -> {y.shape[2:]}")
    print(f"  PASSED\n")


if __name__ == "__main__":
    test_isotropic_upsample()
    test_isotropic_downsample()
    test_create_patch_unet_shapes()
    test_spatial_dims_through_levels()
    test_param_count()
    test_original_unet_unaffected()
    print("=" * 50)
    print("ALL TESTS PASSED")