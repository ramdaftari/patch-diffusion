"""
Isotropic 3D UNet for patch-based diffusion (arXiv 2512.18161).

Only two changes from the original Krainovic et al. UNet:
  1. Upsample: scale_factor=2 in all 3 dims (was keeping Z fixed)
  2. Downsample: stride=2 in all 3 dims (was (1,2,2))

Usage:
    from patch_unet import create_patch_unet
    model = create_patch_unet()  # uses paper defaults
"""

import torch.nn as nn
import torch.nn.functional as F

from unet_arch import unet as unet_module
from unet_arch.unet import UNetModel
from unet_arch.nn import conv_nd, avg_pool_nd


# ---------- Fixed classes ----------

class IsotropicUpsample3D(nn.Module):
    """
    Upsample with isotropic 2x scaling in all spatial dims.
    Original: when dims==3, only upsampled H,W (kept Z fixed).
    Fixed: scale_factor=2 in all 3 dims.
    """

    def __init__(self, channels, use_conv, dims=2, out_channels=None):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.dims = dims
        if use_conv:
            self.conv = conv_nd(dims, self.channels, self.out_channels, 3, padding=1)

    def forward(self, x):
        assert x.shape[1] == self.channels
        # FIX: scale_factor=2 for all dims uniformly
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        if self.use_conv:
            x = self.conv(x)
        return x


class IsotropicDownsample3D(nn.Module):
    """
    Downsample with isotropic stride=2 in all spatial dims.
    Original: when dims==3, stride was (1,2,2).
    Fixed: stride=2 in all 3 dims.
    """

    def __init__(self, channels, use_conv, dims=2, out_channels=None):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.dims = dims
        # FIX: stride=2 uniformly, not (1,2,2)
        stride = 2
        if use_conv:
            self.op = conv_nd(
                dims, self.channels, self.out_channels, 3, stride=stride, padding=1
            )
        else:
            assert self.channels == self.out_channels
            self.op = avg_pool_nd(dims, kernel_size=stride, stride=stride)

    def forward(self, x):
        assert x.shape[1] == self.channels
        return self.op(x)


# ---------- Factory ----------

def create_patch_unet(
    in_channels=5,
    out_channels=1,
    model_channels=64,
    channel_mult=(1, 2, 4, 4),
    num_res_blocks=2,
    attention_resolutions=(8,),
    num_heads=4,
    num_head_channels=64,
    dropout=0.0,
    use_scale_shift_norm=False,
    resblock_updown=False,
    use_fp16=False,
    use_new_attention_order=False,
    dims=3,
):
    """
    Create the 3D UNet matching the Local Patches paper (Table 6).

    Monkey-patches Upsample/Downsample in the original unet module
    to be isotropic, creates the model, then restores the originals.

    Paper defaults (Table 6, supplementary):
        Base channel width:   64
        Channel multipliers:  [1, 2, 4, 4]
        Input channels:       5  (patch + downsample + 3 positional)
        Output channels:      1  (predicted noise on patch)
        Attention resolution:  [8]
        Num residual blocks:   2
        Learning rate:         2e-5  (handled outside, not here)
    """
    # Save originals
    orig_upsample = unet_module.Upsample
    orig_downsample = unet_module.Downsample

    # Monkey-patch with isotropic versions
    unet_module.Upsample = IsotropicUpsample3D
    unet_module.Downsample = IsotropicDownsample3D

    try:
        model = UNetModel(
            in_channels=in_channels,
            model_channels=model_channels,
            out_channels=out_channels,
            num_res_blocks=num_res_blocks,
            attention_resolutions=attention_resolutions,
            dropout=dropout,
            channel_mult=channel_mult,
            conv_resample=True,
            dims=dims,
            use_fp16=use_fp16,
            num_heads=num_heads,
            num_head_channels=num_head_channels,
            use_scale_shift_norm=use_scale_shift_norm,
            resblock_updown=resblock_updown,
            use_new_attention_order=use_new_attention_order,
        )
    finally:
        # Restore originals so any other code using this module is unaffected
        unet_module.Upsample = orig_upsample
        unet_module.Downsample = orig_downsample

    return model