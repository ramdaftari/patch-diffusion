"""
Core operations:
  - zero_pad_volume: pad volume by P on each side
  - extract_patch: grab a P x P x P patch given offset and patch index
  - downsample_to_patch_size: shrink full volume to P x P x P
  - build_positional_arrays: create normalized x, y, z coordinate grids
  - extract_positional_patch: grab positional encoding for a specific patch
  - build_network_input: assemble the 5-channel input for the UNet
  
"""

import torch
import torch.nn.functional as F
from typing import Tuple


def zero_pad_volume(volume: torch.Tensor, patch_size: int) -> torch.Tensor:
    """
    Zero-pad a volume by patch_size on each side along all 3 spatial dims.

    Args:
        volume: (B, C, Nz, Ny, Nx) or (C, Nz, Ny, Nx) or (Nz, Ny, Nx)
        patch_size: P, the patch size

    Returns:
        Padded volume with shape (..., Nz+2P, Ny+2P, Nx+2P)
    """
    P = patch_size
    # F.pad expects (last_dim_left, last_dim_right, ..., first_dim_left, first_dim_right)
    # For 3 spatial dims: (x_left, x_right, y_left, y_right, z_left, z_right)
    padding = (P, P, P, P, P, P)
    return F.pad(volume, padding, mode='constant', value=0)


def compute_num_patches_per_dim(volume_size: int, patch_size: int) -> int:
    """
    Number of non-overlapping patches along one dimension after padding.

    After padding, the dimension is volume_size + 2*P. With offset in {0,...,P-1},
    the first patch starts at offset, and we tile with stride P.
    Number of patches = (volume_size + 2*P - offset) // P, but since offset < P,
    this is (volume_size // P) + 1 for any valid offset.

    The paper says k = N/P patches to tile the original, plus 1 border patch,
    giving k+1 = N/P + 1 patches per dimension.
    """
    return (volume_size // patch_size) + 1


def extract_patch(
    volume: torch.Tensor,
    patch_size: int,
    offset: Tuple[int, int, int],
    patch_idx: Tuple[int, int, int]
) -> torch.Tensor:
    """
    Extract a single P x P x P patch from a (padded) volume.

    The padded volume has already been zero-padded by P on each side.
    Given offset (o1, o2, o3) and patch grid indices (iz, iy, ix),
    the patch starts at:
        z_start = o1 + iz * P
        y_start = o2 + iy * P
        x_start = o3 + ix * P

    Args:
        volume: (..., Nz+2P, Ny+2P, Nx+2P) padded volume
        patch_size: P
        offset: (o1, o2, o3), each in {0, ..., P-1}
        patch_idx: (iz, iy, ix), grid indices for the patch

    Returns:
        (..., P, P, P) patch tensor
    """
    P = patch_size
    oz, oy, ox = offset
    iz, iy, ix = patch_idx

    z_start = oz + iz * P
    y_start = oy + iy * P
    x_start = ox + ix * P

    return volume[..., z_start:z_start+P, y_start:y_start+P, x_start:x_start+P]


def downsample_to_patch_size(
    volume: torch.Tensor,
    patch_size: int
) -> torch.Tensor:
    """
    Downsample a volume to (P, P, P) using trilinear interpolation.

    This produces the global context channel D(x) from the paper.
    The input should be the UN-padded volume (we want global context
    of the actual image, not the zero-padded version).

    Args:
        volume: (B, C, Nz, Ny, Nx) volume
        patch_size: P, target spatial size

    Returns:
        (B, C, P, P, P) downsampled volume
    """
    target_size = (patch_size, patch_size, patch_size)
    return F.interpolate(volume, size=target_size, mode='trilinear', align_corners=False)


def build_positional_arrays(
    volume_shape: Tuple[int, int, int],
    patch_size: int,
    device: torch.device = None
) -> torch.Tensor:
    """
    Build 3D positional encoding arrays for the PADDED volume.

    Each array contains normalized coordinates in [-1, 1] for the
    original (un-padded) volume region. The padded border region
    extends slightly beyond [-1, 1].

    Args:
        volume_shape: (Nz, Ny, Nx) original un-padded volume shape
        patch_size: P
        device: torch device

    Returns:
        (3, Nz+2P, Ny+2P, Nx+2P) tensor of z, y, x coordinates
    """
    P = patch_size
    Nz, Ny, Nx = volume_shape

    # Coordinates for the padded volume
    # Original volume spans indices [P, P+Nz) in the padded volume
    # We want the original volume center to map to 0, edges to +/-1
    # So padded index i maps to: (i - P) / (N/2) - 1 ... no.
    # More precisely: original voxel j (0-indexed) maps to -1 + 2*j/(N-1)
    # Padded voxel i corresponds to original voxel j = i - P
    # So padded voxel i maps to -1 + 2*(i - P)/(N - 1)

    def make_coords(N_orig, N_padded):
        # i ranges over [0, N_padded)
        i = torch.arange(N_padded, dtype=torch.float32, device=device)
        # Map to original voxel index
        j = i - P
        # Normalize to [-1, 1] based on original volume extent
        if N_orig > 1:
            coords = -1.0 + 2.0 * j / (N_orig - 1)
        else:
            coords = torch.zeros_like(i)
        return coords

    z_coords = make_coords(Nz, Nz + 2 * P)  # (Nz+2P,)
    y_coords = make_coords(Ny, Ny + 2 * P)  # (Ny+2P,)
    x_coords = make_coords(Nx, Nx + 2 * P)  # (Nx+2P,)

    # Broadcast to 3D grids: each is (Nz+2P, Ny+2P, Nx+2P)
    pos_z = z_coords[:, None, None].expand(Nz + 2*P, Ny + 2*P, Nx + 2*P)
    pos_y = y_coords[None, :, None].expand(Nz + 2*P, Ny + 2*P, Nx + 2*P)
    pos_x = x_coords[None, None, :].expand(Nz + 2*P, Ny + 2*P, Nx + 2*P)

    # Stack as (3, Nz+2P, Ny+2P, Nx+2P)
    return torch.stack([pos_z, pos_y, pos_x], dim=0)


def sample_random_patch_location(
    volume_shape: Tuple[int, int, int],
    patch_size: int
) -> Tuple[Tuple[int, int, int], Tuple[int, int, int]]:
    """
    Randomly sample a patch offset and patch grid index.

    Args:
        volume_shape: (Nz, Ny, Nx) original un-padded volume shape
        patch_size: P

    Returns:
        offset: (o1, o2, o3), each in {0, ..., P-1}
        patch_idx: (iz, iy, ix), grid indices
    """
    P = patch_size
    Nz, Ny, Nx = volume_shape

    # Random offset
    o1 = torch.randint(0, P, (1,)).item()
    o2 = torch.randint(0, P, (1,)).item()
    o3 = torch.randint(0, P, (1,)).item()

    # Number of patches per dim with this offset
    nz = compute_num_patches_per_dim(Nz, P)
    ny = compute_num_patches_per_dim(Ny, P)
    nx = compute_num_patches_per_dim(Nx, P)

    # Random patch index
    iz = torch.randint(0, nz, (1,)).item()
    iy = torch.randint(0, ny, (1,)).item()
    ix = torch.randint(0, nx, (1,)).item()

    return (o1, o2, o3), (iz, iy, ix)


def build_network_input(
    noisy_patch: torch.Tensor,
    noisy_volume_unpadded: torch.Tensor,
    positional_arrays: torch.Tensor,
    patch_size: int,
    offset: Tuple[int, int, int],
    patch_idx: Tuple[int, int, int]
) -> torch.Tensor:
    """
    Assemble the 5-channel UNet input: [noisy_patch, downsampled_volume, pos_z, pos_y, pos_x].

    Args:
        noisy_patch: (B, 1, P, P, P) extracted noisy patch
        noisy_volume_unpadded: (B, 1, Nz, Ny, Nx) full noisy volume (un-padded)
        positional_arrays: (3, Nz+2P, Ny+2P, Nx+2P) coordinate grids
        patch_size: P
        offset: (o1, o2, o3)
        patch_idx: (iz, iy, ix)

    Returns:
        (B, 5, P, P, P) concatenated input
    """
    B = noisy_patch.shape[0]
    P = patch_size

    # Channel 2: downsample the full noisy volume to PxPxP
    downsampled = downsample_to_patch_size(noisy_volume_unpadded, P)  # (B, 1, P, P, P)

    # Channels 3-5: extract positional patch (same location as the image patch)
    # positional_arrays is (3, Nz+2P, Ny+2P, Nx+2P), extract_patch works on last 3 dims
    pos_patch = extract_patch(positional_arrays, P, offset, patch_idx)  # (3, P, P, P)
    pos_patch = pos_patch.unsqueeze(0).expand(B, -1, -1, -1, -1)  # (B, 3, P, P, P)

    # Concatenate: [patch, downsampled, pos_z, pos_y, pos_x]
    return torch.cat([noisy_patch, downsampled, pos_patch], dim=1)  # (B, 5, P, P, P)