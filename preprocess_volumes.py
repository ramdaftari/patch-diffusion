"""
Preprocess Stanford knee MRI volumes using the ORIGINAL codebase pipeline.

Uses the exact same FastMRIVolumeDataset and FastMRI3DDataTransform
that the codebase uses for reconstruction, to compute 3D MVUE volumes.
Saves them as .pt files for patch-based diffusion training.

"""

import os
import sys
import torch
import argparse
from pathlib import Path

# --- Original codebase imports (runs from codebase dir) ---
from src.datasets.fastmri_volume_dataset import FastMRIVolumeDataset
from src.problem_trafos.dataset_trafo.fastmri_3d_trafo import FastMRI3DDataTransform

# --- Config (matching stanford_3d.yaml and mri3d_recon_calc_mvue.yaml) ---
DATA_ROOT_TRAIN = '/home/ram/trial-project/stanford_fastmri_cor/sag'
DATA_ROOT_VAL = '/home/ram/trial-project/stanford_fastmri_cor/sag_val'
SENSMAP_ROOT = '/home/ram/trial-project/stanford_fastmri_cor/sensmaps'
OUTPUT_DIR = os.path.expanduser('~/patch_diffusion/data')


def create_dataset(data_root):
    """
    Create dataset + transform using the ORIGINAL codebase classes,
    with settings that produce clean 3D MVUE volumes.
    """
    # 3D transform: compute MVUE from fully-sampled kspace
    # mask_enabled=False because we want the clean prior, not undersampled
    transform = FastMRI3DDataTransform(
        which_challenge='multicoil',
        mask_enabled=False,
        mask_type='Poisson2D',
        mask_accelerations=4.0,
        mask_center_fractions=[0.08],
        mask_seed=1234,
        use_seed=True,
        provide_pseudoinverse=False,
        provide_measurement=False,
        target_type='fullysampled_rec',
        multicoil_reduction_op='norm_sum_sensmaps',
        scale_target_by_kspacenorm=False,
        target_scaling_factor=1.0,
        target_interpolate_by_factor=1.0,
        normalize_target=False,
        wrapped_2d=False,
        return_pseudoinverse_as_observation=False,
    )

    # Volume dataset: loads kspace, does readout correction, attaches sensmaps
    # Settings match stanford_3d.yaml exactly
    dataset = FastMRIVolumeDataset(
        root=data_root,
        challenge='multicoil',
        transform=transform,
        dataset_is_3d=True,
        recons_key="reconstruction_rss",
        return_sensmaps=True, 
        sensmaps_key_in_h5=None,  
        sensmap_files_root="/home/ram/trial-project/stanford_fastmri_cor/sensmaps" ,
        readout_dim_is_spatial=True,
        apply_fft1c_on_readout_dim=True,
        apply_fft1c_on_readout_dim_shifts=[True, False],
        readout_dim_keep_spatial=False,
    )

    return dataset


def process_and_save(dataset, output_dir, split_name):
    """Load each volume, take magnitude, normalize, save as .pt"""
    split_dir = os.path.join(output_dir, split_name)
    os.makedirs(split_dir, exist_ok=True)

    for i in range(len(dataset)):
        print(f"  Processing {split_name} volume {i}/{len(dataset)-1}...")

        # target comes from the transform: (Z, Y, X, 2) real-valued complex MVUE
        target = dataset[i]

        # Convert to complex and take magnitude
        # (Z, Y, X, 2) -> (Z, Y, X) complex -> (Z, Y, X) float magnitude
        target_complex = torch.view_as_complex(target)
        magnitude = torch.abs(target_complex)  # (Z, Y, X)

        # Add channel dim: (1, Z, Y, X)
        volume = magnitude.unsqueeze(0).float()

        # Per-volume normalization (zero mean, unit std)
        mean = volume.mean()
        std = volume.std()
        volume = (volume - mean) / (std + 1e-11)

        # Save
        save_path = os.path.join(split_dir, f'vol_{i:02d}.pt')
        torch.save({
            'volume': volume,
            'mean': mean,
            'std': std,
            'source_file': str(dataset.raw_samples[i].fname),
        }, save_path)

        print(f"    Shape: {volume.shape}, "
              f"mean: {volume.mean():.4f}, std: {volume.std():.4f}, "
              f"saved to {save_path}")


if __name__ == '__main__':
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("=" * 60)
    print("Preprocessing training volumes")
    print("=" * 60)
    train_dataset = create_dataset(DATA_ROOT_TRAIN)
    print(f"Found {len(train_dataset)} training volumes\n")
    process_and_save(train_dataset, OUTPUT_DIR, 'train')

    print()
    print("=" * 60)
    print("Preprocessing validation volumes")
    print("=" * 60)
    val_dataset = create_dataset(DATA_ROOT_VAL)
    print(f"Found {len(val_dataset)} validation volumes\n")
    process_and_save(val_dataset, OUTPUT_DIR, 'val')

    print()
    print("=" * 60)
    print("DONE")
    print(f"Saved to: {OUTPUT_DIR}")
    print("=" * 60)