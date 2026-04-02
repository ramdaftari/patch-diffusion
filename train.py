"""
Train 3D patch-based diffusion model (Algorithm 2, arXiv 2512.18161).

Follows the structure of the original trainer in
src/diffmodels/trainer/trainer.py as closely as possible.

Usage:
    cd ~/patch-diffusion
    conda activate res_rob
    python train.py

    # Override defaults:
    python train.py --epochs 500 --batch_size 2 --lr 2e-5
"""

import os
import argparse
import torch
from torch.optim import Adam
from tqdm import tqdm

import wandb

from patch_unet import create_patch_unet
from patch_loss import PatchDiffusionLoss
from mri_dataset import PreprocessedVolumeDataset
from ema import ExponentialMovingAverage


def parse_args():
    parser = argparse.ArgumentParser(description='Train 3D patch diffusion model')

    # Data
    parser.add_argument('--train_dir', type=str,
                        default=os.path.expanduser('~/patch-diffusion/data/train'))
    parser.add_argument('--val_dir', type=str,
                        default=os.path.expanduser('~/patch-diffusion/data/val'))

    # Model (paper Table 6 defaults)
    parser.add_argument('--patch_size', type=int, default=32)
    parser.add_argument('--model_channels', type=int, default=64)
    parser.add_argument('--channel_mult', type=int, nargs='+', default=[1, 2, 4, 4])
    parser.add_argument('--num_res_blocks', type=int, default=2)
    parser.add_argument('--attention_resolutions', type=int, nargs='+', default=[8])

    # Training (paper defaults)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--batch_size', type=int, default=4,
                        help='Number of volumes per step. Paper used 64 on A100 (80GB). '
                             'With 24GB 4090, start with 4.')
    parser.add_argument('--lr', type=float, default=2e-5)
    parser.add_argument('--steps_per_epoch', type=int, default=None,
                        help='Training steps per epoch. Default: num_volumes // batch_size * patches_per_volume_estimate')

    # EMA (matching original trainer)
    parser.add_argument('--use_ema', action='store_true', default=True)
    parser.add_argument('--ema_decay', type=float, default=0.999)
    parser.add_argument('--ema_warm_start_steps', type=int, default=500)

    # Diffusion schedule (matching sde.py)
    parser.add_argument('--beta_min', type=float, default=0.0001)
    parser.add_argument('--beta_max', type=float, default=0.02)
    parser.add_argument('--num_steps', type=int, default=1000)

    # Logging & saving (matching original trainer)
    parser.add_argument('--save_every', type=int, default=10,
                        help='Save checkpoint every N epochs')
    parser.add_argument('--wandb_project', type=str, default='patch_diffusion')
    parser.add_argument('--wandb_entity', type=str, default=None)
    parser.add_argument('--run_name', type=str, default=None)
    parser.add_argument('--save_dir', type=str,
                        default=os.path.expanduser('~/patch-diffusion/checkpoints'))

    return parser.parse_args()


def load_all_volumes(data_dir, device='cpu'):
    """
    Load all preprocessed volumes into memory.
    With 12 volumes of (1, 256, 320, 320) float32 ~ 1.2GB total,
    this fits comfortably in CPU RAM.
    """
    dataset = PreprocessedVolumeDataset(data_dir)
    volumes = []
    for i in range(len(dataset)):
        vol = dataset[i]  # (1, Z, Y, X)
        volumes.append(vol)
    return volumes


def sample_batch(volumes, batch_size, device):
    """
    Randomly sample batch_size volumes (with replacement).
    Returns (batch_size, 1, Z, Y, X) on device.
    """
    indices = torch.randint(0, len(volumes), (batch_size,))
    batch = torch.stack([volumes[i] for i in indices], dim=0)  # (B, 1, Z, Y, X)
    return batch.to(device)


def save_checkpoint(model, ema, epoch, args):
    """Save model and EMA checkpoints, matching original save_model()."""
    os.makedirs(args.save_dir, exist_ok=True)

    is_final = (epoch == args.epochs - 1)

    model_filename = 'model.pt' if is_final else f'model_{epoch}.pt'
    torch.save(model.state_dict(), os.path.join(args.save_dir, model_filename))

    if ema is not None:
        ema_filename = 'ema_model.pt' if is_final else f'ema_model_{epoch}.pt'
        torch.save(ema.state_dict(), os.path.join(args.save_dir, ema_filename))


def eval_on_validation_set(model, loss_fn, val_volumes, device, num_eval_steps=20):
    """
    Evaluate validation loss, matching the original trainer's structure.
    Since we have few val volumes, we run multiple steps with random patches.
    """
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for _ in range(num_eval_steps):
            idx = torch.randint(0, len(val_volumes), (1,)).item()
            x = val_volumes[idx].unsqueeze(0).to(device)  # (1, 1, Z, Y, X)
            loss = loss_fn(x, model)
            val_loss += loss.item()
    return val_loss / num_eval_steps


def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # --- Load data ---
    print("\nLoading training volumes...")
    train_volumes = load_all_volumes(args.train_dir)
    volume_shape = tuple(train_volumes[0].shape[1:])  # (Nz, Ny, Nx)
    print(f"  {len(train_volumes)} volumes, shape: {volume_shape}")

    print("Loading validation volumes...")
    val_volumes = load_all_volumes(args.val_dir)
    print(f"  {len(val_volumes)} volumes")

    # Verify divisibility by patch size
    P = args.patch_size
    Nz, Ny, Nx = volume_shape
    assert Nz % P == 0 and Ny % P == 0 and Nx % P == 0, \
        f"Volume dims {volume_shape} not divisible by patch_size={P}"

    # --- Create model ---
    print(f"\nCreating model (patch_size={P})...")
    model = create_patch_unet(
        model_channels=args.model_channels,
        channel_mult=tuple(args.channel_mult),
        num_res_blocks=args.num_res_blocks,
        attention_resolutions=tuple(args.attention_resolutions),
    ).to(device)
    num_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"  Parameters: {num_params:.2f}M")

    # --- Optimizer (matching original trainer) ---
    optimizer = Adam(model.parameters(), lr=args.lr)

    # --- EMA (matching original trainer) ---
    ema = None
    if args.use_ema:
        ema = ExponentialMovingAverage(
            model.parameters(),
            decay=args.ema_decay
        )

    # --- Loss ---
    loss_fn = PatchDiffusionLoss(
        patch_size=P,
        volume_shape=volume_shape,
        beta_min=args.beta_min,
        beta_max=args.beta_max,
        num_steps=args.num_steps,
    ).to(device)

    # --- Steps per epoch ---
    # Paper trains for 8 days with batch 64 on 90 volumes.
    # With 12 volumes, we use more steps per epoch to compensate.
    if args.steps_per_epoch is None:
        # Rough estimate: enough steps to see ~1000 patches per volume per epoch
        patches_per_vol = (Nz // P) * (Ny // P) * (Nx // P)
        args.steps_per_epoch = max(100, (len(train_volumes) * patches_per_vol) // args.batch_size)
    print(f"  Steps per epoch: {args.steps_per_epoch}")

    # --- W&B ---
    wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        name=args.run_name,
        config=vars(args),
    )

    # --- Training loop (matching original trainer structure) ---
    print(f"\nStarting training: {args.epochs} epochs, batch_size={args.batch_size}")
    print(f"=" * 60)

    grad_step = 0
    for epoch in range(args.epochs):
        avg_loss, num_items = 0, 0

        model.train()
        with tqdm(range(args.steps_per_epoch), desc=f'Epoch {epoch+1}/{args.epochs}') as pbar:
            for step in pbar:

                # Sample random batch of volumes
                x = sample_batch(train_volumes, args.batch_size, device)

                # Compute patch-based loss (Algorithm 2)
                loss = loss_fn(x, model)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                pbar.set_description(
                    f'Epoch {epoch+1}/{args.epochs} loss={loss.item():.1f}',
                    refresh=False
                )

                grad_step += 1
                avg_loss += loss.item() * args.batch_size
                num_items += args.batch_size

        # --- EMA update (matching original trainer) ---
        if args.use_ema and (
            grad_step > args.ema_warm_start_steps or epoch > 0):
            ema.update(model.parameters())

        # --- Save checkpoint (matching original save_model) ---
        should_save = (
            epoch % args.save_every == 0 or epoch == args.epochs - 1)
        if should_save:
            save_checkpoint(model, ema, epoch, args)

        # --- Validation loss (matching original trainer) ---
        val_loss = eval_on_validation_set(model, loss_fn, val_volumes, device)

        # --- W&B logging (matching original trainer) ---
        wandb.log({
            'loss': avg_loss / num_items,
            'val_loss': val_loss,
            'epoch': epoch + 1,
            'step': epoch + 1,
        })

        print(f"  Epoch {epoch+1}: train_loss={avg_loss/num_items:.2f}, val_loss={val_loss:.2f}")

    # --- Final save (matching original trainer) ---
    torch.save(model.state_dict(), os.path.join(args.save_dir, 'last_model.pt'))

    wandb.finish()
    print("\nTraining complete.")


if __name__ == '__main__':
    args = parse_args()
    train(args)