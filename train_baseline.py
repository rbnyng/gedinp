"""
Training script for baseline models (MLP) using foundation model embeddings + lat/lon.

This baseline tests whether the CNP's context aggregation mechanism adds value
beyond simple point-to-point mapping.
"""

import argparse
import json
from pathlib import Path
import pickle

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import numpy as np
import pandas as pd
from tqdm import tqdm

from data.gedi import GEDIQuerier
from data.embeddings import EmbeddingExtractor
from data.spatial_cv import SpatialTileSplitter
from models.baseline import SimpleMLPBaseline, FlatMLPBaseline, baseline_loss
from models.neural_process import compute_metrics


class GEDIBaselineDataset(Dataset):
    """
    Simple dataset for baseline models - direct point-to-point mapping.

    Unlike the Neural Process dataset, this doesn't do context/target splitting.
    Each sample is just a single GEDI shot with its embedding and coordinates.
    """

    def __init__(
        self,
        data_df: pd.DataFrame,
        normalize_coords: bool = True,
        normalize_agbd: bool = True,
        agbd_scale: float = 200.0,
        log_transform_agbd: bool = True,
        augment_coords: bool = False,
        coord_noise_std: float = 0.01
    ):
        """
        Initialize baseline dataset.

        Args:
            data_df: DataFrame with columns: latitude, longitude, agbd, embedding_patch, tile_id
            normalize_coords: Normalize coordinates to [0, 1] within each tile
            normalize_agbd: Normalize AGBD values
            agbd_scale: Scale factor for AGBD normalization
            log_transform_agbd: Apply log(1+x) transform to AGBD
            augment_coords: Add small random noise to coordinates
            coord_noise_std: Standard deviation of coordinate noise
        """
        # Filter out shots without embeddings
        self.data_df = data_df[data_df['embedding_patch'].notna()].copy().reset_index(drop=True)

        self.normalize_coords = normalize_coords
        self.normalize_agbd = normalize_agbd
        self.agbd_scale = agbd_scale
        self.log_transform_agbd = log_transform_agbd
        self.augment_coords = augment_coords
        self.coord_noise_std = coord_noise_std

        # Precompute tile bounds for normalization
        if normalize_coords:
            self.tile_bounds = {}
            for tile_id, group in self.data_df.groupby('tile_id'):
                self.tile_bounds[tile_id] = {
                    'lon_min': group['longitude'].min(),
                    'lon_max': group['longitude'].max(),
                    'lat_min': group['latitude'].min(),
                    'lat_max': group['latitude'].max()
                }

        print(f"Baseline dataset initialized with {len(self.data_df)} shots")

    def __len__(self) -> int:
        return len(self.data_df)

    def __getitem__(self, idx: int):
        """Get a single training sample."""
        row = self.data_df.iloc[idx]

        # Get coordinates
        coords = np.array([row['longitude'], row['latitude']], dtype=np.float32)

        # Normalize coordinates within tile
        if self.normalize_coords:
            tile_id = row['tile_id']
            bounds = self.tile_bounds[tile_id]

            lon_range = bounds['lon_max'] - bounds['lon_min']
            lat_range = bounds['lat_max'] - bounds['lat_min']

            # Avoid division by zero
            lon_range = lon_range if lon_range > 0 else 1.0
            lat_range = lat_range if lat_range > 0 else 1.0

            coords[0] = (coords[0] - bounds['lon_min']) / lon_range
            coords[1] = (coords[1] - bounds['lat_min']) / lat_range

        # Apply coordinate augmentation
        if self.augment_coords:
            coords = coords + np.random.normal(0, self.coord_noise_std, coords.shape)
            coords = np.clip(coords, 0, 1)

        # Get embedding
        embedding = row['embedding_patch'].astype(np.float32)

        # Get AGBD
        agbd = np.array([row['agbd']], dtype=np.float32)

        # Normalize AGBD
        if self.normalize_agbd:
            if self.log_transform_agbd:
                agbd = np.log1p(agbd) / np.log1p(self.agbd_scale)
            else:
                agbd = agbd / self.agbd_scale

        return {
            'coords': torch.from_numpy(coords),
            'embedding': torch.from_numpy(embedding),
            'agbd': torch.from_numpy(agbd)
        }


def parse_args():
    parser = argparse.ArgumentParser(description='Train baseline model')

    # Data arguments
    parser.add_argument('--region_bbox', type=float, nargs=4, required=True,
                        help='Region bounding box: min_lon min_lat max_lon max_lat')
    parser.add_argument('--start_time', type=str, default='2019-01-01',
                        help='Start date for GEDI data (YYYY-MM-DD)')
    parser.add_argument('--end_time', type=str, default='2023-12-31',
                        help='End date for GEDI data (YYYY-MM-DD)')
    parser.add_argument('--embedding_year', type=int, default=2024,
                        help='Year of GeoTessera embeddings')
    parser.add_argument('--cache_dir', type=str, default='./cache',
                        help='Directory for caching tiles and embeddings')

    # Model arguments
    parser.add_argument('--model_type', type=str, default='simple_mlp',
                        choices=['simple_mlp', 'flat_mlp'],
                        help='Type of baseline model')
    parser.add_argument('--patch_size', type=int, default=3,
                        help='Embedding patch size (default: 3x3)')
    parser.add_argument('--hidden_dim', type=int, default=512,
                        help='Hidden layer dimension')
    parser.add_argument('--embedding_feature_dim', type=int, default=128,
                        help='Embedding feature dimension (for simple_mlp)')
    parser.add_argument('--num_hidden_layers', type=int, default=3,
                        help='Number of hidden layers')
    parser.add_argument('--output_uncertainty', action='store_true', default=True,
                        help='Output uncertainty estimates')

    # Training arguments
    parser.add_argument('--batch_size', type=int, default=256,
                        help='Batch size (number of individual shots)')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Learning rate')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of epochs')
    parser.add_argument('--val_ratio', type=float, default=0.15,
                        help='Validation set ratio')
    parser.add_argument('--test_ratio', type=float, default=0.15,
                        help='Test set ratio')
    parser.add_argument('--early_stopping_patience', type=int, default=15,
                        help='Early stopping patience (epochs)')
    parser.add_argument('--lr_scheduler_patience', type=int, default=5,
                        help='LR scheduler patience (epochs)')
    parser.add_argument('--lr_scheduler_factor', type=float, default=0.5,
                        help='LR scheduler reduction factor')

    # Dataset arguments
    parser.add_argument('--log_transform_agbd', action='store_true', default=True,
                        help='Apply log transform to AGBD')
    parser.add_argument('--augment_coords', action='store_true', default=True,
                        help='Add coordinate augmentation')
    parser.add_argument('--coord_noise_std', type=float, default=0.01,
                        help='Standard deviation for coordinate noise')

    # Output arguments
    parser.add_argument('--output_dir', type=str, default='./outputs_baseline',
                        help='Output directory for models and logs')
    parser.add_argument('--save_every', type=int, default=10,
                        help='Save checkpoint every N epochs')

    # Other
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Device to use')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of data loading workers')

    return parser.parse_args()


def set_seed(seed):
    """Set random seeds for reproducibility."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


def train_epoch(model, dataloader, optimizer, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    n_samples = 0

    for batch in tqdm(dataloader, desc='Training'):
        coords = batch['coords'].to(device)
        embeddings = batch['embedding'].to(device)
        agbd = batch['agbd'].to(device)

        optimizer.zero_grad()

        # Forward pass
        pred_mean, pred_log_var = model(coords, embeddings)

        # Compute loss
        loss = baseline_loss(pred_mean, pred_log_var, agbd)

        # Check for NaN/Inf
        if torch.isnan(loss) or torch.isinf(loss):
            print(f"Warning: NaN/Inf loss detected! Skipping batch.")
            continue

        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        total_loss += loss.item() * len(coords)
        n_samples += len(coords)

    return total_loss / max(n_samples, 1)


def validate(model, dataloader, device):
    """Validate the model."""
    model.eval()
    total_loss = 0
    all_preds = []
    all_targets = []
    all_uncertainties = []
    n_samples = 0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc='Validation'):
            coords = batch['coords'].to(device)
            embeddings = batch['embedding'].to(device)
            agbd = batch['agbd'].to(device)

            # Forward pass
            pred_mean, pred_log_var = model(coords, embeddings)

            # Compute loss
            loss = baseline_loss(pred_mean, pred_log_var, agbd)

            # Check for NaN/Inf
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"Warning: NaN/Inf loss detected in validation!")
                continue

            total_loss += loss.item() * len(coords)
            n_samples += len(coords)

            # Collect predictions for metrics
            all_preds.append(pred_mean)
            all_targets.append(agbd)

            if pred_log_var is not None:
                pred_std = torch.exp(0.5 * pred_log_var)
                all_uncertainties.append(pred_std)

    avg_loss = total_loss / max(n_samples, 1)

    # Compute metrics
    all_preds = torch.cat(all_preds, dim=0)
    all_targets = torch.cat(all_targets, dim=0)

    if len(all_uncertainties) > 0:
        all_uncertainties = torch.cat(all_uncertainties, dim=0)
    else:
        all_uncertainties = None

    metrics = compute_metrics(all_preds, all_uncertainties, all_targets)

    return avg_loss, metrics


def main():
    args = parse_args()
    set_seed(args.seed)

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save config
    with open(output_dir / 'config.json', 'w') as f:
        json.dump(vars(args), f, indent=2)

    print("=" * 80)
    print(f"Baseline Model Training ({args.model_type})")
    print("=" * 80)
    print(f"Device: {args.device}")
    print(f"Region: {args.region_bbox}")
    print(f"Output: {output_dir}")
    print()

    # Step 1: Query GEDI data
    print("Step 1: Querying GEDI data...")
    querier = GEDIQuerier()
    gedi_df = querier.query_region_tiles(
        region_bbox=args.region_bbox,
        tile_size=0.1,
        start_time=args.start_time,
        end_time=args.end_time
    )
    print(f"Retrieved {len(gedi_df)} GEDI shots across {gedi_df['tile_id'].nunique()} tiles")
    print()

    if len(gedi_df) == 0:
        print("No GEDI data found in region. Exiting.")
        return

    # Step 2: Extract embeddings
    print("Step 2: Extracting GeoTessera embeddings...")
    extractor = EmbeddingExtractor(
        year=args.embedding_year,
        patch_size=args.patch_size,
        cache_dir=args.cache_dir
    )
    gedi_df = extractor.extract_patches_batch(gedi_df, verbose=True)
    print()

    # Filter out shots without embeddings
    gedi_df = gedi_df[gedi_df['embedding_patch'].notna()]
    print(f"Retained {len(gedi_df)} shots with valid embeddings")
    print()

    # Save processed data
    with open(output_dir / 'processed_data.pkl', 'wb') as f:
        pickle.dump(gedi_df, f)

    # Step 3: Spatial split (use same splitter as CNP for fair comparison)
    print("Step 3: Creating spatial train/val/test split...")
    splitter = SpatialTileSplitter(
        gedi_df,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        random_state=args.seed
    )
    train_df, val_df, test_df = splitter.split()
    print()

    # Save splits
    train_df.to_csv(output_dir / 'train_split.csv', index=False)
    val_df.to_csv(output_dir / 'val_split.csv', index=False)
    test_df.to_csv(output_dir / 'test_split.csv', index=False)

    # Step 4: Create datasets
    print("Step 4: Creating datasets...")
    train_dataset = GEDIBaselineDataset(
        train_df,
        log_transform_agbd=args.log_transform_agbd,
        augment_coords=args.augment_coords,
        coord_noise_std=args.coord_noise_std
    )
    val_dataset = GEDIBaselineDataset(
        val_df,
        log_transform_agbd=args.log_transform_agbd,
        augment_coords=False,  # No augmentation for validation
        coord_noise_std=0.0
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers
    )
    print()

    # Step 5: Initialize model
    print("Step 5: Initializing model...")
    if args.model_type == 'simple_mlp':
        model = SimpleMLPBaseline(
            patch_size=args.patch_size,
            embedding_channels=128,
            embedding_feature_dim=args.embedding_feature_dim,
            hidden_dim=args.hidden_dim,
            output_uncertainty=args.output_uncertainty,
            num_hidden_layers=args.num_hidden_layers
        ).to(args.device)
    elif args.model_type == 'flat_mlp':
        model = FlatMLPBaseline(
            patch_size=args.patch_size,
            embedding_channels=128,
            hidden_dim=args.hidden_dim,
            output_uncertainty=args.output_uncertainty,
            num_hidden_layers=args.num_hidden_layers
        ).to(args.device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {n_params:,}")
    print()

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=args.lr_scheduler_factor,
        patience=args.lr_scheduler_patience
    )

    # Step 6: Training loop
    print("Step 6: Training...")
    best_val_loss = float('inf')
    best_r2 = float('-inf')
    epochs_without_improvement = 0
    train_losses = []
    val_losses = []

    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")
        print("-" * 80)

        # Train
        train_loss = train_epoch(model, train_loader, optimizer, args.device)
        train_losses.append(train_loss)

        # Validate
        val_loss, val_metrics = validate(model, val_loader, args.device)
        val_losses.append(val_loss)

        # Print metrics
        print(f"Train Loss: {train_loss:.6e}")
        print(f"Val Loss:   {val_loss:.6e}")
        if val_metrics:
            print(f"Val RMSE:   {val_metrics.get('rmse', 0):.4f}")
            print(f"Val MAE:    {val_metrics.get('mae', 0):.4f}")
            print(f"Val R²:     {val_metrics.get('r2', 0):.4f}")

        # Get current learning rate
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Learning Rate: {current_lr:.6e}")

        # Step the learning rate scheduler
        scheduler.step(val_loss)

        # Save best model based on validation loss
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_without_improvement = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'val_metrics': val_metrics
            }, output_dir / 'best_model.pt')
            print("✓ Saved best model (lowest val loss)")
        else:
            epochs_without_improvement += 1

        # Save best model based on R²
        current_r2 = val_metrics.get('r2', float('-inf')) if val_metrics else float('-inf')
        if current_r2 > best_r2:
            best_r2 = current_r2
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'val_metrics': val_metrics,
                'r2': current_r2
            }, output_dir / 'best_r2_model.pt')
            print(f"✓ Saved best R² model (R² = {best_r2:.4f})")

        # Early stopping check
        if epochs_without_improvement >= args.early_stopping_patience:
            print(f"\nEarly stopping triggered after {epoch} epochs")
            print(f"No improvement in validation loss for {args.early_stopping_patience} epochs")
            break

        # Save checkpoint
        if epoch % args.save_every == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, output_dir / f'checkpoint_epoch_{epoch}.pt')

    # Save training history
    history = {
        'train_losses': train_losses,
        'val_losses': val_losses
    }
    with open(output_dir / 'history.json', 'w') as f:
        json.dump(history, f, indent=2)

    print("\n" + "=" * 80)
    print("Training complete!")
    print(f"Best validation loss: {best_val_loss:.6e}")
    print(f"Best R² score: {best_r2:.4f}")
    print(f"Models saved to: {output_dir}")
    print("=" * 80)


if __name__ == '__main__':
    main()
