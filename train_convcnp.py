"""
Training script for GEDI ConvCNP model.

Uses dense tile representations with UNet architecture.
"""

import argparse
import json
from pathlib import Path
import pickle

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
from tqdm import tqdm

from data.gedi import GEDIQuerier
from data.embeddings import EmbeddingExtractor
from data.convcnp_dataset import ConvCNPTileDataset, collate_convcnp
from data.spatial_cv import SpatialTileSplitter
from models.convcnp import (
    GEDIConvCNP,
    convcnp_loss,
    compute_metrics
)


def parse_args():
    parser = argparse.ArgumentParser(description='Train GEDI ConvCNP')

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
    parser.add_argument('--feature_dim', type=int, default=128,
                        help='Feature dimension for UNet output')
    parser.add_argument('--base_channels', type=int, default=64,
                        help='Base channels for UNet (default: 64)')
    parser.add_argument('--unet_depth', type=int, default=3,
                        help='UNet depth (default: 3)')
    parser.add_argument('--decoder_hidden_dim', type=int, default=128,
                        help='Decoder MLP hidden dimension')
    parser.add_argument('--use_small_unet', action='store_true',
                        help='Use smaller UNet architecture')
    parser.add_argument('--max_tile_size', type=int, default=512,
                        help='Maximum tile size (downsample if larger)')

    # Training arguments
    parser.add_argument('--batch_size', type=int, default=4,
                        help='Batch size (number of tiles)')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of epochs')
    parser.add_argument('--val_ratio', type=float, default=0.15,
                        help='Validation set ratio')
    parser.add_argument('--test_ratio', type=float, default=0.15,
                        help='Test set ratio')
    parser.add_argument('--min_shots_per_tile', type=int, default=10,
                        help='Minimum GEDI shots per tile')

    # Output arguments
    parser.add_argument('--output_dir', type=str, default='./outputs_convcnp',
                        help='Output directory for models and logs')
    parser.add_argument('--save_every', type=int, default=10,
                        help='Save checkpoint every N epochs')

    # Other
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Device to use')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--num_workers', type=int, default=0,
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
    n_batches = 0

    for batch in tqdm(dataloader, desc='Training'):
        if batch is None:
            continue

        optimizer.zero_grad()

        # Move to device
        tile_embedding = batch['tile_embedding'].to(device)
        context_agbd = batch['context_agbd'].to(device)
        context_mask = batch['context_mask'].to(device)
        target_agbd = batch['target_agbd'].to(device)
        target_mask = batch['target_mask'].to(device)

        # Forward pass
        pred_mean, pred_log_var = model(
            tile_embedding,
            context_agbd,
            context_mask
        )

        # Compute loss (only at target locations)
        loss = convcnp_loss(pred_mean, pred_log_var, target_agbd, target_mask)

        # Backward pass
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

    return total_loss / max(n_batches, 1)


def validate(model, dataloader, device):
    """Validate the model."""
    model.eval()
    total_loss = 0
    all_metrics = []
    n_batches = 0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc='Validation'):
            if batch is None:
                continue

            # Move to device
            tile_embedding = batch['tile_embedding'].to(device)
            context_agbd = batch['context_agbd'].to(device)
            context_mask = batch['context_mask'].to(device)
            target_agbd = batch['target_agbd'].to(device)
            target_mask = batch['target_mask'].to(device)

            # Forward pass
            pred_mean, pred_log_var = model(
                tile_embedding,
                context_agbd,
                context_mask
            )

            # Compute loss
            loss = convcnp_loss(pred_mean, pred_log_var, target_agbd, target_mask)
            total_loss += loss.item()
            n_batches += 1

            # Compute metrics
            pred_std = torch.exp(0.5 * pred_log_var) if pred_log_var is not None else None
            metrics = compute_metrics(pred_mean, pred_std, target_agbd, target_mask)
            if metrics:
                all_metrics.append(metrics)

    avg_loss = total_loss / max(n_batches, 1)

    # Aggregate metrics
    avg_metrics = {}
    if len(all_metrics) > 0:
        for key in all_metrics[0].keys():
            avg_metrics[key] = np.mean([m[key] for m in all_metrics])

    return avg_loss, avg_metrics


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
    print("GEDI ConvCNP Training")
    print("=" * 80)
    print(f"Device: {args.device}")
    print(f"Region: {args.region_bbox}")
    print(f"Output: {output_dir}")
    print(f"Max tile size: {args.max_tile_size}")
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

    if gedi_df is None or len(gedi_df) == 0:
        print("No GEDI data found in region. Exiting.")
        return

    print(f"Found {len(gedi_df)} GEDI shots across {gedi_df['tile_id'].nunique()} tiles")
    print()

    # Step 2: Spatial CV split
    print("Step 2: Creating spatial train/val/test splits...")
    splitter = SpatialTileSplitter(
        gedi_df,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        random_state=args.seed
    )
    train_df, val_df, test_df = splitter.split()

    print(f"Train: {len(train_df)} shots in {train_df['tile_id'].nunique()} tiles")
    print(f"Val:   {len(val_df)} shots in {val_df['tile_id'].nunique()} tiles")
    print(f"Test:  {len(test_df)} shots in {test_df['tile_id'].nunique()} tiles")
    print()

    # Step 3: Create datasets
    print("Step 3: Creating ConvCNP datasets...")
    train_dataset = ConvCNPTileDataset(
        train_df,
        embedding_year=args.embedding_year,
        cache_dir=args.cache_dir,
        min_shots_per_tile=args.min_shots_per_tile,
        max_tile_size=args.max_tile_size
    )

    val_dataset = ConvCNPTileDataset(
        val_df,
        embedding_year=args.embedding_year,
        cache_dir=args.cache_dir,
        min_shots_per_tile=args.min_shots_per_tile,
        max_tile_size=args.max_tile_size
    )

    test_dataset = ConvCNPTileDataset(
        test_df,
        embedding_year=args.embedding_year,
        cache_dir=args.cache_dir,
        min_shots_per_tile=args.min_shots_per_tile,
        max_tile_size=args.max_tile_size
    )

    print()

    # Step 4: Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_convcnp
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_convcnp
    )

    # Step 5: Initialize model
    print("Step 4: Initializing ConvCNP model...")
    model = GEDIConvCNP(
        embedding_channels=128,
        feature_dim=args.feature_dim,
        base_channels=args.base_channels,
        unet_depth=args.unet_depth,
        decoder_hidden_dim=args.decoder_hidden_dim,
        output_uncertainty=True,
        use_small_unet=args.use_small_unet
    ).to(args.device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params / 1e6:.2f}M")
    print()

    # Step 6: Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # Step 7: Training loop
    print("Step 5: Training...")
    best_val_loss = float('inf')
    history = {
        'train_loss': [],
        'val_loss': [],
        'val_metrics': []
    }

    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch + 1}/{args.epochs}")
        print("-" * 40)

        # Train
        train_loss = train_epoch(model, train_loader, optimizer, args.device)
        history['train_loss'].append(train_loss)

        # Validate
        val_loss, val_metrics = validate(model, val_loader, args.device)
        history['val_loss'].append(val_loss)
        history['val_metrics'].append(val_metrics)

        # Print metrics
        print(f"Train Loss: {train_loss:.4f}")
        print(f"Val Loss:   {val_loss:.4f}")
        if val_metrics:
            print(f"Val RMSE:   {val_metrics.get('rmse', 0):.4f}")
            print(f"Val MAE:    {val_metrics.get('mae', 0):.4f}")
            print(f"Val R2:     {val_metrics.get('r2', 0):.4f}")

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'val_metrics': val_metrics,
            }, output_dir / 'best_model.pt')
            print("✓ Saved best model")

        # Save checkpoint
        if (epoch + 1) % args.save_every == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, output_dir / f'checkpoint_epoch_{epoch + 1}.pt')

        # Save history
        with open(output_dir / 'history.pkl', 'wb') as f:
            pickle.dump(history, f)

    print("\n" + "=" * 80)
    print("Training complete!")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Model saved to: {output_dir / 'best_model.pt'}")
    print("=" * 80)


if __name__ == '__main__':
    main()
