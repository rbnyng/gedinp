"""
Training script for GEDI Neural Process model.
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

from config import ExperimentConfig, get_agbd_config
from data.gedi import GEDIQuerier
from data.embeddings import EmbeddingExtractor
from data.dataset import GEDINeuralProcessDataset, collate_neural_process
from data.spatial_cv import SpatialTileSplitter
from models.neural_process import (
    GEDINeuralProcess,
    neural_process_loss,
    compute_metrics
)


def parse_config():
    """Parse command line arguments and create/load config."""
    parser = argparse.ArgumentParser(description='Train GEDI Neural Process')

    # Config file (optional)
    parser.add_argument('--config', type=str, default=None,
                        help='Path to config JSON file (if provided, other args override config values)')

    # Required argument
    parser.add_argument('--region_bbox', type=float, nargs=4, required=True,
                        help='Region bounding box: min_lon min_lat max_lon max_lat')

    # Experiment name
    parser.add_argument('--experiment_name', type=str, default='gedi_cnp',
                        help='Experiment name (creates subdirectory in output_dir)')

    # Optional overrides for common parameters
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Output directory for models and logs')
    parser.add_argument('--cache_dir', type=str, default=None,
                        help='Directory for caching tiles and embeddings')
    parser.add_argument('--device', type=str, default=None,
                        help='Device to use (cuda/cpu)')
    parser.add_argument('--seed', type=int, default=None,
                        help='Random seed')
    parser.add_argument('--epochs', type=int, default=None,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=None,
                        help='Batch size (number of tiles)')
    parser.add_argument('--lr', type=float, default=None,
                        help='Learning rate')

    # Target variable selection
    parser.add_argument('--target', type=str, default='agbd',
                        choices=['agbd', 'rh98', 'cover', 'pai'],
                        help='Target variable to predict')

    args = parser.parse_args()

    # Load config from file if provided, otherwise create default
    if args.config:
        config = ExperimentConfig.load(args.config)
    else:
        # Use preset config based on target
        if args.target == 'agbd':
            config = get_agbd_config()
        else:
            # Import other preset configs
            from config import get_height_config, get_cover_config, get_pai_config
            if args.target == 'rh98':
                config = get_height_config()
            elif args.target == 'cover':
                config = get_cover_config()
            elif args.target == 'pai':
                config = get_pai_config()

    # Override config with command line arguments
    config.region_bbox = tuple(args.region_bbox)
    config.experiment_name = args.experiment_name

    if args.output_dir is not None:
        config.output_dir = args.output_dir
    if args.cache_dir is not None:
        config.cache_dir = args.cache_dir
    if args.device is not None:
        config.device = args.device
    else:
        config.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if args.seed is not None:
        config.seed = args.seed
    if args.epochs is not None:
        config.training.epochs = args.epochs
    if args.batch_size is not None:
        config.training.batch_size = args.batch_size
    if args.lr is not None:
        config.training.learning_rate = args.lr

    return config


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
    n_tiles = 0

    for batch in tqdm(dataloader, desc='Training'):
        optimizer.zero_grad()

        batch_loss = 0
        n_tiles_in_batch = 0

        # Process each tile in the batch
        for i in range(len(batch['context_coords'])):
            context_coords = batch['context_coords'][i].to(device)
            context_embeddings = batch['context_embeddings'][i].to(device)
            context_agbd = batch['context_agbd'][i].to(device)
            target_coords = batch['target_coords'][i].to(device)
            target_embeddings = batch['target_embeddings'][i].to(device)
            target_agbd = batch['target_agbd'][i].to(device)

            # Skip if no target points
            if len(target_coords) == 0:
                continue

            # Forward pass
            pred_mean, pred_log_var = model(
                context_coords,
                context_embeddings,
                context_agbd,
                target_coords,
                target_embeddings
            )

            # Compute loss
            loss = neural_process_loss(pred_mean, pred_log_var, target_agbd)

            # Check for NaN/Inf
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"Warning: NaN/Inf loss detected in training! Skipping batch.")
                continue

            batch_loss += loss
            n_tiles_in_batch += 1

        if n_tiles_in_batch > 0:
            batch_loss = batch_loss / n_tiles_in_batch
            batch_loss.backward()

            # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()

            total_loss += batch_loss.item()
            n_tiles += n_tiles_in_batch

    return total_loss / max(n_tiles, 1)


def validate(model, dataloader, device):
    """Validate the model."""
    model.eval()
    total_loss = 0
    all_metrics = []
    n_tiles = 0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc='Validation'):
            batch_loss = 0
            n_tiles_in_batch = 0

            for i in range(len(batch['context_coords'])):
                context_coords = batch['context_coords'][i].to(device)
                context_embeddings = batch['context_embeddings'][i].to(device)
                context_agbd = batch['context_agbd'][i].to(device)
                target_coords = batch['target_coords'][i].to(device)
                target_embeddings = batch['target_embeddings'][i].to(device)
                target_agbd = batch['target_agbd'][i].to(device)

                if len(target_coords) == 0:
                    continue

                # Forward pass
                pred_mean, pred_log_var = model(
                    context_coords,
                    context_embeddings,
                    context_agbd,
                    target_coords,
                    target_embeddings
                )

                # Compute loss
                loss = neural_process_loss(pred_mean, pred_log_var, target_agbd)

                # Check for NaN/Inf
                if torch.isnan(loss) or torch.isinf(loss):
                    print(f"Warning: NaN/Inf loss detected in validation!")
                    print(f"  pred_mean range: [{pred_mean.min():.4f}, {pred_mean.max():.4f}]")
                    if pred_log_var is not None:
                        print(f"  pred_log_var range: [{pred_log_var.min():.4f}, {pred_log_var.max():.4f}]")
                    print(f"  target range: [{target_agbd.min():.4f}, {target_agbd.max():.4f}]")
                    continue

                batch_loss += loss
                n_tiles_in_batch += 1

                # Compute metrics
                pred_std = torch.exp(0.5 * pred_log_var) if pred_log_var is not None else None
                metrics = compute_metrics(pred_mean, pred_std, target_agbd)
                all_metrics.append(metrics)

            if n_tiles_in_batch > 0:
                batch_loss = batch_loss / n_tiles_in_batch
                total_loss += batch_loss.item()
                n_tiles += n_tiles_in_batch

    avg_loss = total_loss / max(n_tiles, 1)

    # Aggregate metrics
    avg_metrics = {}
    if len(all_metrics) > 0:
        for key in all_metrics[0].keys():
            avg_metrics[key] = np.mean([m[key] for m in all_metrics])

    return avg_loss, avg_metrics


def main():
    config = parse_config()
    set_seed(config.seed)

    # Create output directory
    output_dir = config.get_output_dir()
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save config
    config.save(output_dir / 'config.json')

    print("=" * 80)
    print("GEDI Neural Process Training")
    print("=" * 80)
    print(f"Experiment: {config.experiment_name}")
    print(f"Target: {config.data.target_variable}")
    print(f"Device: {config.device}")
    print(f"Region: {config.region_bbox}")
    print(f"Output: {output_dir}")
    print()

    # Step 1: Query GEDI data
    print("Step 1: Querying GEDI data...")
    querier = GEDIQuerier()
    gedi_df = querier.query_region_tiles(
        region_bbox=config.region_bbox,
        tile_size=config.data.tile_size,
        start_time=config.start_time,
        end_time=config.end_time
    )
    print(f"Retrieved {len(gedi_df)} GEDI shots across {gedi_df['tile_id'].nunique()} tiles")
    print()

    if len(gedi_df) == 0:
        print("No GEDI data found in region. Exiting.")
        return

    # Step 2: Extract embeddings
    print("Step 2: Extracting GeoTessera embeddings...")
    extractor = EmbeddingExtractor(
        year=config.data.embedding_year,
        patch_size=config.model.patch_size,
        cache_dir=config.cache_dir
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

    # Step 3: Spatial split
    print("Step 3: Creating spatial train/val/test split...")
    splitter = SpatialTileSplitter(
        gedi_df,
        val_ratio=config.training.val_ratio,
        test_ratio=config.training.test_ratio,
        random_state=config.seed
    )
    train_df, val_df, test_df = splitter.split()
    print()

    # Save splits
    train_df.to_csv(output_dir / 'train_split.csv', index=False)
    val_df.to_csv(output_dir / 'val_split.csv', index=False)
    test_df.to_csv(output_dir / 'test_split.csv', index=False)

    # Compute global coordinate bounds from training data
    # This ensures consistent normalization across train/val/test sets
    # and allows the model to learn latitude-dependent patterns
    global_bounds = (
        train_df['longitude'].min(),
        train_df['latitude'].min(),
        train_df['longitude'].max(),
        train_df['latitude'].max()
    )
    print(f"Global bounds: lon [{global_bounds[0]:.4f}, {global_bounds[2]:.4f}], "
          f"lat [{global_bounds[1]:.4f}, {global_bounds[3]:.4f}]")

    # Save global bounds to config for future evaluation
    config.global_bounds = global_bounds
    config.save(output_dir / 'config.json')

    # Step 4: Create datasets
    print("Step 4: Creating datasets...")
    train_dataset = GEDINeuralProcessDataset(
        train_df,
        min_shots_per_tile=config.data.min_shots_per_tile,
        log_transform_agbd=config.data.log_transform_target,
        augment_coords=config.augmentation.augment_coords,
        coord_noise_std=config.augmentation.coord_noise_std,
        global_bounds=global_bounds
    )
    val_dataset = GEDINeuralProcessDataset(
        val_df,
        min_shots_per_tile=config.data.min_shots_per_tile,
        log_transform_agbd=config.data.log_transform_target,
        augment_coords=False,  # No augmentation for validation
        coord_noise_std=0.0,
        global_bounds=global_bounds
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.training.batch_size,
        shuffle=True,
        collate_fn=collate_neural_process,
        num_workers=config.num_workers
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.training.batch_size,
        shuffle=False,
        collate_fn=collate_neural_process,
        num_workers=config.num_workers
    )
    print()

    # Step 5: Initialize model
    print("Step 5: Initializing model...")
    model = GEDINeuralProcess(
        patch_size=config.model.patch_size,
        embedding_channels=config.model.embedding_channels,
        embedding_feature_dim=config.model.embedding_feature_dim,
        context_repr_dim=config.model.context_repr_dim,
        hidden_dim=config.model.hidden_dim,
        output_uncertainty=config.model.output_uncertainty,
        use_attention=config.model.use_attention,
        num_attention_heads=config.model.num_attention_heads
    ).to(config.device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {n_params:,}")
    print()

    optimizer = torch.optim.Adam(model.parameters(), lr=config.training.learning_rate)

    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=config.training.lr_scheduler_factor,
        patience=config.training.lr_scheduler_patience
    ) if config.training.use_lr_scheduler else None

    # Step 6: Training loop
    print("Step 6: Training...")
    best_val_loss = float('inf')
    best_r2 = float('-inf')
    epochs_without_improvement = 0
    train_losses = []
    val_losses = []

    for epoch in range(1, config.training.epochs + 1):
        print(f"\nEpoch {epoch}/{config.training.epochs}")
        print("-" * 80)

        # Train
        train_loss = train_epoch(model, train_loader, optimizer, config.device)
        train_losses.append(train_loss)

        # Validate
        val_loss, val_metrics = validate(model, val_loader, config.device)
        val_losses.append(val_loss)

        # Print metrics (using scientific notation for losses)
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
        if scheduler is not None:
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
        if config.training.early_stopping and epochs_without_improvement >= config.training.early_stopping_patience:
            print(f"\nEarly stopping triggered after {epoch} epochs")
            print(f"No improvement in validation loss for {config.training.early_stopping_patience} epochs")
            break

        # Save checkpoint
        if epoch % config.training.save_every == 0:
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
