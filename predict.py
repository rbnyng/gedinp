"""
Prediction script for GEDI Neural Process model.

Generates predictions on a dense grid for one or more tiles and visualizes mean AGBD
and uncertainty estimates.
"""

import argparse
import json
from pathlib import Path
import pickle

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from tqdm import tqdm

from data.gedi import GEDIQuerier
from data.embeddings import EmbeddingExtractor
from models.neural_process import GEDINeuralProcess


def parse_args():
    parser = argparse.ArgumentParser(description='Generate predictions with trained GEDI Neural Process')

    # Model arguments
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint (e.g., outputs/best_model.pt)')
    parser.add_argument('--config', type=str, default=None,
                        help='Path to config.json (default: same dir as checkpoint)')

    # Prediction region arguments
    parser.add_argument('--tile_bbox', type=float, nargs=4, required=True,
                        help='Tile bounding box: min_lon min_lat max_lon max_lat')
    parser.add_argument('--grid_resolution', type=float, default=0.001,
                        help='Spatial resolution for prediction grid (degrees, default: 0.001 ≈ 100m)')

    # Context arguments
    parser.add_argument('--context_radius', type=float, default=0.2,
                        help='Radius around tile to query GEDI context shots (degrees)')
    parser.add_argument('--use_train_only', action='store_true',
                        help='Use only training GEDI shots (requires train_split.csv)')
    parser.add_argument('--start_time', type=str, default='2019-01-01',
                        help='Start date for GEDI data (YYYY-MM-DD)')
    parser.add_argument('--end_time', type=str, default='2023-12-31',
                        help='End date for GEDI data (YYYY-MM-DD)')

    # Inference arguments
    parser.add_argument('--batch_size', type=int, default=1000,
                        help='Batch size for inference (number of query points)')
    parser.add_argument('--embedding_year', type=int, default=2024,
                        help='Year of GeoTessera embeddings')
    parser.add_argument('--cache_dir', type=str, default='./cache',
                        help='Directory for caching embeddings')

    # Output arguments
    parser.add_argument('--output_dir', type=str, default='./predictions',
                        help='Output directory for predictions and visualizations')
    parser.add_argument('--save_predictions', action='store_true',
                        help='Save raw predictions as numpy arrays')

    return parser.parse_args()


def load_model_and_config(checkpoint_path: str, config_path: str = None, device: str = 'cuda'):
    """
    Load trained model and configuration.

    Args:
        checkpoint_path: Path to model checkpoint
        config_path: Path to config.json (if None, inferred from checkpoint)
        device: Device to load model on

    Returns:
        model: Loaded model in eval mode
        config: Configuration dict
        checkpoint: Full checkpoint dict
    """
    checkpoint_path = Path(checkpoint_path)

    # Load config
    if config_path is None:
        config_path = checkpoint_path.parent / 'config.json'

    with open(config_path, 'r') as f:
        config = json.load(f)

    print(f"Loaded config from {config_path}")
    print(f"Architecture mode: {config.get('architecture_mode', 'deterministic')}")

    # Initialize model
    model = GEDINeuralProcess(
        embedding_channels=config.get('embedding_channels', 128),
        patch_size=config.get('patch_size', 3),
        hidden_dim=config.get('hidden_dim', 512),
        embedding_feature_dim=config.get('embedding_feature_dim', 128),
        context_repr_dim=config.get('context_repr_dim', 128),
        latent_dim=config.get('latent_dim', 128),
        architecture_mode=config.get('architecture_mode', 'deterministic'),
        num_attention_heads=config.get('num_attention_heads', 4),
    )

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    print(f"Loaded model from {checkpoint_path}")
    print(f"Checkpoint epoch: {checkpoint.get('epoch', 'N/A')}")
    print(f"Validation loss: {checkpoint.get('val_loss', 'N/A'):.4f}")
    if 'val_metrics' in checkpoint:
        metrics = checkpoint['val_metrics']
        print(f"Validation R²: {metrics.get('r2', 'N/A'):.4f}")
        print(f"Validation RMSE: {metrics.get('rmse', 'N/A'):.4f} Mg/ha")

    return model, config, checkpoint


def query_context_gedi(tile_bbox, context_radius, start_time, end_time,
                       use_train_only=False, train_split_path=None):
    """
    Query GEDI context points around the target tile.

    Args:
        tile_bbox: (min_lon, min_lat, max_lon, max_lat)
        context_radius: Radius to expand search (degrees)
        start_time: Start date for GEDI query
        end_time: End date for GEDI query
        use_train_only: Whether to filter to training tiles only
        train_split_path: Path to train_split.csv

    Returns:
        DataFrame with GEDI context shots
    """
    min_lon, min_lat, max_lon, max_lat = tile_bbox

    # Expand bbox by context radius
    query_bbox = (
        min_lon - context_radius,
        min_lat - context_radius,
        max_lon + context_radius,
        max_lat + context_radius
    )

    print(f"Querying GEDI in bbox: {query_bbox}")
    querier = GEDIQuerier()
    gedi_df = querier.query_region_tiles(
        region_bbox=query_bbox,
        tile_size=0.1,
        start_time=start_time,
        end_time=end_time
    )

    print(f"Found {len(gedi_df)} GEDI shots in context region")

    # Filter to training tiles if requested
    if use_train_only and train_split_path:
        train_tiles = pd.read_csv(train_split_path)['tile_id'].unique()
        gedi_df = gedi_df[gedi_df['tile_id'].isin(train_tiles)]
        print(f"Filtered to {len(gedi_df)} shots from training tiles")

    return gedi_df


def create_query_grid(tile_bbox, grid_resolution):
    """
    Create a uniform grid of query points within the tile.

    Args:
        tile_bbox: (min_lon, min_lat, max_lon, max_lat)
        grid_resolution: Spacing between grid points (degrees)

    Returns:
        query_lons: 1D array of query longitudes
        query_lats: 1D array of query latitudes
        grid_shape: (n_lat, n_lon) shape of the grid
    """
    min_lon, min_lat, max_lon, max_lat = tile_bbox

    # Create grid
    lons = np.arange(min_lon, max_lon, grid_resolution)
    lats = np.arange(min_lat, max_lat, grid_resolution)

    lon_grid, lat_grid = np.meshgrid(lons, lats)

    query_lons = lon_grid.ravel()
    query_lats = lat_grid.ravel()

    grid_shape = lon_grid.shape

    print(f"Created query grid: {grid_shape[0]}x{grid_shape[1]} = {len(query_lons)} points")
    print(f"Grid resolution: {grid_resolution}° (≈{grid_resolution * 111:.0f}m)")

    return query_lons, query_lats, grid_shape


def extract_embeddings(gedi_df, query_lons, query_lats, embedding_year, cache_dir, patch_size=3):
    """
    Extract GeoTessera embeddings for context and query points.

    Args:
        gedi_df: DataFrame with GEDI context points
        query_lons: Query longitudes
        query_lats: Query latitudes
        embedding_year: Year of embeddings
        cache_dir: Cache directory
        patch_size: Embedding patch size

    Returns:
        gedi_df: DataFrame with embedding_patch column added
        query_embeddings: Array of query embeddings (N, patch_size, patch_size, 128)
    """
    extractor = EmbeddingExtractor(
        year=embedding_year,
        patch_size=patch_size,
        cache_dir=cache_dir
    )

    # Extract context embeddings
    print("Extracting embeddings for context points...")
    gedi_df = extractor.extract_patches_batch(gedi_df)
    n_valid_context = gedi_df['embedding_patch'].notna().sum()
    print(f"Successfully extracted {n_valid_context}/{len(gedi_df)} context embeddings")

    # Extract query embeddings
    print("Extracting embeddings for query grid...")
    query_df = pd.DataFrame({
        'longitude': query_lons,
        'latitude': query_lats
    })
    query_df = extractor.extract_patches_batch(query_df)

    # Convert to array
    valid_mask = query_df['embedding_patch'].notna()
    print(f"Successfully extracted {valid_mask.sum()}/{len(query_df)} query embeddings")

    if valid_mask.sum() < len(query_df):
        print("Warning: Some query points have missing embeddings (likely outside coverage)")

    query_embeddings = np.stack(query_df['embedding_patch'].values)

    return gedi_df, query_embeddings, valid_mask.values


def run_inference(model, context_df, query_lons, query_lats, query_embeddings,
                 config, batch_size=1000, device='cuda'):
    """
    Run batched inference on query grid.

    Args:
        model: Trained model
        context_df: DataFrame with context GEDI shots (with embeddings)
        query_lons: Query longitudes
        query_lats: Query latitudes
        query_embeddings: Query embeddings
        config: Model config
        batch_size: Batch size for inference
        device: Device to run on

    Returns:
        pred_means: Predicted mean AGBD (normalized)
        pred_log_vars: Predicted log variances
    """
    # Filter context to valid embeddings
    context_df = context_df[context_df['embedding_patch'].notna()].copy()

    if len(context_df) == 0:
        raise ValueError("No valid context points with embeddings!")

    print(f"Using {len(context_df)} context points for inference")

    # Prepare context data
    context_coords = context_df[['longitude', 'latitude']].values
    context_embeddings = np.stack(context_df['embedding_patch'].values)
    context_agbd = context_df['agbd'].values[:, None]

    # Normalize context data
    global_bounds = (
        config['global_lon_min'],
        config['global_lat_min'],
        config['global_lon_max'],
        config['global_lat_max']
    )

    lon_min, lat_min, lon_max, lat_max = global_bounds
    lon_range = lon_max - lon_min if lon_max > lon_min else 1.0
    lat_range = lat_max - lat_min if lat_max > lat_min else 1.0

    context_coords_norm = context_coords.copy()
    context_coords_norm[:, 0] = (context_coords_norm[:, 0] - lon_min) / lon_range
    context_coords_norm[:, 1] = (context_coords_norm[:, 1] - lat_min) / lat_range

    # Normalize AGBD
    agbd_scale = config.get('agbd_scale', 200.0)
    log_transform_agbd = config.get('log_transform_agbd', True)

    if log_transform_agbd:
        context_agbd_norm = np.log1p(context_agbd) / np.log1p(agbd_scale)
    else:
        context_agbd_norm = context_agbd / agbd_scale

    # Convert to tensors
    context_coords_tensor = torch.from_numpy(context_coords_norm).float().to(device)
    context_embeddings_tensor = torch.from_numpy(context_embeddings).float().to(device)
    context_agbd_tensor = torch.from_numpy(context_agbd_norm).float().to(device)

    # Run batched inference on query points
    n_queries = len(query_lons)
    pred_means = []
    pred_log_vars = []

    print(f"Running inference on {n_queries} query points...")

    with torch.no_grad():
        for i in tqdm(range(0, n_queries, batch_size)):
            # Get batch
            batch_end = min(i + batch_size, n_queries)
            batch_query_lons = query_lons[i:batch_end]
            batch_query_lats = query_lats[i:batch_end]
            batch_query_embeddings = query_embeddings[i:batch_end]

            # Normalize query coordinates
            batch_query_coords = np.column_stack([batch_query_lons, batch_query_lats])
            batch_query_coords_norm = batch_query_coords.copy()
            batch_query_coords_norm[:, 0] = (batch_query_coords_norm[:, 0] - lon_min) / lon_range
            batch_query_coords_norm[:, 1] = (batch_query_coords_norm[:, 1] - lat_min) / lat_range

            # Convert to tensors
            query_coords_tensor = torch.from_numpy(batch_query_coords_norm).float().to(device)
            query_embeddings_tensor = torch.from_numpy(batch_query_embeddings).float().to(device)

            # Forward pass
            # Model expects lists for batching, but we have single "tile" with all context
            mean, log_var = model(
                context_coords=[context_coords_tensor],
                context_embeddings=[context_embeddings_tensor],
                context_agbd=[context_agbd_tensor],
                target_coords=[query_coords_tensor],
                target_embeddings=[query_embeddings_tensor],
                training=False
            )

            # Extract predictions (model returns [batch_size] where batch_size=1 in our case)
            pred_means.append(mean[0].cpu().numpy())
            pred_log_vars.append(log_var[0].cpu().numpy())

    # Concatenate batches
    pred_means = np.concatenate(pred_means)
    pred_log_vars = np.concatenate(pred_log_vars)

    return pred_means, pred_log_vars


def denormalize_predictions(pred_means, pred_log_vars, config):
    """
    Denormalize predictions back to original AGBD units.

    Args:
        pred_means: Normalized predicted means
        pred_log_vars: Predicted log variances
        config: Model config

    Returns:
        agbd_means: AGBD means in Mg/ha
        agbd_stds: AGBD standard deviations in Mg/ha
    """
    agbd_scale = config.get('agbd_scale', 200.0)
    log_transform_agbd = config.get('log_transform_agbd', True)

    if log_transform_agbd:
        # Reverse: y = log(1+x) / log(1+scale)
        # So: x = exp(y * log(1+scale)) - 1
        agbd_means = np.expm1(pred_means * np.log1p(agbd_scale))

        # For uncertainty, use delta method approximation
        # If y ~ N(μ, σ²) and x = f(y), then Var(x) ≈ [f'(μ)]² * σ²
        # f(y) = exp(y * log(1+scale)) - 1
        # f'(y) = log(1+scale) * exp(y * log(1+scale))
        derivative = np.log1p(agbd_scale) * np.exp(pred_means * np.log1p(agbd_scale))
        agbd_vars = np.exp(pred_log_vars) * (derivative ** 2)
        agbd_stds = np.sqrt(agbd_vars)
    else:
        # Simple scaling
        agbd_means = pred_means * agbd_scale
        agbd_stds = np.exp(0.5 * pred_log_vars) * agbd_scale

    return agbd_means, agbd_stds


def visualize_predictions(tile_bbox, grid_shape, agbd_means, agbd_stds,
                         context_df, output_path, valid_mask=None):
    """
    Create visualization of mean AGBD and uncertainty.

    Args:
        tile_bbox: Tile bounding box
        grid_shape: (n_lat, n_lon) shape of grid
        agbd_means: Predicted AGBD means (Mg/ha)
        agbd_stds: Predicted AGBD standard deviations (Mg/ha)
        context_df: DataFrame with context GEDI shots
        output_path: Path to save figure
        valid_mask: Boolean mask for valid predictions (None = all valid)
    """
    min_lon, min_lat, max_lon, max_lat = tile_bbox

    # Reshape to grid
    agbd_grid = agbd_means.reshape(grid_shape)
    std_grid = agbd_stds.reshape(grid_shape)

    # Handle invalid predictions
    if valid_mask is not None:
        valid_grid = valid_mask.reshape(grid_shape)
        agbd_grid = np.where(valid_grid, agbd_grid, np.nan)
        std_grid = np.where(valid_grid, std_grid, np.nan)

    # Create figure
    fig = plt.figure(figsize=(16, 7))
    gs = GridSpec(2, 2, figure=fig, height_ratios=[20, 1], hspace=0.3, wspace=0.3)

    # Mean AGBD subplot
    ax1 = fig.add_subplot(gs[0, 0])
    im1 = ax1.imshow(
        agbd_grid,
        extent=[min_lon, max_lon, min_lat, max_lat],
        origin='lower',
        aspect='auto',
        cmap='YlGn',
        interpolation='nearest'
    )

    # Overlay context points
    if len(context_df) > 0:
        ax1.scatter(
            context_df['longitude'],
            context_df['latitude'],
            c='red',
            s=10,
            alpha=0.6,
            marker='.',
            label='GEDI context'
        )
        ax1.legend(loc='upper right', fontsize=8)

    ax1.set_xlabel('Longitude')
    ax1.set_ylabel('Latitude')
    ax1.set_title('Mean AGBD (Mg/ha)', fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3)

    # Colorbar for mean
    cbar_ax1 = fig.add_subplot(gs[1, 0])
    cbar1 = plt.colorbar(im1, cax=cbar_ax1, orientation='horizontal')
    cbar1.set_label('AGBD (Mg/ha)', fontsize=10)

    # Uncertainty subplot
    ax2 = fig.add_subplot(gs[0, 1])
    im2 = ax2.imshow(
        std_grid,
        extent=[min_lon, max_lon, min_lat, max_lat],
        origin='lower',
        aspect='auto',
        cmap='Reds',
        interpolation='nearest'
    )

    ax2.set_xlabel('Longitude')
    ax2.set_ylabel('Latitude')
    ax2.set_title('Uncertainty (Std Dev, Mg/ha)', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3)

    # Colorbar for uncertainty
    cbar_ax2 = fig.add_subplot(gs[1, 1])
    cbar2 = plt.colorbar(im2, cax=cbar_ax2, orientation='horizontal')
    cbar2.set_label('Standard Deviation (Mg/ha)', fontsize=10)

    # Add title with statistics
    valid_agbd = agbd_means[~np.isnan(agbd_means)] if valid_mask is not None else agbd_means
    valid_std = agbd_stds[~np.isnan(agbd_stds)] if valid_mask is not None else agbd_stds

    stats_text = (
        f"Tile: ({min_lon:.3f}, {min_lat:.3f}) to ({max_lon:.3f}, {max_lat:.3f}) | "
        f"Mean AGBD: {valid_agbd.mean():.1f} ± {valid_std.mean():.1f} Mg/ha | "
        f"Context points: {len(context_df)} | "
        f"Grid: {grid_shape[0]}×{grid_shape[1]}"
    )
    fig.suptitle(stats_text, fontsize=11, y=0.98)

    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved visualization to {output_path}")

    plt.close()


def main():
    args = parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Set device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Load model and config
    print("\n" + "="*50)
    print("LOADING MODEL")
    print("="*50)
    model, config, checkpoint = load_model_and_config(
        args.checkpoint,
        args.config,
        device
    )

    # Query context GEDI points
    print("\n" + "="*50)
    print("QUERYING CONTEXT GEDI DATA")
    print("="*50)
    train_split_path = Path(args.checkpoint).parent / 'train_split.csv' if args.use_train_only else None
    gedi_df = query_context_gedi(
        args.tile_bbox,
        args.context_radius,
        args.start_time,
        args.end_time,
        args.use_train_only,
        train_split_path
    )

    # Create query grid
    print("\n" + "="*50)
    print("CREATING QUERY GRID")
    print("="*50)
    query_lons, query_lats, grid_shape = create_query_grid(
        args.tile_bbox,
        args.grid_resolution
    )

    # Extract embeddings
    print("\n" + "="*50)
    print("EXTRACTING EMBEDDINGS")
    print("="*50)
    gedi_df, query_embeddings, valid_mask = extract_embeddings(
        gedi_df,
        query_lons,
        query_lats,
        args.embedding_year,
        args.cache_dir,
        config.get('patch_size', 3)
    )

    # Run inference
    print("\n" + "="*50)
    print("RUNNING INFERENCE")
    print("="*50)
    pred_means, pred_log_vars = run_inference(
        model,
        gedi_df,
        query_lons,
        query_lats,
        query_embeddings,
        config,
        args.batch_size,
        device
    )

    # Denormalize predictions
    print("\n" + "="*50)
    print("DENORMALIZING PREDICTIONS")
    print("="*50)
    agbd_means, agbd_stds = denormalize_predictions(pred_means, pred_log_vars, config)

    # Print statistics
    print(f"\nPrediction Statistics:")
    print(f"  Mean AGBD: {agbd_means.mean():.2f} Mg/ha")
    print(f"  Std AGBD: {agbd_means.std():.2f} Mg/ha")
    print(f"  Min AGBD: {agbd_means.min():.2f} Mg/ha")
    print(f"  Max AGBD: {agbd_means.max():.2f} Mg/ha")
    print(f"  Mean Uncertainty: {agbd_stds.mean():.2f} Mg/ha")
    print(f"  Coefficient of Variation: {(agbd_stds.mean() / agbd_means.mean() * 100):.1f}%")

    # Visualize
    print("\n" + "="*50)
    print("GENERATING VISUALIZATION")
    print("="*50)
    tile_str = f"tile_{args.tile_bbox[0]:.3f}_{args.tile_bbox[1]:.3f}"
    output_path = output_dir / f"{tile_str}_predictions.png"
    visualize_predictions(
        args.tile_bbox,
        grid_shape,
        agbd_means,
        agbd_stds,
        gedi_df,
        output_path,
        valid_mask
    )

    # Save raw predictions if requested
    if args.save_predictions:
        pred_path = output_dir / f"{tile_str}_predictions.npz"
        np.savez(
            pred_path,
            agbd_means=agbd_means.reshape(grid_shape),
            agbd_stds=agbd_stds.reshape(grid_shape),
            query_lons=query_lons,
            query_lats=query_lats,
            grid_shape=grid_shape,
            tile_bbox=args.tile_bbox
        )
        print(f"Saved raw predictions to {pred_path}")

    print("\n" + "="*50)
    print("DONE!")
    print("="*50)


if __name__ == '__main__':
    main()
