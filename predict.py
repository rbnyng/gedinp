"""
Inference script for generating dense AGB predictions using trained Neural Process.
"""

import argparse
import json
from pathlib import Path
import pickle

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

from data.gedi import GEDIQuerier
from data.embeddings import EmbeddingExtractor
from models.neural_process import GEDINeuralProcess


def parse_args():
    parser = argparse.ArgumentParser(description='Generate AGB predictions')

    # Model arguments
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to trained model checkpoint')
    parser.add_argument('--config_path', type=str, required=True,
                        help='Path to training config.json')

    # Prediction region
    parser.add_argument('--tile_lon', type=float, required=True,
                        help='Tile center longitude')
    parser.add_argument('--tile_lat', type=float, required=True,
                        help='Tile center latitude')
    parser.add_argument('--grid_spacing', type=float, default=0.001,
                        help='Prediction grid spacing in degrees (~100m)')

    # Context data
    parser.add_argument('--context_start_time', type=str, default='2019-01-01',
                        help='Start time for context GEDI data')
    parser.add_argument('--context_end_time', type=str, default='2023-12-31',
                        help='End time for context GEDI data')
    parser.add_argument('--context_buffer', type=float, default=0.1,
                        help='Buffer around tile for context shots (degrees)')

    # Output
    parser.add_argument('--output_dir', type=str, default='./predictions',
                        help='Output directory')
    parser.add_argument('--save_geotiff', action='store_true',
                        help='Save predictions as GeoTIFF')
    parser.add_argument('--visualize', action='store_true',
                        help='Create visualization plots')

    # Other
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Device to use')
    parser.add_argument('--cache_dir', type=str, default='./cache',
                        help='Cache directory for embeddings')

    return parser.parse_args()


def load_model(model_path, config_path, device):
    """Load trained model from checkpoint."""
    # Load config
    with open(config_path, 'r') as f:
        config = json.load(f)

    # Initialize model
    model = GEDINeuralProcess(
        patch_size=config['patch_size'],
        embedding_channels=128,
        embedding_feature_dim=config['embedding_feature_dim'],
        context_repr_dim=config['context_repr_dim'],
        hidden_dim=config['hidden_dim'],
        output_uncertainty=True
    ).to(device)

    # Load weights
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    print(f"Loaded model from {model_path}")
    if 'val_metrics' in checkpoint:
        print(f"Model validation metrics: {checkpoint['val_metrics']}")

    return model, config


def get_context_data(
    tile_lon, tile_lat, buffer,
    start_time, end_time,
    extractor
):
    """Get GEDI context shots around tile."""
    # Query GEDI data with buffer
    querier = GEDIQuerier()
    bbox = (
        tile_lon - buffer,
        tile_lat - buffer,
        tile_lon + buffer,
        tile_lat + buffer
    )

    context_df = querier.query_bbox(
        bbox=bbox,
        start_time=start_time,
        end_time=end_time
    )

    if len(context_df) == 0:
        print("Warning: No GEDI context shots found in region")
        return None

    # Extract embeddings
    context_df = extractor.extract_patches_batch(context_df, verbose=True)
    context_df = context_df[context_df['embedding_patch'].notna()]

    print(f"Found {len(context_df)} context shots with embeddings")

    return context_df


def predict_dense_grid(
    model, context_df, query_lons, query_lats, query_embeddings,
    device, batch_size=256, agbd_scale=200.0
):
    """
    Generate predictions on a dense grid.

    Args:
        model: Trained Neural Process model
        context_df: DataFrame with context GEDI shots
        query_lons: Array of query longitudes
        query_lats: Array of query latitudes
        query_embeddings: Array of query embeddings
        device: Torch device
        batch_size: Batch size for inference
        agbd_scale: Scale factor for denormalization

    Returns:
        (predicted_agbd, predicted_std) arrays
    """
    # Prepare context data
    context_coords = torch.from_numpy(
        context_df[['longitude', 'latitude']].values
    ).float().to(device)

    context_embeddings = torch.from_numpy(
        np.stack(context_df['embedding_patch'].values)
    ).float().to(device)

    context_agbd = torch.from_numpy(
        context_df['agbd'].values[:, None] / agbd_scale
    ).float().to(device)

    # Predictions
    n_queries = len(query_lons)
    pred_means = []
    pred_stds = []

    model.eval()
    with torch.no_grad():
        for i in tqdm(range(0, n_queries, batch_size), desc='Predicting'):
            batch_end = min(i + batch_size, n_queries)

            # Query batch
            query_coords_batch = torch.from_numpy(
                np.column_stack([
                    query_lons[i:batch_end],
                    query_lats[i:batch_end]
                ])
            ).float().to(device)

            query_embeddings_batch = torch.from_numpy(
                query_embeddings[i:batch_end]
            ).float().to(device)

            # Predict
            pred_mean, pred_std = model.predict(
                context_coords,
                context_embeddings,
                context_agbd,
                query_coords_batch,
                query_embeddings_batch
            )

            # Denormalize
            pred_mean = pred_mean * agbd_scale
            pred_std = pred_std * agbd_scale

            pred_means.append(pred_mean.cpu().numpy())
            pred_stds.append(pred_std.cpu().numpy())

    pred_means = np.concatenate(pred_means, axis=0).flatten()
    pred_stds = np.concatenate(pred_stds, axis=0).flatten()

    return pred_means, pred_stds


def save_predictions(
    query_lons, query_lats, pred_means, pred_stds,
    output_dir, tile_lon, tile_lat
):
    """Save predictions as CSV."""
    df = pd.DataFrame({
        'longitude': query_lons,
        'latitude': query_lats,
        'agbd_pred': pred_means,
        'agbd_std': pred_stds
    })

    output_path = output_dir / f'predictions_tile_{tile_lon:.2f}_{tile_lat:.2f}.csv'
    df.to_csv(output_path, index=False)
    print(f"Saved predictions to {output_path}")


def visualize_predictions(
    query_lons, query_lats, pred_means, pred_stds,
    context_df, output_dir, tile_lon, tile_lat
):
    """Create visualization plots."""
    fig, axes = plt.subplots(1, 2, figsize=(18, 5))

    # Plot 1: Predicted AGBD
    sc1 = axes[0].scatter(
        query_lons, query_lats,
        c=pred_means, cmap='YlGn',
        s=5, vmin=0, vmax=200
    )
    axes[0].scatter(
        context_df['longitude'], context_df['latitude'],
        c='red', marker='x', s=0.1, label='GEDI context'
    )
    axes[0].set_xlabel('Longitude')
    axes[0].set_ylabel('Latitude')
    axes[0].set_title('Predicted AGBD (Mg/ha)')
    axes[0].legend()
    plt.colorbar(sc1, ax=axes[0], label='AGBD (Mg/ha)')

    # Plot 2: Uncertainty
    sc2 = axes[1].scatter(
        query_lons, query_lats,
        c=pred_stds, cmap='Reds',
        s=5
    )
    axes[1].set_xlabel('Longitude')
    axes[1].set_ylabel('Latitude')
    axes[1].set_title('Prediction Uncertainty (Std Dev)')
    plt.colorbar(sc2, ax=axes[1], label='Std Dev (Mg/ha)')

    plt.tight_layout()
    output_path = output_dir / f'visualization_tile_{tile_lon:.2f}_{tile_lat:.2f}.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved visualization to {output_path}")
    plt.close()


def main():
    args = parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("GEDI Neural Process Prediction")
    print("=" * 80)
    print(f"Tile: ({args.tile_lon}, {args.tile_lat})")
    print(f"Device: {args.device}")
    print()

    # Load model
    print("Loading model...")
    model, config = load_model(args.model_path, args.config_path, args.device)
    print()

    # Initialize embedding extractor
    print("Initializing embedding extractor...")
    extractor = EmbeddingExtractor(
        year=config['embedding_year'],
        patch_size=config['patch_size'],
        cache_dir=args.cache_dir
    )
    print()

    # Get context data
    print("Fetching context GEDI data...")
    context_df = get_context_data(
        args.tile_lon, args.tile_lat, args.context_buffer,
        args.context_start_time, args.context_end_time,
        extractor
    )

    if context_df is None or len(context_df) == 0:
        print("ERROR: No context data available. Cannot generate predictions.")
        return
    print()

    # Generate query grid
    print("Generating query grid...")
    query_lons, query_lats, query_embeddings = extractor.extract_dense_grid(
        args.tile_lon, args.tile_lat,
        spacing=args.grid_spacing
    )

    if query_lons is None:
        print("ERROR: Could not extract query embeddings.")
        return

    print(f"Generated {len(query_lons)} query points")
    print()

    # Predict
    print("Generating predictions...")
    pred_means, pred_stds = predict_dense_grid(
        model, context_df,
        query_lons, query_lats, query_embeddings,
        args.device
    )
    print()

    # Print statistics
    print("Prediction statistics:")
    print(f"  Mean AGBD: {pred_means.mean():.2f} Mg/ha")
    print(f"  Std AGBD:  {pred_means.std():.2f} Mg/ha")
    print(f"  Min AGBD:  {pred_means.min():.2f} Mg/ha")
    print(f"  Max AGBD:  {pred_means.max():.2f} Mg/ha")
    print(f"  Mean uncertainty: {pred_stds.mean():.2f} Mg/ha")
    print()

    # Save predictions
    print("Saving predictions...")
    save_predictions(
        query_lons, query_lats, pred_means, pred_stds,
        output_dir, args.tile_lon, args.tile_lat
    )
    print()

    # Visualize
    if args.visualize:
        print("Creating visualizations...")
        visualize_predictions(
            query_lons, query_lats, pred_means, pred_stds,
            context_df, output_dir, args.tile_lon, args.tile_lat
        )
        print()

    print("=" * 80)
    print("Prediction complete!")
    print(f"Results saved to: {output_dir}")
    print("=" * 80)


if __name__ == '__main__':
    main()
