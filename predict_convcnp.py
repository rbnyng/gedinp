"""
Inference script for ConvCNP - generates dense AGBD predictions.

Loads trained model and creates full tile predictions.
"""

import argparse
import json
from pathlib import Path
import pickle

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from tqdm import tqdm

from data.gedi import GEDIQuerier
from data.embeddings import EmbeddingExtractor
from models.convcnp import GEDIConvCNP
from pyproj import Transformer
from geotessera import GeoTessera


def parse_args():
    parser = argparse.ArgumentParser(description='ConvCNP Inference')

    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to trained model checkpoint')
    parser.add_argument('--tile_lon', type=float, required=True,
                        help='Tile center longitude')
    parser.add_argument('--tile_lat', type=float, required=True,
                        help='Tile center latitude')
    parser.add_argument('--embedding_year', type=int, default=2024,
                        help='Year of GeoTessera embeddings')
    parser.add_argument('--cache_dir', type=str, default='./cache',
                        help='Cache directory for embeddings')
    parser.add_argument('--output_dir', type=str, default='./predictions',
                        help='Output directory for predictions')
    parser.add_argument('--max_tile_size', type=int, default=512,
                        help='Maximum tile size (downsample if larger)')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Device to use')
    parser.add_argument('--agbd_scale', type=float, default=200.0,
                        help='AGBD normalization scale')

    # Context options
    parser.add_argument('--context_bbox', type=float, nargs=4, default=None,
                        help='Bounding box for context GEDI shots: min_lon min_lat max_lon max_lat')
    parser.add_argument('--start_time', type=str, default='2019-01-01',
                        help='Start date for GEDI context data')
    parser.add_argument('--end_time', type=str, default='2023-12-31',
                        help='End date for GEDI context data')

    return parser.parse_args()


def load_tile_embedding(tile_lon, tile_lat, year, cache_dir, max_size=None):
    """Load GeoTessera embedding tile."""
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    cache_file = cache_dir / f"tile_{tile_lon:.2f}_{tile_lat:.2f}_{year}.pkl"

    # Try cache first
    if cache_file.exists():
        with open(cache_file, 'rb') as f:
            tile_data = pickle.load(f)
    else:
        # Download from GeoTessera
        gt = GeoTessera()
        embedding, crs, transform = gt.fetch_embedding(
            lon=tile_lon,
            lat=tile_lat,
            year=year
        )
        tile_data = (embedding, crs, transform)

        # Save to cache
        with open(cache_file, 'wb') as f:
            pickle.dump(tile_data, f)

    embedding, crs, transform = tile_data
    H, W, C = embedding.shape

    # Optionally downsample
    if max_size and max(H, W) > max_size:
        from scipy.ndimage import zoom
        scale = max_size / max(H, W)
        new_H = int(H * scale)
        new_W = int(W * scale)
        embedding = zoom(embedding, (new_H/H, new_W/W, 1), order=1)

    return embedding, crs, transform


def query_context_gedi(tile_lon, tile_lat, bbox=None, start_time='2019-01-01', end_time='2023-12-31'):
    """Query GEDI shots for context."""
    querier = GEDIQuerier()

    if bbox is None:
        # Default: use tile bounds (0.1 degree tile)
        bbox = (tile_lon - 0.05, tile_lat - 0.05, tile_lon + 0.05, tile_lat + 0.05)

    gedi_df = querier.query_bbox(
        bbox=bbox,
        start_time=start_time,
        end_time=end_time
    )

    return gedi_df


def lonlat_to_pixel(lon, lat, transform, crs):
    """Convert lon/lat to pixel coordinates."""
    transformer = Transformer.from_crs("EPSG:4326", crs, always_xy=True)
    x, y = transformer.transform(lon, lat)
    col, row = ~transform * (x, y)
    return int(row), int(col)


def predict_tile(
    model,
    tile_embedding,
    context_gedi_df,
    crs,
    transform,
    agbd_scale,
    device
):
    """
    Generate dense predictions for a tile.

    Args:
        model: Trained ConvCNP model
        tile_embedding: Tile embedding array (H, W, C)
        context_gedi_df: DataFrame with context GEDI shots
        crs: Tile CRS
        transform: Rasterio transform
        agbd_scale: AGBD normalization scale
        device: Torch device

    Returns:
        pred_mean: (H, W) array of mean predictions
        pred_std: (H, W) array of uncertainty predictions
    """
    H, W, C = tile_embedding.shape

    # Convert to torch tensor
    tile_embedding_torch = torch.from_numpy(tile_embedding.transpose(2, 0, 1)).float()
    tile_embedding_torch = tile_embedding_torch.unsqueeze(0).to(device)  # (1, C, H, W)

    # Create context maps
    context_agbd = np.zeros((H, W), dtype=np.float32)
    context_mask = np.zeros((H, W), dtype=np.float32)

    if context_gedi_df is not None and len(context_gedi_df) > 0:
        for _, shot in context_gedi_df.iterrows():
            try:
                row, col = lonlat_to_pixel(
                    shot['longitude'],
                    shot['latitude'],
                    transform,
                    crs
                )

                if 0 <= row < H and 0 <= col < W:
                    agbd_value = shot['agbd'] / agbd_scale
                    context_agbd[row, col] = agbd_value
                    context_mask[row, col] = 1.0
            except:
                continue

    # Convert to torch
    context_agbd_torch = torch.from_numpy(context_agbd).float().unsqueeze(0).unsqueeze(0).to(device)
    context_mask_torch = torch.from_numpy(context_mask).float().unsqueeze(0).unsqueeze(0).to(device)

    # Predict
    model.eval()
    with torch.no_grad():
        pred_mean, pred_std = model.predict(
            tile_embedding_torch,
            context_agbd_torch,
            context_mask_torch
        )

    # Convert back to numpy
    pred_mean = pred_mean.squeeze().cpu().numpy() * agbd_scale  # Denormalize
    pred_std = pred_std.squeeze().cpu().numpy() * agbd_scale

    return pred_mean, pred_std, context_mask


def visualize_prediction(
    pred_mean,
    pred_std,
    context_mask,
    output_path,
    title="ConvCNP AGBD Prediction"
):
    """Create visualization of predictions."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Custom colormap for biomass
    cmap = plt.cm.YlGn

    # Mean prediction
    im1 = axes[0].imshow(pred_mean, cmap=cmap, vmin=0, vmax=200)
    axes[0].set_title('Predicted AGBD (Mg/ha)', fontsize=12, fontweight='bold')
    axes[0].axis('off')
    plt.colorbar(im1, ax=axes[0], fraction=0.046, pad=0.04)

    # Uncertainty
    im2 = axes[1].imshow(pred_std, cmap='Reds', vmin=0)
    axes[1].set_title('Prediction Uncertainty (Std)', fontsize=12, fontweight='bold')
    axes[1].axis('off')
    plt.colorbar(im2, ax=axes[1], fraction=0.046, pad=0.04)

    # Context locations
    axes[2].imshow(pred_mean, cmap=cmap, vmin=0, vmax=200, alpha=0.5)
    axes[2].imshow(context_mask, cmap='binary', alpha=0.5)
    axes[2].set_title('GEDI Context Locations', fontsize=12, fontweight='bold')
    axes[2].axis('off')

    plt.suptitle(title, fontsize=14, fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Saved visualization to: {output_path}")


def main():
    args = parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("ConvCNP Inference")
    print("=" * 80)
    print(f"Model: {args.model_path}")
    print(f"Tile: ({args.tile_lon:.2f}, {args.tile_lat:.2f})")
    print(f"Device: {args.device}")
    print()

    # Load model
    print("Loading model...")
    checkpoint = torch.load(args.model_path, map_location=args.device, weights_only=False)

    # Load config to get model architecture params
    model_dir = Path(args.model_path).parent
    config_path = model_dir / 'config.json'

    if config_path.exists():
        with open(config_path, 'r') as f:
            config = json.load(f)

        model = GEDIConvCNP(
            embedding_channels=128,
            feature_dim=config.get('feature_dim', 128),
            base_channels=config.get('base_channels', 64),
            unet_depth=config.get('unet_depth', 3),
            decoder_hidden_dim=config.get('decoder_hidden_dim', 128),
            output_uncertainty=True,
            use_small_unet=config.get('use_small_unet', False)
        ).to(args.device)
    else:
        # Use defaults
        print("Warning: config.json not found, using default architecture")
        model = GEDIConvCNP(
            embedding_channels=128,
            feature_dim=128,
            base_channels=64,
            unet_depth=3
        ).to(args.device)

    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print("✓ Model loaded")
    print()

    # Load tile embedding
    print("Loading tile embedding...")
    embedding, crs, transform = load_tile_embedding(
        args.tile_lon,
        args.tile_lat,
        args.embedding_year,
        args.cache_dir,
        max_size=args.max_tile_size
    )
    print(f"✓ Tile shape: {embedding.shape}")
    print()

    # Query context GEDI
    print("Querying context GEDI shots...")
    context_df = query_context_gedi(
        args.tile_lon,
        args.tile_lat,
        bbox=args.context_bbox,
        start_time=args.start_time,
        end_time=args.end_time
    )

    if context_df is not None and len(context_df) > 0:
        print(f"✓ Found {len(context_df)} context shots")
    else:
        print("⚠ No context shots found - using zero-shot prediction")
        context_df = None
    print()

    # Generate predictions
    print("Generating predictions...")
    pred_mean, pred_std, context_mask = predict_tile(
        model,
        embedding,
        context_df,
        crs,
        transform,
        args.agbd_scale,
        args.device
    )
    print("✓ Predictions complete")
    print()

    # Save predictions
    output_base = f"tile_{args.tile_lon:.2f}_{args.tile_lat:.2f}"

    np.save(output_dir / f"{output_base}_mean.npy", pred_mean)
    np.save(output_dir / f"{output_base}_std.npy", pred_std)
    print(f"✓ Saved predictions to {output_dir}")

    # Visualize
    print("Creating visualization...")
    visualize_prediction(
        pred_mean,
        pred_std,
        context_mask,
        output_dir / f"{output_base}_prediction.png",
        title=f"ConvCNP AGBD Prediction - Tile ({args.tile_lon:.2f}, {args.tile_lat:.2f})"
    )

    # Print statistics
    print()
    print("=" * 80)
    print("Prediction Statistics")
    print("=" * 80)
    print(f"Mean AGBD: {pred_mean.mean():.2f} Mg/ha")
    print(f"Std AGBD:  {pred_mean.std():.2f} Mg/ha")
    print(f"Max AGBD:  {pred_mean.max():.2f} Mg/ha")
    print(f"Mean Uncertainty: {pred_std.mean():.2f} Mg/ha")
    if context_df is not None and len(context_df) > 0:
        print(f"Context shots: {len(context_df)}")
    print("=" * 80)


if __name__ == '__main__':
    main()
