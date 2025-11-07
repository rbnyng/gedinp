"""
Generate AGB predictions at 10m resolution using trained GEDI Neural Process model.

This script:
1. Loads a trained model checkpoint
2. Queries nearest N GEDI shots for context
3. Generates a dense prediction grid at specified resolution
4. Runs batched GPU inference
5. Outputs GeoTIFF files (mean + uncertainty)
6. Creates visualization by default
"""

import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
import matplotlib.pyplot as plt
import rasterio
from rasterio.transform import from_bounds
from rasterio.crs import CRS
from scipy.spatial import cKDTree

from data.gedi import GEDIQuerier
from data.embeddings import EmbeddingExtractor
from models.neural_process import GEDINeuralProcess


def parse_args():
    parser = argparse.ArgumentParser(
        description='Generate AGB predictions at 10m resolution'
    )

    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint directory')
    parser.add_argument('--region', type=float, nargs=4, required=True,
                        metavar=('min_lon', 'min_lat', 'max_lon', 'max_lat'),
                        help='Bounding box: min_lon min_lat max_lon max_lat')
    parser.add_argument('--resolution', type=float, default=10.0,
                        help='Output resolution in meters (default: 10m)')
    parser.add_argument('--output_dir', type=str, default='./predictions',
                        help='Output directory (default: ./predictions)')
    parser.add_argument('--n_context', type=int, default=100,
                        help='Number of nearest GEDI shots to use as context (default: 100)')
    parser.add_argument('--batch_size', type=int, default=1024,
                        help='Inference batch size (default: 1024)')
    parser.add_argument('--device', type=str,
                        default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Device (cuda/cpu)')
    parser.add_argument('--no_preview', action='store_true',
                        help='Disable preview generation')
    parser.add_argument('--cache_dir', type=str, default='./cache',
                        help='Cache directory for embeddings')
    parser.add_argument('--gedi_start_time', type=str, default='2022-01-01',
                        help='GEDI query start date (YYYY-MM-DD)')
    parser.add_argument('--gedi_end_time', type=str, default='2022-12-31',
                        help='GEDI query end date (YYYY-MM-DD)')
    parser.add_argument('--embedding_year', type=int, default=2022,
                        help='GeoTessera embedding year (default: 2022)')

    return parser.parse_args()


def load_model_and_config(checkpoint_dir: Path, device: str):
    """
    Load model checkpoint and config.

    Auto-detects config from checkpoint directory.

    Args:
        checkpoint_dir: Path to checkpoint directory
        device: Device to load model on

    Returns:
        (model, config)
    """
    print("Loading model configuration...")
    config_path = checkpoint_dir / 'config.json'
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found at {config_path}")

    with open(config_path, 'r') as f:
        config = json.load(f)

    print(f"Architecture mode: {config.get('architecture_mode', 'deterministic')}")

    # Initialize model
    print("Initializing model...")
    model = GEDINeuralProcess(
        patch_size=config.get('patch_size', 3),
        embedding_channels=128,
        embedding_feature_dim=config.get('embedding_feature_dim', 128),
        context_repr_dim=config.get('context_repr_dim', 128),
        hidden_dim=config.get('hidden_dim', 512),
        latent_dim=config.get('latent_dim', 128),
        output_uncertainty=True,
        architecture_mode=config.get('architecture_mode', 'deterministic'),
        num_attention_heads=config.get('num_attention_heads', 4)
    ).to(device)

    # Load checkpoint - try best_r2_model.pt first, then best_model.pt
    checkpoint_files = ['best_r2_model.pt', 'best_model.pt']
    checkpoint_path = None

    for ckpt_file in checkpoint_files:
        path = checkpoint_dir / ckpt_file
        if path.exists():
            checkpoint_path = path
            break

    if checkpoint_path is None:
        raise FileNotFoundError(
            f"No checkpoint found in {checkpoint_dir}. "
            f"Looked for: {checkpoint_files}"
        )

    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    print(f"Loaded checkpoint from epoch {checkpoint.get('epoch', 'unknown')}")
    if 'val_metrics' in checkpoint:
        print("Validation metrics:")
        for k, v in checkpoint['val_metrics'].items():
            print(f"  {k}: {v:.4f}")

    return model, config


def query_context_gedi(
    region_bbox: tuple,
    n_context: int,
    start_time: str,
    end_time: str
) -> pd.DataFrame:
    """
    Query GEDI shots for context.

    Queries the region and returns up to n_context nearest shots
    to the region center.

    Args:
        region_bbox: (min_lon, min_lat, max_lon, max_lat)
        n_context: Number of context shots to return
        start_time: Start date for GEDI query
        end_time: End date for GEDI query

    Returns:
        DataFrame with GEDI shots
    """
    print(f"\nQuerying GEDI context shots...")
    print(f"Region: {region_bbox}")
    print(f"Requesting {n_context} nearest shots")

    querier = GEDIQuerier()

    # Query a larger region to ensure we get enough shots
    min_lon, min_lat, max_lon, max_lat = region_bbox
    buffer = 0.5  # degrees (~50km)
    buffered_bbox = (
        min_lon - buffer,
        min_lat - buffer,
        max_lon + buffer,
        max_lat + buffer
    )

    gedi_df = querier.query_bbox(
        buffered_bbox,
        start_time=start_time,
        end_time=end_time
    )

    if len(gedi_df) == 0:
        raise ValueError(f"No GEDI data found in region {buffered_bbox}")

    print(f"Found {len(gedi_df)} GEDI shots in buffered region")

    # Select N nearest to region center
    center_lon = (min_lon + max_lon) / 2
    center_lat = (min_lat + max_lat) / 2

    # Compute distances to center
    gedi_coords = gedi_df[['longitude', 'latitude']].values
    center = np.array([[center_lon, center_lat]])

    # Simple euclidean distance (good enough for small regions)
    distances = np.sqrt(
        ((gedi_coords - center) ** 2).sum(axis=1)
    )

    # Sort by distance and take top N
    nearest_indices = np.argsort(distances)[:n_context]
    context_df = gedi_df.iloc[nearest_indices].copy()

    print(f"Selected {len(context_df)} nearest context shots")
    print(f"Context AGBD range: [{context_df['agbd'].min():.1f}, {context_df['agbd'].max():.1f}] Mg/ha")

    return context_df


def generate_prediction_grid(
    region_bbox: tuple,
    resolution_m: float
) -> tuple:
    """
    Generate a dense prediction grid.

    Args:
        region_bbox: (min_lon, min_lat, max_lon, max_lat)
        resolution_m: Resolution in meters

    Returns:
        (lons, lats, n_rows, n_cols) where lons/lats are 1D arrays
    """
    min_lon, min_lat, max_lon, max_lat = region_bbox

    # Convert resolution to degrees (approximate)
    # At equator: 1 degree ≈ 111km
    # This is approximate - for proper handling would need to account for latitude
    meters_per_degree = 111000.0
    resolution_deg = resolution_m / meters_per_degree

    # Adjust for latitude (longitude spacing varies with latitude)
    center_lat = (min_lat + max_lat) / 2
    lon_resolution_deg = resolution_deg / np.cos(np.radians(center_lat))
    lat_resolution_deg = resolution_deg

    # Generate grid
    lons = np.arange(min_lon, max_lon, lon_resolution_deg)
    lats = np.arange(min_lat, max_lat, lat_resolution_deg)

    n_cols = len(lons)
    n_rows = len(lats)

    print(f"\nGenerated prediction grid:")
    print(f"  Resolution: {resolution_m}m (~{resolution_deg:.6f}°)")
    print(f"  Grid size: {n_rows} x {n_cols} = {n_rows * n_cols:,} pixels")
    print(f"  Lon range: [{lons[0]:.6f}, {lons[-1]:.6f}]")
    print(f"  Lat range: [{lats[0]:.6f}, {lats[-1]:.6f}]")

    return lons, lats, n_rows, n_cols


def extract_embeddings(
    coords_df: pd.DataFrame,
    extractor: EmbeddingExtractor,
    desc: str = "Extracting embeddings"
) -> pd.DataFrame:
    """
    Extract embeddings for coordinates.

    Args:
        coords_df: DataFrame with 'longitude', 'latitude' columns
        extractor: EmbeddingExtractor instance
        desc: Progress bar description

    Returns:
        DataFrame with added 'embedding_patch' column
    """
    patches = []
    valid_indices = []

    print(f"\n{desc}...")
    for idx, row in tqdm(coords_df.iterrows(), total=len(coords_df), desc=desc):
        patch = extractor.extract_patch(row['longitude'], row['latitude'])
        if patch is not None:
            patches.append(patch)
            valid_indices.append(idx)

    print(f"Successfully extracted {len(patches)}/{len(coords_df)} embeddings "
          f"({100*len(patches)/len(coords_df):.1f}%)")

    # Filter to valid only
    result_df = coords_df.iloc[valid_indices].copy()
    result_df['embedding_patch'] = patches

    return result_df


def normalize_coords(coords: np.ndarray, global_bounds: tuple) -> np.ndarray:
    """
    Normalize coordinates to [0, 1] using global bounds from training.

    Args:
        coords: (N, 2) array of [lon, lat]
        global_bounds: (lon_min, lat_min, lon_max, lat_max) from training

    Returns:
        Normalized coordinates (N, 2)
    """
    lon_min, lat_min, lon_max, lat_max = global_bounds

    lon_range = lon_max - lon_min if lon_max > lon_min else 1.0
    lat_range = lat_max - lat_min if lat_max > lat_min else 1.0

    normalized = coords.copy()
    normalized[:, 0] = (coords[:, 0] - lon_min) / lon_range
    normalized[:, 1] = (coords[:, 1] - lat_min) / lat_range

    return normalized


def normalize_agbd(agbd: np.ndarray, agbd_scale: float = 200.0) -> np.ndarray:
    """
    Normalize AGBD with log transform (same as training).

    Args:
        agbd: Raw AGBD values
        agbd_scale: Scale factor (default: 200.0)

    Returns:
        Normalized AGBD
    """
    return np.log1p(agbd) / np.log1p(agbd_scale)


def denormalize_agbd(agbd_norm: np.ndarray, agbd_scale: float = 200.0) -> np.ndarray:
    """
    Convert normalized AGBD back to raw values.

    Args:
        agbd_norm: Normalized AGBD
        agbd_scale: Scale factor (default: 200.0)

    Returns:
        Raw AGBD values
    """
    return np.expm1(agbd_norm * np.log1p(agbd_scale))


def denormalize_std(std_norm: np.ndarray, agbd_scale: float = 200.0) -> np.ndarray:
    """
    Convert normalized standard deviation to raw values.

    Since std transforms differently than mean under log transform,
    we use the derivative: d/dx[log(1+x)] = 1/(1+x)

    For small normalized values, approximate std_raw ≈ std_norm * log(1+scale)

    Args:
        std_norm: Normalized standard deviation
        agbd_scale: Scale factor (default: 200.0)

    Returns:
        Approximate raw standard deviation
    """
    # Simple linear scaling (approximate)
    return std_norm * np.log1p(agbd_scale)


def run_inference(
    model: torch.nn.Module,
    context_df: pd.DataFrame,
    query_df: pd.DataFrame,
    global_bounds: tuple,
    batch_size: int,
    device: str
) -> tuple:
    """
    Run batched inference.

    Args:
        model: Neural process model
        context_df: Context GEDI shots with embeddings
        query_df: Query points with embeddings
        global_bounds: Global coordinate bounds
        batch_size: Batch size for inference
        device: Device

    Returns:
        (predictions, uncertainties) as numpy arrays
    """
    print(f"\nRunning inference on {len(query_df)} query points...")
    print(f"Using {len(context_df)} context shots")
    print(f"Batch size: {batch_size}")
    print(f"Device: {device}")

    # Prepare context data (same for all batches)
    context_coords = context_df[['longitude', 'latitude']].values
    context_coords_norm = normalize_coords(context_coords, global_bounds)
    context_embeddings = np.stack(context_df['embedding_patch'].values)
    context_agbd_norm = normalize_agbd(context_df['agbd'].values[:, None])

    # Convert to tensors
    context_coords_t = torch.from_numpy(context_coords_norm).float().to(device)
    context_embeddings_t = torch.from_numpy(context_embeddings).float().to(device)
    context_agbd_t = torch.from_numpy(context_agbd_norm).float().to(device)

    # Prepare query data
    query_coords = query_df[['longitude', 'latitude']].values
    query_coords_norm = normalize_coords(query_coords, global_bounds)
    query_embeddings = np.stack(query_df['embedding_patch'].values)

    # Run inference in batches
    all_predictions = []
    all_uncertainties = []

    n_batches = (len(query_df) + batch_size - 1) // batch_size

    with torch.no_grad():
        for i in tqdm(range(n_batches), desc="Inference"):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, len(query_df))

            # Get batch
            batch_coords = query_coords_norm[start_idx:end_idx]
            batch_embeddings = query_embeddings[start_idx:end_idx]

            # Convert to tensors
            batch_coords_t = torch.from_numpy(batch_coords).float().to(device)
            batch_embeddings_t = torch.from_numpy(batch_embeddings).float().to(device)

            # Forward pass
            pred_mean, pred_std = model.predict(
                context_coords_t,
                context_embeddings_t,
                context_agbd_t,
                batch_coords_t,
                batch_embeddings_t
            )

            # Convert to numpy
            pred_mean_np = pred_mean.cpu().numpy().flatten()
            pred_std_np = pred_std.cpu().numpy().flatten()

            all_predictions.append(pred_mean_np)
            all_uncertainties.append(pred_std_np)

    # Concatenate all batches
    predictions_norm = np.concatenate(all_predictions)
    uncertainties_norm = np.concatenate(all_uncertainties)

    # Denormalize
    predictions = denormalize_agbd(predictions_norm)
    uncertainties = denormalize_std(uncertainties_norm)

    print(f"\nPrediction statistics:")
    print(f"  Mean AGB: {predictions.mean():.2f} Mg/ha")
    print(f"  Std AGB: {predictions.std():.2f} Mg/ha")
    print(f"  Range: [{predictions.min():.2f}, {predictions.max():.2f}] Mg/ha")
    print(f"  Mean uncertainty: {uncertainties.mean():.2f} Mg/ha")

    return predictions, uncertainties


def save_geotiff(
    data: np.ndarray,
    lons: np.ndarray,
    lats: np.ndarray,
    output_path: Path,
    description: str = "AGB"
):
    """
    Save data as GeoTIFF.

    Args:
        data: 1D array of values
        lons: 1D array of longitudes
        lats: 1D array of latitudes
        output_path: Output file path
        description: Data description
    """
    # Reshape to 2D grid
    n_rows = len(lats)
    n_cols = len(lons)
    grid = data.reshape(n_rows, n_cols)

    # Create transform
    transform = from_bounds(
        lons[0], lats[0],
        lons[-1], lats[-1],
        n_cols, n_rows
    )

    # Write GeoTIFF
    with rasterio.open(
        output_path,
        'w',
        driver='GTiff',
        height=n_rows,
        width=n_cols,
        count=1,
        dtype=grid.dtype,
        crs=CRS.from_epsg(4326),
        transform=transform,
        compress='lzw'
    ) as dst:
        dst.write(grid, 1)
        dst.set_band_description(1, description)

    print(f"Saved {description} to: {output_path}")


def save_context_geojson(
    context_df: pd.DataFrame,
    output_path: Path
):
    """
    Save context points as GeoJSON.

    Args:
        context_df: Context GEDI shots
        output_path: Output file path
    """
    import geopandas as gpd
    from shapely.geometry import Point

    # Create GeoDataFrame
    geometry = [Point(lon, lat) for lon, lat in
                zip(context_df['longitude'], context_df['latitude'])]

    gdf = gpd.GeoDataFrame(
        context_df[['agbd']],
        geometry=geometry,
        crs='EPSG:4326'
    )

    gdf.to_file(output_path, driver='GeoJSON')
    print(f"Saved context points to: {output_path}")


def create_visualization(
    predictions: np.ndarray,
    uncertainties: np.ndarray,
    lons: np.ndarray,
    lats: np.ndarray,
    context_df: pd.DataFrame,
    output_path: Path,
    region_bbox: tuple
):
    """
    Create visualization of predictions and uncertainty.

    Args:
        predictions: 1D array of predictions
        uncertainties: 1D array of uncertainties
        lons: 1D array of longitudes
        lats: 1D array of latitudes
        context_df: Context GEDI shots
        output_path: Output file path
        region_bbox: Region bounding box
    """
    print("\nGenerating visualization...")

    # Reshape to 2D
    n_rows = len(lats)
    n_cols = len(lons)
    pred_grid = predictions.reshape(n_rows, n_cols)
    std_grid = uncertainties.reshape(n_rows, n_cols)

    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Plot mean AGB
    ax = axes[0]
    im1 = ax.imshow(
        pred_grid,
        extent=[lons[0], lons[-1], lats[0], lats[-1]],
        origin='lower',
        cmap='YlGn',
        vmin=0,
        vmax=min(200, np.nanpercentile(predictions, 99))
    )

    # Overlay context points
    ax.scatter(
        context_df['longitude'],
        context_df['latitude'],
        c='red',
        s=20,
        marker='x',
        alpha=0.6,
        label=f'GEDI context (n={len(context_df)})'
    )

    ax.set_xlabel('Longitude', fontweight='bold')
    ax.set_ylabel('Latitude', fontweight='bold')
    ax.set_title('Mean AGB Prediction', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    cbar1 = plt.colorbar(im1, ax=ax)
    cbar1.set_label('AGB (Mg/ha)', fontweight='bold')

    # Plot uncertainty
    ax = axes[1]
    im2 = ax.imshow(
        std_grid,
        extent=[lons[0], lons[-1], lats[0], lats[-1]],
        origin='lower',
        cmap='Reds',
        vmin=0,
        vmax=np.nanpercentile(uncertainties, 95)
    )

    # Overlay context points
    ax.scatter(
        context_df['longitude'],
        context_df['latitude'],
        c='blue',
        s=20,
        marker='x',
        alpha=0.6,
        label=f'GEDI context (n={len(context_df)})'
    )

    ax.set_xlabel('Longitude', fontweight='bold')
    ax.set_ylabel('Latitude', fontweight='bold')
    ax.set_title('Prediction Uncertainty (Std)', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    cbar2 = plt.colorbar(im2, ax=ax)
    cbar2.set_label('Uncertainty (Mg/ha)', fontweight='bold')

    # Add summary statistics
    min_lon, min_lat, max_lon, max_lat = region_bbox
    stats_text = (
        f"Region: [{min_lon:.3f}, {min_lat:.3f}] to [{max_lon:.3f}, {max_lat:.3f}]\n"
        f"Mean AGB: {np.nanmean(predictions):.1f} ± {np.nanstd(predictions):.1f} Mg/ha\n"
        f"Mean Uncertainty: {np.nanmean(uncertainties):.1f} Mg/ha\n"
        f"Grid size: {n_rows} × {n_cols}"
    )

    fig.text(0.5, 0.02, stats_text, ha='center', fontsize=10,
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.12)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved visualization to: {output_path}")
    plt.close()


def main():
    args = parse_args()

    print("=" * 80)
    print("GEDI NEURAL PROCESS - AGB PREDICTION")
    print("=" * 80)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Region: {args.region}")
    print(f"Resolution: {args.resolution}m")
    print(f"Context shots: {args.n_context}")
    print(f"Device: {args.device}")
    print("=" * 80)

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate region name for outputs
    min_lon, min_lat, max_lon, max_lat = args.region
    region_name = f"region_{min_lon:.3f}_{min_lat:.3f}_{max_lon:.3f}_{max_lat:.3f}"

    # Step 1: Load model and config
    checkpoint_dir = Path(args.checkpoint)
    model, config = load_model_and_config(checkpoint_dir, args.device)

    global_bounds = tuple(config['global_bounds'])
    print(f"\nGlobal coordinate bounds from training:")
    print(f"  Lon: [{global_bounds[0]:.4f}, {global_bounds[2]:.4f}]")
    print(f"  Lat: [{global_bounds[1]:.4f}, {global_bounds[3]:.4f}]")

    # Step 2: Query GEDI context
    context_df = query_context_gedi(
        tuple(args.region),
        args.n_context,
        args.gedi_start_time,
        args.gedi_end_time
    )

    # Step 3: Initialize embedding extractor
    print(f"\nInitializing GeoTessera extractor (year={args.embedding_year})...")
    extractor = EmbeddingExtractor(
        year=args.embedding_year,
        patch_size=config.get('patch_size', 3),
        cache_dir=args.cache_dir
    )

    # Step 4: Extract context embeddings
    context_df = extract_embeddings(
        context_df,
        extractor,
        desc="Extracting context embeddings"
    )

    if len(context_df) == 0:
        raise ValueError("No valid context embeddings extracted!")

    # Step 5: Generate prediction grid
    lons, lats, n_rows, n_cols = generate_prediction_grid(
        tuple(args.region),
        args.resolution
    )

    # Create grid of query points
    lon_grid, lat_grid = np.meshgrid(lons, lats)
    query_df = pd.DataFrame({
        'longitude': lon_grid.flatten(),
        'latitude': lat_grid.flatten()
    })

    print(f"Total query points: {len(query_df):,}")

    # Step 6: Extract query embeddings
    query_df = extract_embeddings(
        query_df,
        extractor,
        desc="Extracting query embeddings"
    )

    if len(query_df) == 0:
        raise ValueError("No valid query embeddings extracted!")

    print(f"\nWill predict for {len(query_df):,} valid points "
          f"({100*len(query_df)/(n_rows*n_cols):.1f}% of grid)")

    # Step 7: Run inference
    predictions, uncertainties = run_inference(
        model,
        context_df,
        query_df,
        global_bounds,
        args.batch_size,
        args.device
    )

    # Step 8: Reconstruct full grid (fill missing with NaN)
    full_predictions = np.full(n_rows * n_cols, np.nan)
    full_uncertainties = np.full(n_rows * n_cols, np.nan)

    # Map query results back to grid
    query_lons = query_df['longitude'].values
    query_lats = query_df['latitude'].values

    for i, (pred, unc) in enumerate(zip(predictions, uncertainties)):
        # Find grid index
        lon_idx = np.argmin(np.abs(lons - query_lons[i]))
        lat_idx = np.argmin(np.abs(lats - query_lats[i]))
        grid_idx = lat_idx * n_cols + lon_idx

        full_predictions[grid_idx] = pred
        full_uncertainties[grid_idx] = unc

    # Step 9: Save outputs
    print("\nSaving outputs...")

    # Save GeoTIFFs
    save_geotiff(
        full_predictions,
        lons, lats,
        output_dir / f"{region_name}_agb_mean.tif",
        "AGB Mean (Mg/ha)"
    )

    save_geotiff(
        full_uncertainties,
        lons, lats,
        output_dir / f"{region_name}_agb_std.tif",
        "AGB Uncertainty (Mg/ha)"
    )

    # Save context points
    save_context_geojson(
        context_df,
        output_dir / f"{region_name}_context.geojson"
    )

    # Step 10: Generate visualization (unless disabled)
    if not args.no_preview:
        create_visualization(
            full_predictions,
            full_uncertainties,
            lons, lats,
            context_df,
            output_dir / f"{region_name}_preview.png",
            tuple(args.region)
        )

    # Save metadata
    metadata = {
        'region_bbox': args.region,
        'resolution_m': args.resolution,
        'n_context': len(context_df),
        'n_predictions': int((~np.isnan(full_predictions)).sum()),
        'grid_size': [n_rows, n_cols],
        'checkpoint': str(checkpoint_dir),
        'config': config,
        'statistics': {
            'mean_agb': float(np.nanmean(full_predictions)),
            'std_agb': float(np.nanstd(full_predictions)),
            'min_agb': float(np.nanmin(full_predictions)),
            'max_agb': float(np.nanmax(full_predictions)),
            'mean_uncertainty': float(np.nanmean(full_uncertainties))
        }
    }

    with open(output_dir / f"{region_name}_metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)

    print("\n" + "=" * 80)
    print("PREDICTION COMPLETE")
    print("=" * 80)
    print(f"Output directory: {output_dir}")
    print(f"Files generated:")
    print(f"  - {region_name}_agb_mean.tif")
    print(f"  - {region_name}_agb_std.tif")
    print(f"  - {region_name}_context.geojson")
    if not args.no_preview:
        print(f"  - {region_name}_preview.png")
    print(f"  - {region_name}_metadata.json")
    print("=" * 80)


if __name__ == '__main__':
    main()
