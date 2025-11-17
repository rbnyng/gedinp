import argparse
import json
from pathlib import Path
import pickle
from time import time

import numpy as np
import pandas as pd
from tqdm import tqdm

from data.gedi import GEDIQuerier
from data.embeddings import EmbeddingExtractor
from data.spatial_cv import SpatialTileSplitter, BufferedSpatialSplitter
from baselines import (
    RandomForestBaseline,
    XGBoostBaseline,
    IDWBaseline,
    MLPBaseline
)
from utils.normalization import normalize_coords, normalize_agbd, denormalize_agbd
from utils.evaluation import compute_metrics
from scipy.stats import norm
from utils.config import save_config, _make_serializable

# Import preprocessing cache utilities
try:
    from preprocess_data import check_cache_exists, load_cached_data
    CACHE_AVAILABLE = True
except ImportError as e:
    CACHE_AVAILABLE = False
    print(f"Warning: Could not import preprocessing cache utilities: {e}")
except Exception as e:
    CACHE_AVAILABLE = False
    print(f"Warning: Error importing preprocessing cache utilities: {e}")


def parse_args():
    parser = argparse.ArgumentParser(description='Train baseline models for GEDI AGBD prediction')

    # Data arguments
    parser.add_argument('--region_bbox', type=float, nargs=4, required=True,
                        help='Region bounding box: min_lon min_lat max_lon max_lat')
    parser.add_argument('--start_time', type=str, default='2022-01-01',
                        help='Start date for GEDI data (YYYY-MM-DD)')
    parser.add_argument('--end_time', type=str, default='2022-12-31',
                        help='End date for GEDI data (YYYY-MM-DD)')
    parser.add_argument('--embedding_year', type=int, default=2022,
                        help='Year of GeoTessera embeddings')
    parser.add_argument('--cache_dir', type=str, default='./cache',
                        help='Directory for caching tiles and embeddings')

    # Model arguments
    parser.add_argument('--patch_size', type=int, default=3,
                        help='Embedding patch size (default: 3x3)')

    # Random Forest arguments
    parser.add_argument('--rf_n_estimators', type=int, default=100,
                        help='Random Forest: number of trees')
    parser.add_argument('--rf_max_depth', type=int, default=6,
                        help='Random Forest: maximum tree depth')

    # XGBoost arguments
    parser.add_argument('--xgb_n_estimators', type=int, default=100,
                        help='XGBoost: number of boosting rounds')
    parser.add_argument('--xgb_max_depth', type=int, default=6,
                        help='XGBoost: maximum tree depth')
    parser.add_argument('--xgb_learning_rate', type=float, default=0.1,
                        help='XGBoost: learning rate')

    # IDW arguments
    parser.add_argument('--idw_power', type=float, default=2.0,
                        help='IDW: power parameter for distance weighting')
    parser.add_argument('--idw_n_neighbors', type=int, default=10,
                        help='IDW: number of nearest neighbors')

    # MLP arguments
    parser.add_argument('--mlp_hidden_dims', type=int, nargs='+', default=[512, 256, 128],
                        help='MLP: hidden layer dimensions')
    parser.add_argument('--mlp_dropout_rate', type=float, default=0.5,
                        help='MLP MC Dropout: dropout rate')
    parser.add_argument('--mlp_learning_rate', type=float, default=5e-4,
                        help='MLP: learning rate')
    parser.add_argument('--mlp_weight_decay', type=float, default=1e-5,
                        help='MLP MC Dropout: L2 regularization (weight decay)')
    parser.add_argument('--mlp_batch_size', type=int, default=256,
                        help='MLP: batch size')
    parser.add_argument('--mlp_n_epochs', type=int, default=100,
                        help='MLP: number of training epochs')
    parser.add_argument('--mlp_mc_samples', type=int, default=100,
                        help='MLP MC Dropout: number of MC samples for uncertainty')

    # Training arguments
    parser.add_argument('--val_ratio', type=float, default=0.15,
                        help='Validation set ratio')
    parser.add_argument('--test_ratio', type=float, default=0.15,
                        help='Test set ratio')
    parser.add_argument('--buffer_size', type=float, default=0.1,
                        help='Buffer size in degrees for spatial CV')
    parser.add_argument('--agbd_scale', type=float, default=200.0,
                        help='AGBD scale factor for normalization (default: 200.0 Mg/ha)')
    parser.add_argument('--log_transform_agbd', action='store_true', default=True,
                        help='Apply log transform to AGBD')

    # Output arguments
    parser.add_argument('--output_dir', type=str, default='./outputs_baselines',
                        help='Output directory for models and logs')

    # Other
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--models', type=str, nargs='+',
                        default=['rf', 'xgb', 'idw'],
                        choices=['rf', 'xgb', 'idw', 'mlp-dropout'],
                        help='Which models to train (default: rf, xgb, idw)')

    return parser.parse_args()


def prepare_data(df, log_transform=True, agbd_scale=200.0, patch_size=3, embedding_dim=128):
    coords = df[['longitude', 'latitude']].values
    # Convert lists back to numpy arrays if loaded from Parquet
    # Parquet saves flattened embeddings, so reshape them back to (H, W, C)
    embeddings_list = df['embedding_patch'].values
    embeddings = []
    for x in embeddings_list:
        if isinstance(x, list):
            # Flattened embedding from Parquet - reshape it
            arr = np.array(x, dtype=np.float32).reshape(patch_size, patch_size, embedding_dim)
        else:
            # Already a numpy array
            arr = x if x.shape == (patch_size, patch_size, embedding_dim) else x.reshape(patch_size, patch_size, embedding_dim)
        embeddings.append(arr)
    embeddings = np.stack(embeddings)
    agbd = df['agbd'].values

    agbd = normalize_agbd(agbd, agbd_scale=agbd_scale, log_transform=log_transform)

    return coords, embeddings, agbd


def compute_calibration_metrics(predictions, targets, stds):
    """
    Compute calibration metrics for uncertainty quantification.

    Args:
        predictions: (N,) array of predictions (in normalized log space)
        targets: (N,) array of ground truth (in normalized log space)
        stds: (N,) array of predicted standard deviations

    Returns:
        dict with calibration metrics:
            - z_mean: mean of z-scores (ideal: 0)
            - z_std: std of z-scores (ideal: 1)
            - coverage_1sigma: empirical coverage at 1σ (ideal: 68.3%)
            - coverage_2sigma: empirical coverage at 2σ (ideal: 95.4%)
            - coverage_3sigma: empirical coverage at 3σ (ideal: 99.7%)
    """
    # Compute z-scores (standardized residuals)
    z_scores = (targets - predictions) / (stds + 1e-8)

    # Z-score statistics
    z_mean = np.mean(z_scores)
    z_std = np.std(z_scores)

    # Compute empirical coverage at key confidence levels
    abs_z = np.abs(z_scores)
    coverage_1sigma = np.sum(abs_z <= 1.0) / len(z_scores) * 100
    coverage_2sigma = np.sum(abs_z <= 2.0) / len(z_scores) * 100
    coverage_3sigma = np.sum(abs_z <= 3.0) / len(z_scores) * 100

    return {
        'z_mean': z_mean,
        'z_std': z_std,
        'coverage_1sigma': coverage_1sigma,
        'coverage_2sigma': coverage_2sigma,
        'coverage_3sigma': coverage_3sigma,
    }


def print_calibration_metrics(metrics, prefix=""):
    """
    Pretty print calibration metrics.

    Args:
        metrics: Dict with calibration metrics
        prefix: Optional prefix for print statements (e.g., "Validation", "Test")
    """
    if prefix:
        prefix = f"{prefix} - "

    print(f"{prefix}Calibration Metrics:")
    print(f"  Z-scores: μ = {metrics['z_mean']:+.4f} (ideal: 0.0), σ = {metrics['z_std']:.4f} (ideal: 1.0)")
    print(f"  Coverage: 1σ = {metrics['coverage_1sigma']:.1f}% (ideal: 68.3%), "
          f"2σ = {metrics['coverage_2sigma']:.1f}% (ideal: 95.4%), "
          f"3σ = {metrics['coverage_3sigma']:.1f}% (ideal: 99.7%)")


def evaluate_model(model, coords, embeddings, agbd_true, agbd_scale=200.0, log_transform=True):
    """
    Evaluate baseline model and compute metrics in both log and linear space.

    Args:
        model: Baseline model (RF, XGBoost, or IDW)
        coords: Coordinates for prediction
        embeddings: Embeddings for prediction
        agbd_true: True AGBD values in linear space (Mg/ha)
        agbd_scale: AGBD scale factor (default: 200.0)
        log_transform: Whether log transform was used (default: True)

    Returns:
        metrics: Dict with log_rmse, log_mae, log_r2, linear_rmse, linear_mae, and calibration metrics
        pred: Predictions in linear space (Mg/ha)
        pred_std_norm: Predicted std in normalized space
    """
    # Predict (normalized, in log space)
    pred_norm, pred_std_norm = model.predict(coords, embeddings, return_std=True)

    # Normalize true values to log space for log-space metrics
    agbd_true_norm = normalize_agbd(agbd_true, agbd_scale=agbd_scale, log_transform=log_transform)

    # Compute log-space metrics
    log_metrics = compute_metrics(pred_norm, agbd_true_norm)

    # Denormalize predictions to linear space
    pred = denormalize_agbd(pred_norm, agbd_scale=agbd_scale, log_transform=log_transform)

    # Compute linear-space metrics
    linear_metrics = compute_metrics(pred, agbd_true)

    # Compute calibration metrics (in normalized log space where model predicts)
    calibration_metrics = compute_calibration_metrics(pred_norm, agbd_true_norm, pred_std_norm)

    # Combine metrics
    metrics = {
        'log_rmse': log_metrics['rmse'],
        'log_mae': log_metrics['mae'],
        'log_r2': log_metrics['r2'],
        'linear_rmse': linear_metrics['rmse'],
        'linear_mae': linear_metrics['mae'],
        **calibration_metrics  # Add calibration metrics
    }

    return metrics, pred, pred_std_norm


def train_random_forest(train_coords, train_embeddings, train_agbd, args,
                        val_coords=None, val_embeddings=None, val_agbd_norm=None):
    print("\n" + "=" * 80)
    print("Training Random Forest Baseline")
    print("=" * 80)

    model = RandomForestBaseline(
        n_estimators=args.rf_n_estimators,
        max_depth=args.rf_max_depth,
        random_state=args.seed
    )

    print(f"n_estimators: {args.rf_n_estimators}")
    print(f"max_depth: {args.rf_max_depth}")

    start_time = time()
    model.fit(train_coords, train_embeddings, train_agbd)
    train_time = time() - start_time

    print(f"Training completed in {train_time:.2f} seconds")

    return model, train_time


def train_xgboost(train_coords, train_embeddings, train_agbd, args,
                  val_coords=None, val_embeddings=None, val_agbd_norm=None):
    print("\n" + "=" * 80)
    print("Training XGBoost Baseline")
    print("=" * 80)

    model = XGBoostBaseline(
        n_estimators=args.xgb_n_estimators,
        max_depth=args.xgb_max_depth,
        learning_rate=args.xgb_learning_rate,
        random_state=args.seed
    )

    print(f"n_estimators: {args.xgb_n_estimators}")
    print(f"max_depth: {args.xgb_max_depth}")
    print(f"learning_rate: {args.xgb_learning_rate}")

    start_time = time()
    model.fit(train_coords, train_embeddings, train_agbd, fit_quantiles=True)
    train_time = time() - start_time

    print(f"Training completed in {train_time:.2f} seconds")

    return model, train_time


def train_idw(train_coords, train_embeddings, train_agbd, args):
    print("\n" + "=" * 80)
    print("Training IDW Baseline (Spatial Only)")
    print("=" * 80)

    model = IDWBaseline(
        power=args.idw_power,
        n_neighbors=args.idw_n_neighbors
    )

    print(f"power: {args.idw_power}")
    print(f"n_neighbors: {args.idw_n_neighbors}")
    print("Note: IDW ignores embeddings and uses only spatial coordinates")

    start_time = time()
    model.fit(train_coords, train_embeddings, train_agbd)
    train_time = time() - start_time

    print(f"Training completed in {train_time:.2f} seconds")

    return model, train_time


def train_mlp_dropout(train_coords, train_embeddings, train_agbd, args,
                      val_coords=None, val_embeddings=None, val_agbd_norm=None):
    print("\n" + "=" * 80)
    print("Training MLP with MC Dropout Baseline")
    print("=" * 80)

    model = MLPBaseline(
        hidden_dims=args.mlp_hidden_dims,
        dropout_rate=args.mlp_dropout_rate,
        learning_rate=args.mlp_learning_rate,
        weight_decay=args.mlp_weight_decay,
        batch_size=args.mlp_batch_size,
        n_epochs=args.mlp_n_epochs,
        mc_samples=args.mlp_mc_samples,
        random_state=args.seed
    )

    print(f"hidden_dims: {args.mlp_hidden_dims}")
    print(f"dropout_rate: {args.mlp_dropout_rate}")
    print(f"learning_rate: {args.mlp_learning_rate}")
    print(f"weight_decay: {args.mlp_weight_decay}")
    print(f"batch_size: {args.mlp_batch_size}")
    print(f"n_epochs: {args.mlp_n_epochs}")
    print(f"mc_samples: {args.mlp_mc_samples}")

    start_time = time()
    model.fit(train_coords, train_embeddings, train_agbd, verbose=True)
    train_time = time() - start_time

    print(f"Training completed in {train_time:.2f} seconds")

    return model, train_time


def try_load_from_cache(args):
    """
    Try to load preprocessed data from cache.

    Returns:
        (gedi_df, train_df, val_df, test_df, global_bounds) if cache hit, None if cache miss
    """
    if not CACHE_AVAILABLE:
        print("Cache not available (preprocess_data module not loaded)")
        return None

    try:
        # Create a mock args object compatible with preprocess_data module
        class PreprocessArgs:
            pass

        pargs = PreprocessArgs()
        pargs.region_bbox = args.region_bbox
        pargs.start_time = args.start_time
        pargs.end_time = args.end_time
        pargs.embedding_year = args.embedding_year
        pargs.patch_size = args.patch_size
        pargs.cache_dir = args.cache_dir
        pargs.buffer_size = args.buffer_size
        pargs.val_ratio = args.val_ratio
        pargs.test_ratio = args.test_ratio
        pargs.seed = args.seed
        pargs.train_years = None  # Baselines don't use temporal filtering

        # Check if cache exists
        print(f"Checking for cached data (seed={args.seed}, region={args.region_bbox})...")
        cache_exists, cache_path = check_cache_exists(pargs)

        if cache_exists:
            print("=" * 80)
            print("LOADING FROM PREPROCESSED CACHE")
            print("=" * 80)
            print(f"Cache location: {cache_path}")
            print("=" * 80)
            print()

            # Load cached data
            gedi_df, train_df, val_df, test_df, global_bounds, metadata = load_cached_data(cache_path)

            return gedi_df, train_df, val_df, test_df, global_bounds

        print(f"No cache found at: {cache_path}")
        print("Will run full data loading pipeline...")
        return None

    except Exception as e:
        print(f"Warning: Failed to load from cache: {e}")
        import traceback
        traceback.print_exc()
        print("Falling back to standard data loading pipeline")
        return None


def main():
    args = parse_args()
    np.random.seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    save_config(vars(args), output_dir / 'config.json')

    print("=" * 80)
    print("GEDI Baseline Models Training")
    print("=" * 80)
    print(f"Models to train: {', '.join(args.models)}")
    print(f"Region: {args.region_bbox}")
    print(f"Output: {output_dir}")
    print()

    # Try to load from cache first
    cached_data = try_load_from_cache(args)

    if cached_data is not None:
        # Use cached data
        gedi_df, train_df, val_df, test_df, global_bounds = cached_data

        # Save splits to output directory for reference
        def prepare_for_parquet(df):
            df_copy = df.copy()
            df_copy['embedding_patch'] = df_copy['embedding_patch'].apply(
                lambda x: x.flatten().tolist() if x is not None else None
            )
            return df_copy

        prepare_for_parquet(train_df).to_parquet(output_dir / 'train_split.parquet', index=True)
        prepare_for_parquet(val_df).to_parquet(output_dir / 'val_split.parquet', index=True)
        prepare_for_parquet(test_df).to_parquet(output_dir / 'test_split.parquet', index=True)

        with open(output_dir / 'processed_data.pkl', 'wb') as f:
            pickle.dump(gedi_df, f)

        print(f"Copied splits to output directory: {output_dir}")
        print()

    else:
        # No cache - run full data loading pipeline
        print("Step 1: Querying GEDI data...")
        querier = GEDIQuerier(cache_dir=args.cache_dir)
        gedi_df = querier.query_region_tiles(
            region_bbox=args.region_bbox,
            tile_size=0.1,
            start_time=args.start_time,
            end_time=args.end_time,
            max_agbd=500.0  # Cap at 500 Mg/ha to remove unrealistic outliers (e.g., 3000+)
        )
        print(f"Retrieved {len(gedi_df)} GEDI shots across {gedi_df['tile_id'].nunique()} tiles")
        print()

        if len(gedi_df) == 0:
            print("No GEDI data found in region. Exiting.")
            return

        print("Step 2: Extracting GeoTessera embeddings...")
        extractor = EmbeddingExtractor(
            year=args.embedding_year,
            patch_size=args.patch_size,
            cache_dir=args.cache_dir
        )
        gedi_df = extractor.extract_patches_batch(gedi_df, verbose=True)
        print()

        gedi_df = gedi_df[gedi_df['embedding_patch'].notna()]
        print(f"Retained {len(gedi_df)} shots with valid embeddings")
        print()

        with open(output_dir / 'processed_data.pkl', 'wb') as f:
            pickle.dump(gedi_df, f)

        # Step 3: Spatial split
        print("Step 3: Creating spatial train/val/test split...")
        print(f"Using BufferedSpatialSplitter with buffer_size={args.buffer_size}° (~{args.buffer_size*111:.0f}km)")
        splitter = BufferedSpatialSplitter(
            gedi_df,
            buffer_size=args.buffer_size,
            val_ratio=args.val_ratio,
            test_ratio=args.test_ratio,
            random_state=args.seed
        )
        train_df, val_df, test_df = splitter.split()
        print()

        # Save splits as Parquet to preserve embedding vectors
        # Flatten embeddings to 1D arrays to avoid nested structure issues
        def prepare_for_parquet(df):
            df_copy = df.copy()
            # Flatten (H, W, C) embeddings to 1D for Parquet storage
            df_copy['embedding_patch'] = df_copy['embedding_patch'].apply(
                lambda x: x.flatten().tolist() if x is not None else None
            )
            return df_copy

        prepare_for_parquet(train_df).to_parquet(output_dir / 'train_split.parquet', index=True)
        prepare_for_parquet(val_df).to_parquet(output_dir / 'val_split.parquet', index=True)
        prepare_for_parquet(test_df).to_parquet(output_dir / 'test_split.parquet', index=True)

        print(f"Saved splits to Parquet files with flattened embeddings")

        global_bounds = (
            train_df['longitude'].min(),
            train_df['latitude'].min(),
            train_df['longitude'].max(),
            train_df['latitude'].max()
        )
        print(f"Global bounds: lon [{global_bounds[0]:.4f}, {global_bounds[2]:.4f}], "
              f"lat [{global_bounds[1]:.4f}, {global_bounds[3]:.4f}]")
        print()

        config = vars(args)
        config['global_bounds'] = list(global_bounds)
        save_config(config, output_dir / 'config.json')

        # Save preprocessed data to cache for future reuse
        if CACHE_AVAILABLE:
            try:
                print("Saving preprocessed data to cache for future reuse...")
                from preprocess_data import get_cache_dir, compute_cache_key
                from datetime import datetime

                # Create mock args for cache computation
                class PreprocessArgs:
                    pass
                pargs = PreprocessArgs()
                pargs.region_bbox = args.region_bbox
                pargs.start_time = args.start_time
                pargs.end_time = args.end_time
                pargs.embedding_year = args.embedding_year
                pargs.patch_size = args.patch_size
                pargs.cache_dir = args.cache_dir
                pargs.buffer_size = args.buffer_size
                pargs.val_ratio = args.val_ratio
                pargs.test_ratio = args.test_ratio
                pargs.seed = args.seed
                pargs.train_years = None  # Baselines don't use temporal filtering

                cache_path = get_cache_dir(pargs)
                cache_path.mkdir(parents=True, exist_ok=True)

                # Save the data in the same format as preprocess_data.py
                def prepare_for_parquet(df):
                    df_copy = df.copy()
                    df_copy['embedding_patch'] = df_copy['embedding_patch'].apply(
                        lambda x: x.flatten().tolist() if x is not None else None
                    )
                    return df_copy

                prepare_for_parquet(train_df).to_parquet(cache_path / 'train_split.parquet', index=False)
                prepare_for_parquet(val_df).to_parquet(cache_path / 'val_split.parquet', index=False)
                prepare_for_parquet(test_df).to_parquet(cache_path / 'test_split.parquet', index=False)

                with open(cache_path / 'processed_data.pkl', 'wb') as f:
                    pickle.dump(gedi_df, f)

                # Save metadata
                _, cache_hash, key_params = compute_cache_key(pargs)
                metadata = {
                    'timestamp': datetime.now().isoformat(),
                    'cache_hash': cache_hash,
                    'parameters': key_params,
                    'patch_size': args.patch_size,
                    'global_bounds': list(global_bounds),
                    'n_total': len(gedi_df),
                    'n_train': len(train_df),
                    'n_val': len(val_df),
                    'n_test': len(test_df),
                    'n_tiles': int(gedi_df['tile_id'].nunique()),
                }

                with open(cache_path / 'metadata.json', 'w') as f:
                    json.dump(metadata, f, indent=2)

                print(f"✓ Saved to cache: {cache_path}")
                print("  Future runs with same parameters will load from this cache\n")
            except Exception as e:
                print(f"Warning: Failed to save to cache: {e}")
                print("Continuing with training...\n")

    print("Step 4: Preparing data for baseline models...")

    train_coords, train_embeddings, train_agbd_norm = prepare_data(
        train_df, log_transform=args.log_transform_agbd, agbd_scale=args.agbd_scale
    )
    val_coords, val_embeddings, val_agbd_norm = prepare_data(
        val_df, log_transform=args.log_transform_agbd, agbd_scale=args.agbd_scale
    )
    test_coords, test_embeddings, test_agbd_norm = prepare_data(
        test_df, log_transform=args.log_transform_agbd, agbd_scale=args.agbd_scale
    )

    train_coords = normalize_coords(train_coords, global_bounds)
    val_coords = normalize_coords(val_coords, global_bounds)
    test_coords = normalize_coords(test_coords, global_bounds)

    train_agbd = train_df['agbd'].values
    val_agbd = val_df['agbd'].values
    test_agbd = test_df['agbd'].values

    print(f"Training set: {len(train_coords)} shots")
    print(f"Validation set: {len(val_coords)} shots")
    print(f"Test set: {len(test_coords)} shots")
    print(f"Feature dimensions: coords={train_coords.shape[1]}, embeddings={train_embeddings.shape[1:]}")
    print()

    results = {}

    if 'rf' in args.models:
        model_rf, train_time = train_random_forest(
            train_coords, train_embeddings, train_agbd_norm, args,
            val_coords, val_embeddings, val_agbd_norm
        )

        print("\nEvaluating on validation set...")
        val_metrics, val_pred, _ = evaluate_model(
            model_rf, val_coords, val_embeddings, val_agbd, args.agbd_scale, args.log_transform_agbd
        )
        print(f"Validation - Log R²: {val_metrics['log_r2']:.4f}, Log RMSE: {val_metrics['log_rmse']:.4f}, Log MAE: {val_metrics['log_mae']:.4f}")
        print(f"             Linear RMSE: {val_metrics['linear_rmse']:.2f} Mg/ha, Linear MAE: {val_metrics['linear_mae']:.2f} Mg/ha")
        print_calibration_metrics(val_metrics, prefix="Validation")

        print("\nEvaluating on test set...")
        test_metrics, test_pred, _ = evaluate_model(
            model_rf, test_coords, test_embeddings, test_agbd, args.agbd_scale, args.log_transform_agbd
        )
        print(f"Test - Log R²: {test_metrics['log_r2']:.4f}, Log RMSE: {test_metrics['log_rmse']:.4f}, Log MAE: {test_metrics['log_mae']:.4f}")
        print(f"       Linear RMSE: {test_metrics['linear_rmse']:.2f} Mg/ha, Linear MAE: {test_metrics['linear_mae']:.2f} Mg/ha")
        print_calibration_metrics(test_metrics, prefix="Test")

        results['random_forest'] = {
            'train_time': train_time,
            'val_metrics': val_metrics,
            'test_metrics': test_metrics
        }

        with open(output_dir / 'random_forest.pkl', 'wb') as f:
            pickle.dump(model_rf, f)

    if 'xgb' in args.models:
        model_xgb, train_time = train_xgboost(
            train_coords, train_embeddings, train_agbd_norm, args,
            val_coords, val_embeddings, val_agbd_norm
        )

        print("\nEvaluating on validation set...")
        val_metrics, val_pred, _ = evaluate_model(
            model_xgb, val_coords, val_embeddings, val_agbd, args.agbd_scale, args.log_transform_agbd
        )
        print(f"Validation - Log R²: {val_metrics['log_r2']:.4f}, Log RMSE: {val_metrics['log_rmse']:.4f}, Log MAE: {val_metrics['log_mae']:.4f}")
        print(f"             Linear RMSE: {val_metrics['linear_rmse']:.2f} Mg/ha, Linear MAE: {val_metrics['linear_mae']:.2f} Mg/ha")
        print_calibration_metrics(val_metrics, prefix="Validation")

        print("\nEvaluating on test set...")
        test_metrics, test_pred, _ = evaluate_model(
            model_xgb, test_coords, test_embeddings, test_agbd, args.agbd_scale, args.log_transform_agbd
        )
        print(f"Test - Log R²: {test_metrics['log_r2']:.4f}, Log RMSE: {test_metrics['log_rmse']:.4f}, Log MAE: {test_metrics['log_mae']:.4f}")
        print(f"       Linear RMSE: {test_metrics['linear_rmse']:.2f} Mg/ha, Linear MAE: {test_metrics['linear_mae']:.2f} Mg/ha")
        print_calibration_metrics(test_metrics, prefix="Test")

        results['xgboost'] = {
            'train_time': train_time,
            'val_metrics': val_metrics,
            'test_metrics': test_metrics
        }

        with open(output_dir / 'xgboost.pkl', 'wb') as f:
            pickle.dump(model_xgb, f)

    if 'idw' in args.models:
        model_idw, train_time = train_idw(
            train_coords, train_embeddings, train_agbd_norm, args
        )

        print("\nEvaluating on validation set...")
        val_metrics, val_pred, _ = evaluate_model(
            model_idw, val_coords, val_embeddings, val_agbd, args.agbd_scale, args.log_transform_agbd
        )
        print(f"Validation - Log R²: {val_metrics['log_r2']:.4f}, Log RMSE: {val_metrics['log_rmse']:.4f}, Log MAE: {val_metrics['log_mae']:.4f}")
        print(f"             Linear RMSE: {val_metrics['linear_rmse']:.2f} Mg/ha, Linear MAE: {val_metrics['linear_mae']:.2f} Mg/ha")
        print_calibration_metrics(val_metrics, prefix="Validation")

        print("\nEvaluating on test set...")
        test_metrics, test_pred, _ = evaluate_model(
            model_idw, test_coords, test_embeddings, test_agbd, args.agbd_scale, args.log_transform_agbd
        )
        print(f"Test - Log R²: {test_metrics['log_r2']:.4f}, Log RMSE: {test_metrics['log_rmse']:.4f}, Log MAE: {test_metrics['log_mae']:.4f}")
        print(f"       Linear RMSE: {test_metrics['linear_rmse']:.2f} Mg/ha, Linear MAE: {test_metrics['linear_mae']:.2f} Mg/ha")
        print_calibration_metrics(test_metrics, prefix="Test")

        results['idw'] = {
            'train_time': train_time,
            'val_metrics': val_metrics,
            'test_metrics': test_metrics
        }

        with open(output_dir / 'idw.pkl', 'wb') as f:
            pickle.dump(model_idw, f)

    if 'mlp-dropout' in args.models:
        model_mlp_dropout, train_time = train_mlp_dropout(
            train_coords, train_embeddings, train_agbd_norm, args,
            val_coords, val_embeddings, val_agbd_norm
        )

        print("\nEvaluating on validation set...")
        val_metrics, val_pred, _ = evaluate_model(
            model_mlp_dropout, val_coords, val_embeddings, val_agbd, args.agbd_scale, args.log_transform_agbd
        )
        print(f"Validation - Log R²: {val_metrics['log_r2']:.4f}, Log RMSE: {val_metrics['log_rmse']:.4f}, Log MAE: {val_metrics['log_mae']:.4f}")
        print(f"             Linear RMSE: {val_metrics['linear_rmse']:.2f} Mg/ha, Linear MAE: {val_metrics['linear_mae']:.2f} Mg/ha")
        print_calibration_metrics(val_metrics, prefix="Validation")

        print("\nEvaluating on test set...")
        test_metrics, test_pred, _ = evaluate_model(
            model_mlp_dropout, test_coords, test_embeddings, test_agbd, args.agbd_scale, args.log_transform_agbd
        )
        print(f"Test - Log R²: {test_metrics['log_r2']:.4f}, Log RMSE: {test_metrics['log_rmse']:.4f}, Log MAE: {test_metrics['log_mae']:.4f}")
        print(f"       Linear RMSE: {test_metrics['linear_rmse']:.2f} Mg/ha, Linear MAE: {test_metrics['linear_mae']:.2f} Mg/ha")
        print_calibration_metrics(test_metrics, prefix="Test")

        results['mlp_dropout'] = {
            'train_time': train_time,
            'val_metrics': val_metrics,
            'test_metrics': test_metrics
        }

        with open(output_dir / 'mlp_dropout.pkl', 'wb') as f:
            pickle.dump(model_mlp_dropout, f)

    print("\n" + "=" * 80)
    print("SUMMARY OF RESULTS")
    print("=" * 80)

    summary_table = []
    for model_name, model_results in results.items():
        row = {
            'Model': model_name.upper(),
            'Train Time (s)': f"{model_results['train_time']:.2f}",
            'Val Log R²': f"{model_results['val_metrics']['log_r2']:.4f}",
            'Val Log RMSE': f"{model_results['val_metrics']['log_rmse']:.4f}",
            'Val Log MAE': f"{model_results['val_metrics']['log_mae']:.4f}",
            'Val RMSE (Mg/ha)': f"{model_results['val_metrics']['linear_rmse']:.2f}",
            'Val MAE (Mg/ha)': f"{model_results['val_metrics']['linear_mae']:.2f}",
            'Test Log R²': f"{model_results['test_metrics']['log_r2']:.4f}",
            'Test Log RMSE': f"{model_results['test_metrics']['log_rmse']:.4f}",
            'Test Log MAE': f"{model_results['test_metrics']['log_mae']:.4f}",
            'Test RMSE (Mg/ha)': f"{model_results['test_metrics']['linear_rmse']:.2f}",
            'Test MAE (Mg/ha)': f"{model_results['test_metrics']['linear_mae']:.2f}",
        }
        summary_table.append(row)

    if summary_table:
        df_summary = pd.DataFrame(summary_table)
        print(df_summary.to_string(index=False))

    # Print calibration summary
    print("\n" + "=" * 80)
    print("CALIBRATION SUMMARY (Test Set)")
    print("=" * 80)
    print(f"{'Model':<20} {'Z-score μ':>12} {'Z-score σ':>12} {'1σ Cov%':>10} {'2σ Cov%':>10} {'3σ Cov%':>10}")
    print(f"{'':20} {'(ideal: 0)':>12} {'(ideal: 1)':>12} {'(68.3%)':>10} {'(95.4%)':>10} {'(99.7%)':>10}")
    print("-" * 80)
    for model_name, model_results in results.items():
        m = model_results['test_metrics']
        print(f"{model_name.upper():<20} {m['z_mean']:>+12.4f} {m['z_std']:>12.4f} "
              f"{m['coverage_1sigma']:>10.1f} {m['coverage_2sigma']:>10.1f} {m['coverage_3sigma']:>10.1f}")

    print("=" * 80)

    with open(output_dir / 'results.json', 'w') as f:
        json.dump(_make_serializable(results), f, indent=2)

    print(f"\nResults saved to: {output_dir}")
    print("Files:")
    print("  - config.json")
    print("  - results.json")
    for model in args.models:
        if model == 'rf':
            print("  - random_forest.pkl")
        elif model == 'xgb':
            print("  - xgboost.pkl")
        elif model == 'idw':
            print("  - idw.pkl")
        elif model == 'mlp-dropout':
            print("  - mlp_dropout.pkl")
    print("=" * 80)

if __name__ == '__main__':
    main()
