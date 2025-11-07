"""
Training script for tree-based baseline models (XGBoost, Random Forest).

These models use the raw flattened embeddings + lat/lon features and train
much faster than neural networks, providing a quick baseline comparison.
"""

import argparse
import json
from pathlib import Path
import pickle
import time

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tqdm import tqdm

try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False
    print("Warning: XGBoost not installed. Only Random Forest will be available.")
    print("Install with: pip install xgboost")

from data.gedi import GEDIQuerier
from data.embeddings import EmbeddingExtractor
from data.spatial_cv import SpatialTileSplitter


def extract_features(df, normalize_coords=True, flatten_embeddings=True):
    """
    Extract features from dataframe for tree-based models.

    Args:
        df: DataFrame with embedding_patch, latitude, longitude, tile_id
        normalize_coords: Normalize coordinates to [0, 1] within each tile
        flatten_embeddings: Flatten the embedding patches

    Returns:
        X: Feature matrix (n_samples, n_features)
        coords_raw: Raw coordinates for reference
    """
    n_samples = len(df)

    # Get coordinates
    coords = df[['longitude', 'latitude']].values.astype(np.float32)
    coords_raw = coords.copy()

    # Normalize coordinates within tiles if requested
    if normalize_coords:
        coords_normalized = coords.copy()
        for tile_id in df['tile_id'].unique():
            tile_mask = df['tile_id'] == tile_id
            tile_coords = coords[tile_mask]

            lon_min, lon_max = tile_coords[:, 0].min(), tile_coords[:, 0].max()
            lat_min, lat_max = tile_coords[:, 1].min(), tile_coords[:, 1].max()

            lon_range = lon_max - lon_min if lon_max > lon_min else 1.0
            lat_range = lat_max - lat_min if lat_max > lat_min else 1.0

            coords_normalized[tile_mask, 0] = (tile_coords[:, 0] - lon_min) / lon_range
            coords_normalized[tile_mask, 1] = (tile_coords[:, 1] - lat_min) / lat_range

        coords = coords_normalized

    # Get embeddings
    embeddings = np.stack(df['embedding_patch'].values)  # (n, patch_size, patch_size, channels)

    if flatten_embeddings:
        # Flatten embeddings: (n, patch_size, patch_size, channels) -> (n, patch_size*patch_size*channels)
        embeddings_flat = embeddings.reshape(n_samples, -1)
        print(f"Flattened embeddings shape: {embeddings_flat.shape}")
    else:
        # Use mean-pooled embeddings: (n, patch_size, patch_size, channels) -> (n, channels)
        embeddings_flat = embeddings.mean(axis=(1, 2))
        print(f"Mean-pooled embeddings shape: {embeddings_flat.shape}")

    # Concatenate features: [embeddings, lon, lat]
    X = np.concatenate([embeddings_flat, coords], axis=1)

    print(f"Feature matrix shape: {X.shape} ({X.shape[1]} features)")

    return X, coords_raw


def extract_targets(df, log_transform=True, scale=200.0):
    """Extract and transform target values."""
    agbd = df['agbd'].values.astype(np.float32)

    if log_transform:
        agbd = np.log1p(agbd) / np.log1p(scale)

    return agbd


def compute_metrics(y_pred, y_true):
    """Compute evaluation metrics."""
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    return {
        'mse': float(mse),
        'rmse': float(rmse),
        'mae': float(mae),
        'r2': float(r2)
    }


def train_xgboost(X_train, y_train, X_val, y_val, args):
    """Train XGBoost model."""
    if not HAS_XGBOOST:
        raise ImportError("XGBoost is not installed. Install with: pip install xgboost")

    print("\nTraining XGBoost model...")
    print(f"Parameters: n_estimators={args.n_estimators}, max_depth={args.max_depth}, "
          f"learning_rate={args.learning_rate}")

    model = xgb.XGBRegressor(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        learning_rate=args.learning_rate,
        subsample=args.subsample,
        colsample_bytree=args.colsample_bytree,
        random_state=args.seed,
        n_jobs=args.n_jobs,
        early_stopping_rounds=args.early_stopping_rounds,
        eval_metric='rmse'
    )

    start_time = time.time()

    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=args.verbose
    )

    train_time = time.time() - start_time
    print(f"Training completed in {train_time:.2f} seconds")

    return model, train_time


def train_random_forest(X_train, y_train, args):
    """Train Random Forest model."""
    print("\nTraining Random Forest model...")
    print(f"Parameters: n_estimators={args.n_estimators}, max_depth={args.max_depth}")

    model = RandomForestRegressor(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth if args.max_depth > 0 else None,
        min_samples_split=args.min_samples_split,
        min_samples_leaf=args.min_samples_leaf,
        random_state=args.seed,
        n_jobs=args.n_jobs,
        verbose=args.verbose
    )

    start_time = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - start_time

    print(f"Training completed in {train_time:.2f} seconds")

    return model, train_time


def parse_args():
    parser = argparse.ArgumentParser(description='Train tree-based baseline models')

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
    parser.add_argument('--model_type', type=str, default='xgboost',
                        choices=['xgboost', 'random_forest', 'both'],
                        help='Type of tree model to train')
    parser.add_argument('--n_estimators', type=int, default=100,
                        help='Number of trees/estimators')
    parser.add_argument('--max_depth', type=int, default=10,
                        help='Maximum tree depth (0 for unlimited in RF)')
    parser.add_argument('--learning_rate', type=float, default=0.1,
                        help='Learning rate for XGBoost')
    parser.add_argument('--subsample', type=float, default=0.8,
                        help='Subsample ratio for XGBoost')
    parser.add_argument('--colsample_bytree', type=float, default=0.8,
                        help='Column subsample ratio for XGBoost')
    parser.add_argument('--min_samples_split', type=int, default=2,
                        help='Min samples to split for Random Forest')
    parser.add_argument('--min_samples_leaf', type=int, default=1,
                        help='Min samples in leaf for Random Forest')
    parser.add_argument('--early_stopping_rounds', type=int, default=10,
                        help='Early stopping rounds for XGBoost')

    # Feature arguments
    parser.add_argument('--flatten_embeddings', action='store_true', default=True,
                        help='Flatten embedding patches (vs mean pooling)')
    parser.add_argument('--normalize_coords', action='store_true', default=True,
                        help='Normalize coordinates within tiles')

    # Training arguments
    parser.add_argument('--val_ratio', type=float, default=0.15,
                        help='Validation set ratio')
    parser.add_argument('--test_ratio', type=float, default=0.15,
                        help='Test set ratio')
    parser.add_argument('--log_transform_agbd', action='store_true', default=True,
                        help='Apply log transform to AGBD')

    # Output arguments
    parser.add_argument('--output_dir', type=str, default='./outputs_tree',
                        help='Output directory for models and logs')

    # Other
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--n_jobs', type=int, default=-1,
                        help='Number of parallel jobs (-1 for all cores)')
    parser.add_argument('--verbose', type=int, default=1,
                        help='Verbosity level')

    return parser.parse_args()


def main():
    args = parse_args()
    np.random.seed(args.seed)

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save config
    with open(output_dir / 'config.json', 'w') as f:
        json.dump(vars(args), f, indent=2)

    print("=" * 80)
    print(f"Tree-Based Baseline Training ({args.model_type})")
    print("=" * 80)
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
        patch_size=3,
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

    # Step 3: Spatial split
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

    # Step 4: Extract features
    print("Step 4: Extracting features...")
    print("\nTrain set:")
    X_train, coords_train = extract_features(
        train_df,
        normalize_coords=args.normalize_coords,
        flatten_embeddings=args.flatten_embeddings
    )
    y_train = extract_targets(train_df, log_transform=args.log_transform_agbd)

    print("\nValidation set:")
    X_val, coords_val = extract_features(
        val_df,
        normalize_coords=args.normalize_coords,
        flatten_embeddings=args.flatten_embeddings
    )
    y_val = extract_targets(val_df, log_transform=args.log_transform_agbd)

    print("\nTest set:")
    X_test, coords_test = extract_features(
        test_df,
        normalize_coords=args.normalize_coords,
        flatten_embeddings=args.flatten_embeddings
    )
    y_test = extract_targets(test_df, log_transform=args.log_transform_agbd)
    print()

    # Save features for later use
    np.savez(
        output_dir / 'features.npz',
        X_train=X_train, y_train=y_train,
        X_val=X_val, y_val=y_val,
        X_test=X_test, y_test=y_test
    )

    # Step 5: Train models
    results = {}

    if args.model_type in ['xgboost', 'both']:
        if HAS_XGBOOST:
            print("=" * 80)
            print("Training XGBoost")
            print("=" * 80)

            xgb_model, xgb_train_time = train_xgboost(X_train, y_train, X_val, y_val, args)

            # Evaluate
            print("\nEvaluating XGBoost...")
            y_train_pred = xgb_model.predict(X_train)
            y_val_pred = xgb_model.predict(X_val)
            y_test_pred = xgb_model.predict(X_test)

            train_metrics = compute_metrics(y_train_pred, y_train)
            val_metrics = compute_metrics(y_val_pred, y_val)
            test_metrics = compute_metrics(y_test_pred, y_test)

            print(f"\nXGBoost Results:")
            print(f"Train - RMSE: {train_metrics['rmse']:.4f}, MAE: {train_metrics['mae']:.4f}, R²: {train_metrics['r2']:.4f}")
            print(f"Val   - RMSE: {val_metrics['rmse']:.4f}, MAE: {val_metrics['mae']:.4f}, R²: {val_metrics['r2']:.4f}")
            print(f"Test  - RMSE: {test_metrics['rmse']:.4f}, MAE: {test_metrics['mae']:.4f}, R²: {test_metrics['r2']:.4f}")

            # Save model
            xgb_model.save_model(str(output_dir / 'xgboost_model.json'))

            results['xgboost'] = {
                'train': train_metrics,
                'val': val_metrics,
                'test': test_metrics,
                'train_time': xgb_train_time
            }
        else:
            print("Skipping XGBoost (not installed)")

    if args.model_type in ['random_forest', 'both']:
        print("\n" + "=" * 80)
        print("Training Random Forest")
        print("=" * 80)

        rf_model, rf_train_time = train_random_forest(X_train, y_train, args)

        # Evaluate
        print("\nEvaluating Random Forest...")
        y_train_pred = rf_model.predict(X_train)
        y_val_pred = rf_model.predict(X_val)
        y_test_pred = rf_model.predict(X_test)

        train_metrics = compute_metrics(y_train_pred, y_train)
        val_metrics = compute_metrics(y_val_pred, y_val)
        test_metrics = compute_metrics(y_test_pred, y_test)

        print(f"\nRandom Forest Results:")
        print(f"Train - RMSE: {train_metrics['rmse']:.4f}, MAE: {train_metrics['mae']:.4f}, R²: {train_metrics['r2']:.4f}")
        print(f"Val   - RMSE: {val_metrics['rmse']:.4f}, MAE: {val_metrics['mae']:.4f}, R²: {val_metrics['r2']:.4f}")
        print(f"Test  - RMSE: {test_metrics['rmse']:.4f}, MAE: {test_metrics['mae']:.4f}, R²: {test_metrics['r2']:.4f}")

        # Save model
        with open(output_dir / 'random_forest_model.pkl', 'wb') as f:
            pickle.dump(rf_model, f)

        results['random_forest'] = {
            'train': train_metrics,
            'val': val_metrics,
            'test': test_metrics,
            'train_time': rf_train_time
        }

    # Save results
    with open(output_dir / 'results.json', 'w') as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 80)
    print("Training complete!")
    print(f"Results saved to: {output_dir}")
    print("=" * 80)


if __name__ == '__main__':
    main()
