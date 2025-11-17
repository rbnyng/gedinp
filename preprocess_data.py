#!/usr/bin/env python3
"""
Data Preprocessing Script with Caching

This script performs the expensive data processing pipeline (GEDI querying,
embedding extraction, spatial splitting) and caches the results for reuse
across multiple training runs with the same data configuration.

Usage:
    python preprocess_data.py \\
        --region_bbox -70 44 -69 45 \\
        --start_time 2022-01-01 \\
        --end_time 2022-12-31 \\
        --embedding_year 2022 \\
        --cache_dir ./cache \\
        --seed 42

This is automatically called by the training harness, but can also be run
independently to pre-cache data for multiple experiments.
"""

import argparse
import hashlib
import json
import pickle
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd

from data.gedi import GEDIQuerier
from data.embeddings import EmbeddingExtractor
from data.spatial_cv import BufferedSpatialSplitter
from utils.config import save_config


def parse_args():
    parser = argparse.ArgumentParser(
        description='Preprocess and cache GEDI data with embeddings and spatial splits',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    # Data arguments
    parser.add_argument('--region_bbox', type=float, nargs=4, required=True,
                        help='Region bounding box: min_lon min_lat max_lon max_lat')
    parser.add_argument('--start_time', type=str, default='2022-01-01',
                        help='Start date for GEDI data (YYYY-MM-DD)')
    parser.add_argument('--end_time', type=str, default='2022-12-31',
                        help='End date for GEDI data (YYYY-MM-DD)')
    parser.add_argument('--embedding_year', type=int, default=2022,
                        help='Year of GeoTessera embeddings')
    parser.add_argument('--patch_size', type=int, default=3,
                        help='Embedding patch size (default: 3x3)')
    parser.add_argument('--cache_dir', type=str, required=True,
                        help='Directory for caching tiles and embeddings')

    # Spatial split arguments
    parser.add_argument('--buffer_size', type=float, default=0.1,
                        help='Buffer size in degrees for spatial CV (default: 0.1)')
    parser.add_argument('--val_ratio', type=float, default=0.15,
                        help='Validation set ratio (default: 0.15)')
    parser.add_argument('--test_ratio', type=float, default=0.15,
                        help='Test set ratio (default: 0.15)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')

    # Optional temporal filtering
    parser.add_argument('--train_years', type=int, nargs='+', default=None,
                        help='Filter to specific years (e.g., 2022 2023)')

    # Control arguments
    parser.add_argument('--force', action='store_true',
                        help='Force reprocessing even if cache exists')
    parser.add_argument('--verbose', action='store_true', default=True,
                        help='Verbose output')

    return parser.parse_args()


def compute_cache_key(args):
    """
    Compute a deterministic cache key based on preprocessing parameters.

    Returns:
        cache_key: String identifier for this preprocessing configuration
        cache_hash: Short hash for directory naming
    """
    # Create a deterministic dictionary of parameters that affect the output
    key_params = {
        'region_bbox': args.region_bbox,
        'start_time': args.start_time,
        'end_time': args.end_time,
        'embedding_year': args.embedding_year,
        'patch_size': args.patch_size,
        'buffer_size': args.buffer_size,
        'val_ratio': args.val_ratio,
        'test_ratio': args.test_ratio,
        'seed': args.seed,
        'train_years': args.train_years,
    }

    # Create a stable JSON string
    key_str = json.dumps(key_params, sort_keys=True)

    # Compute hash
    cache_hash = hashlib.md5(key_str.encode()).hexdigest()[:12]

    # Create human-readable cache key
    bbox_str = '_'.join(f"{x:.2f}" for x in args.region_bbox)
    cache_key = f"region_{bbox_str}_seed_{args.seed}_year_{args.embedding_year}_buf_{args.buffer_size}"

    return cache_key, cache_hash, key_params


def get_cache_dir(args):
    """Get the cache directory path for this preprocessing configuration."""
    cache_key, cache_hash, _ = compute_cache_key(args)
    cache_path = Path(args.cache_dir) / 'preprocessed' / f"{cache_hash}_{cache_key}"
    return cache_path


def check_cache_exists(args):
    """Check if preprocessed data exists and is valid."""
    cache_path = get_cache_dir(args)

    if not cache_path.exists():
        return False, cache_path

    # Check for required files
    required_files = [
        'metadata.json',
        'train_split.parquet',
        'val_split.parquet',
        'test_split.parquet',
        'processed_data.pkl'
    ]

    for fname in required_files:
        if not (cache_path / fname).exists():
            return False, cache_path

    return True, cache_path


def load_cached_data(cache_path):
    """Load preprocessed data from cache."""
    print(f"Loading cached preprocessed data from: {cache_path}")

    # Load metadata
    with open(cache_path / 'metadata.json', 'r') as f:
        metadata = json.load(f)

    # Load full processed data (with embeddings as objects)
    with open(cache_path / 'processed_data.pkl', 'rb') as f:
        gedi_df = pickle.load(f)

    # Load splits (need to reshape embeddings from flattened form)
    train_df = pd.read_parquet(cache_path / 'train_split.parquet')
    val_df = pd.read_parquet(cache_path / 'val_split.parquet')
    test_df = pd.read_parquet(cache_path / 'test_split.parquet')

    # Reshape embeddings from 1D back to (H, W, C)
    patch_size = metadata['patch_size']
    embedding_dim = 128  # GeoTessera embedding dimension

    def reshape_embedding(x):
        if x is None:
            return None
        arr = np.array(x)
        return arr.reshape(patch_size, patch_size, embedding_dim)

    train_df['embedding_patch'] = train_df['embedding_patch'].apply(reshape_embedding)
    val_df['embedding_patch'] = val_df['embedding_patch'].apply(reshape_embedding)
    test_df['embedding_patch'] = test_df['embedding_patch'].apply(reshape_embedding)

    global_bounds = tuple(metadata['global_bounds'])

    print(f"Loaded {len(gedi_df)} total shots")
    print(f"  Train: {len(train_df)} shots")
    print(f"  Val: {len(val_df)} shots")
    print(f"  Test: {len(test_df)} shots")
    print(f"Cached on: {metadata['timestamp']}")

    return gedi_df, train_df, val_df, test_df, global_bounds, metadata


def preprocess_and_cache(args):
    """Run the full preprocessing pipeline and cache results."""
    cache_path = get_cache_dir(args)
    cache_path.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("DATA PREPROCESSING PIPELINE")
    print("=" * 80)
    print(f"Region: {args.region_bbox}")
    print(f"Time range: {args.start_time} to {args.end_time}")
    print(f"Embedding year: {args.embedding_year}")
    print(f"Patch size: {args.patch_size}x{args.patch_size}")
    print(f"Buffer size: {args.buffer_size}°")
    print(f"Seed: {args.seed}")
    print(f"Cache location: {cache_path}")
    print("=" * 80)
    print()

    # Step 1: Query GEDI data
    print("Step 1: Querying GEDI data...")
    querier = GEDIQuerier(cache_dir=args.cache_dir)
    gedi_df = querier.query_region_tiles(
        region_bbox=args.region_bbox,
        tile_size=0.1,
        start_time=args.start_time,
        end_time=args.end_time,
        verbose=args.verbose
    )
    print(f"Found {len(gedi_df)} GEDI shots across {gedi_df['tile_id'].nunique()} tiles")
    print()

    if len(gedi_df) == 0:
        print("ERROR: No GEDI data found in region. Cannot proceed.")
        return None

    # Optional temporal filtering
    if args.train_years is not None:
        print(f"Filtering to training years: {args.train_years}")

        # Find timestamp column
        if 'timestamp' in gedi_df.columns:
            gedi_df['year'] = pd.to_datetime(gedi_df['timestamp']).dt.year
        elif hasattr(gedi_df.index, 'year'):
            gedi_df['year'] = gedi_df.index.year
        else:
            try:
                gedi_df['year'] = pd.to_datetime(gedi_df.index).year
            except:
                print("Warning: Could not determine year for temporal filtering")
                args.train_years = None

        if args.train_years is not None:
            n_before = len(gedi_df)
            gedi_df = gedi_df[gedi_df['year'].isin(args.train_years)]
            n_after = len(gedi_df)
            print(f"Filtered from {n_before} to {n_after} shots ({n_after/n_before*100:.1f}% retained)")
            print(f"Shots per year: {dict(gedi_df['year'].value_counts().sort_index())}")
            print()

    # Step 2: Extract embeddings
    print("Step 2: Extracting GeoTessera embeddings...")
    extractor = EmbeddingExtractor(
        year=args.embedding_year,
        patch_size=args.patch_size,
        cache_dir=args.cache_dir
    )
    gedi_df = extractor.extract_patches_batch(gedi_df, verbose=args.verbose)
    print()

    gedi_df = gedi_df[gedi_df['embedding_patch'].notna()]
    print(f"Retained {len(gedi_df)} shots with valid embeddings")
    print()

    # Step 3: Create spatial splits
    print("Step 3: Creating spatial train/val/test split...")
    print(f"Using BufferedSpatialSplitter with buffer_size={args.buffer_size}° "
          f"(~{args.buffer_size*111:.0f}km)")

    splitter = BufferedSpatialSplitter(
        gedi_df,
        buffer_size=args.buffer_size,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        random_state=args.seed
    )
    train_df, val_df, test_df = splitter.split()
    print()

    # Compute global bounds from training data
    global_bounds = (
        train_df['longitude'].min(),
        train_df['latitude'].min(),
        train_df['longitude'].max(),
        train_df['latitude'].max()
    )
    print(f"Global bounds: lon [{global_bounds[0]:.4f}, {global_bounds[2]:.4f}], "
          f"lat [{global_bounds[1]:.4f}, {global_bounds[3]:.4f}]")
    print()

    # Step 4: Save to cache
    print("Step 4: Saving to cache...")

    # Save full processed data as pickle (preserves numpy arrays as objects)
    with open(cache_path / 'processed_data.pkl', 'wb') as f:
        pickle.dump(gedi_df, f)

    # Save splits as Parquet (flatten embeddings for storage)
    def prepare_for_parquet(df):
        df_copy = df.copy()
        df_copy['embedding_patch'] = df_copy['embedding_patch'].apply(
            lambda x: x.flatten().tolist() if x is not None else None
        )
        return df_copy

    prepare_for_parquet(train_df).to_parquet(cache_path / 'train_split.parquet', index=False)
    prepare_for_parquet(val_df).to_parquet(cache_path / 'val_split.parquet', index=False)
    prepare_for_parquet(test_df).to_parquet(cache_path / 'test_split.parquet', index=False)

    # Save metadata
    _, cache_hash, key_params = compute_cache_key(args)
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

    print(f"Saved preprocessed data to: {cache_path}")
    print(f"  - processed_data.pkl ({len(gedi_df)} shots)")
    print(f"  - train_split.parquet ({len(train_df)} shots)")
    print(f"  - val_split.parquet ({len(val_df)} shots)")
    print(f"  - test_split.parquet ({len(test_df)} shots)")
    print(f"  - metadata.json")
    print()

    return gedi_df, train_df, val_df, test_df, global_bounds, metadata


def main():
    args = parse_args()

    # Check if cache exists
    cache_exists, cache_path = check_cache_exists(args)

    if cache_exists and not args.force:
        print("=" * 80)
        print("CACHED DATA FOUND")
        print("=" * 80)
        print(f"Cache location: {cache_path}")
        print("Use --force to reprocess")
        print("=" * 80)
        print()

        # Load and validate cache
        gedi_df, train_df, val_df, test_df, global_bounds, metadata = load_cached_data(cache_path)

        print("Cache validation successful!")
        return

    if args.force and cache_exists:
        print(f"Force flag set - will reprocess data")
        print()

    # Run preprocessing pipeline
    result = preprocess_and_cache(args)

    if result is None:
        print("ERROR: Preprocessing failed")
        return 1

    gedi_df, train_df, val_df, test_df, global_bounds, metadata = result

    print("=" * 80)
    print("PREPROCESSING COMPLETE")
    print("=" * 80)
    print(f"Total shots: {len(gedi_df)}")
    print(f"Train: {len(train_df)} ({len(train_df)/len(gedi_df)*100:.1f}%)")
    print(f"Val: {len(val_df)} ({len(val_df)/len(gedi_df)*100:.1f}%)")
    print(f"Test: {len(test_df)} ({len(test_df)/len(gedi_df)*100:.1f}%)")
    print(f"Cache: {cache_path}")
    print("=" * 80)

    return 0


if __name__ == '__main__':
    exit(main())
