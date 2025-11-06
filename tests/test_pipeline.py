"""
Comprehensive test suite for the data processing pipeline.

Tests each component step-by-step with visual validation.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path

# Import our modules
from data.gedi import GEDIQuerier, get_gedi_statistics
from data.embeddings import EmbeddingExtractor
from data.dataset import GEDINeuralProcessDataset
from data.spatial_cv import SpatialTileSplitter


# Test region: Zambia (known to have GEDI data)
TEST_REGION = {
    'name': 'Zambia Forest',
    'bbox': (30.256, -15.853, 30.422, -15.625),  # Small region for testing
    'tile_lon': 30.35,
    'tile_lat': -15.75
}


def print_section(title):
    """Print formatted section header."""
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80 + "\n")


def test_1_gedi_query():
    """Test 1: GEDI data querying."""
    print_section("TEST 1: GEDI Data Query")

    try:
        querier = GEDIQuerier()
        print("✓ GEDIQuerier initialized")
    except Exception as e:
        print(f"✗ Failed to initialize GEDIQuerier: {e}")
        return None

    # Test single tile query
    print(f"\nQuerying tile at ({TEST_REGION['tile_lon']}, {TEST_REGION['tile_lat']})...")

    try:
        gedi_df = querier.query_tile(
            tile_lon=TEST_REGION['tile_lon'],
            tile_lat=TEST_REGION['tile_lat'],
            tile_size=0.1,
            start_time="2019-01-01",
            end_time="2023-12-31"
        )
        print(f"✓ Query successful: {len(gedi_df)} shots returned")
    except Exception as e:
        print(f"✗ Query failed: {e}")
        print("\nTrying full region query instead...")
        try:
            gedi_df = querier.query_bbox(
                bbox=TEST_REGION['bbox'],
                start_time="2019-01-01",
                end_time="2023-12-31"
            )
            print(f"✓ Region query successful: {len(gedi_df)} shots returned")
        except Exception as e2:
            print(f"✗ Region query also failed: {e2}")
            return None

    if len(gedi_df) == 0:
        print("⚠ Warning: No GEDI data found in region")
        return None

    # Inspect data
    print("\n--- DataFrame Info ---")
    print(f"Shape: {gedi_df.shape}")
    print(f"Columns: {list(gedi_df.columns)}")
    print(f"\nFirst few rows:")
    print(gedi_df.head())

    print(f"\n--- Data Types ---")
    print(gedi_df.dtypes)

    print(f"\n--- Missing Values ---")
    print(gedi_df.isnull().sum())

    # Check for AGBD column
    agbd_columns = [col for col in gedi_df.columns if 'agbd' in col.lower()]
    print(f"\n--- AGBD Columns Found ---")
    print(agbd_columns)

    if 'agbd' in gedi_df.columns:
        print(f"\n--- AGBD Statistics ---")
        print(gedi_df['agbd'].describe())
        print(f"✓ AGBD column exists and has data")
    else:
        print(f"⚠ Warning: 'agbd' column not found. Available columns: {list(gedi_df.columns)}")

    # Spatial extent
    print(f"\n--- Spatial Extent ---")
    print(f"Longitude range: [{gedi_df['longitude'].min():.4f}, {gedi_df['longitude'].max():.4f}]")
    print(f"Latitude range: [{gedi_df['latitude'].min():.4f}, {gedi_df['latitude'].max():.4f}]")

    # Get statistics
    stats = get_gedi_statistics(gedi_df)
    print(f"\n--- Statistics ---")
    for key, value in stats.items():
        print(f"{key}: {value}")

    return gedi_df


def test_2_tile_coordinates():
    """Test 2: Tile coordinate calculations."""
    print_section("TEST 2: Tile Coordinate Math")

    extractor = EmbeddingExtractor(year=2024, patch_size=3)

    # Test various coordinates
    test_coords = [
        (30.256, -15.853),
        (30.35, -15.75),
        (0.0, 0.0),
        (0.123, 52.456),
    ]

    print("Testing tile center calculations:")
    print(f"{'Input Lon':<12} {'Input Lat':<12} {'Tile Lon':<12} {'Tile Lat':<12}")
    print("-" * 50)

    for lon, lat in test_coords:
        tile_lon, tile_lat = extractor._get_tile_coords(lon, lat)
        print(f"{lon:<12.4f} {lat:<12.4f} {tile_lon:<12.4f} {tile_lat:<12.4f}")

    # Verify tile size
    print("\nVerifying 0.1° tile size:")
    tile_lon, tile_lat = extractor._get_tile_coords(TEST_REGION['tile_lon'], TEST_REGION['tile_lat'])
    print(f"Tile center: ({tile_lon}, {tile_lat})")
    print(f"Tile bounds: [{tile_lon-0.05}, {tile_lon+0.05}] x [{tile_lat-0.05}, {tile_lat+0.05}]")
    print(f"Tile size: ~{0.1 * 111:.1f} km x ~{0.1 * 111:.1f} km (at equator)")

    return tile_lon, tile_lat


def test_3_embedding_extraction(tile_lon, tile_lat):
    """Test 3: GeoTessera embedding extraction."""
    print_section("TEST 3: Embedding Extraction")

    print(f"Fetching GeoTessera tile at ({tile_lon}, {tile_lat})...")

    try:
        extractor = EmbeddingExtractor(
            year=2024,
            patch_size=3,
            cache_dir='./test_cache'
        )
        print("✓ EmbeddingExtractor initialized")
    except Exception as e:
        print(f"✗ Failed to initialize: {e}")
        return None

    try:
        tile_data = extractor._load_tile(tile_lon, tile_lat)

        if tile_data is None:
            print("✗ Failed to load tile (returned None)")
            return None

        embedding, crs, transform = tile_data
        print("✓ Tile loaded successfully")

    except Exception as e:
        print(f"✗ Tile loading failed: {e}")
        import traceback
        traceback.print_exc()
        return None

    # Inspect embedding
    print(f"\n--- Embedding Info ---")
    print(f"Shape: {embedding.shape}")
    print(f"Dtype: {embedding.dtype}")
    print(f"Value range: [{embedding.min():.4f}, {embedding.max():.4f}]")
    print(f"Mean: {embedding.mean():.4f}")
    print(f"Std: {embedding.std():.4f}")

    print(f"\n--- CRS Info ---")
    print(f"CRS: {crs}")

    print(f"\n--- Transform Info ---")
    print(f"Transform: {transform}")

    # Check for NaN/Inf
    print(f"\n--- Data Quality ---")
    print(f"NaN values: {np.isnan(embedding).sum()}")
    print(f"Inf values: {np.isinf(embedding).sum()}")

    # Test coordinate conversion
    print(f"\n--- Testing Coordinate Conversion ---")
    test_lon, test_lat = tile_lon, tile_lat  # Center of tile
    try:
        row, col = extractor._lonlat_to_pixel(test_lon, test_lat, transform, crs)
        print(f"Tile center ({test_lon}, {test_lat}) → pixel ({row}, {col})")
        print(f"Expected center: ~({embedding.shape[0]//2}, {embedding.shape[1]//2})")

        # Check if reasonable
        if 0 <= row < embedding.shape[0] and 0 <= col < embedding.shape[1]:
            print("✓ Coordinate conversion looks reasonable")
        else:
            print(f"⚠ Warning: Converted coordinates out of bounds")
    except Exception as e:
        print(f"✗ Coordinate conversion failed: {e}")

    return extractor, embedding, crs, transform


def test_4_patch_extraction(extractor, gedi_df):
    """Test 4: Extract patches for GEDI shots."""
    print_section("TEST 4: Patch Extraction")

    if gedi_df is None or len(gedi_df) == 0:
        print("⚠ No GEDI data available, skipping patch extraction")
        return None

    # Test on first 5 shots
    n_test = min(5, len(gedi_df))
    print(f"Testing patch extraction on {n_test} GEDI shots...\n")

    results = []
    for idx in range(n_test):
        row = gedi_df.iloc[idx]
        lon, lat = row['longitude'], row['latitude']

        print(f"Shot {idx+1}: ({lon:.4f}, {lat:.4f})")

        try:
            patch = extractor.extract_patch(lon, lat)

            if patch is not None:
                print(f"  ✓ Extracted patch shape: {patch.shape}")
                print(f"  Value range: [{patch.min():.4f}, {patch.max():.4f}]")
                results.append({'success': True, 'patch': patch, 'lon': lon, 'lat': lat})
            else:
                print(f"  ✗ Patch extraction returned None")
                results.append({'success': False, 'lon': lon, 'lat': lat})
        except Exception as e:
            print(f"  ✗ Error: {e}")
            results.append({'success': False, 'lon': lon, 'lat': lat})

    success_count = sum(1 for r in results if r['success'])
    print(f"\n--- Summary ---")
    print(f"Successful extractions: {success_count}/{n_test} ({100*success_count/n_test:.1f}%)")

    return results


def test_5_spatial_alignment(extractor, gedi_df, embedding):
    """Test 5: Visualize spatial alignment."""
    print_section("TEST 5: Spatial Alignment Visualization")

    if gedi_df is None or len(gedi_df) == 0:
        print("⚠ No GEDI data available for visualization")
        return

    # Extract patches for all shots
    print("Extracting patches for all GEDI shots...")
    gedi_df_with_patches = extractor.extract_patches_batch(gedi_df.copy(), verbose=True)

    # Filter successful extractions
    valid_df = gedi_df_with_patches[gedi_df_with_patches['embedding_patch'].notna()]

    print(f"\nValid patches: {len(valid_df)}/{len(gedi_df)} ({100*len(valid_df)/len(gedi_df):.1f}%)")

    # Create visualization
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    # Plot 1: GEDI shot locations
    ax = axes[0, 0]
    ax.scatter(gedi_df['longitude'], gedi_df['latitude'], c='red', s=50, alpha=0.6, label='All GEDI shots')
    ax.scatter(valid_df['longitude'], valid_df['latitude'], c='green', s=20, label='Valid patches')
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.set_title('GEDI Shot Locations')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 2: AGBD distribution
    ax = axes[0, 1]
    if 'agbd' in gedi_df.columns:
        ax.hist(gedi_df['agbd'], bins=30, alpha=0.7, label='All shots')
        ax.hist(valid_df['agbd'], bins=30, alpha=0.7, label='Valid patches')
        ax.set_xlabel('AGBD (Mg/ha)')
        ax.set_ylabel('Frequency')
        ax.set_title('AGBD Distribution')
        ax.legend()
    else:
        ax.text(0.5, 0.5, 'AGBD column not found', ha='center', va='center')
        ax.set_title('AGBD Distribution (N/A)')

    # Plot 3: Embedding visualization (mean across channels)
    ax = axes[0, 2]
    embedding_mean = embedding.mean(axis=2)
    im = ax.imshow(embedding_mean, cmap='viridis', aspect='auto')
    ax.set_title('Embedding (mean across channels)')
    ax.set_xlabel('Column')
    ax.set_ylabel('Row')
    plt.colorbar(im, ax=ax)

    # Plot 4-6: Sample patches
    for i, ax in enumerate(axes[1, :]):
        if i < len(valid_df):
            patch = valid_df.iloc[i]['embedding_patch']
            patch_mean = patch.mean(axis=2)

            im = ax.imshow(patch_mean, cmap='viridis')
            agbd_val = valid_df.iloc[i]['agbd'] if 'agbd' in valid_df.columns else 'N/A'
            ax.set_title(f'Patch {i+1} (AGBD={agbd_val:.1f})')
            plt.colorbar(im, ax=ax)
        else:
            ax.text(0.5, 0.5, 'No patch available', ha='center', va='center')
            ax.set_title(f'Patch {i+1}')

    plt.tight_layout()

    # Save figure
    output_dir = Path('./test_outputs')
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / 'spatial_alignment.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n✓ Visualization saved to: {output_path}")
    plt.close()

    return valid_df


def test_6_full_pipeline(valid_df):
    """Test 6: Full pipeline integration."""
    print_section("TEST 6: Full Pipeline Integration")

    if valid_df is None or len(valid_df) == 0:
        print("⚠ No valid data for pipeline test")
        return

    # Add tile_id if not present
    if 'tile_id' not in valid_df.columns:
        valid_df['tile_id'] = 'test_tile_1'

    print(f"Testing with {len(valid_df)} valid GEDI shots")

    # Test dataset creation
    print("\n--- Creating Dataset ---")
    try:
        dataset = GEDINeuralProcessDataset(
            valid_df,
            min_shots_per_tile=5,
            context_ratio_range=(0.3, 0.7)
        )
        print(f"✓ Dataset created with {len(dataset)} tiles")

        if len(dataset) > 0:
            # Test sampling
            print("\n--- Testing Data Loading ---")
            sample = dataset[0]

            print(f"Sample keys: {list(sample.keys())}")
            for key, value in sample.items():
                print(f"  {key}: shape={value.shape}, dtype={value.dtype}")

            print(f"\n✓ Dataset sampling works correctly")
        else:
            print(f"⚠ Dataset has no tiles (need min_shots_per_tile=5)")

    except Exception as e:
        print(f"✗ Dataset creation failed: {e}")
        import traceback
        traceback.print_exc()
        return

    # Test spatial split (if we have tile_id with multiple tiles)
    if 'tile_id' in valid_df.columns and valid_df['tile_id'].nunique() > 2:
        print("\n--- Testing Spatial Split ---")
        try:
            splitter = SpatialTileSplitter(valid_df, val_ratio=0.2, test_ratio=0.2)
            train_df, val_df, test_df = splitter.split()

            print(f"✓ Spatial split successful")
            print(f"  Train: {len(train_df)} shots")
            print(f"  Val: {len(val_df)} shots")
            print(f"  Test: {len(test_df)} shots")
        except Exception as e:
            print(f"⚠ Spatial split test skipped: {e}")
    else:
        print("\n⚠ Skipping spatial split test (need multiple tiles)")

    print("\n✓ Full pipeline integration test complete")


def main():
    """Run all tests."""
    print_section("GEDI Neural Process - Pipeline Testing")
    print(f"Test region: {TEST_REGION['name']}")
    print(f"Bounding box: {TEST_REGION['bbox']}")

    # Test 1: GEDI query
    gedi_df = test_1_gedi_query()

    # Test 2: Tile coordinates
    tile_lon, tile_lat = test_2_tile_coordinates()

    # Test 3: Embedding extraction
    result = test_3_embedding_extraction(tile_lon, tile_lat)
    if result is None:
        print("\n⚠ Stopping tests - embedding extraction failed")
        return

    extractor, embedding, crs, transform = result

    # Test 4: Patch extraction
    patch_results = test_4_patch_extraction(extractor, gedi_df)

    # Test 5: Spatial alignment
    valid_df = test_5_spatial_alignment(extractor, gedi_df, embedding)

    # Test 6: Full pipeline
    test_6_full_pipeline(valid_df)

    print_section("TESTING COMPLETE")
    print("Check ./test_outputs/ for visualizations")


if __name__ == '__main__':
    main()
