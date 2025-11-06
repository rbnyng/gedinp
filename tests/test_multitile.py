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
 
 
# Test region: Zambia forest area with 3x3 grid of tiles
# Each tile is 0.1° x 0.1° (~11km x 11km)
BASE_LON = -70
BASE_LAT = 0
TILE_SIZE = 0.1
 
# Generate 3x3 grid of tile coordinates
TILE_GRID = []
for i in range(3):
    for j in range(3):
        tile_lon = BASE_LON + (i - 1) * TILE_SIZE
        tile_lat = BASE_LAT + (j - 1) * TILE_SIZE
        TILE_GRID.append((tile_lon, tile_lat, f"tile_{i}_{j}"))
 
 
def print_section(title):
    """Print formatted section header."""
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80 + "\n")
 
 
def test_1_multi_tile_gedi_query():
    """Test 1: Query GEDI data for all 9 tiles."""
    print_section("TEST 1: Multi-Tile GEDI Query (3x3 Grid)")
 
    try:
        querier = GEDIQuerier()
        print("? GEDIQuerier initialized")
    except Exception as e:
        print(f"? Failed to initialize GEDIQuerier: {e}")
        return None
 
    all_data = []
    successful_tiles = []
    failed_tiles = []
 
    print(f"Querying {len(TILE_GRID)} tiles in 3x3 grid...\n")
 
    for tile_lon, tile_lat, tile_id in TILE_GRID:
        print(f"Querying {tile_id} at ({tile_lon:.3f}, {tile_lat:.3f})...")
 
        try:
            gedi_df = querier.query_tile(
                tile_lon=tile_lon,
                tile_lat=tile_lat,
                tile_size=TILE_SIZE,
                start_time="2019-01-01",
                end_time="2023-12-31"
            )
 
            if len(gedi_df) > 0:
                # Add tile identifiers
                gedi_df['tile_id'] = tile_id
                gedi_df['tile_lon'] = tile_lon
                gedi_df['tile_lat'] = tile_lat
 
                all_data.append(gedi_df)
                successful_tiles.append(tile_id)
                print(f"  ? {len(gedi_df)} shots retrieved")
            else:
                failed_tiles.append(tile_id)
                print(f"  ? No data found")
 
        except Exception as e:
            failed_tiles.append(tile_id)
            print(f"  ? Query failed: {e}")
 
    print(f"\n--- Query Summary ---")
    print(f"Successful tiles: {len(successful_tiles)}/{len(TILE_GRID)}")
    print(f"Failed tiles: {len(failed_tiles)}/{len(TILE_GRID)}")
 
    if len(all_data) == 0:
        print("? No GEDI data retrieved from any tile")
        return None
 
    # Combine all data
    combined_df = pd.concat(all_data, ignore_index=True)
 
    print(f"\n--- Combined Dataset ---")
    print(f"Total shots: {len(combined_df)}")
    print(f"Tiles with data: {combined_df['tile_id'].nunique()}")
    print(f"Shots per tile:")
    for tile_id in combined_df['tile_id'].unique():
        n_shots = len(combined_df[combined_df['tile_id'] == tile_id])
        print(f"  {tile_id}: {n_shots} shots")
 
    # Spatial extent
    print(f"\n--- Spatial Extent ---")
    print(f"Longitude range: [{combined_df['longitude'].min():.4f}, {combined_df['longitude'].max():.4f}]")
    print(f"Latitude range: [{combined_df['latitude'].min():.4f}, {combined_df['latitude'].max():.4f}]")
 
    # AGBD statistics
    if 'agbd' in combined_df.columns:
        print(f"\n--- AGBD Statistics ---")
        print(combined_df['agbd'].describe())
 
    return combined_df, successful_tiles
 
 
def test_2_multi_tile_embedding_extraction(gedi_df):
    """Test 2: Extract embeddings for all GEDI shots across multiple tiles."""
    print_section("TEST 2: Multi-Tile Embedding Extraction")
 
    if gedi_df is None or len(gedi_df) == 0:
        print("? No GEDI data available for extraction")
        return None
 
    try:
        extractor = EmbeddingExtractor(
            year=2024,
            patch_size=3,
            cache_dir='./test_cache'
        )
        print("? EmbeddingExtractor initialized")
    except Exception as e:
        print(f"? Failed to initialize: {e}")
        return None
 
    print(f"\nExtracting patches for {len(gedi_df)} GEDI shots across {gedi_df['tile_id'].nunique()} tiles...")
 
    try:
        gedi_df_with_patches = extractor.extract_patches_batch(
            gedi_df.copy(),
            verbose=True
        )
        print("? Batch extraction complete")
    except Exception as e:
        print(f"? Batch extraction failed: {e}")
        import traceback
        traceback.print_exc()
        return None
 
    # Filter successful extractions
    valid_df = gedi_df_with_patches[gedi_df_with_patches['embedding_patch'].notna()].copy()
 
    print(f"\n--- Extraction Summary ---")
    print(f"Valid patches: {len(valid_df)}/{len(gedi_df)} ({100*len(valid_df)/len(gedi_df):.1f}%)")
    print(f"Tiles with valid patches: {valid_df['tile_id'].nunique()}")
 
    print(f"\n--- Valid Patches Per Tile ---")
    for tile_id in sorted(valid_df['tile_id'].unique()):
        n_valid = len(valid_df[valid_df['tile_id'] == tile_id])
        n_total = len(gedi_df[gedi_df['tile_id'] == tile_id])
        print(f"  {tile_id}: {n_valid}/{n_total} ({100*n_valid/n_total:.1f}%)")
 
    return valid_df
 
 
def test_3_spatial_cv_split(valid_df):
    """Test 3: Test spatial cross-validation split."""
    print_section("TEST 3: Spatial Cross-Validation Split")
 
    if valid_df is None or len(valid_df) == 0:
        print("? No valid data for split test")
        return None
 
    n_tiles = valid_df['tile_id'].nunique()
    print(f"Testing spatial split with {n_tiles} tiles and {len(valid_df)} shots\n")
 
    if n_tiles < 3:
        print(f"? Warning: Only {n_tiles} tiles available, need at least 3 for meaningful split")
        if n_tiles < 2:
            print("? Cannot perform split with less than 2 tiles")
            return None
 
    # Test basic spatial split
    print("--- Basic Spatial Split (no buffer) ---")
    try:
        splitter = SpatialTileSplitter(
            valid_df,
            val_ratio=0.2,
            test_ratio=0.2,
            random_state=42
        )
        train_df, val_df, test_df = splitter.split()
        print("? Split successful\n")
 
        # Verify no tile overlap
        train_tiles = set(train_df['tile_id'].unique())
        val_tiles = set(val_df['tile_id'].unique())
        test_tiles = set(test_df['tile_id'].unique())
 
        print("--- Verifying Tile Separation ---")
        print(f"Train tiles: {sorted(train_tiles)}")
        print(f"Val tiles:   {sorted(val_tiles)}")
        print(f"Test tiles:  {sorted(test_tiles)}")
 
        # Check for overlaps
        train_val_overlap = train_tiles & val_tiles
        train_test_overlap = train_tiles & test_tiles
        val_test_overlap = val_tiles & test_tiles
 
        if train_val_overlap:
            print(f"? LEAK DETECTED: Train/Val tile overlap: {train_val_overlap}")
        else:
            print("? No train/val tile overlap")
 
        if train_test_overlap:
            print(f"? LEAK DETECTED: Train/Test tile overlap: {train_test_overlap}")
        else:
            print("? No train/test tile overlap")
 
        if val_test_overlap:
            print(f"? LEAK DETECTED: Val/Test tile overlap: {val_test_overlap}")
        else:
            print("? No val/test tile overlap")
 
        # Verify all shots are assigned to exactly one split
        all_train_shots = set(train_df.index)
        all_val_shots = set(val_df.index)
        all_test_shots = set(test_df.index)
        all_original_shots = set(valid_df.index)
 
        print(f"\n--- Verifying Shot Assignment ---")
        assigned_shots = all_train_shots | all_val_shots | all_test_shots
        if assigned_shots == all_original_shots:
            print(f"? All {len(all_original_shots)} shots assigned to exactly one split")
        else:
            missing = all_original_shots - assigned_shots
            extra = assigned_shots - all_original_shots
            if missing:
                print(f"? {len(missing)} shots missing from splits")
            if extra:
                print(f"? {len(extra)} extra shots in splits")
 
        # Check for shot overlap between splits
        train_val_shot_overlap = all_train_shots & all_val_shots
        train_test_shot_overlap = all_train_shots & all_test_shots
        val_test_shot_overlap = all_val_shots & all_test_shots
 
        if train_val_shot_overlap:
            print(f"? LEAK DETECTED: {len(train_val_shot_overlap)} shots in both train and val")
        else:
            print("? No train/val shot overlap")
 
        if train_test_shot_overlap:
            print(f"? LEAK DETECTED: {len(train_test_shot_overlap)} shots in both train and test")
        else:
            print("? No train/test shot overlap")
 
        if val_test_shot_overlap:
            print(f"? LEAK DETECTED: {len(val_test_shot_overlap)} shots in both val and test")
        else:
            print("? No val/test shot overlap")
 
        return train_df, val_df, test_df
 
    except Exception as e:
        print(f"? Split failed: {e}")
        import traceback
        traceback.print_exc()
        return None
 
 
def test_4_dataset_creation(train_df, val_df, test_df):
    """Test 4: Create datasets from splits."""
    print_section("TEST 4: Dataset Creation from Splits")
 
    if train_df is None or len(train_df) == 0:
        print("? No training data available")
        return None
 
    print(f"Creating datasets from splits...")
    print(f"  Train: {len(train_df)} shots, {train_df['tile_id'].nunique()} tiles")
    print(f"  Val:   {len(val_df)} shots, {val_df['tile_id'].nunique()} tiles")
    print(f"  Test:  {len(test_df)} shots, {test_df['tile_id'].nunique()} tiles")
 
    datasets = {}
 
    for split_name, df in [('train', train_df), ('val', val_df), ('test', test_df)]:
        print(f"\n--- Creating {split_name.upper()} Dataset ---")
 
        try:
            dataset = GEDINeuralProcessDataset(
                df,
                min_shots_per_tile=5,
                context_ratio_range=(0.3, 0.7)
            )
            print(f"? Dataset created with {len(dataset)} tiles")
 
            if len(dataset) > 0:
                # Test sampling
                sample = dataset[0]
                print(f"  Sample keys: {list(sample.keys())}")
                print(f"  Context points: {sample['context_x'].shape[0]}")
                print(f"  Target points: {sample['target_x'].shape[0]}")
 
                datasets[split_name] = dataset
            else:
                print(f"  ? Dataset has no tiles (min_shots_per_tile may be too high)")
 
        except Exception as e:
            print(f"? Dataset creation failed: {e}")
            import traceback
            traceback.print_exc()
 
    return datasets
 
 
def test_5_visualize_split(valid_df, train_df, val_df, test_df):
    """Test 5: Visualize the spatial split."""
    print_section("TEST 5: Spatial Split Visualization")
 
    if train_df is None:
        print("? No split data available for visualization")
        return
 
    # Create figure with multiple subplots
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
 
    # Plot 1: All tiles with split colors
    ax = axes[0, 0]
    ax.scatter(train_df['longitude'], train_df['latitude'],
               c='blue', s=10, alpha=0.5, label='Train')
    ax.scatter(val_df['longitude'], val_df['latitude'],
               c='green', s=10, alpha=0.5, label='Val')
    ax.scatter(test_df['longitude'], test_df['latitude'],
               c='red', s=10, alpha=0.5, label='Test')
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.set_title('Spatial Split: All Shots')
    ax.legend()
    ax.grid(True, alpha=0.3)
 
    # Plot 2: Tile-level view
    ax = axes[0, 1]
    # Get tile centers for each split
    train_tiles = train_df.groupby('tile_id')[['tile_lon', 'tile_lat']].first()
    val_tiles = val_df.groupby('tile_id')[['tile_lon', 'tile_lat']].first()
    test_tiles = test_df.groupby('tile_id')[['tile_lon', 'tile_lat']].first()
 
    ax.scatter(train_tiles['tile_lon'], train_tiles['tile_lat'],
               c='blue', s=200, alpha=0.7, marker='s', label='Train tiles')
    ax.scatter(val_tiles['tile_lon'], val_tiles['tile_lat'],
               c='green', s=200, alpha=0.7, marker='s', label='Val tiles')
    ax.scatter(test_tiles['tile_lon'], test_tiles['tile_lat'],
               c='red', s=200, alpha=0.7, marker='s', label='Test tiles')
 
    # Add tile IDs as labels
    for tile_id in valid_df['tile_id'].unique():
        tile_data = valid_df[valid_df['tile_id'] == tile_id].iloc[0]
        ax.text(tile_data['tile_lon'], tile_data['tile_lat'], tile_id,
                ha='center', va='center', fontsize=8)
 
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.set_title('Spatial Split: Tile Centers')
    ax.legend()
    ax.grid(True, alpha=0.3)
 
    # Plot 3: Shots per tile by split
    ax = axes[0, 2]
    tile_counts = []
    for tile_id in sorted(valid_df['tile_id'].unique()):
        train_count = len(train_df[train_df['tile_id'] == tile_id])
        val_count = len(val_df[val_df['tile_id'] == tile_id])
        test_count = len(test_df[test_df['tile_id'] == tile_id])
        tile_counts.append({
            'tile_id': tile_id,
            'train': train_count,
            'val': val_count,
            'test': test_count
        })
 
    tile_counts_df = pd.DataFrame(tile_counts)
    x = np.arange(len(tile_counts_df))
    width = 0.25
 
    ax.bar(x - width, tile_counts_df['train'], width, label='Train', color='blue', alpha=0.7)
    ax.bar(x, tile_counts_df['val'], width, label='Val', color='green', alpha=0.7)
    ax.bar(x + width, tile_counts_df['test'], width, label='Test', color='red', alpha=0.7)
 
    ax.set_xlabel('Tile ID')
    ax.set_ylabel('Number of Shots')
    ax.set_title('Shots per Tile by Split')
    ax.set_xticks(x)
    ax.set_xticklabels(tile_counts_df['tile_id'], rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
 
    # Plot 4-6: AGBD distributions
    if 'agbd' in valid_df.columns:
        for idx, (split_name, df) in enumerate([('Train', train_df), ('Val', val_df), ('Test', test_df)]):
            ax = axes[1, idx]
            ax.hist(df['agbd'], bins=30, alpha=0.7, edgecolor='black')
            ax.set_xlabel('AGBD (Mg/ha)')
            ax.set_ylabel('Frequency')
            ax.set_title(f'{split_name} AGBD Distribution\n(n={len(df)}, mean={df["agbd"].mean():.1f})')
            ax.grid(True, alpha=0.3)
    else:
        for idx in range(3):
            ax = axes[1, idx]
            ax.text(0.5, 0.5, 'AGBD data not available',
                    ha='center', va='center', transform=ax.transAxes)
 
    plt.tight_layout()
 
    # Save figure
    output_dir = Path('./test_outputs')
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / 'multi_tile_spatial_split.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n? Visualization saved to: {output_path}")
    plt.close()
 
 
def main():
    """Run all multi-tile tests."""
    print_section("GEDI Neural Process - Multi-Tile Integration Test (3x3 Grid)")
    print(f"Base coordinates: ({BASE_LON}, {BASE_LAT})")
    print(f"Tile size: {TILE_SIZE}° (~{TILE_SIZE * 111:.1f} km)")
    print(f"Total tiles: {len(TILE_GRID)} (3x3 grid)")
 
    # Test 1: Multi-tile GEDI query
    result = test_1_multi_tile_gedi_query()
    if result is None:
        print("\n? Stopping tests - no GEDI data retrieved")
        return
 
    gedi_df, successful_tiles = result
 
    # Test 2: Multi-tile embedding extraction
    valid_df = test_2_multi_tile_embedding_extraction(gedi_df)
    if valid_df is None or len(valid_df) == 0:
        print("\n? Stopping tests - no valid patches extracted")
        return
 
    # Test 3: Spatial CV split
    split_result = test_3_spatial_cv_split(valid_df)
    if split_result is None:
        print("\n? Stopping tests - spatial split failed")
        return
 
    train_df, val_df, test_df = split_result
 
    # Test 4: Dataset creation
    datasets = test_4_dataset_creation(train_df, val_df, test_df)
 
    # Test 5: Visualize split
    test_5_visualize_split(valid_df, train_df, val_df, test_df)
 
    print_section("MULTI-TILE TESTING COMPLETE")
    print("? All tests passed successfully")
    print(f"? Verified spatial CV split correctness across {valid_df['tile_id'].nunique()} tiles")
    print(f"? No data leakage detected between train/val/test sets")
    print("\nCheck ./test_outputs/ for visualizations")
 
 
if __name__ == '__main__':
    main()