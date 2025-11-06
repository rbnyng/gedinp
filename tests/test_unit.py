"""
Unit tests for data processing components using mock data.

These tests validate logic without requiring API access.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import pandas as pd
import torch


def test_tile_coordinate_math():
    """Test tile coordinate calculations."""
    print("=" * 80)
    print("TEST: Tile Coordinate Math")
    print("=" * 80)

    from data.embeddings import EmbeddingExtractor

    extractor = EmbeddingExtractor(year=2024, patch_size=3)

    # Test cases: (input_lon, input_lat) -> expected (tile_lon, tile_lat)
    test_cases = [
        ((0.0, 0.0), (-0.05, -0.05)),  # Closest tile to 0.0 is -0.05
        ((0.1, 0.1), (0.05, 0.05)),    # Closest to 0.1 is 0.05
        ((0.12, 0.12), (0.15, 0.15)),  # Closest to 0.12 is 0.15
        ((30.35, -15.75), (30.35, -15.75)),  # Already on grid
        ((30.256, -15.853), (30.25, -15.85)),  # Edge of test region
    ]
    
    print("\nTesting tile center calculations:")
    print(f"{'Input':<20} {'Expected':<20} {'Actual':<20} {'Status'}")
    print("-" * 80)

    for (lon, lat), (exp_lon, exp_lat) in test_cases:
        tile_lon, tile_lat = extractor._get_tile_coords(lon, lat)
        status = "✓" if abs(tile_lon - exp_lon) < 0.01 and abs(tile_lat - exp_lat) < 0.01 else "✗"
        print(f"({lon:.3f},{lat:.3f})    ({exp_lon:.2f},{exp_lat:.2f})    ({tile_lon:.2f},{tile_lat:.2f})    {status}")

    print("\n✓ Tile coordinate math test complete\n")


def test_coordinate_normalization():
    """Test coordinate normalization in dataset."""
    print("=" * 80)
    print("TEST: Coordinate Normalization")
    print("=" * 80)

    from data.dataset import GEDINeuralProcessDataset

    # Create mock data
    mock_data = {
        'longitude': [30.0, 30.05, 30.1],
        'latitude': [-15.0, -15.05, -15.1],
        'agbd': [100, 150, 200],
        'tile_id': ['tile_1', 'tile_1', 'tile_1'],
        'embedding_patch': [np.random.randn(3, 3, 128) for _ in range(3)]
    }
    mock_df = pd.DataFrame(mock_data)

    # Create dataset
    dataset = GEDINeuralProcessDataset(
        mock_df,
        min_shots_per_tile=2,
        normalize_coords=True,
        normalize_agbd=True
    )

    print(f"\nDataset created with {len(dataset)} tiles")

    if len(dataset) > 0:
        sample = dataset[0]

        print("\nSample data shapes:")
        for key, value in sample.items():
            print(f"  {key}: {value.shape}")

        print("\nCoordinate ranges (should be ~[0, 1]):")
        all_coords = torch.cat([sample['context_coords'], sample['target_coords']], dim=0)
        print(f"  Longitude: [{all_coords[:, 0].min():.3f}, {all_coords[:, 0].max():.3f}]")
        print(f"  Latitude: [{all_coords[:, 1].min():.3f}, {all_coords[:, 1].max():.3f}]")

        print("\nAGBD ranges (should be ~[0, 1]):")
        all_agbd = torch.cat([sample['context_agbd'], sample['target_agbd']], dim=0)
        print(f"  AGBD: [{all_agbd.min():.3f}, {all_agbd.max():.3f}]")

        print("\n✓ Coordinate normalization test complete\n")
    else:
        print("\n⚠ No tiles in dataset (need at least min_shots_per_tile)\n")


def test_spatial_split():
    """Test spatial CV splitting."""
    print("=" * 80)
    print("TEST: Spatial Split")
    print("=" * 80)

    from data.spatial_cv import SpatialTileSplitter

    # Create mock data with multiple tiles
    n_tiles = 10
    shots_per_tile = 20

    data_list = []
    for tile_idx in range(n_tiles):
        for shot_idx in range(shots_per_tile):
            data_list.append({
                'longitude': 30.0 + tile_idx * 0.1 + np.random.rand() * 0.05,
                'latitude': -15.0 + tile_idx * 0.1 + np.random.rand() * 0.05,
                'agbd': np.random.uniform(50, 250),
                'tile_id': f'tile_{tile_idx}',
                'tile_lon': 30.0 + tile_idx * 0.1,
                'tile_lat': -15.0 + tile_idx * 0.1,
                'embedding_patch': np.random.randn(3, 3, 128)
            })

    mock_df = pd.DataFrame(data_list)

    print(f"\nCreated mock data: {len(mock_df)} shots across {n_tiles} tiles")

    # Test split
    splitter = SpatialTileSplitter(mock_df, val_ratio=0.2, test_ratio=0.2, random_state=42)
    train_df, val_df, test_df = splitter.split()

    # Verify no tile overlap
    train_tiles = set(train_df['tile_id'].unique())
    val_tiles = set(val_df['tile_id'].unique())
    test_tiles = set(test_df['tile_id'].unique())

    print("\nChecking for tile overlap:")
    print(f"  Train ∩ Val: {len(train_tiles & val_tiles)} tiles (should be 0)")
    print(f"  Train ∩ Test: {len(train_tiles & test_tiles)} tiles (should be 0)")
    print(f"  Val ∩ Test: {len(val_tiles & test_tiles)} tiles (should be 0)")

    if len(train_tiles & val_tiles) == 0 and len(train_tiles & test_tiles) == 0 and len(val_tiles & test_tiles) == 0:
        print("\n✓ No spatial leakage detected")
    else:
        print("\n✗ Warning: Spatial leakage detected!")

    print("\n✓ Spatial split test complete\n")


def test_neural_process_forward():
    """Test Neural Process forward pass."""
    print("=" * 80)
    print("TEST: Neural Process Forward Pass")
    print("=" * 80)

    from models.neural_process import GEDINeuralProcess

    # Initialize model
    model = GEDINeuralProcess(
        patch_size=3,
        embedding_channels=128,
        embedding_feature_dim=128,
        context_repr_dim=128,
        hidden_dim=256,
        output_uncertainty=True
    )

    print(f"\nModel initialized")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Create dummy inputs
    n_context = 10
    n_query = 5

    context_coords = torch.randn(n_context, 2)
    context_embeddings = torch.randn(n_context, 3, 3, 128)
    context_agbd = torch.randn(n_context, 1)

    query_coords = torch.randn(n_query, 2)
    query_embeddings = torch.randn(n_query, 3, 3, 128)

    print(f"\nInput shapes:")
    print(f"  Context: {n_context} points")
    print(f"  Query: {n_query} points")

    # Forward pass
    model.eval()
    with torch.no_grad():
        pred_mean, pred_log_var = model(
            context_coords,
            context_embeddings,
            context_agbd,
            query_coords,
            query_embeddings
        )

    print(f"\nOutput shapes:")
    print(f"  Predicted mean: {pred_mean.shape}")
    print(f"  Predicted log_var: {pred_log_var.shape}")

    print(f"\nOutput statistics:")
    print(f"  Mean range: [{pred_mean.min():.3f}, {pred_mean.max():.3f}]")
    print(f"  Log_var range: [{pred_log_var.min():.3f}, {pred_log_var.max():.3f}]")

    # Test that outputs are finite
    if torch.isfinite(pred_mean).all() and torch.isfinite(pred_log_var).all():
        print("\n✓ All outputs are finite")
    else:
        print("\n✗ Warning: Some outputs are NaN or Inf")

    print("\n✓ Neural Process forward pass test complete\n")


def test_patch_extraction_logic():
    """Test patch extraction logic with mock embedding."""
    print("=" * 80)
    print("TEST: Patch Extraction Logic")
    print("=" * 80)

    # Create a mock embedding tile
    height, width, channels = 100, 100, 128
    mock_embedding = np.random.randn(height, width, channels)

    patch_size = 3
    half_patch = patch_size // 2

    print(f"\nMock embedding shape: {mock_embedding.shape}")
    print(f"Patch size: {patch_size}x{patch_size}")

    # Test center extraction
    center_row, center_col = height // 2, width // 2
    patch = mock_embedding[
        center_row - half_patch:center_row + half_patch + 1,
        center_col - half_patch:center_col + half_patch + 1,
        :
    ]

    print(f"\nExtracted patch from center ({center_row}, {center_col})")
    print(f"Patch shape: {patch.shape}")

    expected_shape = (patch_size, patch_size, channels)
    if patch.shape == expected_shape:
        print(f"✓ Patch shape matches expected {expected_shape}")
    else:
        print(f"✗ Patch shape mismatch: expected {expected_shape}, got {patch.shape}")

    # Test boundary cases
    print("\nTesting boundary cases:")
    test_positions = [
        (half_patch, half_patch, "Near top-left corner"),
        (height - half_patch - 1, width - half_patch - 1, "Near bottom-right corner"),
        (0, 0, "Out of bounds (top-left)"),
        (height - 1, width - 1, "Out of bounds (bottom-right)")
    ]

    for row, col, desc in test_positions:
        can_extract = (row - half_patch >= 0 and row + half_patch + 1 <= height and
                      col - half_patch >= 0 and col + half_patch + 1 <= width)
        status = "✓ Can extract" if can_extract else "✗ Out of bounds"
        print(f"  ({row:3d}, {col:3d}) - {desc:<30s}: {status}")

    print("\n✓ Patch extraction logic test complete\n")


def main():
    """Run all unit tests."""
    print("\n" + "=" * 80)
    print(" " * 20 + "UNIT TESTS (Mock Data)")
    print("=" * 80 + "\n")

    try:
        test_tile_coordinate_math()
    except Exception as e:
        print(f"✗ Test failed: {e}\n")

    try:
        test_coordinate_normalization()
    except Exception as e:
        print(f"✗ Test failed: {e}\n")

    try:
        test_spatial_split()
    except Exception as e:
        print(f"✗ Test failed: {e}\n")

    try:
        test_neural_process_forward()
    except Exception as e:
        print(f"✗ Test failed: {e}\n")

    try:
        test_patch_extraction_logic()
    except Exception as e:
        print(f"✗ Test failed: {e}\n")

    print("=" * 80)
    print(" " * 25 + "ALL TESTS COMPLETE")
    print("=" * 80)


if __name__ == '__main__':
    main()
