"""
Quick test to verify GEDI memory fixes work.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

import logging

# Set up logging to see what's happening
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

from data.gedi import GEDIQuerier

# Test region: Zambia (same as original test)
TEST_REGION = {
    'name': 'Zambia Forest',
    'bbox': (30.256, -15.853, 30.422, -15.625),
    'tile_lon': 30.35,
    'tile_lat': -15.75
}

def test_gedi_query():
    """Test GEDI data querying with memory fixes."""
    print("=" * 80)
    print("  TEST: GEDI Data Query with Memory Fixes")
    print("=" * 80)
    print()

    try:
        # Initialize with lower memory budget
        querier = GEDIQuerier(memory_budget_mb=256)
        print("✓ GEDIQuerier initialized with 256 MB memory budget")
    except Exception as e:
        print(f"✗ Failed to initialize GEDIQuerier: {e}")
        return None

    # Test single tile query
    print(f"\nQuerying tile at ({TEST_REGION['tile_lon']}, {TEST_REGION['tile_lat']})...")
    print(f"Tile size: 0.1°")
    print()

    try:
        gedi_df = querier.query_tile(
            tile_lon=TEST_REGION['tile_lon'],
            tile_lat=TEST_REGION['tile_lat'],
            tile_size=0.1,
            start_time="2019-01-01",
            end_time="2023-12-31"
        )
        print(f"✓ Query successful: {len(gedi_df)} shots returned")

        if len(gedi_df) > 0:
            print(f"\nFirst few rows:")
            print(gedi_df.head())
            print(f"\nColumns: {list(gedi_df.columns)}")

            if 'agbd' in gedi_df.columns:
                print(f"\nAGBD statistics:")
                print(gedi_df['agbd'].describe())

        return gedi_df

    except MemoryError as e:
        print(f"✗ Memory error: {e}")
        print("\nThe memory fix didn't fully resolve the issue.")
        print("This may be a limitation of the gediDB library itself.")
        return None
    except Exception as e:
        print(f"✗ Query failed: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == '__main__':
    result = test_gedi_query()
    if result is not None:
        print("\n" + "=" * 80)
        print("  TEST PASSED")
        print("=" * 80)
    else:
        print("\n" + "=" * 80)
        print("  TEST FAILED")
        print("=" * 80)
