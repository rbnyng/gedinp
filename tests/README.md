# Testing Suite

This directory contains tests for validating the GEDI Neural Process pipeline.

## Test Files

### `test_unit.py` - Unit Tests (No API calls)
Tests core logic using mock data, no network required.

**Tests:**
- Tile coordinate calculations
- Coordinate normalization
- Spatial CV splitting
- Neural Process forward pass
- Patch extraction logic

**Run:**
```bash
python tests/test_unit.py
```

### `test_pipeline.py` - Integration Tests (Requires API access)
Tests full pipeline with real gediDB and GeoTessera APIs.

**Tests:**
1. GEDI data querying from gediDB
2. Tile coordinate math verification
3. GeoTessera embedding extraction
4. Patch extraction for GEDI shots
5. Spatial alignment visualization
6. Full pipeline integration

**Run:**
```bash
python tests/test_pipeline.py
```

**Note:** This test requires:
- Network access to gediDB S3 bucket
- Network access to GeoTessera
- May take several minutes depending on data size
- Creates visualizations in `./test_outputs/`

## Known Issues Fixed

### Issue 1: GeoDataFrame Requirement
**Problem:** gediDB expects `geometry` parameter as GeoDataFrame, not shapely geometry.

**Fix:** Convert shapely box to GeoDataFrame with CRS:
```python
bbox_geom = box(*bbox)
roi = gpd.GeoDataFrame([1], geometry=[bbox_geom], crs="EPSG:4326")
```

### Issue 2: Tile Coordinate Snapping
**Problem:** GeoTessera tiles are at 0.05, 0.15, 0.25, etc. (not 0.0, 0.1, 0.2).

**Fix:** Proper rounding logic in `_get_tile_coords()`:
```python
tile_lon = round(lon / 0.1) * 0.1 + 0.05
tile_lat = round(lat / 0.1) * 0.1 + 0.05
```

## Test Output

Unit tests should show:
```
================================================================================
                    UNIT TESTS (Mock Data)
================================================================================

TEST: Tile Coordinate Math
✓ All coordinate calculations pass

TEST: Coordinate Normalization
✓ Values normalized to [0, 1] range

TEST: Spatial Split
✓ No spatial leakage detected

TEST: Neural Process Forward Pass
✓ All outputs are finite

TEST: Patch Extraction Logic
✓ Boundary cases handled correctly
```

Integration tests should show:
```
TEST 1: GEDI Data Query
✓ Retrieved N GEDI shots
✓ AGBD column exists

TEST 2: Tile Coordinate Math
✓ Tile centers calculated correctly

TEST 3: Embedding Extraction
✓ Tile loaded successfully
✓ Shape: (H, W, 128)

TEST 4: Patch Extraction
✓ Patches extracted for all shots

TEST 5: Spatial Alignment
✓ Visualization saved to ./test_outputs/spatial_alignment.png

TEST 6: Full Pipeline Integration
✓ Dataset created successfully
✓ Spatial split works correctly
```

## Quick Validation

For quick validation without network access:
```bash
# Run unit tests only
python tests/test_unit.py

# This should complete in < 10 seconds
```

For full pipeline validation:
```bash
# Run integration tests (requires network)
python tests/test_pipeline.py

# This may take 1-5 minutes depending on:
# - Network speed
# - Number of GEDI shots in test region
# - GeoTessera tile download time
```
