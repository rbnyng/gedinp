# Testing Suite Summary

## What Was Created

### 1. Bug Fixes

**Issue Discovered:** gediDB requires GeoDataFrame for geometry parameter, not shapely Box.

**Fixed in `data/gedi.py`:**
```python
# Before (incorrect):
roi = box(*bbox)
provider.get_data(geometry=roi, ...)

# After (correct):
bbox_geom = box(*bbox)
roi = gpd.GeoDataFrame([1], geometry=[bbox_geom], crs="EPSG:4326")
provider.get_data(geometry=roi, ...)
```

### 2. Test Suite Components

#### **`tests/test_unit.py`** - Unit Tests (No API Required)
Tests core logic with mock data - runs in seconds without network access.

**Tests included:**
- ✓ Tile coordinate calculations and grid snapping
- ✓ Coordinate normalization to [0, 1] range
- ✓ Spatial CV splitting without tile leakage
- ✓ Neural Process forward pass and output shapes
- ✓ Patch extraction boundary conditions

**Run:**
```bash
python tests/test_unit.py
```

**Expected output:**
```
================================================================================
                    UNIT TESTS (Mock Data)
================================================================================

TEST: Tile Coordinate Math
✓ All coordinate calculations pass

TEST: Coordinate Normalization
✓ Values normalized correctly

TEST: Spatial Split
✓ No spatial leakage detected

TEST: Neural Process Forward Pass
✓ Model parameters: 1,234,567
✓ All outputs are finite

TEST: Patch Extraction Logic
✓ Boundary cases handled correctly

================================================================================
                     ALL TESTS COMPLETE
================================================================================
```

#### **`tests/test_pipeline.py`** - Integration Tests (Requires Network)
Tests full pipeline with real gediDB and GeoTessera APIs.

**Tests included:**
1. GEDI L4A data querying with quality filtering
2. Tile coordinate math with GeoTessera grid
3. GeoTessera embedding extraction and caching
4. 3x3 patch extraction at GEDI shot locations
5. Spatial alignment visualization (creates plots)
6. Full pipeline integration (dataset + splits)

**Run:**
```bash
python tests/test_pipeline.py
```

**Outputs:**
- Console logs showing data statistics
- `./test_outputs/spatial_alignment.png` - Visualization of GEDI shots, embeddings, and patches

**Expected flow:**
```
TEST 1: GEDI Data Query
✓ GEDIQuerier initialized
✓ Query successful: N shots returned
✓ AGBD column exists

TEST 2: Tile Coordinate Math
✓ Tile centers calculated correctly

TEST 3: Embedding Extraction
✓ Tile loaded successfully
✓ Shape: (H, W, 128)
✓ Coordinate conversion works

TEST 4: Patch Extraction
✓ Extracted patches for test shots

TEST 5: Spatial Alignment
✓ Visualization saved

TEST 6: Full Pipeline Integration
✓ Dataset created successfully
✓ Spatial split works
```

### 3. Documentation

#### **`TROUBLESHOOTING.md`**
Comprehensive guide covering:
- Installation issues and dependency conflicts
- Data query problems (authentication, no data found)
- Embedding extraction debugging
- Training issues (NaN loss, OOM, slow performance)
- Prediction and visualization issues
- Step-by-step debugging workflows

#### **`tests/README.md`**
Test suite documentation:
- What each test file does
- How to run tests
- Expected outputs
- Known issues and fixes

## How to Use

### Quick Start (No Network)

Test the logic without API calls:
```bash
python tests/test_unit.py
```

This validates:
- All coordinate transformations work
- Model architecture is correct
- Dataset logic is sound
- No obvious bugs in core components

**Time:** ~10 seconds

### Full Validation (Requires Network)

Test with real APIs:
```bash
python tests/test_pipeline.py
```

This validates:
- gediDB connectivity and data format
- GeoTessera connectivity and embedding format
- Spatial alignment between GEDI and embeddings
- End-to-end pipeline functionality

**Time:** 1-5 minutes (depends on network and data size)

**Note:** First run downloads GeoTessera tiles. Subsequent runs use cache.

### When Things Go Wrong

1. **Check unit tests first:**
   ```bash
   python tests/test_unit.py
   ```
   If these fail, there's a logic bug (not API issue)

2. **Check integration tests:**
   ```bash
   python tests/test_pipeline.py
   ```
   If these fail, check network/API access

3. **Consult troubleshooting guide:**
   ```bash
   cat TROUBLESHOOTING.md
   ```
   Find your specific error and solution

### Debugging Specific Components

Test individual components:

```python
# Test GEDI query only
from data.gedi import GEDIQuerier
querier = GEDIQuerier()
df = querier.query_bbox((30.25, -15.85, 30.35, -15.75))
print(f"Found {len(df)} shots")
print(df.head())

# Test embedding extraction only
from data.embeddings import EmbeddingExtractor
extractor = EmbeddingExtractor(year=2024, patch_size=3)
patch = extractor.extract_patch(30.35, -15.75)
print(f"Patch: {patch.shape if patch is not None else 'None'}")

# Test model only
from models.neural_process import GEDINeuralProcess
import torch
model = GEDINeuralProcess(patch_size=3, hidden_dim=256)
print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
```

## What We Learned

### Issues We Anticipated and Found

✓ **GeoDataFrame requirement** - gediDB needs GeoDataFrame, not shapely geometry
✓ **Coordinate system** - GeoTessera tiles centered at 0.05, 0.15, not 0.0, 0.1
✓ **Boundary conditions** - Patches near tile edges need validation
✓ **Data format** - Need to verify AGBD column names and types

### Best Practices Implemented

1. **Test without network first** - Unit tests catch logic bugs immediately
2. **Mock data for development** - Faster iteration, no API dependencies
3. **Comprehensive error handling** - Try/except with helpful messages
4. **Visualization for validation** - Plot spatial alignment to catch subtle bugs
5. **Documentation upfront** - Troubleshooting guide before issues occur

## Next Steps

### Before Training

1. Run unit tests to validate installation:
   ```bash
   python tests/test_unit.py
   ```

2. Run integration tests on your target region:
   ```bash
   python tests/test_pipeline.py
   ```

3. Check visualizations in `./test_outputs/`

4. If issues found, consult `TROUBLESHOOTING.md`

### During Training

Monitor for common issues:
- NaN losses → reduce learning rate
- OOM errors → reduce batch_size or hidden_dim
- Slow training → check if using GPU
- Poor validation metrics → check spatial CV splits

### After Training

Validate predictions:
```bash
python predict.py \
    --model_path ./outputs/best_model.pt \
    --config_path ./outputs/config.json \
    --tile_lon 30.35 \
    --tile_lat -15.75 \
    --visualize
```

Check:
- AGBD values in reasonable range (0-300 Mg/ha)
- Uncertainty values make sense
- Visualizations look correct

## Files Summary

```
gedinp/
├── data/
│   └── gedi.py              [FIXED] - Now uses GeoDataFrame
├── tests/
│   ├── test_unit.py         [NEW] - Unit tests with mock data
│   ├── test_pipeline.py     [NEW] - Integration tests with APIs
│   └── README.md            [NEW] - Test documentation
├── TROUBLESHOOTING.md       [NEW] - Comprehensive troubleshooting
└── TESTING_SUMMARY.md       [NEW] - This file
```

## Commit History

1. **Initial implementation** - Full pipeline with data, models, training
2. **Implementation summary** - Documentation of approach
3. **Test suite + fixes** - Unit tests, integration tests, bug fixes, troubleshooting guide

All code committed and pushed to:
`claude/gedi-agb-interpolation-neural-processes-011CUrwoN7TL5ZXbUQdFN4Rc`

---

**Ready to test!** Start with `python tests/test_unit.py` to validate everything without needing API access.
