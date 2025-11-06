# Troubleshooting Guide

Common issues and solutions for the GEDI Neural Process pipeline.

## Installation Issues

### Issue: `ModuleNotFoundError: No module named 'gedidb'`
**Solution:**
```bash
pip install gedidb geotessera
```

### Issue: Packaging conflicts during install
**Solution:** Use `--ignore-installed` flag:
```bash
pip install --ignore-installed gedidb geotessera torch
```

### Issue: PyTorch installation
**Solution:** Install CPU or GPU version:
```bash
# CPU version
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# GPU version (CUDA 11.8)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

## Data Query Issues

### Issue: TileDB MemoryError - "Unable to allocate 4.00 GiB for an array"
**Cause:** TileDB's default memory budget (5-10 GB) can cause allocation errors with large GEDI queries.

**Solution:** FIXED in `data/gedi.py` - TileDB memory budget is now configurable:
```python
# Initialize with custom memory budget (default: 512 MB)
querier = GEDIQuerier(memory_budget_mb=512)

# For very memory-constrained environments:
querier = GEDIQuerier(memory_budget_mb=256)

# For larger queries (if you have the RAM):
querier = GEDIQuerier(memory_budget_mb=1024)
```

The fix includes:
- Configurable TileDB memory budget (`sm.memory_budget`, `sm.memory_budget_var`)
- Reduced tile cache size to prevent memory spikes
- Limited concurrent operations to reduce memory footprint
- Automatic fallback to chunked queries on memory errors
- Better error messages with actionable suggestions

### Issue: "For 'bounding_box' queries, a valid GeoDataFrame must be provided"
**Cause:** gediDB expects GeoDataFrame, not shapely geometry.

**Solution:** Already fixed in `data/gedi.py`:
```python
# Incorrect (old):
roi = box(*bbox)

# Correct (new):
bbox_geom = box(*bbox)
roi = gpd.GeoDataFrame([1], geometry=[bbox_geom], crs="EPSG:4326")
```

### Issue: No GEDI data found in region
**Possible causes:**
1. Region has no GEDI coverage
2. Date range doesn't include GEDI mission dates (2019+)
3. Quality filtering too strict

**Solutions:**
- Try a known GEDI-rich region (Amazon, Central Africa)
- Expand date range: `start_time="2019-01-01"`, `end_time="2023-12-31"`
- Disable quality filter: `quality_filter=False`

### Issue: gediDB authentication error
**Solution:** Some GEDI products may require NASA Earthdata credentials.
Check gediDB documentation: https://gedidb.readthedocs.io

## Embedding Extraction Issues

### Issue: GeoTessera tile not found
**Possible causes:**
1. Tile not available for requested year
2. Coordinates outside GeoTessera coverage
3. Network issues

**Solutions:**
- Try year 2024 (most recent)
- Check coverage: `geotessera coverage --year 2024`
- Try different region

### Issue: Coordinate conversion gives out-of-bounds pixels
**Cause:** Mismatch between coordinate system and tile transform.

**Debug:**
```python
extractor = EmbeddingExtractor(year=2024, patch_size=3)
tile_lon, tile_lat = extractor._get_tile_coords(lon, lat)
tile_data = extractor._load_tile(tile_lon, tile_lat)
embedding, crs, transform = tile_data

print(f"Tile shape: {embedding.shape}")
print(f"CRS: {crs}")
print(f"Transform: {transform}")

row, col = extractor._lonlat_to_pixel(lon, lat, transform)
print(f"Pixel coords: ({row}, {col})")
```

### Issue: All patches return None
**Cause:** GEDI shots near tile boundaries or coordinate mismatch.

**Debug:** Check if shots are actually within tile bounds:
```python
print(f"Tile center: ({tile_lon}, {tile_lat})")
print(f"Tile bounds: [{tile_lon-0.05}, {tile_lon+0.05}] x [{tile_lat-0.05}, {tile_lat+0.05}]")
print(f"Shot coords: ({shot_lon}, {shot_lat})")
```

## Training Issues

### Issue: "Dataset has no tiles (need min_shots_per_tile)"
**Cause:** Not enough GEDI shots per tile after filtering.

**Solutions:**
- Reduce `min_shots_per_tile` (try 5 instead of 10)
- Expand spatial region
- Disable quality filtering
- Expand date range

### Issue: CUDA out of memory
**Solutions:**
- Reduce `batch_size`
- Reduce `hidden_dim`
- Use CPU: `--device cpu`

### Issue: Training loss is NaN
**Possible causes:**
1. Learning rate too high
2. Data not normalized properly
3. Numerical instability in loss

**Solutions:**
- Reduce learning rate: `--lr 1e-5`
- Check data ranges: AGBD should be [0, 200], coords [0, 1]
- Add gradient clipping:
```python
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

## Prediction Issues

### Issue: Predictions all same value
**Cause:** Model not trained properly or context data issues.

**Debug:**
- Check training loss curve
- Verify context GEDI shots are loaded
- Check embedding extraction worked

### Issue: Uncertainty values too small/large
**Cause:** Log variance output unbounded.

**Solution:** Add bounds in decoder:
```python
log_var = torch.clamp(self.log_var_head(x), min=-10, max=10)
```

## Performance Issues

### Issue: Embedding extraction very slow
**Solutions:**
- Use caching: `--cache_dir ./cache`
- Cached tiles load instantly
- First download will be slow

### Issue: Training very slow
**Solutions:**
- Use GPU if available
- Reduce `batch_size` if I/O bound
- Reduce `num_workers` if CPU bound
- Use smaller `hidden_dim`

## Data Quality Issues

### Issue: High RMSE on validation
**Possible causes:**
1. Spatial leakage (tiles too close)
2. Not enough training data
3. Model underfitting

**Solutions:**
- Use BufferedSpatialSplitter with larger buffer
- Expand training region
- Increase model capacity: `--hidden_dim 512`
- Train longer: `--epochs 200`

### Issue: Predictions outside valid AGBD range
**Solution:** Add output activation:
```python
pred_mean = F.relu(self.mean_head(x))  # Ensure positive AGBD
```

## Visualization Issues

### Issue: "No display found" when creating plots
**Solution:** Use non-interactive backend:
```python
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
```

Already fixed in test scripts.

## Getting Help

1. **Check logs:** Enable verbose output in data modules
2. **Run unit tests:** `python tests/test_unit.py` - validates logic without API calls
3. **Run integration tests:** `python tests/test_pipeline.py` - tests full pipeline
4. **Check API docs:**
   - gediDB: https://gedidb.readthedocs.io
   - GeoTessera: https://github.com/ucam-eo/geotessera

## Common Workflow Issues

### Starting from scratch

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Test installation (no API calls)
python tests/test_unit.py

# 3. Test with real data (requires network)
python tests/test_pipeline.py

# 4. Train on small region first
python train.py \
    --region_bbox 30.256 -15.853 30.422 -15.625 \
    --epochs 10 \
    --batch_size 4 \
    --output_dir ./test_run

# 5. Check outputs
ls ./test_run/
# Should see: config.json, best_model.pt, history.json, etc.
```

### Debugging specific components

```python
# Test GEDI query only
from data.gedi import GEDIQuerier
querier = GEDIQuerier()
df = querier.query_bbox((30.25, -15.85, 30.35, -15.75))
print(f"Found {len(df)} shots")

# Test embedding extraction only
from data.embeddings import EmbeddingExtractor
extractor = EmbeddingExtractor(year=2024, patch_size=3)
patch = extractor.extract_patch(30.35, -15.75)
print(f"Patch shape: {patch.shape if patch is not None else 'None'}")

# Test model only
from models.neural_process import GEDINeuralProcess
import torch
model = GEDINeuralProcess(patch_size=3, hidden_dim=128)
# ... test with dummy data
```
