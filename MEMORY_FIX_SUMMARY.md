# GEDI TileDB Memory Error Fix

## Problem

The GEDI data querying was failing with massive memory allocation errors:

```
MemoryError: Unable to allocate 4.00 GiB for an array with shape (4294967296,) and data type uint8
MemoryError: Unable to allocate 16.0 GiB for an array with shape (2147483648,) and data type uint64
```

These errors occurred when trying to query even small bounding boxes (0.1° tiles) from the GEDI database.

## Root Cause

TileDB (the storage backend used by gediDB) has default memory budget settings that allow it to try allocating 5-10 GB of memory for queries. When combined with certain query patterns or S3 configurations, this can lead to:

1. Attempts to allocate arrays with billions of elements (2^31 to 2^32)
2. System memory exhaustion
3. Query failures even for small spatial extents

## Solution

### 1. TileDB Memory Budget Configuration

Added configurable memory limits to the `GEDIQuerier` class:

```python
class GEDIQuerier:
    def __init__(
        self,
        storage_type: str = 's3',
        s3_bucket: str = "dog.gedidb.gedi-l2-l4-v002",
        url: str = "https://s3.gfz-potsdam.de",
        local_path: Optional[str] = None,
        memory_budget_mb: int = 512  # NEW PARAMETER
    ):
        # Configure TileDB memory limits
        memory_budget_bytes = memory_budget_mb * 1024 * 1024
        config = tiledb.Config()
        config["sm.memory_budget"] = str(memory_budget_bytes)
        config["sm.memory_budget_var"] = str(memory_budget_bytes * 2)
        config["sm.tile_cache_size"] = str(memory_budget_bytes // 2)
        config["sm.compute_concurrency_level"] = "1"
        config["sm.io_concurrency_level"] = "1"

        tiledb.default_ctx(config=config)
        # ... rest of initialization
```

### 2. Enhanced Error Handling

Added better error detection and reporting:

```python
except Exception as e:
    # Check if it's a TileDB memory error
    if "MemoryError" in str(e) or "Unable to allocate" in str(e):
        logger.error(f"TileDB memory allocation error: {e}")
        raise MemoryError(str(e))
```

### 3. Automatic Chunked Query Fallback

Implemented automatic fallback to chunked queries when memory errors occur:

```python
def query_tile(self, tile_lon, tile_lat, tile_size=0.1, **kwargs):
    # ... bbox calculation
    try:
        return self.query_bbox(bbox, **kwargs)
    except MemoryError as e:
        logger.warning("Memory error, falling back to chunked approach")
        return self._query_bbox_chunked(bbox, chunk_size=0.05, **kwargs)
```

### 4. Chunked Query Implementation

Added a chunked query method that splits large bounding boxes:

```python
def _query_bbox_chunked(self, bbox, chunk_size=0.1, **kwargs):
    """Query large bounding box by splitting into smaller chunks."""
    # Splits bbox into grid of smaller chunks
    # Queries each chunk separately
    # Merges results and removes duplicates
```

### 5. Improved Logging

Added comprehensive logging throughout the query process:

```python
import logging
logger = logging.getLogger(__name__)

# Log all query parameters
logger.info(f"Querying GEDI data for bbox {bbox}")
logger.info(f"Time range: {start_time} to {end_time}")
logger.info(f"Retrieved {len(df)} shots before filtering")
```

## Usage

### Default Usage (512 MB budget)

```python
from data.gedi import GEDIQuerier

querier = GEDIQuerier()
gedi_df = querier.query_tile(30.35, -15.75, tile_size=0.1)
```

### Memory-Constrained Environments

```python
# Use 256 MB budget
querier = GEDIQuerier(memory_budget_mb=256)
gedi_df = querier.query_tile(30.35, -15.75)
```

### Larger Queries (if RAM available)

```python
# Use 1024 MB budget
querier = GEDIQuerier(memory_budget_mb=1024)
gedi_df = querier.query_bbox((30.0, -16.0, 31.0, -15.0))
```

### Manual Chunked Queries

```python
# Force chunked query for very large regions
querier = GEDIQuerier()
gedi_df = querier.query_tile(30.35, -15.75, use_chunked=True)
```

## Testing

The fix has been validated to eliminate the memory allocation errors. The query now either:

1. Succeeds with the configured memory budget
2. Automatically falls back to chunked queries
3. Provides clear error messages if memory is still insufficient

Test results show that queries no longer attempt to allocate 4-16 GB arrays, and instead respect the configured memory budget.

## Files Modified

- `data/gedi.py`: Main implementation of memory fixes
- `TROUBLESHOOTING.md`: Added documentation of the issue and fix
- `test_memory_fix.py`: Test script to verify the fix

## Configuration Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `memory_budget_mb` | 512 | Total memory budget for TileDB in MB |
| `sm.memory_budget` | 512 MB | Fixed-size attribute memory limit |
| `sm.memory_budget_var` | 1024 MB | Variable-size attribute memory limit (2x budget) |
| `sm.tile_cache_size` | 256 MB | Tile cache size (0.5x budget) |
| `sm.compute_concurrency_level` | 1 | Concurrent compute threads |
| `sm.io_concurrency_level` | 1 | Concurrent I/O threads |

## Known Limitations

1. Very memory-constrained environments (< 256 MB) may still encounter issues
2. Chunked queries are slower due to multiple API calls
3. Network errors can still occur independently of memory configuration

## Future Improvements

1. Adaptive chunk sizing based on available memory
2. Progress callbacks for long-running chunked queries
3. Caching of chunk results to avoid duplicate queries
4. Parallel chunk queries (if memory allows)

## References

- TileDB Configuration Documentation: https://docs.tiledb.com/main/how-to/configuration
- gediDB Repository: https://github.com/simonbesnard1/gedidb
- TileDB-Py Memory Issues: https://github.com/TileDB-Inc/TileDB-Py/issues
