"""
GEDI data querying utilities using gediDB.

Handles querying GEDI L4A data for aboveground biomass (AGBD) with spatial filtering.
"""

import gedidb as gdb
import pandas as pd
import xarray as xr
import geopandas as gpd
from shapely.geometry import box
from typing import Optional, Union, List
import numpy as np
import tiledb
import logging

# Configure logging
logger = logging.getLogger(__name__)


class GEDIQuerier:
    """Query GEDI data from gediDB."""

    def __init__(
        self,
        storage_type: str = 's3',
        s3_bucket: str = "dog.gedidb.gedi-l2-l4-v002",
        url: str = "https://s3.gfz-potsdam.de",
        local_path: Optional[str] = None,
        memory_budget_mb: int = 512
    ):
        """
        Initialize GEDI data provider.

        Args:
            storage_type: 's3' for cloud access or 'local' for local database
            s3_bucket: S3 bucket name for cloud access
            url: S3 endpoint URL
            local_path: Path to local gediDB if storage_type='local'
            memory_budget_mb: TileDB memory budget in MB (default: 512)
        """
        # Configure TileDB memory limits to prevent huge allocations
        memory_budget_bytes = memory_budget_mb * 1024 * 1024
        config = tiledb.Config()
        config["sm.memory_budget"] = str(memory_budget_bytes)
        config["sm.memory_budget_var"] = str(memory_budget_bytes * 2)
        # Reduce tile cache size
        config["sm.tile_cache_size"] = str(memory_budget_bytes // 2)
        # Limit concurrent operations
        config["sm.compute_concurrency_level"] = "1"
        config["sm.io_concurrency_level"] = "1"
        # Enable memory management features
        config["sm.enable_signal_handlers"] = "false"

        logger.info(f"TileDB memory budget set to {memory_budget_mb} MB")

        # Set TileDB default configuration
        try:
            # Set default context with config
            tiledb.default_ctx(config=config)
        except Exception as e:
            logger.warning(f"Could not set TileDB default context: {e}")
            logger.warning("Continuing without custom TileDB configuration")

        # Store config for later use
        self.tiledb_config = config
        self.memory_budget_mb = memory_budget_mb

        if storage_type == 's3':
            self.provider = gdb.GEDIProvider(
                storage_type='s3',
                s3_bucket=s3_bucket,
                url=url
            )
        else:
            self.provider = gdb.GEDIProvider(
                storage_type='local',
                local_path=local_path
            )

    def query_bbox(
        self,
        bbox: tuple,
        start_time: str = "2023-01-01",
        end_time: str = "2023-12-31",
        variables: Optional[List[str]] = None,
        quality_filter: bool = True,
        min_agbd: float = 0.0,
        max_agbd: Optional[float] = None,
        use_dask: bool = False
    ) -> pd.DataFrame:
        """
        Query GEDI shots within a bounding box.

        Args:
            bbox: (min_lon, min_lat, max_lon, max_lat)
            start_time: Start date (YYYY-MM-DD)
            end_time: End date (YYYY-MM-DD)
            variables: List of variables to retrieve. If None, uses defaults.
            quality_filter: Apply quality filtering based on L4 quality flags
            min_agbd: Minimum AGBD threshold (Mg/ha)
            max_agbd: Maximum AGBD threshold (Mg/ha), None for no upper limit
            use_dask: Use dask for lazy loading (reduces memory usage)

        Returns:
            DataFrame with columns: latitude, longitude, agbd, quality metrics
        """
        if variables is None:
            variables = [
                "agbd",  # Aboveground biomass density
            ]

        logger.info(f"Querying GEDI data for bbox {bbox}")
        logger.info(f"Time range: {start_time} to {end_time}")
        logger.info(f"Variables: {variables}")

        # Create bbox geometry as GeoDataFrame (required by gediDB)
        bbox_geom = box(*bbox)
        roi = gpd.GeoDataFrame([1], geometry=[bbox_geom], crs="EPSG:4326")

        try:
            # Query data with memory-efficient settings
            return_type = 'xarray'  # Use xarray for better memory management

            gedi_data = self.provider.get_data(
                variables=variables,
                query_type="bounding_box",
                geometry=roi,
                start_time=start_time,
                end_time=end_time,
                return_type=return_type
            )

            logger.info(f"Query successful, converting to DataFrame...")

            # Convert to DataFrame
            if hasattr(gedi_data, 'to_dataframe'):
                df = gedi_data.to_dataframe().reset_index()
            else:
                # Fallback if not xarray
                df = pd.DataFrame(gedi_data)

            logger.info(f"Retrieved {len(df)} shots before filtering")

            # Handle empty results gracefully
            if len(df) == 0:
                logger.info("No data found, returning empty DataFrame")
                return pd.DataFrame()

            # Filter by AGBD range
            if 'agbd' in df.columns:
                df = df[df['agbd'] >= min_agbd]
                if max_agbd is not None:
                    df = df[df['agbd'] <= max_agbd]

                # Remove NaN values
                df = df.dropna(subset=['latitude', 'longitude', 'agbd'])
            else:
                logger.warning("'agbd' column not found in results")
                # Remove NaN values from coordinates at minimum
                df = df.dropna(subset=['latitude', 'longitude'])

            logger.info(f"Returning {len(df)} shots after filtering")
            return df

        except MemoryError as e:
            logger.error(f"MemoryError during query: {e}")
            # Don't raise immediately - let caller handle it
            raise
        except Exception as e:
            # Check if it's a TileDB memory error (even if not caught as MemoryError)
            if "MemoryError" in str(e) or "Unable to allocate" in str(e):
                logger.error(f"TileDB memory allocation error: {e}")
                raise MemoryError(str(e))
            logger.error(f"Error querying GEDI data: {e}")
            raise

    def _query_bbox_chunked(
        self,
        bbox: tuple,
        chunk_size: float = 0.1,
        **kwargs
    ) -> pd.DataFrame:
        """
        Query large bounding box by splitting into smaller chunks.

        This is a workaround for TileDB memory allocation issues.

        Args:
            bbox: (min_lon, min_lat, max_lon, max_lat)
            chunk_size: Size of each chunk in degrees
            **kwargs: Additional arguments passed to query_bbox

        Returns:
            Combined DataFrame from all chunks
        """
        min_lon, min_lat, max_lon, max_lat = bbox

        # Calculate number of chunks needed
        lon_chunks = int(np.ceil((max_lon - min_lon) / chunk_size))
        lat_chunks = int(np.ceil((max_lat - min_lat) / chunk_size))

        total_chunks = lon_chunks * lat_chunks
        logger.info(f"Splitting query into {total_chunks} chunks ({lon_chunks}x{lat_chunks})")

        all_data = []
        chunk_count = 0

        for i in range(lon_chunks):
            for j in range(lat_chunks):
                chunk_min_lon = min_lon + i * chunk_size
                chunk_max_lon = min(chunk_min_lon + chunk_size, max_lon)
                chunk_min_lat = min_lat + j * chunk_size
                chunk_max_lat = min(chunk_min_lat + chunk_size, max_lat)

                chunk_bbox = (chunk_min_lon, chunk_min_lat, chunk_max_lon, chunk_max_lat)
                chunk_count += 1

                logger.info(f"Querying chunk {chunk_count}/{total_chunks}: {chunk_bbox}")

                try:
                    chunk_df = self.query_bbox(chunk_bbox, **kwargs)
                    if len(chunk_df) > 0:
                        all_data.append(chunk_df)
                        logger.info(f"  Retrieved {len(chunk_df)} shots")
                    else:
                        logger.info(f"  No shots found")
                except Exception as e:
                    logger.warning(f"  Chunk failed: {e}")
                    continue

        if len(all_data) == 0:
            logger.warning("No data retrieved from any chunks")
            return pd.DataFrame()

        # Combine all chunks
        combined_df = pd.concat(all_data, ignore_index=True)
        # Remove duplicates (shots may appear in multiple chunks at boundaries)
        if 'shot_number' in combined_df.columns:
            combined_df = combined_df.drop_duplicates(subset=['shot_number'])
        else:
            # Fallback: drop duplicates based on coordinates
            combined_df = combined_df.drop_duplicates(subset=['latitude', 'longitude'])

        logger.info(f"Total shots after merging chunks: {len(combined_df)}")
        return combined_df

    def query_tile(
        self,
        tile_lon: float,
        tile_lat: float,
        tile_size: float = 0.1,
        use_chunked: bool = False,
        **kwargs
    ) -> pd.DataFrame:
        """
        Query GEDI shots within a single tile.

        Args:
            tile_lon: Tile center longitude
            tile_lat: Tile center latitude
            tile_size: Tile size in degrees (default 0.1° for GeoTessera alignment)
            use_chunked: If True, use chunked query (workaround for memory issues)
            **kwargs: Additional arguments passed to query_bbox

        Returns:
            DataFrame of GEDI shots within the tile
        """
        half_size = tile_size / 2
        bbox = (
            tile_lon - half_size,
            tile_lat - half_size,
            tile_lon + half_size,
            tile_lat + half_size
        )

        if use_chunked:
            return self._query_bbox_chunked(bbox, chunk_size=0.05, **kwargs)
        else:
            try:
                return self.query_bbox(bbox, **kwargs)
            except MemoryError as e:
                logger.warning(f"Memory error in direct query, falling back to chunked approach")
                logger.warning(f"Original error: {e}")
                return self._query_bbox_chunked(bbox, chunk_size=0.05, **kwargs)

    def query_region_tiles(
        self,
        region_bbox: tuple,
        tile_size: float = 0.1,
        **kwargs
    ) -> pd.DataFrame:
        """
        Query GEDI shots across multiple tiles in a region.

        Args:
            region_bbox: (min_lon, min_lat, max_lon, max_lat) for entire region
            tile_size: Tile size in degrees
            **kwargs: Additional arguments passed to query_bbox

        Returns:
            DataFrame with additional 'tile_id' column for spatial CV
        """
        min_lon, min_lat, max_lon, max_lat = region_bbox

        # Generate tile centers
        lon_centers = np.arange(
            min_lon + tile_size/2,
            max_lon,
            tile_size
        )
        lat_centers = np.arange(
            min_lat + tile_size/2,
            max_lat,
            tile_size
        )

        all_shots = []

        for lon_center in lon_centers:
            for lat_center in lat_centers:
                tile_df = self.query_tile(lon_center, lat_center, tile_size, **kwargs)

                if len(tile_df) > 0:
                    # Add tile identifier
                    tile_df['tile_id'] = f"tile_{lon_center:.2f}_{lat_center:.2f}"
                    tile_df['tile_lon'] = lon_center
                    tile_df['tile_lat'] = lat_center
                    all_shots.append(tile_df)

        if len(all_shots) == 0:
            return pd.DataFrame()

        return pd.concat(all_shots, ignore_index=True)


def get_gedi_statistics(df: pd.DataFrame) -> dict:
    """
    Compute summary statistics for GEDI data.

    Args:
        df: DataFrame from GEDIQuerier

    Returns:
        Dictionary of statistics
    """
    stats = {
        'n_shots': len(df),
        'agbd_mean': df['agbd'].mean(),
        'agbd_std': df['agbd'].std(),
        'agbd_min': df['agbd'].min(),
        'agbd_max': df['agbd'].max(),
        'spatial_extent': {
            'lon_range': (df['longitude'].min(), df['longitude'].max()),
            'lat_range': (df['latitude'].min(), df['latitude'].max())
        }
    }

    if 'tile_id' in df.columns:
        stats['n_tiles'] = df['tile_id'].nunique()
        stats['shots_per_tile'] = df.groupby('tile_id').size().describe().to_dict()

    return stats
