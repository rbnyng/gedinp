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


class GEDIQuerier:
    """Query GEDI data from gediDB."""

    def __init__(
        self,
        storage_type: str = 's3',
        s3_bucket: str = "dog.gedidb.gedi-l2-l4-v002",
        url: str = "https://s3.gfz-potsdam.de",
        local_path: Optional[str] = None
    ):
        """
        Initialize GEDI data provider.

        Args:
            storage_type: 's3' for cloud access or 'local' for local database
            s3_bucket: S3 bucket name for cloud access
            url: S3 endpoint URL
            local_path: Path to local gediDB if storage_type='local'
        """
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
        start_time: str = "2019-01-01",
        end_time: str = "2023-12-31",
        variables: Optional[List[str]] = None,
        quality_filter: bool = True,
        min_agbd: float = 0.0,
        max_agbd: Optional[float] = None
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

        Returns:
            DataFrame with columns: latitude, longitude, agbd, quality metrics
        """
        if variables is None:
            variables = [
                "latitude", "longitude", "agbd",
                "l4_quality_flag", "sensitivity",
                "shot_number", "beam"
            ]

        # Create bbox geometry as GeoDataFrame (required by gediDB)
        bbox_geom = box(*bbox)
        roi = gpd.GeoDataFrame([1], geometry=[bbox_geom], crs="EPSG:4326")

        # Query data
        gedi_data = self.provider.get_data(
            variables=variables,
            query_type="bounding_box",
            geometry=roi,
            start_time=start_time,
            end_time=end_time,
            return_type='xarray'
        )

        # Convert to DataFrame
        df = gedi_data.to_dataframe().reset_index()

        # Apply quality filtering
        if quality_filter and 'l4_quality_flag' in df.columns:
            # Keep only high quality shots (flag == 1 typically indicates good quality)
            df = df[df['l4_quality_flag'] == 1]

        # Filter by AGBD range
        df = df[df['agbd'] >= min_agbd]
        if max_agbd is not None:
            df = df[df['agbd'] <= max_agbd]

        # Remove NaN values
        df = df.dropna(subset=['latitude', 'longitude', 'agbd'])

        return df

    def query_tile(
        self,
        tile_lon: float,
        tile_lat: float,
        tile_size: float = 0.1,
        **kwargs
    ) -> pd.DataFrame:
        """
        Query GEDI shots within a single tile.

        Args:
            tile_lon: Tile center longitude
            tile_lat: Tile center latitude
            tile_size: Tile size in degrees (default 0.1° for GeoTessera alignment)
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
        return self.query_bbox(bbox, **kwargs)

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
