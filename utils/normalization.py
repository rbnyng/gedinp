"""
Normalization and denormalization utilities for GEDI AGBD data.

This module provides functions for normalizing and denormalizing:
- Coordinates (longitude, latitude)
- AGBD values (Above Ground Biomass Density)
- Standard deviations (uncertainty estimates)
"""

import numpy as np
from typing import Tuple, Union
import pandas as pd


def compute_agbd_scale(
    agbd: Union[np.ndarray, pd.Series],
    percentile: float = 99.0,
    max_cap: float = 500.0,
    default_scale: float = 200.0
) -> float:
    """
    Compute data-driven AGBD scale based on percentile of training data.

    Args:
        agbd: AGBD values from training data (Mg/ha)
        percentile: Percentile to use for scale (default: 99.0)
        max_cap: Maximum cap on scale value (default: 500.0 Mg/ha)
        default_scale: Default scale if data is empty or invalid (default: 200.0)

    Returns:
        Computed scale factor (float)
    """
    if isinstance(agbd, pd.Series):
        agbd = agbd.values

    # Remove NaN values
    agbd_clean = agbd[~np.isnan(agbd)]

    if len(agbd_clean) == 0:
        print(f"Warning: No valid AGBD values found. Using default scale: {default_scale}")
        return default_scale

    # Compute percentile and apply cap
    scale = np.percentile(agbd_clean, percentile)
    scale = min(scale, max_cap)

    # Ensure scale is reasonable (at least 50 Mg/ha)
    scale = max(scale, 50.0)

    print(f"Computed AGBD scale: {scale:.2f} Mg/ha (from {percentile}th percentile, capped at {max_cap})")
    print(f"  Data range: [{agbd_clean.min():.2f}, {agbd_clean.max():.2f}] Mg/ha")
    print(f"  Mean: {agbd_clean.mean():.2f} Mg/ha, Median: {np.median(agbd_clean):.2f} Mg/ha")

    return float(scale)


def normalize_coords(coords: np.ndarray, global_bounds: Tuple[float, float, float, float]) -> np.ndarray:
    """
    Normalize coordinates to [0, 1] range using global bounds.

    Args:
        coords: (N, 2) array of [lon, lat] coordinates
        global_bounds: (lon_min, lat_min, lon_max, lat_max) tuple

    Returns:
        Normalized coordinates (N, 2) in [0, 1] range
    """
    lon_min, lat_min, lon_max, lat_max = global_bounds

    lon_range = lon_max - lon_min if lon_max > lon_min else 1.0
    lat_range = lat_max - lat_min if lat_max > lat_min else 1.0

    normalized = coords.copy()
    normalized[:, 0] = (coords[:, 0] - lon_min) / lon_range
    normalized[:, 1] = (coords[:, 1] - lat_min) / lat_range

    return normalized


def normalize_agbd(agbd: np.ndarray, agbd_scale: float = 200.0, log_transform: bool = True) -> np.ndarray:
    """
    Normalize AGBD values.

    Args:
        agbd: Raw AGBD values in Mg/ha
        agbd_scale: Scale factor for normalization (default: 200.0)
        log_transform: If True, apply log1p transform before normalizing (default: True)

    Returns:
        Normalized AGBD values
    """
    if log_transform:
        return np.log1p(agbd) / np.log1p(agbd_scale)
    else:
        return agbd / agbd_scale


def denormalize_agbd(agbd_norm: np.ndarray, agbd_scale: float = 200.0, log_transform: bool = True) -> np.ndarray:
    """
    Denormalize AGBD values back to raw Mg/ha.

    Args:
        agbd_norm: Normalized AGBD values
        agbd_scale: Scale factor used in normalization (default: 200.0)
        log_transform: If True, apply expm1 to reverse log1p transform (default: True)

    Returns:
        Raw AGBD values in Mg/ha
    """
    if log_transform:
        return np.expm1(agbd_norm * np.log1p(agbd_scale))
    else:
        return agbd_norm * agbd_scale


def denormalize_std(
    std_norm: np.ndarray,
    agbd_norm: np.ndarray,
    agbd_scale: float = 200.0,
    simple_transform: bool = False
) -> np.ndarray:
    """
    Convert normalized standard deviation to raw values (Mg/ha).

    For log-transformed data, the standard deviation transforms according to the
    derivative of the log transform. This function implements the proper
    mathematical transformation.

    Uses the derivative of the log transform at the predicted mean:
    d/dx[log(1+x)] = 1/(1+x)

    For log-normal distributions, the standard deviation transforms as:
    std_raw ≈ std_norm * log(1+scale) * (1 + mean_raw)

    Args:
        std_norm: Normalized standard deviation
        agbd_norm: Normalized AGBD mean values (for proper scaling)
        agbd_scale: Scale factor (default: 200.0)
        simple_transform: If True, use simple scaling without derivative correction (default: False)

    Returns:
        Raw standard deviation in Mg/ha
    """
    if simple_transform:
        # Simple transform: just scale by log(1+scale)
        return std_norm * np.log1p(agbd_scale)
    else:
        # Proper transform using derivative of log at the mean
        # Denormalize the mean first to get the scale factor
        mean_raw = denormalize_agbd(agbd_norm, agbd_scale)

        # Transform std using derivative of log at the mean
        # For log(1+x), derivative is 1/(1+x), but we're in normalized space
        # so we need to scale by log(1+scale) and multiply by (1+mean_raw)
        std_raw = std_norm * np.log1p(agbd_scale) * (1 + mean_raw)

        return std_raw
