"""Transform functions for AGBD and coordinate normalization/denormalization."""

import numpy as np


def normalize_agbd(agbd: np.ndarray, agbd_scale: float = 200.0) -> np.ndarray:
    """
    Normalize AGBD values using log transform.

    Args:
        agbd: Raw AGBD values in Mg/ha
        agbd_scale: Scale factor (default: 200.0)

    Returns:
        Normalized AGBD values
    """
    return np.log1p(agbd) / np.log1p(agbd_scale)


def denormalize_agbd(agbd_norm: np.ndarray, agbd_scale: float = 200.0, log_transform: bool = True) -> np.ndarray:
    """
    Denormalize AGBD values back to raw values (Mg/ha).

    Args:
        agbd_norm: Normalized AGBD values
        agbd_scale: Scale factor (default: 200.0)
        log_transform: Whether to use log transform (default: True)
                       If False, uses linear scaling instead

    Returns:
        Raw AGBD values in Mg/ha
    """
    if log_transform:
        return np.expm1(agbd_norm * np.log1p(agbd_scale))
    else:
        return agbd_norm * agbd_scale


def denormalize_std(std_norm: np.ndarray, agbd_norm: np.ndarray, agbd_scale: float = 200.0) -> np.ndarray:
    """
    Convert normalized standard deviation to raw values (Mg/ha).

    Uses the derivative of the log transform at the predicted mean:
    d/dx[log(1+x)] = 1/(1+x)

    For log-normal distributions, the standard deviation transforms as:
    std_raw ≈ std_norm * log(1+scale) * (1 + mean_raw)

    Args:
        std_norm: Normalized standard deviation
        agbd_norm: Normalized AGBD mean values (for proper scaling)
        agbd_scale: Scale factor (default: 200.0)

    Returns:
        Raw standard deviation in Mg/ha
    """
    # Denormalize the mean first to get the scale factor
    mean_raw = denormalize_agbd(agbd_norm, agbd_scale)

    # Transform std using derivative of log at the mean
    # For log(1+x), derivative is 1/(1+x), but we're in normalized space
    # so we need to scale by log(1+scale) and multiply by (1+mean_raw)
    std_raw = std_norm * np.log1p(agbd_scale) * (1 + mean_raw)

    return std_raw


def normalize_coords(coords: np.ndarray, global_bounds: tuple) -> np.ndarray:
    """
    Normalize coordinates to [0, 1] range based on global bounds.

    Args:
        coords: Array of shape (N, 2) with [lon, lat] coordinates
        global_bounds: Tuple of (lon_min, lat_min, lon_max, lat_max)

    Returns:
        Normalized coordinates in [0, 1] range
    """
    lon_min, lat_min, lon_max, lat_max = global_bounds
    lon_range = lon_max - lon_min if lon_max > lon_min else 1.0
    lat_range = lat_max - lat_min if lat_max > lat_min else 1.0

    normalized = coords.copy()
    normalized[:, 0] = (coords[:, 0] - lon_min) / lon_range
    normalized[:, 1] = (coords[:, 1] - lat_min) / lat_range

    return normalized
