"""
PyTorch Dataset for ConvCNP training with dense tile representations.

Creates sparse tile tensors with AGBD values, masks, and dense embeddings.
"""

import torch
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
from typing import Optional, Dict, Tuple, List
import random
from pathlib import Path
import pickle
from pyproj import Transformer
from geotessera import GeoTessera


class ConvCNPTileDataset(Dataset):
    """
    Dataset for ConvCNP training with full tile representations.

    Each sample is a complete tile with:
    - Dense GeoTessera embeddings (H x W x 128)
    - Sparse AGBD values at GEDI shot locations
    - Binary mask indicating GEDI shot locations

    Creates context/target splits by masking random GEDI shots.
    """

    def __init__(
        self,
        data_df: pd.DataFrame,
        embedding_year: int = 2024,
        cache_dir: Optional[str] = None,
        min_shots_per_tile: int = 10,
        context_ratio_range: Tuple[float, float] = (0.1, 0.9),
        normalize_agbd: bool = True,
        agbd_scale: float = 200.0,
        max_tile_size: Optional[int] = None
    ):
        """
        Initialize ConvCNP dataset.

        Args:
            data_df: DataFrame with columns: latitude, longitude, agbd, tile_id
            embedding_year: Year of GeoTessera embeddings
            cache_dir: Directory for caching tiles
            min_shots_per_tile: Minimum number of GEDI shots per tile
            context_ratio_range: Range of context/total ratios for training (min, max)
            normalize_agbd: Normalize AGBD values
            agbd_scale: Scale factor for AGBD normalization
            max_tile_size: Maximum tile dimension (for downsampling large tiles)
        """
        self.data_df = data_df.copy()
        self.embedding_year = embedding_year
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.context_ratio_range = context_ratio_range
        self.normalize_agbd = normalize_agbd
        self.agbd_scale = agbd_scale
        self.max_tile_size = max_tile_size

        # Initialize GeoTessera
        self.gt = GeoTessera()

        # Cache for tiles
        self.tile_cache: Dict[Tuple[float, float], Tuple[np.ndarray, object, object]] = {}
        self.transformer_cache: Dict[str, Transformer] = {}

        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Group by tiles and filter
        self.tiles_data = []
        for tile_id, group in self.data_df.groupby('tile_id'):
            if len(group) >= min_shots_per_tile:
                # Extract tile coordinates from first shot
                first_shot = group.iloc[0]
                tile_lon, tile_lat = self._get_tile_coords(
                    first_shot['longitude'],
                    first_shot['latitude']
                )

                self.tiles_data.append({
                    'tile_id': tile_id,
                    'tile_lon': tile_lon,
                    'tile_lat': tile_lat,
                    'shots': group
                })

        print(f"ConvCNP Dataset initialized with {len(self.tiles_data)} tiles")
        if len(self.tiles_data) > 0:
            shots_per_tile = [len(t['shots']) for t in self.tiles_data]
            print(f"Shots per tile: min={min(shots_per_tile)}, "
                  f"max={max(shots_per_tile)}, mean={np.mean(shots_per_tile):.1f}")

    def _get_tile_coords(self, lon: float, lat: float) -> Tuple[float, float]:
        """Get tile center coordinates for a given point."""
        tile_lon = round((lon - 0.05) / 0.1) * 0.1 + 0.05
        tile_lat = round((lat - 0.05) / 0.1) * 0.1 + 0.05
        return tile_lon, tile_lat

    def _load_tile(
        self,
        tile_lon: float,
        tile_lat: float
    ) -> Optional[Tuple[np.ndarray, object, object]]:
        """Load a tile from cache or download it."""
        tile_key = (tile_lon, tile_lat)

        # Check memory cache
        if tile_key in self.tile_cache:
            return self.tile_cache[tile_key]

        # Check disk cache
        if self.cache_dir:
            cache_file = self.cache_dir / f"tile_{tile_lon:.2f}_{tile_lat:.2f}_{self.embedding_year}.pkl"
            if cache_file.exists():
                with open(cache_file, 'rb') as f:
                    tile_data = pickle.load(f)
                    self.tile_cache[tile_key] = tile_data
                    return tile_data

        # Download from GeoTessera
        try:
            embedding, crs, transform = self.gt.fetch_embedding(
                lon=tile_lon,
                lat=tile_lat,
                year=self.embedding_year
            )

            tile_data = (embedding, crs, transform)
            self.tile_cache[tile_key] = tile_data

            # Save to disk cache
            if self.cache_dir:
                cache_file = self.cache_dir / f"tile_{tile_lon:.2f}_{tile_lat:.2f}_{self.embedding_year}.pkl"
                with open(cache_file, 'wb') as f:
                    pickle.dump(tile_data, f)

            return tile_data

        except Exception as e:
            print(f"Warning: Could not fetch tile at ({tile_lon}, {tile_lat}): {e}")
            return None

    def _lonlat_to_pixel(
        self,
        lon: float,
        lat: float,
        transform,
        crs
    ) -> Tuple[int, int]:
        """Convert lon/lat to pixel coordinates."""
        crs_str = str(crs)

        # Get or create transformer
        if crs_str not in self.transformer_cache:
            self.transformer_cache[crs_str] = Transformer.from_crs(
                "EPSG:4326",
                crs,
                always_xy=True
            )

        transformer = self.transformer_cache[crs_str]
        x, y = transformer.transform(lon, lat)
        col, row = ~transform * (x, y)

        return int(row), int(col)

    def __len__(self) -> int:
        return len(self.tiles_data)

    def __getitem__(self, idx: int) -> Optional[Dict[str, torch.Tensor]]:
        """
        Get a training sample.

        Returns a dict with:
            - tile_embedding: (C, H, W) dense embeddings
            - context_agbd: (H, W) sparse AGBD values (0 where no data)
            - context_mask: (H, W) binary mask (1 at context locations)
            - target_agbd: (H, W) sparse AGBD values at target locations
            - target_mask: (H, W) binary mask (1 at target locations)
        """
        tile_info = self.tiles_data[idx]

        # Load tile
        tile_data = self._load_tile(tile_info['tile_lon'], tile_info['tile_lat'])
        if tile_data is None:
            return None

        embedding, crs, transform = tile_data

        # embedding shape: (H, W, C) where C=128
        H, W, C = embedding.shape

        # Optionally downsample large tiles
        if self.max_tile_size and max(H, W) > self.max_tile_size:
            scale = self.max_tile_size / max(H, W)
            new_H = int(H * scale)
            new_W = int(W * scale)

            # Simple downsampling using numpy (average pooling)
            from scipy.ndimage import zoom
            embedding = zoom(embedding, (new_H/H, new_W/W, 1), order=1)
            H, W = new_H, new_W

        # Convert to (C, H, W) format for PyTorch
        tile_embedding = embedding.transpose(2, 0, 1)  # (C, H, W)

        # Create sparse AGBD and mask arrays
        agbd_map = np.zeros((H, W), dtype=np.float32)
        mask_map = np.zeros((H, W), dtype=np.float32)

        # Place GEDI shots on the map
        shots_df = tile_info['shots']
        valid_pixels = []

        for _, shot in shots_df.iterrows():
            try:
                row, col = self._lonlat_to_pixel(
                    shot['longitude'],
                    shot['latitude'],
                    transform,
                    crs
                )

                # Check bounds
                if 0 <= row < H and 0 <= col < W:
                    agbd_value = shot['agbd']
                    if self.normalize_agbd:
                        agbd_value = agbd_value / self.agbd_scale

                    agbd_map[row, col] = agbd_value
                    mask_map[row, col] = 1.0
                    valid_pixels.append((row, col, agbd_value))

            except Exception as e:
                continue

        # Need at least 2 shots for context/target split
        if len(valid_pixels) < 2:
            return None

        # Create context/target split
        n_total = len(valid_pixels)
        context_ratio = random.uniform(*self.context_ratio_range)
        n_context = max(1, int(n_total * context_ratio))

        # Randomly select context indices
        context_indices = set(random.sample(range(n_total), n_context))

        # Create context and target maps
        context_agbd = np.zeros((H, W), dtype=np.float32)
        context_mask = np.zeros((H, W), dtype=np.float32)
        target_agbd = np.zeros((H, W), dtype=np.float32)
        target_mask = np.zeros((H, W), dtype=np.float32)

        for i, (row, col, agbd_value) in enumerate(valid_pixels):
            if i in context_indices:
                context_agbd[row, col] = agbd_value
                context_mask[row, col] = 1.0
            else:
                target_agbd[row, col] = agbd_value
                target_mask[row, col] = 1.0

        return {
            'tile_embedding': torch.from_numpy(tile_embedding).float(),
            'context_agbd': torch.from_numpy(context_agbd).float().unsqueeze(0),  # (1, H, W)
            'context_mask': torch.from_numpy(context_mask).float().unsqueeze(0),  # (1, H, W)
            'target_agbd': torch.from_numpy(target_agbd).float().unsqueeze(0),    # (1, H, W)
            'target_mask': torch.from_numpy(target_mask).float().unsqueeze(0),    # (1, H, W)
        }


def collate_convcnp(batch: List[Optional[Dict]]) -> Optional[Dict[str, torch.Tensor]]:
    """
    Collate function for ConvCNP dataset.

    Filters out None samples and stacks tensors with padding to handle variable sizes.
    """
    # Filter out None samples
    batch = [b for b in batch if b is not None]

    if len(batch) == 0:
        return None

    # Find maximum spatial dimensions in the batch
    max_h = max(b['tile_embedding'].shape[1] for b in batch)
    max_w = max(b['tile_embedding'].shape[2] for b in batch)

    # Pad each sample to the maximum dimensions
    def pad_to_size(tensor, target_h, target_w):
        """Pad tensor to target spatial dimensions (H, W)."""
        # tensor shape: (C, H, W) or (1, H, W)
        c, h, w = tensor.shape
        pad_h = target_h - h
        pad_w = target_w - w
        # pad: (left, right, top, bottom)
        return torch.nn.functional.pad(tensor, (0, pad_w, 0, pad_h), mode='constant', value=0)

    # Pad and stack tensors
    return {
        'tile_embedding': torch.stack([pad_to_size(b['tile_embedding'], max_h, max_w) for b in batch]),
        'context_agbd': torch.stack([pad_to_size(b['context_agbd'], max_h, max_w) for b in batch]),
        'context_mask': torch.stack([pad_to_size(b['context_mask'], max_h, max_w) for b in batch]),
        'target_agbd': torch.stack([pad_to_size(b['target_agbd'], max_h, max_w) for b in batch]),
        'target_mask': torch.stack([pad_to_size(b['target_mask'], max_h, max_w) for b in batch]),
    }
