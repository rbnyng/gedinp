import numpy as np
import pandas as pd
from typing import Tuple, List, Dict, Optional
from sklearn.model_selection import KFold
import random


class SpatialTileSplitter:
    def __init__(
        self,
        data_df: pd.DataFrame,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
        random_state: int = 42
    ):
        self.data_df = data_df
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.random_state = random_state

        self.tile_ids = data_df['tile_id'].unique()
        self.n_tiles = len(self.tile_ids)

        random.seed(random_state)
        np.random.seed(random_state)

    def split(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        # Sort tiles to consistent order before shuffling
        sorted_tiles = np.sort(self.tile_ids)

        # Shuffle tiles
        shuffled_tiles = sorted_tiles.copy()
        np.random.shuffle(shuffled_tiles)

        # split sizes
        n_test = max(1, int(self.n_tiles * self.test_ratio)) if self.test_ratio > 0 else 0
        n_val = max(1, int(self.n_tiles * self.val_ratio)) if self.val_ratio > 0 else 0
        n_train = self.n_tiles - n_test - n_val

        # Split tiles
        train_tiles = shuffled_tiles[:n_train]
        val_tiles = shuffled_tiles[n_train:n_train + n_val]
        test_tiles = shuffled_tiles[n_train + n_val:]

        # dataframe splits
        train_df = self.data_df[self.data_df['tile_id'].isin(train_tiles)]
        val_df = self.data_df[self.data_df['tile_id'].isin(val_tiles)]
        test_df = self.data_df[self.data_df['tile_id'].isin(test_tiles)]

        print(f"Spatial split created:")
        print(f"  Train: {len(train_tiles)} tiles, {len(train_df)} shots")
        print(f"  Val:   {len(val_tiles)} tiles, {len(val_df)} shots")
        print(f"  Test:  {len(test_tiles)} tiles, {len(test_df)} shots")

        return train_df, val_df, test_df

    def k_fold_split(self, n_folds: int = 5) -> List[Tuple[pd.DataFrame, pd.DataFrame]]:
        kf = KFold(n_splits=n_folds, shuffle=True, random_state=self.random_state)

        splits = []
        for train_idx, val_idx in kf.split(self.tile_ids):
            train_tiles = self.tile_ids[train_idx]
            val_tiles = self.tile_ids[val_idx]

            train_df = self.data_df[self.data_df['tile_id'].isin(train_tiles)]
            val_df = self.data_df[self.data_df['tile_id'].isin(val_tiles)]

            splits.append((train_df, val_df))

        print(f"Created {n_folds}-fold spatial CV")
        for i, (train_df, val_df) in enumerate(splits):
            print(f"  Fold {i+1}: Train {len(train_df)} shots, Val {len(val_df)} shots")

        return splits


class BufferedSpatialSplitter:
    def __init__(
        self,
        data_df: pd.DataFrame,
        buffer_size: float = 0.1,  # degrees
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
        random_state: int = 42
    ):
        """
        Initialize buffered splitter.

        Args:
            data_df: DataFrame with 'tile_lon', 'tile_lat', 'tile_id' columns
            buffer_size: Buffer distance in degrees between splits
            val_ratio: Fraction of tiles for validation
            test_ratio: Fraction of tiles for test
            random_state: Random seed
        """
        self.data_df = data_df
        self.buffer_size = buffer_size
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.random_state = random_state

        random.seed(random_state)
        np.random.seed(random_state)

        # tile centers
        self.tile_info = (
            data_df[['tile_id', 'tile_lon', 'tile_lat']]
            .drop_duplicates()
            .set_index('tile_id')
        )

    def _compute_distance(self, tile1: str, tile2: str) -> float:
        lon1, lat1 = self.tile_info.loc[tile1, ['tile_lon', 'tile_lat']]
        lon2, lat2 = self.tile_info.loc[tile2, ['tile_lon', 'tile_lat']]

        # Euclidean distance (good enough for small regions)
        return np.sqrt((lon2 - lon1)**2 + (lat2 - lat1)**2)

    def split(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        tile_ids = np.sort(self.tile_info.index.values)
        n_tiles = len(tile_ids)

        np.random.shuffle(tile_ids)
        n_test = max(1, int(n_tiles * self.test_ratio)) if self.test_ratio > 0 else 0
        test_tiles = tile_ids[:n_test]

        # Find tiles within buffer of test tiles
        test_buffer_tiles = set()
        for test_tile in test_tiles:
            for tile in tile_ids:
                if tile != test_tile:
                    dist = self._compute_distance(test_tile, tile)
                    if dist < self.buffer_size:
                        test_buffer_tiles.add(tile)

        # Remaining tiles excluding buffer
        remaining_tiles = [t for t in tile_ids if t not in test_tiles and t not in test_buffer_tiles]

        if len(remaining_tiles) < 2:
            print("Warning: Not enough tiles for buffered split, falling back to simple split")
            return SpatialTileSplitter(
                self.data_df, self.val_ratio, self.test_ratio, self.random_state
            ).split()

        # Select validation tiles from remaining
        if self.val_ratio > 0:
            n_val = max(1, int(len(remaining_tiles) * self.val_ratio / (1 - self.test_ratio)))
            n_val = min(n_val, len(remaining_tiles) - 1)
        else:
            n_val = 0
        val_tiles = remaining_tiles[:n_val]

        # Find tiles within buffer of val tiles
        val_buffer_tiles = set()
        for val_tile in val_tiles:
            for tile in remaining_tiles:
                if tile != val_tile:
                    dist = self._compute_distance(val_tile, tile)
                    if dist < self.buffer_size:
                        val_buffer_tiles.add(tile)

        # Train tiles
        train_tiles = [
            t for t in remaining_tiles
            if t not in val_tiles and t not in val_buffer_tiles
        ]

        train_df = self.data_df[self.data_df['tile_id'].isin(train_tiles)]
        val_df = self.data_df[self.data_df['tile_id'].isin(val_tiles)]
        test_df = self.data_df[self.data_df['tile_id'].isin(test_tiles)]

        print(f"Buffered spatial split created (buffer={self.buffer_size}°):")
        print(f"  Train: {len(train_tiles)} tiles, {len(train_df)} shots")
        print(f"  Val:   {len(val_tiles)} tiles, {len(val_df)} shots")
        print(f"  Test:  {len(test_tiles)} tiles, {len(test_df)} shots")
        print(f"  Excluded (buffers): {len(test_buffer_tiles) + len(val_buffer_tiles)} tiles")

        return train_df, val_df, test_df


def analyze_spatial_split(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame
) -> Dict:
    def get_extent(df):
        return {
            'lon_range': (df['longitude'].min(), df['longitude'].max()),
            'lat_range': (df['latitude'].min(), df['latitude'].max()),
            'center': (df['longitude'].mean(), df['latitude'].mean())
        }

    analysis = {
        'train': {
            'n_tiles': df['tile_id'].nunique(),
            'n_shots': len(train_df),
            'extent': get_extent(train_df),
            'agbd_stats': {
                'mean': train_df['agbd'].mean(),
                'std': train_df['agbd'].std(),
                'min': train_df['agbd'].min(),
                'max': train_df['agbd'].max()
            }
        },
        'val': {
            'n_tiles': val_df['tile_id'].nunique(),
            'n_shots': len(val_df),
            'extent': get_extent(val_df),
            'agbd_stats': {
                'mean': val_df['agbd'].mean(),
                'std': val_df['agbd'].std(),
                'min': val_df['agbd'].min(),
                'max': val_df['agbd'].max()
            }
        },
        'test': {
            'n_tiles': test_df['tile_id'].nunique(),
            'n_shots': len(test_df),
            'extent': get_extent(test_df),
            'agbd_stats': {
                'mean': test_df['agbd'].mean(),
                'std': test_df['agbd'].std(),
                'min': test_df['agbd'].min(),
                'max': test_df['agbd'].max()
            }
        }
    }

    return analysis
