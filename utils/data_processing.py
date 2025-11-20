"""
Data processing utilities for GEDI Neural Process training.

This module provides common data processing functions used across
training and evaluation scripts.
"""

import pandas as pd
import numpy as np
from typing import Optional


def prepare_embeddings_for_parquet(df: pd.DataFrame) -> pd.DataFrame:
    """
    Flatten embedding patches for Parquet serialization.

    Parquet doesn't handle nested numpy arrays well, so this function
    flattens (H, W, C) embedding patches to 1D lists for storage.

    Args:
        df: DataFrame with 'embedding_patch' column containing numpy arrays

    Returns:
        Copy of DataFrame with flattened embeddings as lists

    Example:
        >>> df_copy = prepare_embeddings_for_parquet(train_df)
        >>> df_copy.to_parquet('train_split.parquet', index=False)
    """
    df_copy = df.copy()
    df_copy['embedding_patch'] = df_copy['embedding_patch'].apply(
        lambda x: x.flatten().tolist() if x is not None else None
    )
    return df_copy
