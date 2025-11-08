# Refactoring Implementation Guide

This guide provides step-by-step instructions for implementing the refactoring recommendations.

## Phase 1: High Priority (80+ lines of duplication)

### Step 1.1: Create `utils/model.py`

**Scope:** Consolidate model loading, initialization, and checkpoint operations

**Functions to implement:**

```python
# utils/model.py

from pathlib import Path
from typing import Optional, List, Dict, Tuple
import torch
import json
from models.neural_process import GEDINeuralProcess

def initialize_model_from_config(
    config: Dict,
    device: str
) -> GEDINeuralProcess:
    """
    Initialize GEDINeuralProcess model from configuration dictionary.
    
    Args:
        config: Configuration dict with model parameters
        device: Device to place model on ('cuda' or 'cpu')
    
    Returns:
        Initialized GEDINeuralProcess model on specified device
    """
    model = GEDINeuralProcess(
        patch_size=config.get('patch_size', 3),
        embedding_channels=128,
        embedding_feature_dim=config.get('embedding_feature_dim', 128),
        context_repr_dim=config.get('context_repr_dim', 128),
        hidden_dim=config.get('hidden_dim', 512),
        latent_dim=config.get('latent_dim', 128),
        output_uncertainty=True,
        architecture_mode=config.get('architecture_mode', 'deterministic'),
        num_attention_heads=config.get('num_attention_heads', 4)
    ).to(device)
    
    return model


def load_checkpoint(
    model: torch.nn.Module,
    checkpoint_path: Path,
    device: str,
    fallback_paths: Optional[List[Path]] = None
) -> Dict:
    """
    Load checkpoint into model with optional fallback paths.
    
    Args:
        model: Model to load state into
        checkpoint_path: Primary checkpoint path
        device: Device for loading
        fallback_paths: Optional list of fallback checkpoint paths
    
    Returns:
        Loaded checkpoint dictionary
    
    Raises:
        FileNotFoundError: If no checkpoint found at any path
    """
    # Try primary path
    if checkpoint_path.exists():
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        return checkpoint
    
    # Try fallback paths
    if fallback_paths:
        for fallback_path in fallback_paths:
            if fallback_path.exists():
                checkpoint = torch.load(fallback_path, map_location=device, weights_only=False)
                model.load_state_dict(checkpoint['model_state_dict'])
                return checkpoint
    
    # No checkpoint found
    paths_tried = [str(checkpoint_path)]
    if fallback_paths:
        paths_tried.extend([str(p) for p in fallback_paths])
    
    raise FileNotFoundError(
        f"No checkpoint found. Looked in:\n" + "\n".join(paths_tried)
    )


def find_and_load_checkpoint(
    model: torch.nn.Module,
    model_dir: Path,
    device: str,
    checkpoint_names: Optional[List[str]] = None
) -> Dict:
    """
    Find and load best checkpoint from model directory with fallbacks.
    
    Args:
        model: Model to load state into
        model_dir: Directory containing checkpoint files
        device: Device for loading
        checkpoint_names: List of checkpoint names to try (in order)
                         Default: ['best_r2_model.pt', 'best_model.pt']
    
    Returns:
        Loaded checkpoint dictionary
    """
    if checkpoint_names is None:
        checkpoint_names = ['best_r2_model.pt', 'best_model.pt']
    
    fallback_paths = [model_dir / name for name in checkpoint_names[1:]]
    primary_path = model_dir / checkpoint_names[0]
    
    return load_checkpoint(model, primary_path, device, fallback_paths)
```

**Files that will be updated:**
- `/home/user/gedinp/predict.py` - Replace `load_model_and_config()` with new utility
- `/home/user/gedinp/evaluate.py` - Replace checkpoint loading code
- `/home/user/gedinp/evaluate_temporal.py` - Replace checkpoint loading code
- `/home/user/gedinp/diagnostics.py` - Replace checkpoint loading code
- `/home/user/gedinp/train.py` - Replace model initialization code

---

### Step 1.2: Create `utils/config.py`

**Scope:** Consolidate config loading, saving, and management

**Functions to implement:**

```python
# utils/config.py

from pathlib import Path
from typing import Dict, Any
import json
import logging

logger = logging.getLogger(__name__)

def load_config(config_path: Path) -> Dict[str, Any]:
    """
    Load JSON configuration file.
    
    Args:
        config_path: Path to config.json file
    
    Returns:
        Configuration dictionary
    
    Raises:
        FileNotFoundError: If config file doesn't exist
        json.JSONDecodeError: If config file is invalid JSON
    """
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found at {config_path}")
    
    with open(config_path, 'r') as f:
        try:
            config = json.load(f)
        except json.JSONDecodeError as e:
            raise json.JSONDecodeError(
                f"Invalid JSON in {config_path}: {e.msg}",
                e.doc,
                e.pos
            )
    
    logger.info(f"Loaded config from {config_path}")
    return config


def load_model_config(model_dir: Path) -> Dict[str, Any]:
    """
    Load config from model directory.
    
    Args:
        model_dir: Directory containing config.json
    
    Returns:
        Configuration dictionary
    """
    return load_config(model_dir / 'config.json')


def save_config(config: Dict[str, Any], output_path: Path) -> None:
    """
    Save configuration to JSON file.
    
    Args:
        config: Configuration dictionary to save
        output_path: Path where to save config.json
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    logger.info(f"Saved config to {output_path}")


def setup_output_directory(output_dir: Path) -> Path:
    """
    Create output directory.
    
    Args:
        output_dir: Path to create
    
    Returns:
        Created output directory path
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Created output directory: {output_dir}")
    return output_dir


def save_args_as_config(args_dict: Dict, output_dir: Path) -> Path:
    """
    Save command-line arguments as config.json.
    
    Args:
        args_dict: Dictionary of arguments (e.g., vars(args))
        output_dir: Directory to save config in
    
    Returns:
        Path to saved config
    """
    config_path = output_dir / 'config.json'
    save_config(args_dict, config_path)
    return config_path
```

**Files that will be updated:**
- `/home/user/gedinp/train.py` - Replace config loading/saving
- `/home/user/gedinp/train_baselines.py` - Replace config loading/saving
- `/home/user/gedinp/run_ablation_study.py` - Replace config loading/saving
- `/home/user/gedinp/predict.py` - Replace config loading
- `/home/user/gedinp/evaluate.py` - Replace config loading
- `/home/user/gedinp/evaluate_temporal.py` - Replace config loading

---

## Phase 2: Medium Priority (150+ lines of duplication)

### Step 2.1: Create `utils/data_loading.py`

**Scope:** Consolidate GEDI querying, embedding extraction, and dataset creation

**Key functions:**

```python
# utils/data_loading.py

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, List, Tuple, Dict
from torch.utils.data import DataLoader
from tqdm import tqdm

from data.gedi import GEDIQuerier
from data.embeddings import EmbeddingExtractor
from data.dataset import GEDINeuralProcessDataset, collate_neural_process

def query_gedi_data(
    region_bbox: Tuple[float, float, float, float],
    start_time: str,
    end_time: str,
    filter_years: Optional[List[int]] = None,
    verbose: bool = True
) -> pd.DataFrame:
    """
    Query GEDI data with optional temporal filtering.
    
    Args:
        region_bbox: (min_lon, min_lat, max_lon, max_lat)
        start_time: Start date (YYYY-MM-DD)
        end_time: End date (YYYY-MM-DD)
        filter_years: Optional years to filter to
        verbose: Print progress information
    
    Returns:
        DataFrame with GEDI shots
    """
    if verbose:
        print("Querying GEDI data...")
    
    querier = GEDIQuerier()
    gedi_df = querier.query_region_tiles(
        region_bbox=region_bbox,
        tile_size=0.1,
        start_time=start_time,
        end_time=end_time
    )
    
    if verbose:
        print(f"Retrieved {len(gedi_df)} GEDI shots across {gedi_df['tile_id'].nunique()} tiles")
    
    # Temporal filtering if requested
    if filter_years is not None:
        gedi_df = filter_dataframe_by_years(gedi_df, filter_years, verbose=verbose)
    
    return gedi_df


def extract_year_from_dataframe(df: pd.DataFrame) -> pd.Series:
    """
    Extract year from datetime column trying multiple column names.
    
    Args:
        df: DataFrame with datetime column
    
    Returns:
        Series with year values
    
    Raises:
        ValueError: If no datetime column found
    """
    # Try common column names
    for col_name in ['time', 'date_time', 'datetime']:
        if col_name in df.columns:
            return pd.to_datetime(df[col_name]).dt.year
    
    # Try index as datetime
    try:
        return pd.to_datetime(df.index).year
    except:
        pass
    
    raise ValueError(
        f"Could not find datetime column. Available: {list(df.columns)}"
    )


def filter_dataframe_by_years(
    df: pd.DataFrame,
    years: List[int],
    verbose: bool = True
) -> pd.DataFrame:
    """
    Filter dataframe to only include specified years.
    
    Args:
        df: DataFrame to filter
        years: List of years to keep
        verbose: Print filtering statistics
    
    Returns:
        Filtered DataFrame
    """
    year_series = extract_year_from_dataframe(df)
    
    n_before = len(df)
    df_filtered = df[year_series.isin(years)].copy()
    n_after = len(df_filtered)
    
    if verbose:
        print(f"Temporal filtering: {n_before} -> {n_after} shots ({100*n_after/n_before:.1f}%)")
        print(f"Years in data: {sorted(year_series.unique().tolist())}")
    
    return df_filtered


def extract_embeddings_for_dataframe(
    df: pd.DataFrame,
    extractor: EmbeddingExtractor,
    batch_mode: bool = True,
    verbose: bool = True,
    desc: str = "Extracting embeddings"
) -> pd.DataFrame:
    """
    Extract embeddings for all rows in dataframe.
    
    Args:
        df: DataFrame with longitude/latitude columns
        extractor: EmbeddingExtractor instance
        batch_mode: Use batch extraction if True, else single-shot
        verbose: Print progress
        desc: Description for progress bar
    
    Returns:
        DataFrame with embedding_patch column added
    """
    if batch_mode and hasattr(extractor, 'extract_patches_batch'):
        # Use efficient batch method
        df = extractor.extract_patches_batch(df, verbose=verbose)
    else:
        # Fall back to single-shot extraction
        patches = []
        valid_indices = []
        
        if verbose:
            print(f"\n{desc}...")
        
        iterator = tqdm(df.iterrows(), total=len(df), desc=desc) if verbose else df.iterrows()
        
        for idx, row in iterator:
            patch = extractor.extract_patch(row['longitude'], row['latitude'])
            if patch is not None:
                patches.append(patch)
                valid_indices.append(idx)
        
        if verbose:
            success_rate = 100 * len(patches) / len(df)
            print(f"Successfully extracted {len(patches)}/{len(df)} embeddings ({success_rate:.1f}%)")
        
        df = df.loc[valid_indices].copy()
        df['embedding_patch'] = patches
    
    return df


def create_neural_process_dataloader(
    df: pd.DataFrame,
    config: Dict,
    global_bounds: Tuple[float, float, float, float],
    shuffle: bool = True,
    augment: bool = True,
    batch_size: Optional[int] = None,
    num_workers: int = 0
) -> DataLoader:
    """
    Create DataLoader for GEDINeuralProcessDataset.
    
    Args:
        df: DataFrame with GEDI data
        config: Configuration dict with dataset parameters
        global_bounds: Global coordinate bounds for normalization
        shuffle: Whether to shuffle data
        augment: Whether to apply coordinate augmentation
        batch_size: Batch size (uses config['batch_size'] if None)
        num_workers: Number of data loading workers
    
    Returns:
        DataLoader instance
    """
    if batch_size is None:
        batch_size = config.get('batch_size', 16)
    
    dataset = GEDINeuralProcessDataset(
        df,
        min_shots_per_tile=config.get('min_shots_per_tile', 10),
        log_transform_agbd=config.get('log_transform_agbd', True),
        augment_coords=augment,
        coord_noise_std=config.get('coord_noise_std', 0.01) if augment else 0.0,
        global_bounds=global_bounds
    )
    
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_neural_process,
        num_workers=num_workers
    )
    
    return loader
```

**Files that will be updated:**
- `/home/user/gedinp/train.py` - Replace GEDI querying, embedding extraction, dataset creation
- `/home/user/gedinp/train_baselines.py` - Replace GEDI querying, embedding extraction
- `/home/user/gedinp/evaluate_temporal.py` - Replace GEDI querying, embedding extraction, dataset creation
- `/home/user/gedinp/predict.py` - Optional: replace GEDI querying (but keep custom embedding extraction for now)

---

### Step 2.2: Create `utils/arguments.py`

**Scope:** Consolidate common argument parsing

**Key functions:**

```python
# utils/arguments.py

import argparse
import torch

def get_default_device() -> str:
    """Get CUDA device or CPU."""
    return 'cuda' if torch.cuda.is_available() else 'cpu'


def add_common_data_arguments(parser: argparse.ArgumentParser) -> None:
    """Add region_bbox, start_time, end_time, embedding_year, cache_dir arguments."""
    parser.add_argument('--region_bbox', type=float, nargs=4, required=True,
                        help='Region bounding box: min_lon min_lat max_lon max_lat')
    parser.add_argument('--start_time', type=str, default='2019-01-01',
                        help='Start date for GEDI data (YYYY-MM-DD)')
    parser.add_argument('--end_time', type=str, default='2023-12-31',
                        help='End date for GEDI data (YYYY-MM-DD)')
    parser.add_argument('--embedding_year', type=int, default=2024,
                        help='Year of GeoTessera embeddings')
    parser.add_argument('--cache_dir', type=str, default='./cache',
                        help='Directory for caching tiles and embeddings')


def add_device_argument(parser: argparse.ArgumentParser) -> None:
    """Add --device argument with proper default."""
    parser.add_argument('--device', type=str, default=get_default_device(),
                        help='Device (cuda/cpu)')


def add_seed_argument(parser: argparse.ArgumentParser, required: bool = False) -> None:
    """Add --seed argument."""
    parser.add_argument('--seed', type=int, default=42,
                        required=required,
                        help='Random seed')


def add_model_arguments(parser: argparse.ArgumentParser) -> None:
    """Add model architecture arguments."""
    parser.add_argument('--patch_size', type=int, default=3,
                        help='Embedding patch size (default: 3x3)')
    parser.add_argument('--hidden_dim', type=int, default=512,
                        help='Hidden layer dimension')
    parser.add_argument('--embedding_feature_dim', type=int, default=128,
                        help='Embedding feature dimension')
    parser.add_argument('--context_repr_dim', type=int, default=128,
                        help='Context representation dimension')
    parser.add_argument('--latent_dim', type=int, default=128,
                        help='Latent variable dimension')
    parser.add_argument('--num_attention_heads', type=int, default=4,
                        help='Number of attention heads')
```

**Usage example:**

```python
# Instead of:
parser = argparse.ArgumentParser(description='Train GEDI Neural Process')
parser.add_argument('--region_bbox', type=float, nargs=4, required=True, ...)
parser.add_argument('--start_time', type=str, default='2019-01-01', ...)
# ... 20+ more lines

# Use:
from utils.arguments import add_common_data_arguments, add_device_argument, add_seed_argument

parser = argparse.ArgumentParser(description='Train GEDI Neural Process')
add_common_data_arguments(parser)
add_device_argument(parser)
add_seed_argument(parser)
```

---

## Phase 3: Low Priority

### Step 3.1: Create `utils/common.py`

**Key functions:**

```python
def set_random_seed(seed: int) -> None:
    """Set seed for PyTorch, NumPy, and CUDA."""
    import torch
    import numpy as np
    
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


def make_serializable(obj) -> object:
    """Convert numpy/torch types to Python natives for JSON serialization."""
    import numpy as np
    
    if isinstance(obj, dict):
        return {key: make_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [make_serializable(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(make_serializable(item) for item in obj)
    elif isinstance(obj, (np.integer, np.int32, np.int64)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.bool_):
        return bool(obj)
    else:
        return obj
```

---

## Migration Checklist

### Phase 1
- [ ] Create `utils/model.py`
- [ ] Update `predict.py` to use model utilities
- [ ] Update `evaluate.py` to use model utilities
- [ ] Update `evaluate_temporal.py` to use model utilities
- [ ] Update `diagnostics.py` to use model utilities
- [ ] Update `train.py` to use model utilities
- [ ] Create `utils/config.py`
- [ ] Update all 7 files to use config utilities
- [ ] Run all scripts to verify functionality

### Phase 2
- [ ] Create `utils/data_loading.py`
- [ ] Update `train.py` to use data loading utilities
- [ ] Update `train_baselines.py` to use data loading utilities
- [ ] Update `evaluate_temporal.py` to use data loading utilities
- [ ] Create `utils/arguments.py`
- [ ] Refactor argument parsing in all scripts
- [ ] Run all scripts to verify functionality

### Phase 3
- [ ] Create `utils/common.py`
- [ ] Update `train.py` to use common utilities
- [ ] Update `train_baselines.py` to use common utilities
- [ ] Update `evaluate_temporal.py` to use common utilities
- [ ] Enhance `utils/evaluation.py` for baseline models
- [ ] Run all scripts to verify functionality
- [ ] Run full test suite

---

## Testing Strategy

After each phase:

1. **Unit Tests** - Test new utility functions in isolation
2. **Integration Tests** - Test updated scripts run without errors
3. **Regression Tests** - Verify outputs are identical to before refactoring

Example test commands:
```bash
# Test config utilities
python -m pytest tests/test_config.py

# Test model utilities
python -m pytest tests/test_model.py

# Test data loading
python -m pytest tests/test_data_loading.py

# Integration test - run a quick training
python train.py --region_bbox -180 -90 180 90 --epochs 1 --output_dir /tmp/test_output
```

---

## Benefits Summary

**Phase 1:**
- Reduce 80+ lines of duplication
- Single point of change for model initialization
- Consistent checkpoint loading with fallbacks

**Phase 2:**
- Reduce 150+ lines of duplication
- Reusable data pipeline components
- Standardized argument parsing

**Phase 3:**
- Reduce 45+ lines of duplication
- Utilities available across all scripts
- Improved code maintainability

**Total Impact:**
- 275+ lines of duplicated code eliminated
- 5+ fewer places to update when making changes
- Improved code quality and maintainability
- Better separation of concerns

