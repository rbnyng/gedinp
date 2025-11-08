# GEDI Codebase Refactoring Analysis Report

## Overview
Analysis of 2,867 lines of code across 7 main scripts and multiple utility modules to identify duplicate functions, common patterns, and refactoring opportunities.

## Key Findings Summary
- **5 instances** of repeated model loading and initialization (predict.py, evaluate.py, evaluate_temporal.py, diagnostics.py, train.py)
- **3 files** with duplicate GEDI data querying patterns
- **4 files** with similar dataset creation and embedding extraction patterns
- **3+ files** with duplicated argument parsing for common parameters
- **Multiple utility functions** used across scripts that should be centralized

---

## 1. MODEL LOADING AND INITIALIZATION DUPLICATION

### Current Status
The `GEDINeuralProcess` model initialization code is duplicated across 5 files with nearly identical logic.

### Examples:

**File: /home/user/gedinp/predict.py (lines 68-78)**
```python
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
```

**File: /home/user/gedinp/evaluate.py (lines 114-124)** - IDENTICAL
**File: /home/user/gedinp/evaluate_temporal.py (lines 179-189)** - IDENTICAL
**File: /home/user/gedinp/diagnostics.py (lines 523-533)** - IDENTICAL
**File: /home/user/gedinp/train.py (lines 419-429)** - IDENTICAL

### Impact
- 5x duplication (10 lines each = 50 lines total)
- Any changes to model initialization must be made in 5 places
- Risk of inconsistency between files

### Refactoring Opportunity
**Create `utils/model.py`** with helper function:
```python
def initialize_model_from_config(config: dict, device: str) -> GEDINeuralProcess:
    """Initialize GEDINeuralProcess model from config dict."""
```

---

## 2. MODEL CONFIGURATION LOADING DUPLICATION

### Current Status
Config loading is duplicated across scripts:

**Files affected:**
- /home/user/gedinp/predict.py (lines 56-62)
- /home/user/gedinp/evaluate.py (lines 41-42)
- /home/user/gedinp/evaluate_temporal.py (lines 74-75)
- /home/user/gedinp/train.py (lines 277-278)
- /home/user/gedinp/run_ablation_study.py (lines 43-44, 136-137)
- /home/user/gedinp/train_baselines.py (lines 181-182, 249-250)
- /home/user/gedinp/diagnostics.py (multiple locations)

### Example Pattern
```python
with open(config_path, 'r') as f:
    config = json.load(f)
```

Also duplicated:
- Config saving: `json.dump(vars(args), f, indent=2)` appears in 3 files
- Config path construction: `model_dir / 'config.json'` pattern repeated

### Impact
- 7+ files with config loading code
- 3+ files with config saving code
- Error handling inconsistent across files

### Refactoring Opportunity
**Create `utils/config.py`** with helper functions:
```python
def load_config(config_path: Path) -> dict:
    """Load JSON config file with error handling."""

def save_config(config: dict, output_path: Path) -> None:
    """Save config dict as JSON with error handling."""

def load_model_config(model_dir: Path) -> dict:
    """Load config from model directory."""
```

---

## 3. CHECKPOINT LOADING DUPLICATION

### Current Status
Model checkpoint loading code is duplicated with minor variations:

**File: /home/user/gedinp/predict.py (lines 80-98)**
```python
checkpoint_files = ['best_r2_model.pt', 'best_model.pt']
checkpoint_path = None

for ckpt_file in checkpoint_files:
    path = checkpoint_dir / ckpt_file
    if path.exists():
        checkpoint_path = path
        break

if checkpoint_path is None:
    raise FileNotFoundError(...)

checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
model.load_state_dict(checkpoint['model_state_dict'])
```

**File: /home/user/gedinp/evaluate.py (lines 127-130)** - Similar but simpler
**File: /home/user/gedinp/evaluate_temporal.py (lines 192-195)** - Similar
**File: /home/user/gedinp/diagnostics.py (lines 550-562)** - Similar

### Impact
- 4 instances of checkpoint loading logic
- predict.py has most robust implementation (fallback logic)
- Other files don't have fallback, fragile to checkpoint naming changes

### Refactoring Opportunity
**Add to `utils/model.py`:**
```python
def load_checkpoint(
    model: torch.nn.Module,
    checkpoint_path: Path,
    device: str,
    fallback_paths: Optional[List[Path]] = None
) -> dict:
    """Load model checkpoint with optional fallback paths."""
```

---

## 4. GEDI DATA QUERYING PATTERN DUPLICATION

### Current Status
GEDI data querying follows similar pattern in 3 files:

**File: /home/user/gedinp/train.py (lines 288-296)**
```python
print("Step 1: Querying GEDI data...")
querier = GEDIQuerier()
gedi_df = querier.query_region_tiles(
    region_bbox=args.region_bbox,
    tile_size=0.1,
    start_time=args.start_time,
    end_time=args.end_time
)
print(f"Retrieved {len(gedi_df)} GEDI shots across {gedi_df['tile_id'].nunique()} tiles")
```

**File: /home/user/gedinp/train_baselines.py (lines 193-200)** - IDENTICAL
**File: /home/user/gedinp/evaluate_temporal.py (lines 105-115)** - Similar (loops for multiple years)

### Additional Duplication
Temporal year filtering logic is only in train.py (lines 298-323) but is a common pattern that predict.py and evaluate_temporal.py might benefit from.

### Impact
- Code reuse across 3 files
- Logging/status printing inconsistent
- Temporal filtering logic isolated in train.py

### Refactoring Opportunity
**Create `utils/data_loading.py`:**
```python
def query_gedi_data(
    region_bbox: tuple,
    start_time: str,
    end_time: str,
    filter_years: Optional[List[int]] = None
) -> pd.DataFrame:
    """Query GEDI data with optional temporal filtering."""

def query_gedi_for_temporal_evaluation(
    region_bbox: tuple,
    test_years: List[int]
) -> pd.DataFrame:
    """Query GEDI data for temporal validation across multiple years."""
```

---

## 5. EMBEDDING EXTRACTION PATTERN DUPLICATION

### Current Status
Embedding extraction follows similar pattern in 4 files:

**File: /home/user/gedinp/predict.py (lines 197-218)**
```python
def extract_embeddings(coords_df, extractor, desc="Extracting embeddings"):
    patches = []
    valid_indices = []
    for idx, row in tqdm(coords_df.iterrows(), total=len(coords_df), desc=desc):
        patch = extractor.extract_patch(row['longitude'], row['latitude'])
        if patch is not None:
            patches.append(patch)
            valid_indices.append(idx)
    result_df = coords_df.loc[valid_indices].copy()
    result_df['embedding_patch'] = patches
    return result_df
```

**File: /home/user/gedinp/train.py** - Uses `extractor.extract_patches_batch()` (line 338)
**File: /home/user/gedinp/train_baselines.py** - Uses `extractor.extract_patches_batch()` (line 213)
**File: /home/user/gedinp/evaluate_temporal.py** - Uses `extractor.extract_patches_batch()` (line 141)

### Impact
- predict.py has custom single-shot extraction
- Other files use batch extraction method
- Inconsistent approach to embedding extraction

### Refactoring Opportunity
**Add to `utils/data_loading.py`:**
```python
def extract_embeddings_for_dataframe(
    df: pd.DataFrame,
    extractor: EmbeddingExtractor,
    batch_mode: bool = True,
    verbose: bool = True
) -> pd.DataFrame:
    """Extract embeddings for all rows in dataframe with optional batching."""
```

---

## 6. DATASET AND DATALOADER CREATION DUPLICATION

### Current Status
Similar code for creating datasets and dataloaders in train.py, train_baselines.py, evaluate_temporal.py:

**File: /home/user/gedinp/train.py (lines 384-414)**
```python
train_dataset = GEDINeuralProcessDataset(
    train_df,
    min_shots_per_tile=args.min_shots_per_tile,
    log_transform_agbd=args.log_transform_agbd,
    augment_coords=args.augment_coords,
    coord_noise_std=args.coord_noise_std,
    global_bounds=global_bounds
)
train_loader = DataLoader(
    train_dataset,
    batch_size=args.batch_size,
    shuffle=True,
    collate_fn=collate_neural_process,
    num_workers=args.num_workers
)
```

**File: /home/user/gedinp/evaluate_temporal.py (lines 158-173)** - Very similar
**File: /home/user/gedinp/train_baselines.py** - Different dataset type but similar structure

### Impact
- 10+ lines of nearly identical code repeated
- Dataloader configuration scattered across files
- Inconsistent default parameters

### Refactoring Opportunity
**Add to `utils/data_loading.py`:**
```python
def create_neural_process_dataloader(
    df: pd.DataFrame,
    config: dict,
    global_bounds: tuple,
    shuffle: bool = True,
    augment: bool = True
) -> DataLoader:
    """Create DataLoader for GEDINeuralProcessDataset with config."""
```

---

## 7. ARGUMENT PARSING PATTERN DUPLICATION

### Current Status
Common arguments repeated across multiple `parse_args()` functions:

**Repeated in all/most files:**
- `--region_bbox` (4 files: train.py, predict.py, train_baselines.py, run_ablation_study.py)
- `--start_time` / `--end_time` (5 files: train.py, predict.py, evaluate_temporal.py, train_baselines.py, run_ablation_study.py)
- `--embedding_year` (4 files: train.py, predict.py, evaluate_temporal.py, train_baselines.py)
- `--cache_dir` (4 files: train.py, predict.py, evaluate_temporal.py, train_baselines.py)
- `--device` (5 files - all with same default logic)
- `--seed` (3 files: train.py, train_baselines.py, run_ablation_study.py)

**File: /home/user/gedinp/train.py (lines 25-112)** - 87 lines
**File: /home/user/gedinp/predict.py (lines 20-52)** - 32 lines
**File: /home/user/gedinp/train_baselines.py (lines 19-77)** - 58 lines
**File: /home/user/gedinp/evaluate_temporal.py (lines 40-65)** - 25 lines
**File: /home/user/gedinp/run_ablation_study.py (lines 226-255)** - 29 lines

### Device Default Duplication
```python
'default': 'cuda' if torch.cuda.is_available() else 'cpu'
```
Appears in: predict.py (line 39), evaluate.py (line 32), train.py (line 105), evaluate_temporal.py (line 59)

### Impact
- ~20-30 lines of duplicated argument definitions per file
- Inconsistent help text
- Default values scattered
- Changes to common args require updates in 4+ places

### Refactoring Opportunity
**Create `utils/arguments.py`:**
```python
def add_common_data_arguments(parser):
    """Add region_bbox, start_time, end_time, embedding_year, cache_dir arguments."""
    parser.add_argument('--region_bbox', ...)
    parser.add_argument('--start_time', ...)
    # ... etc

def add_device_argument(parser):
    """Add --device argument with proper default."""
    parser.add_argument('--device', ...)

def add_seed_argument(parser):
    """Add --seed argument."""
    parser.add_argument('--seed', ...)

def get_default_device():
    """Get CUDA device or CPU."""
    return 'cuda' if torch.cuda.is_available() else 'cpu'
```

Usage:
```python
parser = argparse.ArgumentParser(...)
add_common_data_arguments(parser)
add_device_argument(parser)
```

---

## 8. UTILITY FUNCTION DUPLICATION

### A. Set Seed Function
**File: /home/user/gedinp/train.py (lines 115-119)**
```python
def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
```

Used only in train.py but pattern appears in train_baselines.py (lines 176):
```python
np.random.seed(args.seed)  # Incomplete
```

### B. Convert to Serializable Function
**File: /home/user/gedinp/evaluate_temporal.py (lines 20-36)**
```python
def convert_to_serializable(obj):
    if isinstance(obj, dict):
        return {key: convert_to_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_serializable(item) for item in obj]
    # ... numpy type conversions
```

This function is essential for JSON serialization but only in evaluate_temporal.py. The same issue would occur in other evaluation scripts.

### Impact
- set_seed not consistently applied (train_baselines.py missing torch seeding)
- convert_to_serializable only in one file but needed for JSON serialization
- Risk of JSON serialization errors in other scripts

### Refactoring Opportunity
**Add to `utils/common.py` or `utils/misc.py`:**
```python
def set_random_seed(seed: int) -> None:
    """Set seed for PyTorch, NumPy, and CUDA."""

def make_serializable(obj: Any) -> Any:
    """Convert numpy/torch types to Python natives for JSON serialization."""
```

---

## 9. OUTPUT DIRECTORY AND FILE HANDLING DUPLICATION

### Current Status
Output directory creation and file saving patterns repeated:

**File: /home/user/gedinp/train.py (lines 274-278)**
```python
output_dir = Path(args.output_dir)
output_dir.mkdir(parents=True, exist_ok=True)
with open(output_dir / 'config.json', 'w') as f:
    json.dump(vars(args), f, indent=2)
```

**File: /home/user/gedinp/train_baselines.py (lines 178-182)** - IDENTICAL
**File: /home/user/gedinp/run_ablation_study.py (lines 257-261)** - IDENTICAL
**File: /home/user/gedinp/predict.py (lines 422-423)** - Similar
**File: /home/user/gedinp/evaluate_temporal.py** - Different approach

### Impact
- 5+ lines duplicated in 3+ files
- No consistent error handling for directory creation
- Config saving logic scattered

### Refactoring Opportunity
**Add to `utils/config.py`:**
```python
def setup_output_directory(output_dir: Path) -> Path:
    """Create output directory and save config template."""

def save_config_to_dir(config: dict, output_dir: Path) -> None:
    """Save config JSON to output directory."""
```

---

## 10. METRICS EVALUATION DUPLICATION

### Current Status
There are two separate metric computation functions:

**File: /home/user/gedinp/utils/evaluation.py (lines 19-71)**
```python
def compute_metrics(pred, true, pred_std=None) -> Dict[str, float]:
    # RMSE, MAE, R² computation
```

**File: /home/user/gedinp/train_baselines.py (lines 90-100)**
```python
def evaluate_model(model, coords, embeddings, agbd_true, agbd_scale=200.0, log_transform=True):
    pred_norm, pred_std_norm = model.predict(coords, embeddings, return_std=True)
    pred = denormalize_agbd(pred_norm, agbd_scale=agbd_scale, log_transform=log_transform)
    metrics = compute_metrics(pred, agbd_true)
    return metrics, pred, pred_std_norm
```

The `evaluate_model` in train_baselines.py is specific to baseline models and wraps `compute_metrics` with denormalization, but:
- Uses different model.predict() signature (baseline models vs neural process)
- Has different normalization parameters
- Duplicates normalization/denormalization logic

### Impact
- Two different evaluation paradigms scattered
- Baseline model evaluation isolated
- Code duplication in normalization application

### Refactoring Opportunity
**Enhance `utils/evaluation.py`:**
```python
def evaluate_baseline_model(
    model,
    coords: np.ndarray,
    embeddings: np.ndarray,
    agbd_true: np.ndarray,
    agbd_scale: float = 200.0,
    log_transform: bool = True
) -> Tuple[Dict[str, float], np.ndarray, np.ndarray]:
    """Evaluate baseline model with automatic denormalization."""
```

---

## 11. TEMPORAL DATA HANDLING

### Current Status
Temporal data handling and year filtering is duplicated between files:

**File: /home/user/gedinp/train.py (lines 298-323)**
- Complex year extraction logic from multiple possible column names
- Filtering logic

**File: /home/user/gedinp/evaluate_temporal.py (lines 107-122)**
- Loops through test years and queries separately
- Different approach than train.py

### Impact
- Year extraction logic duplicated/inconsistent
- Temporal filtering not reusable

### Refactoring Opportunity
**Add to `utils/data_loading.py`:**
```python
def extract_year_from_dataframe(df: pd.DataFrame) -> pd.Series:
    """Extract year from datetime column (tries multiple column names)."""

def filter_dataframe_by_years(df: pd.DataFrame, years: List[int]) -> pd.DataFrame:
    """Filter dataframe to only include specified years."""
```

---

## Summary Table of Refactoring Opportunities

| Category | Files Affected | Lines Duplicated | Priority | New Module |
|----------|---------------|-----------------|----------|-----------|
| Model initialization | 5 | 50 | HIGH | utils/model.py |
| Config loading/saving | 7+ | 30+ | HIGH | utils/config.py |
| Checkpoint loading | 4 | 40+ | HIGH | utils/model.py |
| GEDI querying | 3 | 20+ | MEDIUM | utils/data_loading.py |
| Embedding extraction | 4 | 20+ | MEDIUM | utils/data_loading.py |
| Dataset creation | 3 | 30+ | MEDIUM | utils/data_loading.py |
| Argument parsing | 5+ | 100+ | MEDIUM | utils/arguments.py |
| Utility functions | 2 | 15+ | LOW | utils/common.py |
| Output handling | 3+ | 15+ | LOW | utils/config.py |
| Metrics evaluation | 2 | 10+ | LOW | utils/evaluation.py |
| Temporal handling | 2 | 20+ | LOW | utils/data_loading.py |

---

## Recommended Implementation Order

1. **Phase 1 (HIGH PRIORITY)**
   - Create `utils/model.py` - consolidate model loading/initialization
   - Create `utils/config.py` - consolidate config operations
   - This will immediately reduce 80+ lines of duplication

2. **Phase 2 (MEDIUM PRIORITY)**
   - Create `utils/data_loading.py` - consolidate GEDI, embedding, dataset operations
   - Create `utils/arguments.py` - consolidate argument parsing
   - This will reduce 150+ lines of duplication

3. **Phase 3 (LOW PRIORITY)**
   - Create `utils/common.py` - utility functions
   - Enhance `utils/evaluation.py` for baseline models
   - Polish and refactor edge cases

---

## Code Quality Improvements

Beyond consolidation, consider:
- Consistent error handling across all modules
- Standardized logging approach
- Type hints in all utility functions
- Unit tests for utility functions
- Documentation for new modules

