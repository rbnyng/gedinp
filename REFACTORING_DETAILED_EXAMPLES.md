# Detailed Code Examples - Duplicate Patterns

## 1. Model Initialization - Exact Duplicates

### Location 1: predict.py (lines 68-78)
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

### Location 2: evaluate.py (lines 114-124) - IDENTICAL

### Location 3: evaluate_temporal.py (lines 179-189) - IDENTICAL

### Location 4: train.py (lines 419-429) - IDENTICAL

### Location 5: diagnostics.py (lines 523-533) - IDENTICAL

**Total Duplication: 5x (50 lines of nearly identical code)**

---

## 2. Config Loading Pattern

### Pattern 1: Loading config
```python
# train.py (lines 277-278)
with open(output_dir / 'config.json', 'w') as f:
    json.dump(vars(args), f, indent=2)

# train_baselines.py (lines 181-182) - IDENTICAL
# run_ablation_study.py (lines 260-261) - IDENTICAL
```

### Pattern 2: Saving config
```python
# evaluate_temporal.py (lines 74-75)
with open(model_dir / 'config.json', 'r') as f:
    config = json.load(f)

# evaluate.py (lines 41-42) - Similar
# predict.py (lines 57-62) - Similar with error handling
```

**Total Files: 7+ with variations**

---

## 3. Argument Parsing Duplication

### Common Arguments Across Multiple Files

#### --region_bbox (appears in 4 files)
```python
# train.py (lines 29-30)
parser.add_argument('--region_bbox', type=float, nargs=4, required=True,
                    help='Region bounding box: min_lon min_lat max_lon max_lat')

# predict.py (lines 27-29) - IDENTICAL
# train_baselines.py (lines 23-24) - IDENTICAL
# run_ablation_study.py (lines 229-230) - IDENTICAL
```

#### --start_time / --end_time (appears in 5 files)
```python
# train.py (lines 31-34)
parser.add_argument('--start_time', type=str, default='2019-01-01',
                    help='Start date for GEDI data (YYYY-MM-DD)')
parser.add_argument('--end_time', type=str, default='2023-12-31',
                    help='End date for GEDI data (YYYY-MM-DD)')

# predict.py (lines 45-48) - Nearly identical (different defaults)
# evaluate_temporal.py - Similar
# train_baselines.py - Similar
# run_ablation_study.py - Similar
```

#### --device (appears in 5 files with same default)
```python
# predict.py (lines 38-40)
parser.add_argument('--device', type=str,
                    default='cuda' if torch.cuda.is_available() else 'cpu',
                    help='Device (cuda/cpu)')

# evaluate.py (lines 32) - IDENTICAL
# train.py (lines 105-106) - IDENTICAL
# evaluate_temporal.py (lines 59) - IDENTICAL
# diagnostics.py - Not explicitly defined but used
```

**Total Duplication: ~100+ lines of argument code**

---

## 4. GEDI Data Querying Pattern

### Pattern: Query and Print Stats
```python
# train.py (lines 289-296)
querier = GEDIQuerier()
gedi_df = querier.query_region_tiles(
    region_bbox=args.region_bbox,
    tile_size=0.1,
    start_time=args.start_time,
    end_time=args.end_time
)
print(f"Retrieved {len(gedi_df)} GEDI shots across {gedi_df['tile_id'].nunique()} tiles")

# train_baselines.py (lines 193-200) - IDENTICAL
# evaluate_temporal.py (lines 110-115) - Similar (in loop)
```

### Temporal Filtering Pattern (only in train.py - lines 298-323)
```python
if args.train_years is not None:
    print(f"\nApplying temporal filtering: using only years {args.train_years}")
    
    # Extract year from timestamp if available
    if 'time' in gedi_df.columns:
        gedi_df['year'] = pd.to_datetime(gedi_df['time']).dt.year
    elif 'date_time' in gedi_df.columns:
        gedi_df['year'] = pd.to_datetime(gedi_df['date_time']).dt.year
    elif 'datetime' in gedi_df.columns:
        gedi_df['year'] = pd.to_datetime(gedi_df['datetime']).dt.year
    else:
        try:
            gedi_df['year'] = pd.to_datetime(gedi_df.index).year
        except:
            print("Warning: Could not find timestamp column...")
    
    if args.train_years is not None:
        n_before = len(gedi_df)
        gedi_df = gedi_df[gedi_df['year'].isin(args.train_years)]
        n_after = len(gedi_df)
        print(f"Filtered from {n_before} to {n_after} shots...")
```

**This pattern is ONLY in train.py but would be useful in other files**

---

## 5. Embedding Extraction Duplication

### Pattern 1: Manual Loop (predict.py - lines 197-218)
```python
def extract_embeddings(coords_df, extractor, desc="Extracting embeddings"):
    patches = []
    valid_indices = []
    
    print(f"\n{desc}...")
    for idx, row in tqdm(coords_df.iterrows(), total=len(coords_df), desc=desc):
        patch = extractor.extract_patch(row['longitude'], row['latitude'])
        if patch is not None:
            patches.append(patch)
            valid_indices.append(idx)
    
    print(f"Successfully extracted {len(patches)}/{len(coords_df)} embeddings ({100*len(patches)/len(coords_df):.1f}%)")
    
    result_df = coords_df.loc[valid_indices].copy()
    result_df['embedding_patch'] = patches
    
    return result_df
```

### Pattern 2: Batch Method (train.py - line 338)
```python
gedi_df = extractor.extract_patches_batch(gedi_df, verbose=True)
```

**Other files also use batch method:**
- train_baselines.py (line 213)
- evaluate_temporal.py (line 141)

**Inconsistency: predict.py has unique implementation, others use batch**

---

## 6. Dataset and DataLoader Creation

### Pattern (train.py - lines 384-414)
```python
train_dataset = GEDINeuralProcessDataset(
    train_df,
    min_shots_per_tile=args.min_shots_per_tile,
    log_transform_agbd=args.log_transform_agbd,
    augment_coords=args.augment_coords,
    coord_noise_std=args.coord_noise_std,
    global_bounds=global_bounds
)
val_dataset = GEDINeuralProcessDataset(
    val_df,
    min_shots_per_tile=args.min_shots_per_tile,
    log_transform_agbd=args.log_transform_agbd,
    augment_coords=False,
    coord_noise_std=0.0,
    global_bounds=global_bounds
)

train_loader = DataLoader(
    train_dataset,
    batch_size=args.batch_size,
    shuffle=True,
    collate_fn=collate_neural_process,
    num_workers=args.num_workers
)
val_loader = DataLoader(
    val_dataset,
    batch_size=args.batch_size,
    shuffle=False,
    collate_fn=collate_neural_process,
    num_workers=args.num_workers
)
```

### Similar Pattern (evaluate_temporal.py - lines 158-173)
```python
eval_dataset = GEDINeuralProcessDataset(
    gedi_df,
    min_shots_per_tile=config.get('min_shots_per_tile', 10),
    log_transform_agbd=config.get('log_transform_agbd', True),
    augment_coords=False,
    coord_noise_std=0.0,
    global_bounds=global_bounds
)

eval_loader = DataLoader(
    eval_dataset,
    batch_size=args.batch_size,
    shuffle=False,
    collate_fn=collate_neural_process,
    num_workers=args.num_workers
)
```

**Total: 3 files with very similar patterns**

---

## 7. Utility Functions Scattered

### set_seed() - Only in train.py (lines 115-119)
```python
def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
```

**Used in: Only train.py**

**Partial implementation in: train_baselines.py (line 176) - Missing torch.cuda.manual_seed()**
```python
np.random.seed(args.seed)  # Incomplete!
```

### convert_to_serializable() - Only in evaluate_temporal.py (lines 20-36)
```python
def convert_to_serializable(obj):
    if isinstance(obj, dict):
        return {key: convert_to_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_serializable(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(convert_to_serializable(item) for item in obj)
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

**Used in: Only evaluate_temporal.py (line 232) for JSON serialization**

**Problem: This function is needed in other scripts but doesn't exist elsewhere**

---

## 8. Checkpoint Loading Variations

### Best Implementation (predict.py - lines 80-98)
```python
checkpoint_files = ['best_r2_model.pt', 'best_model.pt']
checkpoint_path = None

for ckpt_file in checkpoint_files:
    path = checkpoint_dir / ckpt_file
    if path.exists():
        checkpoint_path = path
        break

if checkpoint_path is None:
    raise FileNotFoundError(
        f"No checkpoint found in {checkpoint_dir}. "
        f"Looked for: {checkpoint_files}"
    )

print(f"Loading checkpoint: {checkpoint_path}")
checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
model.load_state_dict(checkpoint['model_state_dict'])
```

### Simpler Implementation (evaluate.py - lines 127-130)
```python
checkpoint_path = model_dir / args.checkpoint
print(f"Loading checkpoint from: {checkpoint_path}")
checkpoint = torch.load(checkpoint_path, map_location=args.device,weights_only=False)
model.load_state_dict(checkpoint['model_state_dict'])
```

**Issue: evaluate.py will fail if checkpoint doesn't exist, no fallback**

### Similar (evaluate_temporal.py - lines 192-195)
```python
checkpoint_path = model_dir / args.checkpoint
print(f"Loading checkpoint from: {checkpoint_path}")
checkpoint = torch.load(checkpoint_path, map_location=args.device, weights_only=False)
model.load_state_dict(checkpoint['model_state_dict'])
```

### Similar (diagnostics.py - lines 550-562)
```python
checkpoint_files = ['best_r2_model.pt', 'best_model.pt']
checkpoint_path = None

for ckpt_file in checkpoint_files:
    path = model_dir / ckpt_file
    if path.exists():
        checkpoint_path = path
        break

if checkpoint_path is None:
    raise FileNotFoundError(f"No checkpoint found in {model_dir}")

checkpoint = torch.load(checkpoint_path, map_location=device)
```

**Total: 4 implementations with variations - should be unified**

---

## 9. Device Default Handling

Appears 5 times across files:

```python
# predict.py (line 39)
'default': 'cuda' if torch.cuda.is_available() else 'cpu',

# evaluate.py (line 32)
default='cuda' if torch.cuda.is_available() else 'cpu')

# train.py (line 105)
default='cuda' if torch.cuda.is_available() else 'cpu',

# evaluate_temporal.py (line 59)
default='cuda' if torch.cuda.is_available() else 'cpu')

# diagnostics.py - not shown but likely similar
```

---

## 10. Output Directory Creation Pattern

### Pattern (repeated in 3 files)
```python
# train.py (lines 274-278)
output_dir = Path(args.output_dir)
output_dir.mkdir(parents=True, exist_ok=True)

with open(output_dir / 'config.json', 'w') as f:
    json.dump(vars(args), f, indent=2)

# train_baselines.py (lines 178-182) - IDENTICAL
# run_ablation_study.py (lines 257-261) - IDENTICAL
# predict.py (lines 422-423) - Similar
```

---

## Summary Statistics

| Pattern | Instances | Total Lines | Files |
|---------|-----------|------------|-------|
| Model initialization | 5 | 50 | 5 |
| Config loading | 7+ | 30+ | 7+ |
| Checkpoint loading | 4 | 40+ | 4 |
| GEDI querying | 3 | 20+ | 3 |
| Argument parsing (common args) | 5+ | 100+ | 5+ |
| Device default | 5 | 5 | 5 |
| Output directory setup | 3 | 15 | 3+ |
| **TOTAL** | **~32** | **260+** | **7 scripts** |

