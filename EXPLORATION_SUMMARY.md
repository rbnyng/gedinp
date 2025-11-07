# GEDI Neural Process Codebase Exploration Summary

## Quick Overview

This is a **Conditional Neural Process (CNP)** implementation for predicting Aboveground Biomass Density (AGBD) from GEDI satellite lidar data using GeoTessera foundation model embeddings.

**Key files:** `/home/user/gedinp/models/neural_process.py`, `/home/user/gedinp/train.py`, `/home/user/gedinp/data/dataset.py`

---

## 1. CNP/Neural Process Implementation

### What is a Conditional Neural Process?

A CNP is a meta-learning model that:
- Takes a set of **context points** (with observations) and their features
- Makes predictions for **target points** using learned representations
- Can handle variable numbers of context/target points
- Outputs both predictions AND uncertainty estimates

### Architecture Components

```
GEDINeuralProcess
├── EmbeddingEncoder (CNN)
│   └─ Input: 3×3 spatial patches of 128-D GeoTessera embeddings
│   └─ Output: 128-D feature vectors
│
├── ContextEncoder (MLP)
│   └─ Input: lon/lat (2D) + embedding features (128D) + AGBD (1D)
│   └─ Output: 128-D context representations
│
├── AttentionAggregator (Multi-head Attention)
│   └─ Aggregates context based on each query point
│   └─ 4 attention heads, cross-attention mechanism
│
└── Decoder (MLP)
    └─ Input: lon/lat + embedding features + aggregated context
    └─ Output: AGBD mean prediction + log-variance (uncertainty)
```

### Key Advantages
- **Context-aware**: Predictions depend on available context
- **Uncertainty quantification**: Learns to predict uncertainty via Gaussian NLL loss
- **Attention-based**: Can selectively aggregate relevant context information
- **Flexible**: Handles variable numbers of context/target points per tile

---

## 2. Embeddings and Features

### Feature Stack

Each GEDI shot is represented by:

| Feature | Dimension | Source | Processing |
|---------|-----------|--------|-----------|
| Longitude | 1 | GEDI | Normalized to [0,1] within tile |
| Latitude | 1 | GEDI | Normalized to [0,1] within tile |
| GeoTessera Embedding | 128 (3×3 spatial) | Foundation model | Encoded to 128D via CNN |
| AGBD (ground truth) | 1 | GEDI L4A | Log-transformed + normalized by 200.0 |

### Total Input Dimensions
- **Context point**: 2 + 128 + 1 = 131D (coords + embedding + AGBD)
- **Query point**: 2 + 128 = 130D (coords + embedding)
- **After aggregation**: 2 + 128 + 128 = 258D (coords + embedding + context_repr)

### Normalization Strategy
```
AGBD normalization:
  Raw AGBD (Mg/ha): [45.2, 150.3, 200.1, ...]
    ↓ log(1+x)
  Log-transformed: [3.82, 5.01, 5.30, ...]
    ↓ / log(1+200) ≈ 5.61
  Normalized: [0.68, 0.89, 0.94, ...] → trained on [0,1]
    ↓ (at inference: * log(1+200))
  Denormalized: [3.8, 5.0, 5.3, ...]
    ↓ exp(x) - 1
  Final prediction: [45, 150, 200] Mg/ha ✓

Coordinate normalization:
  Within each tile's bounding box, normalize lon/lat to [0,1]
  → Ensures model sees relative spatial positions
```

---

## 3. Training Loop and Metrics

### Training Pipeline

```
1. Query GEDI shots in region
   └─ Returns: latitude, longitude, agbd, tile_id
   
2. Extract GeoTessera embeddings (3×3 patches, 128D)
   └─ Cached locally to avoid re-downloading
   
3. Spatial train/val/test split (at TILE level)
   └─ 70% train, 15% val, 15% test
   └─ Prevents data leakage
   
4. Per-tile context/target split
   └─ Random split: context_ratio ~ U[0.3, 0.7]
   └─ Example: 10 shots → 6 context + 4 target
   
5. Forward pass
   └─ Encode embeddings (shared encoder)
   └─ Encode context: (coords + emb_features + agbd) → repr
   └─ Aggregate context via multi-head attention
   └─ Decode: (coords + emb_features + aggregated_context) → AGBD + uncertainty
   
6. Loss computation
   └─ Gaussian NLL: 0.5 * (log_var + exp(-log_var) * (pred - target)²)
   
7. Backprop + Adam optimizer
   └─ Gradient clipping: max_norm=1.0 (stability)
   └─ Learning rate scheduler: ReduceLROnPlateau (factor=0.5, patience=5)
```

### Evaluation Metrics

Computed per-tile (then averaged):
- **RMSE**: √(mean((pred - target)²))
- **MAE**: mean(|pred - target|)
- **R²**: 1 - SS_res/SS_tot
- **Mean Uncertainty**: average predicted std dev

### Training Configuration
- **Optimizer**: Adam (lr=5e-4)
- **Batch size**: 16 tiles
- **Epochs**: 100 (with early stopping at 15 epochs no improvement)
- **Best model selection**: 
  - Primary: lowest validation loss
  - Secondary: highest R² score

---

## 4. Data Format and Context/Target Structure

### DataFrame Schema

After querying and embedding extraction:
```
latitude (float)        [WGS84 degrees]
longitude (float)       [WGS84 degrees]
agbd (float)           [Mg/ha]
embedding_patch (np.ndarray)  [3, 3, 128]
tile_id (str)          ["tile_30.35_-15.75"]
tile_lon (float)       [tile center longitude]
tile_lat (float)       [tile center latitude]
```

### Dataset Sample Structure

Each `__getitem__` call returns one tile:

```python
{
    # Context points (30-70% of tile's shots)
    'context_coords': torch.tensor(shape=[n_context, 2]),
    'context_embeddings': torch.tensor(shape=[n_context, 3, 3, 128]),
    'context_agbd': torch.tensor(shape=[n_context, 1]),  # normalized
    
    # Target points (remaining 30-70%)
    'target_coords': torch.tensor(shape=[n_target, 2]),
    'target_embeddings': torch.tensor(shape=[n_target, 3, 3, 128]),
    'target_agbd': torch.tensor(shape=[n_target, 1]),    # normalized
}
```

### Key Processing Steps

1. **Coordinate Augmentation** (training only)
   - Gaussian noise: N(0, 0.01) added to normalized coords
   - Clipped to [0, 1] range

2. **Log Transform** (if enabled, default: True)
   - AGBD: log(1 + x) / log(1 + 200)

3. **Context/Target Split** (random each epoch)
   - Context ratio: uniform[0.3, 0.7]
   - Ensures model trains on variable context sizes

4. **Collation**
   - DataLoader returns **lists of tensors** (one per tile)
   - Handles variable-length sequences elegantly

---

## 5. Existing Baseline Models

**No baseline models currently exist.**

The codebase contains:
- ✓ GEDINeuralProcess (attention-based CNP)
- ✓ Full training/inference pipeline
- ✗ No simple baselines (e.g., mean AGBD, spatial interpolation)
- ✗ No alternative architectures (e.g., pure MLP, random forest)

**Implication**: To establish baselines, you would need to implement:
- **Mean predictor**: Just predict mean AGBD for all locations
- **Kriging**: Spatial interpolation (e.g., ordinary kriging)
- **Linear regression**: On features (coords + embedding features)
- **XGBoost/Random Forest**: Gradient boosting baselines
- **Simple MLP**: Without attention/context aggregation

---

## 6. Model Hyperparameters

### Architecture
- `patch_size`: 3 (3×3 spatial patches)
- `embedding_channels`: 128 (from GeoTessera)
- `embedding_feature_dim`: 128 (CNN output)
- `context_repr_dim`: 128 (encoder output)
- `hidden_dim`: 512 (MLP layers)
- `num_attention_heads`: 4
- `output_uncertainty`: True (learns variance)
- `use_attention`: True (vs mean pooling)

### Data Processing
- `log_transform_agbd`: True (log-normalize AGBD)
- `agbd_scale`: 200.0 (normalization factor)
- `normalize_coords`: True ([0,1] normalization)
- `augment_coords`: True (training only)
- `coord_noise_std`: 0.01 (augmentation noise)
- `context_ratio_range`: (0.3, 0.7) (random split)
- `min_shots_per_tile`: 10 (filter tiles)

### Training
- `batch_size`: 16 tiles
- `lr`: 5e-4
- `epochs`: 100
- `early_stopping_patience`: 15
- `lr_scheduler_patience`: 5
- `lr_scheduler_factor`: 0.5
- `grad_clip_max_norm`: 1.0

---

## 7. File Organization

```
/home/user/gedinp/
├── models/
│   ├── __init__.py
│   └── neural_process.py          # Main CNP implementation
├── data/
│   ├── __init__.py
│   ├── gedi.py                    # GEDI data querying
│   ├── embeddings.py              # GeoTessera extraction
│   ├── dataset.py                 # PyTorch Dataset
│   └── spatial_cv.py              # Train/val/test splitting
├── train.py                       # Training pipeline
├── predict.py                     # Inference/predictions
├── example_usage.py               # Usage examples
└── tests/
    ├── test_unit.py
    ├── test_pipeline.py
    └── test_multitile.py
```

---

## 8. Usage Example

### Training

```bash
python train.py \
    --region_bbox 30.256 -15.853 30.422 -15.625 \
    --embedding_year 2024 \
    --batch_size 16 \
    --epochs 100 \
    --output_dir ./outputs
```

### Prediction

```bash
python predict.py \
    --model_path ./outputs/best_model.pt \
    --config_path ./outputs/config.json \
    --tile_lon 30.35 \
    --tile_lat -15.75 \
    --visualize \
    --output_dir ./predictions
```

---

## 9. Documentation Files

I've created three detailed documentation files:

1. **CODEBASE_ANALYSIS.md** (9.2 KB)
   - Detailed component breakdown
   - Feature dimensions at each stage
   - Normalization strategies
   - Training dynamics
   - All hyperparameters

2. **ARCHITECTURE_FLOW.md** (24 KB)
   - Visual ASCII diagrams of model architecture
   - End-to-end data processing pipeline
   - Feature dimension tracking
   - Normalization/denormalization examples
   - Training dynamics illustration

3. **KEY_CODE_SNIPPETS.md** (15 KB)
   - Actual code from repository
   - Forward pass implementation
   - Loss function details
   - Training loop structure
   - Data preparation procedures
   - Embedding extraction process

---

## 10. Next Steps for MLP Baseline

To create an "MLP baseline" comparison, you would:

1. **Create a simple baseline model**
   ```python
   class MLPBaseline(nn.Module):
       def __init__(self, input_dim=258):
           super().__init__()
           self.fc1 = nn.Linear(input_dim, 256)
           self.fc2 = nn.Linear(256, 128)
           self.fc3 = nn.Linear(128, 2)  # mean + log_var
       
       def forward(self, context_coords, context_embeddings, context_agbd,
                   query_coords, query_embeddings):
           # Ignore context, just predict from query
           x = torch.cat([query_coords, query_embeddings], dim=-1)
           x = F.relu(self.fc1(x))
           x = F.relu(self.fc2(x))
           mean = x[:, :1]
           log_var = x[:, 1:]
           return mean, log_var
   ```

2. **Use the same training pipeline** (`train.py`) with the baseline model

3. **Compare metrics** (RMSE, MAE, R²) against the CNP

---

## Key Insights

1. **Spatial Properties**: Tile-level splits (0.1° × 0.1°) ensure no spatial leakage
2. **Attention Mechanism**: Each query point gets a weighted aggregation of context
3. **Uncertainty Quantification**: Model learns both predictions and confidence
4. **Foundation Model Integration**: GeoTessera embeddings capture rich satellite imagery features
5. **Flexible Context**: Model can work with variable numbers of context/target points

