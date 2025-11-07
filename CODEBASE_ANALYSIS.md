# GEDI Neural Process Codebase Analysis

## 1. Neural Process Implementation

### Architecture Overview
The implementation uses a **Conditional Neural Process (CNP)** with the following components:

#### Components (neural_process.py)

1. **EmbeddingEncoder** (lines 7-74)
   - Processes GeoTessera embedding patches (3x3 spatial patches with 128 channels)
   - Architecture: 3 convolutional layers with batch norm + residual connections
   - Input: (batch, 3, 3, 128) embedding patches
   - Output: (batch, 128) feature vectors
   - Uses adaptive average pooling + 2-layer MLP

2. **ContextEncoder** (lines 77-140)
   - Encodes context points: coordinates + embedding features + AGBD value
   - Input combines: lon/lat (2D), embedding features (128D), AGBD (1D) → 131D total
   - 3 fully connected layers with LayerNorm and residual connections
   - Output: (batch, 128) context representations

3. **Decoder** (lines 143-220)
   - Predicts AGBD at query points
   - Input: query coords + embedding features + aggregated context (2 + 128 + 128 = 258D)
   - Outputs:
     - Mean prediction (1D)
     - Log-variance for uncertainty (1D) - clamped to [-7, 5] for numerical stability
   - Uses 3 FC layers with LayerNorm

4. **AttentionAggregator** (lines 223-277)
   - Multi-head attention for context aggregation
   - Query: projected query representations
   - Key/Value: context representations
   - Default: 4 attention heads, 0.1 dropout

5. **GEDINeuralProcess** (lines 280-414)
   - Main model orchestrating all components
   - Forward pass:
     1. Encode both context and query embeddings
     2. Encode context points (coords + features + AGBD)
     3. Aggregate context via attention or mean pooling
     4. Decode query points with aggregated context
   - Returns: (predicted_mean, predicted_log_var)

### Loss Function (lines 446-472)
- **Gaussian Negative Log-Likelihood Loss**
- Formula: 0.5 * (log_var + exp(-log_var) * (target - mean)²)
- Falls back to MSE if uncertainty not predicted

### Metrics (lines 475-514)
Computed metrics:
- RMSE (Root Mean Squared Error)
- MAE (Mean Absolute Error)
- R² Score
- MSE (Mean Squared Error)
- Mean Uncertainty (if available)

---

## 2. Features and Embeddings

### Input Features

#### 2.1 GeoTessera Foundation Model Embeddings
- **Source**: GeoTessera (foundation model for satellite imagery)
- **Format**: 3×3 spatial patches with 128 channels
- **Year**: Configurable (default: 2024)
- **Extraction**: EmbeddingExtractor class (embeddings.py)

#### 2.2 Spatial Coordinates
- **Format**: (longitude, latitude)
- **Processing**:
  - Normalized to [0, 1] range within tile bounds
  - Optional coordinate augmentation during training (Gaussian noise, std=0.01)
  - Clip augmented coordinates to valid range

#### 2.3 Aboveground Biomass Density (AGBD)
- **Source**: GEDI L4A data
- **Units**: Mg/ha (megagrams per hectare)
- **Processing**:
  - Log transform: log(1 + AGBD)
  - Normalize by scale factor (default: 200.0)
  - Denormalize during prediction: pred * 200.0

### Feature Flow in Neural Process
```
Context Point Features:
  coords (2D) + embedding_features (128D) + agbd (1D) → context_repr (128D)
  
Query Point Features:
  coords (2D) + embedding_features (128D) + aggregated_context (128D) → prediction
```

---

## 3. Training Loop and Metrics

### Training Configuration (train.py)

#### Data Flow
1. **Data Querying**: GEDIQuerier.query_region_tiles()
   - Returns GEDI shots for region with tile assignments
   - Spatial tiling at 0.1° resolution (matches GeoTessera tiles)

2. **Embedding Extraction**: EmbeddingExtractor.extract_patches_batch()
   - Extracts 3×3 patches around each GEDI shot
   - Caches tiles to disk for efficiency

3. **Spatial Train/Val/Test Split**: SpatialTileSplitter
   - Splits at **tile level** (not individual shots) to prevent data leakage
   - Default: 70% train, 15% val, 15% test
   - Each tile = 0.1° × 0.1° spatial region

4. **Dataset Creation**: GEDINeuralProcessDataset
   - Per-tile context/target splits
   - Random context ratio: 30-70% of shots per tile
   - Min shots per tile: 10 (configurable)

### Training Loop (lines 110-167)
```python
def train_epoch(model, dataloader, optimizer, device):
    - Per-tile processing (batch = list of tiles)
    - Random context/target splits within each tile
    - Forward pass → loss computation
    - Backprop with gradient clipping (max_norm=1.0)
    - NaN/Inf checks
    - Averaging loss over tiles
```

### Validation Loop (lines 170-235)
```python
def validate(model, dataloader, device):
    - Per-tile evaluation
    - Computes loss and metrics (RMSE, MAE, R², uncertainty)
    - Aggregates metrics across tiles
```

### Training Configuration
- **Optimizer**: Adam (lr=5e-4)
- **Learning Rate Scheduler**: ReduceLROnPlateau
  - Factor: 0.5, Patience: 5 epochs
- **Early Stopping**: Patience 15 epochs (no val loss improvement)
- **Batch Size**: Tiles (default 16)
- **Epochs**: Default 100
- **Gradient Clipping**: max_norm=1.0

### Model Selection
- Best model based on **validation loss**
- Alternate best model based on **R² score**
- Checkpoints saved every 10 epochs

### Logged Metrics
- Train loss (per epoch)
- Val loss (per epoch)
- Val RMSE, MAE, R²
- Current learning rate
- Mean prediction uncertainty

---

## 4. Data Format and Context/Target Structure

### GEDI DataFrame Schema (after processing)
```
Columns after querying + embedding extraction:
- latitude: float (WGS84)
- longitude: float (WGS84)
- agbd: float (Mg/ha, ground truth)
- embedding_patch: ndarray (3, 3, 128) - GeoTessera embeddings
- tile_id: str (e.g., "tile_30.35_-15.75")
- tile_lon: float (tile center)
- tile_lat: float (tile center)
- [other GEDI quality metrics from L4A]
```

### Dataset Sample (GEDINeuralProcessDataset.__getitem__)
```
Returns a dictionary with:
{
  'context_coords': tensor (n_context, 2) - lon/lat
  'context_embeddings': tensor (n_context, 3, 3, 128)
  'context_agbd': tensor (n_context, 1) - normalized
  
  'target_coords': tensor (n_target, 2) - lon/lat
  'target_embeddings': tensor (n_target, 3, 3, 128)
  'target_agbd': tensor (n_target, 1) - normalized
}
```

### Context/Target Split Strategy
- Per-tile random split
- Context ratio: uniformly sampled from [0.3, 0.7]
- Ensures min 1 context point (or skip if no targets)
- Training: augment coordinates, no augmentation for val/test

### Normalization
- **Coordinates**: Normalized to [0, 1] within tile bounds
  - If single point: range = 1.0 (avoid div by zero)
- **AGBD**:
  - Log transform: log(1 + x)
  - Scale: divide by 200.0 (typical max AGBD)
  - Denormalization: multiply by 200.0 + exp (for predictions)

### Data Collation (collate_neural_process)
- Returns **lists of tensors** (one per tile)
- Handles variable numbers of shots per tile
- No padding/masking required (per-tile processing in training loop)

---

## 5. Existing Baseline Models

**No baseline models exist in the codebase.**

The repository contains:
- ✓ GEDINeuralProcess (CNP implementation)
- ✓ Utilities (data loading, embedding extraction)
- ✓ Training pipeline
- ✗ No simple baselines (e.g., mean predictor, linear regression)
- ✗ No traditional spatial interpolation methods
- ✗ No alternative neural architectures

**Implication**: All comparisons against "baselines" would need to be implemented separately.

---

## 6. Key Files Summary

| File | Purpose |
|------|---------|
| neural_process.py | Core CNN/attention-based neural process model |
| train.py | End-to-end training pipeline |
| dataset.py | PyTorch Dataset for Neural Process |
| embeddings.py | GeoTessera embedding extraction and caching |
| gedi.py | GEDI data querying via gediDB |
| spatial_cv.py | Tile-based spatial cross-validation |
| predict.py | Dense grid prediction and visualization |

---

## 7. Hyperparameters

### Model Architecture
- patch_size: 3 (GeoTessera 3×3 patches)
- embedding_channels: 128 (from GeoTessera)
- embedding_feature_dim: 128 (encoded patch feature dim)
- context_repr_dim: 128 (context representation dim)
- hidden_dim: 512 (default)
- output_uncertainty: True
- use_attention: True
- num_attention_heads: 4

### Data Processing
- normalize_coords: True
- normalize_agbd: True
- agbd_scale: 200.0
- log_transform_agbd: True
- augment_coords: True (training only)
- coord_noise_std: 0.01
- context_ratio_range: (0.3, 0.7)
- min_shots_per_tile: 10

### Training
- batch_size: 16 (tiles)
- lr: 5e-4
- epochs: 100
- early_stopping_patience: 15
- lr_scheduler_patience: 5
- lr_scheduler_factor: 0.5

---

## 8. Important Implementation Details

### Memory and Efficiency
- **Tile Caching**: Embeddings cached to disk (pickle) to avoid re-downloads
- **Memory Budgets**: TileDB configured with 512MB budget for GEDI queries
- **Chunked Querying**: Falls back to chunked queries (0.05° chunks) on memory errors
- **Per-tile Processing**: Models work on individual tiles (avoids large batch sizes)

### Data Validation
- Removes shots with missing embeddings
- Removes NaN coordinates or AGBD values
- Skips batches with NaN/Inf losses
- Filters GEDI data by min/max AGBD thresholds

### Spatial Properties
- Tile size: 0.1° × 0.1° (matches GeoTessera and GEDI query granularity)
- Coordinate normalization: Within-tile normalization ensures model sees relative positions
- Spatial CV: Tile-level splits prevent data leakage across train/val/test

