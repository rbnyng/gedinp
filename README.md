# GEDI AGB Interpolation with Neural Processes

Interpolating GEDI Aboveground Biomass (AGB) measurements using Neural Processes with geospatial foundation model embeddings as context.

## Approach

GEDI provides highly accurate point samples of AGB (~25m footprints), but with sparse spatial coverage. This project uses:

- **Neural Processes**: For sparse-to-dense interpolation with uncertainty quantification
- **GeoTessera Foundation Model Embeddings**: Rich 128-channel spatial features at 10m resolution from Sentinel-1/2
- **3x3 Embedding Patches**: 30m × 30m patches around each GEDI shot for spatial context
- **Tile-based Spatial CV**: 0.1° × 0.1° tiles for robust cross-validation

## Project Structure

```
gedinp/
├── data/
│   ├── gedi.py          # GEDI data querying from gediDB
│   ├── embeddings.py    # GeoTessera embedding extraction
│   ├── dataset.py       # PyTorch Dataset for training
│   └── spatial_cv.py    # Tile-based spatial cross-validation
├── models/
│   └── neural_process.py # Neural Process architecture
├── baselines/
│   └── models.py        # Baseline models (RF, XGBoost, IDW)
├── train.py             # Neural Process training script
├── train_baselines.py   # Baseline models training script
├── predict.py           # Inference script
└── requirements.txt
```

## APIs Used

- [gediDB](https://github.com/simonbesnard1/gedidb): Efficient GEDI L2A-L4C data access
- [GeoTessera](https://github.com/ucam-eo/geotessera): Geospatial foundation model embeddings

## Installation

```bash
pip install -r requirements.txt
```

## Quick Start

### 1. Training a Model

Train a Neural Process model on a region of interest:

```bash
python train.py \
    --region_bbox 30.256 -15.853 30.422 -15.625 \
    --start_time 2019-01-01 \
    --end_time 2023-12-31 \
    --embedding_year 2024 \
    --epochs 100 \
    --batch_size 8 \
    --hidden_dim 256 \
    --output_dir ./outputs \
    --cache_dir ./cache
```

This will:
- Query GEDI L4A data for the specified region
- Extract 3x3 GeoTessera embedding patches for each GEDI shot
- Create tile-based train/val/test splits
- Train the Neural Process model
- Save checkpoints and metrics to `./outputs`

### 2. Generating Predictions

Generate dense AGB predictions at 10m resolution for a custom region:

```bash
python predict.py \
    --checkpoint ./outputs \
    --region 30.30 -15.80 30.40 -15.70 \
    --resolution 10 \
    --n_context 100 \
    --batch_size 1024 \
    --output_dir ./predictions
```

This will:
- Load the trained model and config (auto-detected from checkpoint directory)
- Query 100 nearest GEDI shots as context
- Generate predictions on a dense grid at 10m resolution
- Save GeoTIFF files (mean AGB + uncertainty)
- Save context points as GeoJSON
- Generate visualization preview (mean + uncertainty side-by-side)

**Key Options:**
- `--checkpoint`: Path to trained model directory (contains config.json and best_model.pt)
- `--region`: Bounding box [min_lon, min_lat, max_lon, max_lat]
- `--resolution`: Output resolution in meters (default: 10m)
- `--n_context`: Number of nearest GEDI shots for context (default: 100)
- `--batch_size`: GPU batch size for inference (default: 1024)
- `--device`: Use 'cuda' or 'cpu' (auto-detected by default)
- `--no_preview`: Disable visualization generation
- `--embedding_year`: GeoTessera year to use (default: 2024)

**Outputs:**
```
predictions/
├── region_<bbox>_agb_mean.tif       # Mean AGB predictions (GeoTIFF)
├── region_<bbox>_agb_std.tif        # Uncertainty estimates (GeoTIFF)
├── region_<bbox>_context.geojson    # GEDI context points
├── region_<bbox>_preview.png        # Visualization
└── region_<bbox>_metadata.json      # Prediction metadata
```

### 3. Python API

Use the components directly in your code:

```python
from data import GEDIQuerier, EmbeddingExtractor, SpatialTileSplitter
from models import GEDINeuralProcess

# Query GEDI data
querier = GEDIQuerier()
gedi_df = querier.query_region_tiles(
    region_bbox=(30.256, -15.853, 30.422, -15.625),
    tile_size=0.1
)

# Extract embeddings
extractor = EmbeddingExtractor(year=2024, patch_size=3)
gedi_df = extractor.extract_patches_batch(gedi_df)

# Create spatial splits
splitter = SpatialTileSplitter(gedi_df)
train_df, val_df, test_df = splitter.split()

# Initialize model
model = GEDINeuralProcess(
    patch_size=3,
    embedding_feature_dim=128,
    hidden_dim=256
)
```

See `example_usage.py` for more examples.

### 4. Training Baseline Models

For ablation studies, train baseline models (Random Forest, XGBoost, IDW) to compare with Neural Process:

```bash
python train_baselines.py \
    --region_bbox 30.256 -15.853 30.422 -15.625 \
    --start_time 2019-01-01 \
    --end_time 2023-12-31 \
    --embedding_year 2024 \
    --models rf xgb idw \
    --output_dir ./outputs_baselines \
    --cache_dir ./cache
```

This will:
- Use the same GEDI data and GeoTessera embeddings as Neural Process
- Apply identical spatial train/val/test splits for fair comparison
- Train all three baseline models:
  - **Random Forest**: Uses coords + flattened embeddings as features
  - **XGBoost**: Uses coords + flattened embeddings with quantile regression for uncertainty
  - **IDW (Inverse Distance Weighting)**: Spatial-only baseline (ignores embeddings)
- Save trained models and evaluation metrics to `./outputs_baselines`

**Baseline Model Options:**
- `--models`: Which baselines to train (`rf`, `xgb`, `idw`). Default: all three
- `--rf_n_estimators`: Random Forest trees (default: 100)
- `--rf_max_depth`: Random Forest max depth (default: None)
- `--xgb_n_estimators`: XGBoost rounds (default: 100)
- `--xgb_max_depth`: XGBoost max depth (default: 6)
- `--xgb_learning_rate`: XGBoost learning rate (default: 0.1)
- `--idw_power`: IDW distance power (default: 2.0)
- `--idw_n_neighbors`: IDW neighbors (default: 10)

**Outputs:**
```
outputs_baselines/
├── config.json              # Training configuration
├── results.json             # Summary of all model metrics
├── random_forest.pkl        # Trained RF model
├── xgboost.pkl              # Trained XGBoost model
├── idw.pkl                  # Trained IDW model
├── train_split.csv          # Training data split
├── val_split.csv            # Validation data split
└── test_split.csv           # Test data split
```

The baselines provide important ablation context:
- **RF/XGBoost vs Neural Process**: Tests if meta-learning provides benefits
- **IDW vs RF/XGBoost**: Shows value-add of satellite embeddings vs spatial-only interpolation

### 5. Temporal Validation

Test whether your model generalizes to future years not seen during training:

```bash
# Step 1: Train on historical years (2019-2021)
python train.py \
    --region_bbox 30.256 -15.853 30.422 -15.625 \
    --train_years 2019 2020 2021 \
    --epochs 100 \
    --output_dir ./outputs_temporal

# Step 2: Evaluate on future years (2022-2023)
python evaluate_temporal.py \
    --model_dir ./outputs_temporal \
    --test_years 2022 2023 \
    --checkpoint best_r2_model.pt

# Step 3: Compare spatial vs temporal performance
python compare_spatial_temporal.py \
    --model_dir ./outputs_temporal \
    --temporal_suffix years_2022_2023
```

This workflow:
- **Training**: Uses only 2019-2021 GEDI data (with spatial CV)
- **Temporal test**: Evaluates on completely held-out years (2022-2023)
- **Comparison**: Generates side-by-side spatial vs temporal performance plots

**Why temporal validation matters:**
- Tests real-world deployment scenario (train on past, predict future)
- Assesses robustness to temporal distribution shift
- Often required by reviewers for publication

See [`docs/TEMPORAL_VALIDATION.md`](docs/TEMPORAL_VALIDATION.md) for detailed guide and best practices.

## Key Features

- **Spatial Cross-Validation**: Tile-based splits ensure no data leakage
- **Uncertainty Quantification**: Neural Processes output prediction uncertainty
- **Efficient Caching**: Tiles and embeddings are cached to avoid redundant downloads
- **Flexible Architecture**: Configurable model dimensions and hyperparameters
- **Dense Prediction**: Generate wall-to-wall AGB maps from sparse GEDI samples

## Model Architecture

The Neural Process consists of:

1. **Embedding Encoder**: CNN to encode 3×3×128 GeoTessera patches → 128-d features
2. **Context Encoder**: MLP to encode (coordinates, embedding features, AGBD) → context representations
3. **Aggregator**: Mean pooling over context representations → global context
4. **Decoder**: MLP to decode (query coordinates, query embedding, global context) → AGBD prediction + uncertainty

## Citation

If you use this code, please cite:

- GEDI mission: [Dubayah et al. 2020](https://doi.org/10.1016/j.rse.2020.111817)
- Neural Processes: [Garnelo et al. 2018](https://arxiv.org/abs/1807.01622)
- GeoTessera: [ucam-eo/geotessera](https://github.com/ucam-eo/geotessera)
- gediDB: [simonbesnard1/gedidb](https://github.com/simonbesnard1/gedidb)
