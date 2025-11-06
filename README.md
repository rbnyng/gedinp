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
├── train.py             # Training script
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

Generate dense AGB predictions for a tile:

```bash
python predict.py \
    --model_path ./outputs/best_model.pt \
    --config_path ./outputs/config.json \
    --tile_lon 30.35 \
    --tile_lat -15.75 \
    --grid_spacing 0.001 \
    --visualize \
    --output_dir ./predictions
```

This will:
- Load the trained model
- Query nearby GEDI shots as context
- Generate predictions on a dense grid (~100m spacing)
- Save predictions as CSV
- Optionally create visualization plots

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
