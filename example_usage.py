"""
Example usage of the GEDI Neural Process pipeline.

This script demonstrates the full workflow:
1. Query GEDI data for a region
2. Extract GeoTessera embeddings
3. Create train/val/test splits
4. Train a Neural Process model
5. Generate predictions
"""

# Example 1: Query and explore GEDI data
from data import GEDIQuerier, get_gedi_statistics

# Define region (example: Zambia)
region_bbox = (30.256, -15.853, 30.422, -15.625)

# Query GEDI data
querier = GEDIQuerier()
gedi_df = querier.query_region_tiles(
    region_bbox=region_bbox,
    tile_size=0.1,
    start_time="2019-01-01",
    end_time="2023-12-31"
)

print(f"Found {len(gedi_df)} GEDI shots")
stats = get_gedi_statistics(gedi_df)
print(f"Statistics: {stats}")


# Example 2: Extract embeddings for GEDI shots
from data import EmbeddingExtractor

extractor = EmbeddingExtractor(
    year=2024,
    patch_size=3,
    cache_dir='./cache'
)

# Extract 3x3 patches for each GEDI shot
gedi_df = extractor.extract_patches_batch(gedi_df, verbose=True)


# Example 3: Create spatial splits
from data import SpatialTileSplitter, analyze_spatial_split

splitter = SpatialTileSplitter(
    gedi_df,
    val_ratio=0.15,
    test_ratio=0.15,
    random_state=42
)

train_df, val_df, test_df = splitter.split()
analysis = analyze_spatial_split(train_df, val_df, test_df)
print(f"Split analysis: {analysis}")


# Example 4: Training (use train.py script)
print("""
To train the model, run:

python train.py \\
    --region_bbox 30.256 -15.853 30.422 -15.625 \\
    --embedding_year 2024 \\
    --epochs 100 \\
    --batch_size 8 \\
    --output_dir ./outputs
""")


# Example 5: Prediction (use predict.py script)
print("""
To generate predictions for a tile, run:

python predict.py \\
    --model_path ./outputs/best_model.pt \\
    --config_path ./outputs/config.json \\
    --tile_lon 30.35 \\
    --tile_lat -15.75 \\
    --visualize \\
    --output_dir ./predictions
""")


# Example 6: Direct model usage
import torch
from models import GEDINeuralProcess

# Initialize model
model = GEDINeuralProcess(
    patch_size=3,
    embedding_channels=128,
    embedding_feature_dim=128,
    context_repr_dim=128,
    hidden_dim=256,
    output_uncertainty=True
)

# Example forward pass (with dummy data)
n_context = 10
n_query = 5

context_coords = torch.randn(n_context, 2)
context_embeddings = torch.randn(n_context, 3, 3, 128)
context_agbd = torch.randn(n_context, 1)

query_coords = torch.randn(n_query, 2)
query_embeddings = torch.randn(n_query, 3, 3, 128)

pred_mean, pred_log_var = model(
    context_coords,
    context_embeddings,
    context_agbd,
    query_coords,
    query_embeddings
)

print(f"Prediction shape: {pred_mean.shape}")  # (5, 1)
print(f"Uncertainty shape: {pred_log_var.shape}")  # (5, 1)
