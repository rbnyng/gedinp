# GEDI Neural Process: Architecture & Data Flow

## Model Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────┐
│                     GEDINeuralProcess                               │
├─────────────────────────────────────────────────────────────────────┤
│                                                                       │
│  Context Path:                  Query Path:                          │
│  ─────────────                  ──────────                          │
│                                                                       │
│  context_coords ──────┐         query_coords ──────┐                │
│  context_embeddings ──┼────┐    query_embeddings ──┼────┐           │
│  context_agbd ────────┘    │                        │    │           │
│                            │                        │    │           │
│                  EmbeddingEncoder                   │    │           │
│                      │                              │    │           │
│            context_emb_features                     │    │           │
│                      │                              │    │           │
│           ┌──────────┘                              │    │           │
│           │                         EmbeddingEncoder    │           │
│           │                              │             │           │
│           │                      query_emb_features    │           │
│           │                              │             │           │
│  ┌────────▼─────────┐            ┌──────▼───────┐     │           │
│  │  ContextEncoder  │            │ QueryProj    │     │           │
│  │                  │            │ (if attention)│    │           │
│  │  coords+         │            │              │     │           │
│  │  emb_features+   │            │  coords+     │     │           │
│  │  agbd ──>        │            │  emb_features│     │           │
│  │  context_repr    │            │  ──>         │     │           │
│  └────────┬─────────┘            │  query_repr  │     │           │
│           │                      └──────┬───────┘     │           │
│           │                             │             │           │
│           │                 ┌───────────▼─────────┐   │           │
│           │                 │ AttentionAggregator │   │           │
│           └────────────────▶│  (multihead attn)   │   │           │
│                             │                     │   │           │
│                             │  aggregated_context │   │           │
│                             └───────────┬─────────┘   │           │
│                                         │             │           │
│           ┌─────────────────────────────┘             │           │
│           │                                            │           │
│           ├─────────────────────────────────────────┬─┘           │
│           │                                         │              │
│  ┌────────▼────────────────────────────┐  context  │              │
│  │          Decoder                    │◀──repr───┘              │
│  │                                     │                         │
│  │  query_coords +                     │                         │
│  │  query_emb_features +               │                         │
│  │  aggregated_context                 │                         │
│  │        ──>                          │                         │
│  │  [pred_mean, pred_log_var]          │                         │
│  └────────┬─────────────────────────────┘                         │
│           │                                                        │
│    ┌──────▼────────┐                                              │
│    │ NLL Loss +    │                                              │
│    │ Uncertainty   │                                              │
│    └───────────────┘                                              │
│                                                                    │
└────────────────────────────────────────────────────────────────────┘
```

## Data Processing Pipeline

```
┌──────────────────────────────────────────────────────────────────┐
│ Step 1: Query GEDI Data                                          │
├──────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Region Bbox (min_lon, min_lat, max_lon, max_lat)              │
│           │                                                     │
│           ▼                                                     │
│  GEDIQuerier.query_region_tiles()                              │
│           │                                                     │
│  ┌────────▼────────────────────────────────────────────┐       │
│  │ GEDI DataFrame:                                    │       │
│  │ ┌──────────────────────────────────────────────┐   │       │
│  │ │ latitude   │ longitude  │ agbd  │ tile_id  │   │       │
│  │ ├──────────────────────────────────────────────┤   │       │
│  │ │ -15.853   │ 30.256     │ 145.2│ tile_30..│   │       │
│  │ │ -15.854   │ 30.257     │ 152.1│ tile_30..│   │       │
│  │ │ ...       │ ...        │ ...  │ ...      │   │       │
│  │ └──────────────────────────────────────────────┘   │       │
│  └───────────────────┬────────────────────────────────┘       │
│                      │                                        │
│                      ▼                                        │
├──────────────────────────────────────────────────────────────────┤
│ Step 2: Extract GeoTessera Embeddings                           │
├──────────────────────────────────────────────────────────────────┤
│                                                                  │
│  EmbeddingExtractor.extract_patches_batch()                     │
│           │                                                     │
│           ├─ Get tile coordinates for each GEDI shot            │
│           ├─ Load/cache GeoTessera tiles (0.1° × 0.1°)         │
│           ├─ Convert lon/lat to pixel coordinates              │
│           └─ Extract 3×3 patches (128 channels)                │
│                                                                  │
│  Updated DataFrame with 'embedding_patch' column:               │
│  ┌─────────────────────────────────────────────────┐           │
│  │ ... │ embedding_patch (3, 3, 128) │ ...        │           │
│  │ ... │ [[[[0.34, ...], ...], ...]]  │ ...        │           │
│  └─────────────────────────────────────────────────┘           │
│                                                                  │
│                      ▼                                        │
├──────────────────────────────────────────────────────────────────┤
│ Step 3: Spatial Train/Val/Test Split                            │
├──────────────────────────────────────────────────────────────────┤
│                                                                  │
│  SpatialTileSplitter.split()                                    │
│           │                                                     │
│           ├─ 70% tiles → Train (e.g., 12 tiles, 3000 shots)    │
│           ├─ 15% tiles → Val   (e.g., 3 tiles, 750 shots)      │
│           └─ 15% tiles → Test  (e.g., 3 tiles, 750 shots)      │
│                                                                  │
│           ▼                                        │
├──────────────────────────────────────────────────────────────────┤
│ Step 4: Per-Tile Context/Target Split (During Training)         │
├──────────────────────────────────────────────────────────────────┤
│                                                                  │
│  For each tile with N GEDI shots:                              │
│                                                                  │
│  1. Normalize coordinates to [0, 1] within tile bounds          │
│  2. Sample context ratio r ~ U[0.3, 0.7]                       │
│  3. Randomly select round(N*r) shots as context                │
│  4. Remaining shots as target                                   │
│                                                                  │
│  Example (10 shots, r=0.6):                                    │
│  ┌─────────────────────────────────┐                          │
│  │ Tile (0.1° × 0.1°)             │                          │
│  │                                  │                          │
│  │ • = Context (6 shots)           │                          │
│  │ ◆ = Target (4 shots)            │                          │
│  │                                  │                          │
│  │    •      ◆                      │                          │
│  │        •                         │                          │
│  │  •   •      •      ◆             │                          │
│  │        ◆        ◆                │                          │
│  └─────────────────────────────────┘                          │
│                                                                  │
│  Output for each tile:                                          │
│  {                                                              │
│    'context_coords': (6, 2)                                    │
│    'context_embeddings': (6, 3, 3, 128)                       │
│    'context_agbd': (6, 1) - log-normalized                    │
│    'target_coords': (4, 2)                                     │
│    'target_embeddings': (4, 3, 3, 128)                        │
│    'target_agbd': (4, 1) - log-normalized                     │
│  }                                                              │
│                                                                  │
│                      ▼                                        │
├──────────────────────────────────────────────────────────────────┤
│ Step 5: Neural Process Forward Pass                             │
├──────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Input:                                                         │
│    - context_coords: (6, 2)                                     │
│    - context_embeddings: (6, 3, 3, 128)                        │
│    - context_agbd: (6, 1)                                       │
│    - target_coords: (4, 2)                                      │
│    - target_embeddings: (4, 3, 3, 128)                         │
│                                                                  │
│  Processing:                                                    │
│    1. Embed all embeddings (context + target)                  │
│       context_emb_features: (6, 128)                           │
│       target_emb_features: (4, 128)                            │
│                                                                  │
│    2. Encode context points                                     │
│       context_repr = ContextEncoder(                           │
│         coords=(6,2), emb_features=(6,128), agbd=(6,1)        │
│       ) → (6, 128)                                             │
│                                                                  │
│    3. Aggregate context with attention                         │
│       aggregated_context = AttentionAggregator(                │
│         query=target_coords+target_emb_features,               │
│         context=context_repr                                   │
│       ) → (4, 128)                                             │
│                                                                  │
│    4. Decode target points                                      │
│       pred_mean, pred_log_var = Decoder(                       │
│         coords=(4,2), emb_features=(4,128),                   │
│         context_repr=(4,128)                                   │
│       ) → (4, 1), (4, 1)                                       │
│                                                                  │
│  Output:                                                        │
│    - pred_mean: (4, 1) - predicted log-normalized AGBD        │
│    - pred_log_var: (4, 1) - uncertainty (log scale)           │
│                                                                  │
│                      ▼                                        │
├──────────────────────────────────────────────────────────────────┤
│ Step 6: Loss & Backprop                                         │
├──────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Loss = NLLGaussian(pred_mean, pred_log_var, target_agbd)     │
│       = mean(0.5 * (log_var +                                 │
│                     exp(-log_var) * (target - mean)²))        │
│                                                                  │
│  Backward pass → Gradient Clipping (max_norm=1.0)             │
│  Adam Optimizer step                                            │
│                                                                  │
└──────────────────────────────────────────────────────────────────┘
```

## Feature Dimensions at Each Stage

```
Context Point:
  coords (2D: lon, lat)
    ↓
  EmbeddingEncoder
    ↓
  emb_features (128D)
  
  ────────────────────────────
  
  [coords (2D) + emb_features (128D) + agbd (1D)] → 131D
    ↓
  ContextEncoder
    ↓
  context_repr (128D)


Query Point:
  coords (2D: lon, lat)
    ↓
  EmbeddingEncoder
    ↓
  emb_features (128D)
  
  ────────────────────────────
  
  [coords (2D) + emb_features (128D)] → 130D
    ↓
  QueryProj (if attention)
    ↓
  query_repr (128D)


Aggregation:
  context_repr (6 points × 128D)
    ↓
  AttentionAggregator (multihead attention)
    ↓
  aggregated_context (4 query points × 128D)


Decoder:
  [coords (2D) + emb_features (128D) + aggregated_context (128D)] → 258D
    ↓
  3× FC layers with LayerNorm
    ↓
  [pred_mean (1D), pred_log_var (1D)]
```

## Normalization & Denormalization

```
┌─────────────────────────────────────────────────────────────────┐
│                    TRAINING                                     │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  GEDI AGBD (raw): [45.2, 150.3, 200.1, ...] Mg/ha             │
│         │                                                      │
│         ├─ Log transform: log(1 + x)                           │
│         │  [3.82, 5.01, 5.30, ...]                            │
│         │                                                      │
│         └─ Normalize by scale=200: divide by log(1+200)       │
│            [0.68, 0.89, 0.94, ...]                            │
│                                                                 │
│  Model trains on normalized values in [0, 1]                  │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                   INFERENCE                                      │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Predicted (normalized): [0.72, 0.85, 0.91, ...]             │
│         │                                                      │
│         └─ Denormalize: multiply by log(1+200) ≈ 5.61        │
│            [4.04, 4.77, 5.11, ...]                            │
│                                                                 │
│         └─ Inverse log transform: exp(x) - 1                  │
│            [56.3, 117.1, 165.8, ...] Mg/ha                   │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│              COORDINATE NORMALIZATION                           │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Raw coordinates (WGS84):                                       │
│    lon: [30.2560, 30.2570, 30.2580]                           │
│    lat: [-15.8530, -15.8540, -15.8550]                        │
│         │                                                      │
│         ├─ Within-tile min/max:                                │
│         │    lon_range = 30.2580 - 30.2560 = 0.002           │
│         │    lat_range = -15.8530 - (-15.8550) = 0.002       │
│         │                                                      │
│         └─ Normalize to [0, 1]:                                │
│            lon_norm = (lon - 30.2560) / 0.002                 │
│            lat_norm = (lat - (-15.8550)) / 0.002              │
│                                                                 │
│  Normalized: [0.0, 0.5, 1.0], [1.0, 0.5, 0.0]                │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

## Training Dynamics

```
Batch (16 tiles) from DataLoader
       │
       ├─ Tile 1: 8 context points, 3 target points
       ├─ Tile 2: 5 context points, 2 target points
       ├─ Tile 3: 12 context points, 5 target points
       └─ ... (16 tiles total)
       │
       ▼
Per-tile forward passes (independent)
       │
       ├─ Tile 1: loss₁
       ├─ Tile 2: loss₂
       ├─ Tile 3: loss₃
       └─ ...
       │
       ▼
Average loss across tiles with targets
       │
       ▼
Single backward pass + optimizer step
```

