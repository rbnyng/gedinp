# GEDI Neural Process Codebase Exploration - Quick Reference

## Documents Created

I've created comprehensive documentation about the GEDI Neural Process codebase:

### 1. EXPLORATION_SUMMARY.md (this directory)
**Quick start guide with key insights**
- 1-page overview of CNP architecture
- Feature stack explanation
- Training pipeline walkthrough
- Data format specification
- Baseline implementation guide

### 2. CODEBASE_ANALYSIS.md
**Detailed technical breakdown (9.2 KB)**
- 1. Neural Process Implementation (5 component breakdown)
- 2. Features and Embeddings (GeoTessera, coords, AGBD)
- 3. Training Loop and Metrics (complete pipeline)
- 4. Data Format and Context/Target Structure
- 5. Existing Baseline Models (none currently)
- 6. Key Files Summary (file-by-file guide)
- 7. All Hyperparameters
- 8. Implementation Details (memory, validation, spatial properties)

### 3. ARCHITECTURE_FLOW.md
**Visual ASCII diagrams and data flow (24 KB)**
- Neural Process Architecture Diagram
- End-to-end Data Processing Pipeline (8 steps illustrated)
- Feature Dimensions at Each Stage
- Normalization & Denormalization Examples
- Training Dynamics

### 4. KEY_CODE_SNIPPETS.md
**Actual code excerpts from repository (15 KB)**
1. Neural Process Forward Pass (lines 354-414)
2. Loss Function (lines 446-472)
3. Training Loop (lines 110-167)
4. Data Preparation (lines 108-172)
5. Embedding Extraction (lines 159-206)
6. Spatial Cross-Validation (lines 50-81)
7. Metrics Computation (lines 475-514)
8. Complete Training Example

---

## Key Takeaways

### What is Being Done?
The codebase implements a **Conditional Neural Process (CNP)** to predict AGBD (Aboveground Biomass Density) from GEDI satellite lidar data using GeoTessera foundation model embeddings.

### Architecture Components
```
Input: GEDI shots with coords + GeoTessera embeddings + AGBD labels
  ↓
1. EmbeddingEncoder (CNN): 3×3 embeddings → 128D features
2. ContextEncoder (MLP): coords + emb_features + agbd → 128D representations
3. AttentionAggregator: Multi-head attention to weight context
4. Decoder (MLP): coords + emb_features + aggregated_context → AGBD + uncertainty
  ↓
Output: Mean AGBD prediction + uncertainty (log-variance)
```

### Key Features Used
- **Spatial Coordinates**: lon/lat (normalized to [0,1] within tile)
- **GeoTessera Embeddings**: 3×3 patches with 128 channels (foundation model)
- **AGBD (target)**: Log-transformed and normalized (max ~200 Mg/ha)

### Training Pipeline
```
Region → Query GEDI → Extract Embeddings → Spatial Split
  ↓
  Per-Tile Processing:
    - Random context/target split (30-70%)
    - Forward pass through model
    - Gaussian NLL loss (learns uncertainty)
    - Backprop with gradient clipping
  ↓
  Validation each epoch
  ↓
  Save best model (by val loss or R²)
```

### Evaluation Metrics
- RMSE, MAE, R², Mean Uncertainty
- Computed per-tile, then averaged

### No Baselines Currently Exist
To add baselines, implement:
- Simple mean predictor
- Linear regression on embedding features
- Spatial interpolation (kriging)
- Tree-based methods (XGBoost, random forest)
- MLP without context aggregation

---

## Important Implementation Details

### Data Normalization
```
AGBD: raw → log(1+x) → divide by log(1+200) → model trains on [0,1]
Coords: within-tile normalization to [0,1] (ensures relative position matters)
```

### Context/Target Split (Per Tile)
```
Example: 10 GEDI shots in a tile
  Context ratio ~ Uniform[0.3, 0.7] → e.g., 0.6
  → 6 context shots + 4 target shots
  → Model learns from context, predicts target
  → This varies each epoch for regularization
```

### Spatial Train/Val/Test Split
```
Split at TILE LEVEL (not individual shots)
→ 70% tiles for training
→ 15% tiles for validation  
→ 15% tiles for testing
→ Prevents spatial leakage
```

### Uncertainty Quantification
```
Model outputs: mean AND log-variance
Loss function: Gaussian NLL = 0.5 * (log_var + exp(-log_var) * error²)
→ Model learns when to be confident vs uncertain
→ Predictions include uncertainty estimates
```

---

## File Structure

```
/home/user/gedinp/
│
├── Core Model
│   ├── models/neural_process.py      # GEDINeuralProcess class
│   ├── train.py                      # Training script
│   └── predict.py                    # Inference script
│
├── Data Processing
│   ├── data/gedi.py                  # GEDI data querying
│   ├── data/embeddings.py            # GeoTessera extraction
│   ├── data/dataset.py               # PyTorch Dataset class
│   └── data/spatial_cv.py            # Train/val/test splits
│
├── Documentation (Created by Exploration)
│   ├── EXPLORATION_SUMMARY.md        # This file (overview)
│   ├── CODEBASE_ANALYSIS.md          # Detailed analysis
│   ├── ARCHITECTURE_FLOW.md          # Diagrams & flow
│   └── KEY_CODE_SNIPPETS.md          # Code excerpts
│
└── Tests & Examples
    ├── tests/test_unit.py
    ├── tests/test_pipeline.py
    └── example_usage.py
```

---

## How to Use This Documentation

### For Understanding the Model
Start with → EXPLORATION_SUMMARY.md (quick overview)
Then read → KEY_CODE_SNIPPETS.md (see actual code)
Deep dive → CODEBASE_ANALYSIS.md (detailed breakdown)

### For Architecture/Data Flow
Read → ARCHITECTURE_FLOW.md (visual diagrams)
Shows: Pipeline, feature dims, normalization, training dynamics

### For Modifying/Extending
Reference → KEY_CODE_SNIPPETS.md (copy-paste ready code)
Check → CODEBASE_ANALYSIS.md (all hyperparameters)

### For Creating Baselines
Start at → Section 5 in CODEBASE_ANALYSIS.md
Then → Section 10 in EXPLORATION_SUMMARY.md
Code example at → KEY_CODE_SNIPPETS.md section 1

---

## Quick Reference: Key Numbers

| Aspect | Value |
|--------|-------|
| Embedding dim | 128 (3×3 patches) |
| Encoded emb dim | 128 |
| Context repr dim | 128 |
| Hidden layer dim | 512 |
| Attention heads | 4 |
| Tile size | 0.1° × 0.1° |
| Context ratio | U[0.3, 0.7] |
| Learning rate | 5e-4 |
| Batch size | 16 tiles |
| AGBD scale | 200.0 Mg/ha |
| Coord noise (aug) | σ=0.01 |
| Gradient clip | max_norm=1.0 |
| Patience (ES) | 15 epochs |

---

## Questions Answered

### 1. What is the neural process implementation like?
A: Conditional Neural Process with 4 components:
- EmbeddingEncoder (CNN for patches)
- ContextEncoder (encode observed points)
- AttentionAggregator (weight context)
- Decoder (predict target AGBD + uncertainty)

### 2. What embeddings/features are used?
A: Three types:
- GeoTessera (128D, foundation model)
- Coordinates (2D, lon/lat normalized)
- AGBD (1D, log-normalized, ground truth only for context)

### 3. How does training work?
A: Per-tile with random context/target split:
- Context points (30-70%) provide information
- Target points (30-70%) are predicted
- Loss: Gaussian NLL (learns uncertainty)
- Optimizer: Adam with LR scheduler and early stopping

### 4. How is data structured?
A: Each tile is (N, [coords, emb, agbd])
Split into context and target sets randomly each epoch
Normalized: coords → [0,1], agbd → log then [0,1]

### 5. Are there baselines?
A: No. Implement: mean, linear regression, kriging, XGBoost, etc.

---

## Next Steps

1. **Read EXPLORATION_SUMMARY.md** for overview
2. **Read ARCHITECTURE_FLOW.md** for data flow
3. **Skim KEY_CODE_SNIPPETS.md** for implementation details
4. **Reference CODEBASE_ANALYSIS.md** as needed for specifics
5. **Implement your baseline model** (see section 10 in EXPLORATION_SUMMARY.md)
6. **Run training** with baseline and CNP for comparison
7. **Analyze results** using the metrics (RMSE, MAE, R²)

---

**Generated**: 2025-11-07  
**Files**: 4 documentation files (48 KB total)  
**Content**: Complete codebase exploration with diagrams, code snippets, and analysis

