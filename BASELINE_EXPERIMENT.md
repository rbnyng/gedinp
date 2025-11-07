# Baseline Experiment: Testing CNP Value-Add

## Motivation

This experiment tests whether the Conditional Neural Process (CNP) architecture adds value over a simple baseline that uses the same features (foundation model embeddings + lat/lon) with a basic MLP.

**Key Question:** If a simple MLP achieves similar performance, the CNP's context aggregation mechanism may not be necessary.

## Models Compared

### 1. CNP (Current Implementation)
- **Architecture:** EmbeddingEncoder (CNN) → ContextEncoder (MLP) → AttentionAggregator → Decoder (MLP)
- **Training:** Context/target splits within tiles
- **Features:** 128D embeddings + 2D lat/lon
- **Complexity:** ~1-2M parameters

### 2. Simple MLP Baseline
- **Architecture:** EmbeddingEncoder (CNN, same as CNP) → Simple MLP
- **Training:** Direct point-to-point mapping (no context/target splits)
- **Features:** 128D embeddings + 2D lat/lon (same as CNP)
- **Complexity:** ~500K-1M parameters

### 3. Flat MLP Baseline (Alternative)
- **Architecture:** Flatten embedding patch → Simple MLP
- **Training:** Direct point-to-point mapping
- **Features:** 1152D flattened embeddings + 2D lat/lon
- **Complexity:** Higher parameter count due to larger input

### 4. XGBoost Baseline
- **Architecture:** Gradient boosted decision trees
- **Training:** Direct point-to-point mapping (fast, minutes not hours)
- **Features:** 1154D flattened embeddings + 2D lat/lon
- **Complexity:** Controlled by n_estimators and max_depth

### 5. Random Forest Baseline
- **Architecture:** Ensemble of decision trees
- **Training:** Direct point-to-point mapping (fast)
- **Features:** 1154D flattened embeddings + 2D lat/lon
- **Complexity:** Controlled by n_estimators and max_depth

## Usage

### Train Baseline Model

```bash
python train_baseline.py \
    --region_bbox -122.5 37.5 -122.0 38.0 \
    --model_type simple_mlp \
    --hidden_dim 512 \
    --embedding_feature_dim 128 \
    --batch_size 256 \
    --lr 1e-3 \
    --epochs 100 \
    --output_dir ./outputs_baseline
```

**Model Types:**
- `simple_mlp`: Uses same CNN encoder as CNP (fair comparison)
- `flat_mlp`: Flattens embedding patch (tests if CNN helps)

### Train Tree-Based Baseline (XGBoost/Random Forest)

```bash
python train_tree_baseline.py \
    --region_bbox -122.5 37.5 -122.0 38.0 \
    --model_type both \
    --n_estimators 100 \
    --max_depth 10 \
    --output_dir ./outputs_tree
```

**Model Types:**
- `xgboost`: XGBoost gradient boosting (requires: pip install xgboost)
- `random_forest`: Random Forest ensemble
- `both`: Train both models

**Advantages:**
- **Fast training:** Minutes instead of hours
- **No GPU needed:** Runs efficiently on CPU
- **Strong baseline:** Often competitive with neural networks
- **Interpretable:** Can extract feature importances

### Compare All Models

Compare CNP, MLP baseline, and tree baselines:

```bash
python compare_all_models.py \
    --cnp_dir ./outputs \
    --mlp_dir ./outputs_baseline \
    --tree_dir ./outputs_tree \
    --output_file all_models_comparison.json
```

Or compare just CNP vs MLP:

```bash
python compare_models.py \
    --cnp_output_dir ./outputs \
    --baseline_output_dir ./outputs_baseline \
    --output_file comparison_results.json
```

This will output:
- Test set performance for all models
- Side-by-side comparison table
- Percentage differences
- Best model identification
- Interpretation of results

## Expected Outcomes

### If Baseline Performs Similarly (R² difference < 5%)
**Conclusion:** CNP's context aggregation doesn't add significant value
- Consider using simpler baseline for production
- Faster training and inference
- Easier to interpret

### If CNP Performs Better (R² difference > 5%)
**Conclusion:** Context aggregation is valuable
- Spatial patterns matter for prediction
- CNP's attention mechanism helps
- Complex architecture is justified

### If Baseline Performs Better
**Conclusion:** CNP may be overfitting or poorly tuned
- Try adjusting CNP hyperparameters
- Reduce model complexity
- Check for training issues

## Implementation Details

### Fair Comparison Guarantees

1. **Same Features:** Both models use identical embeddings and coordinates
2. **Same Data Splits:** Both use identical train/val/test splits
3. **Same Preprocessing:** Both use same normalization and log transforms
4. **Same Encoder:** Simple MLP baseline uses same CNN encoder as CNP
5. **Same Metrics:** Both evaluated with RMSE, MAE, R²

### Key Differences

| Aspect | CNP | MLP Baseline | Tree Baseline |
|--------|-----|--------------|---------------|
| Training | Context/target splits | Direct mapping | Direct mapping |
| Architecture | 4 components + attention | Single MLP | Decision trees |
| Training Time | Hours (GPU) | Hours (GPU) | Minutes (CPU) |
| Batch Size | 16 tiles | 256 points | N/A |
| Aggregation | Attention over context | None | Tree splits |
| Inference | Requires context set | Direct prediction | Direct prediction |
| Uncertainty | Learned variance | Learned variance | None (RF: variance across trees) |

## Files

- `models/baseline.py` - MLP baseline model implementations (SimpleMLPBaseline, FlatMLPBaseline)
- `train_baseline.py` - Training script for MLP baselines
- `train_tree_baseline.py` - Training script for tree-based baselines (XGBoost, Random Forest)
- `compare_models.py` - Compare CNP vs MLP baseline
- `compare_all_models.py` - Compare all models (CNP, MLP, XGBoost, RF)
- `BASELINE_EXPERIMENT.md` - This document

## Next Steps

1. **Quick Start:** Train tree baselines first (fastest, minutes on CPU)
2. **Compare:** If tree models do well, train MLP baseline
3. **Final Test:** If simple baselines perform well, compare with CNP
4. **Interpret Results:**
   - If tree/MLP performs similarly to CNP → Use simpler model
   - If CNP significantly better → Context aggregation is valuable
   - If baselines better → Investigate CNP training or hyperparameters

## Recommended Training Order

1. **Start with XGBoost/RF** (fastest, ~5-10 minutes)
   - Establishes a quick performance baseline
   - No GPU required
   - If these work well, CNP may be overkill

2. **Then MLP baseline** (~1-2 hours on GPU)
   - Tests if neural network helps vs tree models
   - Uses same features as CNP for fair comparison

3. **Finally compare with CNP** (if needed)
   - Only if simpler models are insufficient
   - Determines value of context aggregation

## Dependencies

### For MLP Baselines
```bash
# Already installed if you have the CNP setup
torch, numpy, pandas
```

### For Tree Baselines
```bash
# Install XGBoost (optional, Random Forest works without it)
pip install xgboost

# scikit-learn (likely already installed)
pip install scikit-learn
```
