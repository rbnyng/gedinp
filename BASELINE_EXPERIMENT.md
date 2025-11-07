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

### Compare Models

After training both CNP and baseline:

```bash
python compare_models.py \
    --cnp_output_dir ./outputs \
    --baseline_output_dir ./outputs_baseline \
    --output_file comparison_results.json
```

This will output:
- Test set performance for both models
- Side-by-side comparison
- Percentage differences
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

| Aspect | CNP | Baseline |
|--------|-----|----------|
| Training | Context/target splits | Direct mapping |
| Architecture | 4 components + attention | Single MLP |
| Batch Size | 16 tiles | 256 individual points |
| Aggregation | Attention over context | None |
| Inference | Requires context set | Direct prediction |

## Files

- `models/baseline.py` - Baseline model implementations
- `train_baseline.py` - Training script for baselines
- `compare_models.py` - Model comparison script
- `BASELINE_EXPERIMENT.md` - This document

## Next Steps

1. Train baseline on your data
2. Compare with existing CNP results
3. Based on comparison:
   - If similar: Consider using baseline or investigating why CNP doesn't help
   - If CNP better: Document the performance gain and continue with CNP
   - If baseline better: Debug CNP training or tune hyperparameters

## Optional: XGBoost/Random Forest

For an even simpler baseline, you could extract the embeddings + coordinates and train XGBoost or Random Forest. This would be the fastest to train and could provide additional insights.

To add this:
1. Extract features from the dataset
2. Train XGBoost/RF with scikit-learn
3. Compare performance

This is left as a future enhancement if the MLP baseline shows promise.
