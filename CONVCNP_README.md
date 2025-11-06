# ConvCNP Implementation for GEDI AGB Prediction

## Overview

This implementation upgrades the baseline Conditional Neural Process (CNP) to a **Convolutional Conditional Neural Process (ConvCNP)** architecture. This is a significant improvement that addresses the key limitation of mean pooling in the baseline model.

## Why ConvCNP?

### The Problem with Baseline CNP

The baseline CNP (`models/neural_process.py`) uses **mean pooling** to aggregate context representations:

```python
# Line 282 in models/neural_process.py
aggregated_context = context_repr.mean(dim=0, keepdim=True)
```

**This is a critical bottleneck** because:
- ❌ Destroys all spatial structure
- ❌ Treats nearby and distant points identically
- ❌ Cannot capture spatial autocorrelation
- ❌ Limits model capacity for high-fidelity predictions

### The ConvCNP Solution

ConvCNP leverages the fact that you already have **dense gridded embeddings** covering entire tiles:

✅ **Preserves spatial structure** - No information bottleneck
✅ **Efficient for dense predictions** - Process tile once, predict everywhere
✅ **Strong inductive bias** - Convolutions understand 2D locality naturally
✅ **Scalable** - O(1) prediction cost per pixel after encoding

## Architecture

### Input Representation

For each tile, we create a sparse multi-channel tensor:

```
Input: (B, 130, H, W)
├─ Channel 0:     Sparse AGBD values (only at GEDI locations)
├─ Channel 1:     Binary mask (1 = has GEDI, 0 = needs prediction)
└─ Channels 2-129: Dense GeoTessera embeddings (full tile coverage)
```

### Model Pipeline

```
1. Input Encoding
   └─ Sparse AGBD + Mask + Dense Embeddings → (B, 130, H, W)

2. UNet Encoder
   ├─ Downsampling path (captures global context)
   ├─ Upsampling path (restores resolution)
   └─ Skip connections (preserves details)
   └─ Output: Dense feature map (B, feature_dim, H, W)

3. Decoder
   ├─ Per-pixel MLP
   └─ Outputs: (mean, log_variance) for every pixel
```

### Key Components

**UNet Architecture** (`models/unet.py`):
- Encoder-decoder with skip connections
- Multiple scales for context aggregation
- Batch normalization + ReLU activations
- Configurable depth and channels

**ConvCNP Model** (`models/convcnp.py`):
- Wraps UNet encoder
- Lightweight decoder MLP
- Gaussian likelihood for uncertainty

**Dataset** (`data/convcnp_dataset.py`):
- Loads full embedding tiles
- Creates sparse AGBD/mask representations
- Context/target splits for training

## File Structure

```
New Files (ConvCNP):
├── data/convcnp_dataset.py       # Sparse tile dataset
├── models/unet.py                # UNet encoder architecture
├── models/convcnp.py             # ConvCNP model + loss + metrics
├── train_convcnp.py              # Training script
├── predict_convcnp.py            # Dense prediction inference
└── test_convcnp.py               # Unit tests

Existing Files (Baseline):
├── models/neural_process.py      # Baseline CNP (kept for comparison)
├── train.py                      # Baseline training
└── ...
```

## Usage

### 1. Training

```bash
python train_convcnp.py \
  --region_bbox 30.256 -15.853 30.422 -15.625 \
  --embedding_year 2024 \
  --cache_dir ./cache \
  --output_dir ./outputs_convcnp \
  --batch_size 4 \
  --lr 1e-4 \
  --epochs 100 \
  --feature_dim 128 \
  --base_channels 64 \
  --unet_depth 3 \
  --max_tile_size 512
```

**Key Arguments**:
- `--region_bbox`: Bounding box for GEDI data (min_lon min_lat max_lon max_lat)
- `--max_tile_size`: Downsample tiles larger than this (default: 512)
- `--feature_dim`: UNet output feature dimension (default: 128)
- `--base_channels`: UNet base channels (default: 64)
- `--unet_depth`: Number of UNet levels (default: 3)
- `--use_small_unet`: Use smaller UNet for faster training

### 2. Inference (Dense Predictions)

```bash
python predict_convcnp.py \
  --model_path ./outputs_convcnp/best_model.pt \
  --tile_lon 30.35 \
  --tile_lat -15.75 \
  --embedding_year 2024 \
  --cache_dir ./cache \
  --output_dir ./predictions \
  --context_bbox 30.256 -15.853 30.422 -15.625
```

**Outputs**:
- `tile_*.npy`: Dense AGBD predictions (mean and std)
- `tile_*.png`: Visualization with context locations

### 3. Testing

```bash
python test_convcnp.py
```

Runs unit tests for:
- UNet forward/backward pass
- ConvCNP model
- Loss computation
- Metrics
- Gradient flow
- Variable input sizes

## Model Sizes

| Configuration | Parameters | Memory (256x256) | Use Case |
|--------------|------------|------------------|----------|
| Small UNet   | ~5M        | ~2 GB           | Fast prototyping |
| Standard (depth=3) | ~15M | ~4 GB      | Balanced |
| Large (depth=4)    | ~40M | ~8 GB      | Maximum capacity |

## Comparison: ConvCNP vs Baseline CNP

| Aspect | Baseline CNP | ConvCNP |
|--------|-------------|---------|
| **Context Aggregation** | Mean pooling (bottleneck) | UNet (preserves structure) |
| **Spatial Understanding** | Must learn from coordinates | Built-in via convolutions |
| **Dense Predictions** | Slow (per-point decoding) | Fast (single pass) |
| **Scalability** | O(N) for N targets | O(1) per pixel after encoding |
| **Memory (training)** | ~1 GB | ~4-8 GB |
| **Best For** | Few scattered predictions | Dense map generation |

## Performance Tips

### Memory Optimization

1. **Reduce tile size**: Use `--max_tile_size 256` or `512`
2. **Smaller UNet**: Use `--use_small_unet` flag
3. **Reduce batch size**: `--batch_size 2` or `1`
4. **Fewer channels**: `--base_channels 32`

### Speed Optimization

1. **Shallow UNet**: `--unet_depth 2`
2. **Lower resolution**: `--max_tile_size 256`
3. **More workers**: `--num_workers 4` (if you have CPU cores)

### Quality Optimization

1. **Deeper UNet**: `--unet_depth 4`
2. **More channels**: `--base_channels 128`
3. **Higher resolution**: `--max_tile_size 1024` (if GPU has memory)

## Training Tips

### Context Ratio

The model learns by masking random GEDI shots:
- **Context**: Used as input (with AGBD values)
- **Target**: Hidden from model (used for loss)

Default range: 10-90% context shots per tile. This forces the model to interpolate.

### Loss Function

Negative log-likelihood with predicted uncertainty:

```python
loss = 0.5 * (log_var + exp(-log_var) * (target - pred)^2)
```

Benefits:
- Penalizes overconfident wrong predictions
- Rewards well-calibrated uncertainty
- Handles heteroscedastic noise

## Next Steps

### After Training

1. **Evaluate on test set**:
   ```bash
   # Modify train_convcnp.py to add test evaluation
   ```

2. **Compare to baseline**:
   - Train baseline CNP with same data
   - Compare RMSE, MAE, R² scores
   - Analyze prediction quality visually

3. **Generate maps**:
   - Use `predict_convcnp.py` for multiple tiles
   - Create regional biomass maps
   - Analyze uncertainty patterns

### Potential Improvements

1. **Multi-scale Features**:
   - Extract features at multiple UNet levels
   - Helps capture both local and global patterns

2. **Attention in Decoder**:
   - Cross-attention between features and targets
   - May improve fine-grained predictions

3. **Data Augmentation**:
   - Random rotations/flips during training
   - Helps with generalization

4. **Conditional Batch Normalization**:
   - Condition on context density
   - Handles varying shot distributions better

## Troubleshooting

### Out of Memory

**Symptom**: CUDA out of memory error

**Solutions**:
1. Reduce `--max_tile_size` to 256 or 128
2. Use `--batch_size 1`
3. Enable `--use_small_unet`
4. Reduce `--base_channels` to 32 or 16

### Slow Training

**Symptom**: Very slow iterations

**Solutions**:
1. Check tile sizes (print in dataset) - downsample if >512
2. Reduce `--unet_depth` to 2
3. Use GPU if available
4. Profile with PyTorch profiler

### Poor Predictions

**Symptom**: High RMSE, low R²

**Solutions**:
1. Check if embeddings are loading correctly
2. Verify GEDI shots are on tile (check coordinates)
3. Increase model capacity (`--base_channels 128`)
4. Train for more epochs
5. Check for data quality issues (outliers in AGBD)

## Technical Details

### Why Not Attentive Neural Process (ANP)?

While ANP handles irregular points natively, it has critical limitations:

❌ **Spatial blindness**: Must learn 2D geometry from scratch
❌ **O(N²) complexity**: Prohibitive for dense predictions
❌ **Inefficient**: Computes attention for every target pixel

ConvCNP is superior because:
✅ **Built-in spatial bias**: Free understanding of locality
✅ **O(1) per pixel**: Massively parallelizable
✅ **Perfect for grids**: Matches embedding structure

### Why This Works for GEDI

GEDI data has two key properties:

1. **Sparse observations**: ~10-100 shots per 10km² tile
2. **Dense embeddings**: Full 10m resolution coverage

ConvCNP exploits both:
- Embeddings guide interpolation between sparse GEDI points
- Convolutions capture spatial patterns in biomass distribution

## Citation

If you use this implementation, please cite:

```bibtex
@article{convcnp2018,
  title={Convolutional Conditional Neural Processes},
  author={Gordon, Jonathan and Bruinsma, Wessel P and Foong, Andrew YK and Requeima, James and Dubois, Yann and Turner, Richard E},
  journal={arXiv preprint arXiv:1810.13428},
  year={2018}
}
```

## License

Same as parent repository.

## Questions?

- Check `test_convcnp.py` for usage examples
- Review `models/convcnp.py` for architecture details
- Compare with `models/neural_process.py` to understand improvements
