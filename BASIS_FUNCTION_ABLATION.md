# Basis Function Ablation Study Guide

This guide describes how to run ablation experiments with different basis functions for coordinate encoding in the ANP model.

## Overview

The ANP now supports multiple basis function types for encoding spatial coordinates:

1. **`none`** (baseline): Raw normalized coordinates
2. **`fourier_random`**: Random Fourier features with fixed frequencies
3. **`fourier_learnable`**: Fourier features with learnable frequencies
4. **`hybrid`**: Concatenation of raw coordinates + Fourier features

## New Command-Line Arguments

```bash
--basis_function_type {none,fourier_random,fourier_learnable,hybrid}
    Type of basis function (default: 'none')

--basis_num_frequencies INT
    Number of Fourier frequency components (default: 32)
    Output dimension = 2 * num_frequencies (sin + cos)

--basis_frequency_scale FLOAT
    Scale for frequency initialization (default: 1.0)
    Frequencies initialized as: B ~ N(0, scale^2)

--basis_learnable
    Make frequencies learnable (only for fourier_random)
    Note: fourier_learnable automatically makes frequencies learnable
```

## Recommended Ablation Experiments

### Experiment 1: Baseline vs. Random Fourier

Compare raw coordinates against fixed Fourier features:

```bash
# Baseline (current behavior)
python train.py \
  --region_bbox <bbox> \
  --architecture_mode anp \
  --basis_function_type none \
  --output_dir runs/baseline

# Random Fourier (32 frequencies)
python train.py \
  --region_bbox <bbox> \
  --architecture_mode anp \
  --basis_function_type fourier_random \
  --basis_num_frequencies 32 \
  --basis_frequency_scale 1.0 \
  --output_dir runs/fourier_32

# Random Fourier (64 frequencies)
python train.py \
  --region_bbox <bbox> \
  --architecture_mode anp \
  --basis_function_type fourier_random \
  --basis_num_frequencies 64 \
  --basis_frequency_scale 1.0 \
  --output_dir runs/fourier_64
```

### Experiment 2: Fixed vs. Learnable Frequencies

Test whether learning frequencies improves performance:

```bash
# Fixed frequencies
python train.py \
  --region_bbox <bbox> \
  --architecture_mode anp \
  --basis_function_type fourier_random \
  --basis_num_frequencies 32 \
  --output_dir runs/fourier_fixed

# Learnable frequencies
python train.py \
  --region_bbox <bbox> \
  --architecture_mode anp \
  --basis_function_type fourier_learnable \
  --basis_num_frequencies 32 \
  --output_dir runs/fourier_learnable
```

### Experiment 3: Hybrid Encoding

Compare pure Fourier vs. hybrid (raw + Fourier):

```bash
# Pure Fourier
python train.py \
  --region_bbox <bbox> \
  --architecture_mode anp \
  --basis_function_type fourier_random \
  --basis_num_frequencies 32 \
  --output_dir runs/pure_fourier

# Hybrid (raw + Fourier)
python train.py \
  --region_bbox <bbox> \
  --architecture_mode anp \
  --basis_function_type hybrid \
  --basis_num_frequencies 32 \
  --output_dir runs/hybrid
```

### Experiment 4: Frequency Scale Sensitivity

Test impact of frequency initialization scale:

```bash
for scale in 0.5 1.0 2.0; do
  python train.py \
    --region_bbox <bbox> \
    --architecture_mode anp \
    --basis_function_type fourier_random \
    --basis_num_frequencies 32 \
    --basis_frequency_scale $scale \
    --output_dir runs/fourier_scale_${scale}
done
```

## Evaluation Metrics

For each experiment, track:

1. **Interpolation performance**: R², RMSE on test set (same region)
2. **Extrapolation performance**: R², RMSE on held-out regions (if available)
3. **Training efficiency**: Convergence speed, final loss
4. **Model size**: Number of parameters
5. **Uncertainty calibration**: Predicted std vs. actual error

Use the spatial extrapolation evaluation script:

```bash
python evaluate_spatial_extrapolation.py \
  --checkpoint_dir runs/fourier_32 \
  --output_dir evals/fourier_32
```

## Expected Outcomes

Based on similar work in neural fields and spatial interpolation:

- **Fourier features vs. raw coords**: 5-10% improvement in interpolation, 10-30% in extrapolation
- **Learnable vs. fixed**: Small improvement (2-5%), may overfit on small datasets
- **Hybrid encoding**: Best of both worlds, robust performance
- **More frequencies**: Diminishing returns beyond 32-64 frequencies

## Parameter Count Impact

| Basis Type | Num Frequencies | Output Dim | Added Parameters |
|------------|-----------------|------------|------------------|
| none | - | 2 | 0 |
| fourier_random | 32 | 64 | 0 (fixed) |
| fourier_learnable | 32 | 64 | 64 (2×32) |
| hybrid | 32 | 66 | 0 |

Note: Parameter increase primarily affects:
- ContextEncoder input layer
- Decoder input layer
- Query projection (for attention)

## Implementation Details

### Where Basis Functions Are Applied

1. **Context Encoding**: Coordinates → Basis Functions → ContextEncoder
2. **Query Encoding**: Coordinates → Basis Functions → Decoder
3. **Attention Queries**: Coordinates → Basis Functions → Query Projection

The same basis function encoder is shared across all three locations to ensure consistency.

### Coordinate Normalization

Coordinates are normalized to [0, 1] range BEFORE basis function encoding:

```
lon, lat → normalize to [0,1] → basis_function_encoder → MLP
```

This ensures that Fourier features operate on a consistent input range.

### Random Seed Handling

For reproducibility, set random seed before training:

```bash
python train.py --seed 42 ...
```

Note: Random Fourier features use the global random state, so different seeds will produce different frequency matrices.

## Troubleshooting

### NaN losses with Fourier features

- Try reducing `--basis_frequency_scale` (e.g., 0.5)
- Increase `--lr_scheduler_patience` for more stable training
- Check if coordinates are properly normalized to [0, 1]

### Poor performance with learnable frequencies

- May indicate overfitting - try fixed frequencies first
- Reduce number of frequencies
- Increase weight decay: `--weight_decay 0.05`

### Memory issues with many frequencies

- Reduce `--basis_num_frequencies` (try 16 or 32)
- Reduce batch size: `--batch_size 8`
- Use hybrid encoding only if needed

## References

- [Fourier Features Let Networks Learn High Frequency Functions](https://arxiv.org/abs/2006.10739)
- [NeRF: Neural Radiance Fields](https://arxiv.org/abs/2003.08934)
- [Attention is All You Need](https://arxiv.org/abs/1706.03762) (positional encoding)
