# Variance Calibration for Baseline Models

## Summary

Added variance calibration to Random Forest and XGBoost baseline models using temperature scaling.

## What Was Added

### 1. Temperature Scaling Implementation

Both `RandomForestBaseline` and `XGBoostBaseline` in `baselines/models.py` now have:

- **`temperature` attribute**: Initialized to 1.0 (no scaling by default)
- **`calibrate()` method**: Optimizes temperature on validation set to minimize NLL
- **Automatic scaling**: `predict()` automatically applies temperature to std estimates

### 2. Calibration Method

```python
def calibrate(coords, embeddings, agbd, bounds=(0.1, 10.0)) -> float
```

**Purpose**: Find optimal temperature T that minimizes negative log-likelihood:

```
NLL = 0.5 * (log(T^2 * σ^2) + (y - ŷ)^2 / (T^2 * σ^2))
```

**Process**:
1. Get uncalibrated predictions and std from model
2. Define NLL as function of temperature T
3. Optimize T using scipy's bounded scalar minimization
4. Store optimal T in model.temperature
5. Future predictions automatically scale std by T

**Result**: Calibrated uncertainties better reflect actual prediction errors

### 3. Training Integration

Updated `train_baselines.py` to:
- Pass validation data to training functions
- Call `.calibrate()` after model training, before evaluation
- Print calibration results (optimal T and final NLL)

## How It Works

### Random Forest
- **Original**: std = ensemble variance (tree disagreement)
- **Calibrated**: std = ensemble variance × T
- **Effect**: Adjusts for systematic over/under-confidence in ensemble

### XGBoost
- **Original**: std ≈ (Q₀.₉₅ - Q₀.₀₅) / (2 × 1.96)
- **Calibrated**: std = quantile-based std × T
- **Effect**: Corrects quantile interval → Gaussian std approximation

## Why Temperature Scaling?

1. **Simple**: Single scalar parameter (T)
2. **Fast**: Optimization takes < 1 second
3. **Effective**: Proven method for calibrating uncertainties
4. **No retraining**: Applied post-hoc to trained models
5. **Preserves order**: Relative uncertainties maintained

## Usage

### Training with Calibration (Automatic)

```bash
python train_baselines.py \
  --region_bbox -123 45 -122 46 \
  --models rf xgb
```

Calibration now runs automatically after training each model.

### Manual Calibration

```python
from baselines.models import RandomForestBaseline

# Train model
model = RandomForestBaseline()
model.fit(train_coords, train_embeddings, train_agbd)

# Calibrate on validation set
model.calibrate(val_coords, val_embeddings, val_agbd)

# Predictions now use calibrated uncertainties
pred, std_calibrated = model.predict(test_coords, test_embeddings)
```

## Testing

Run the test suite to verify calibration:

```bash
python test_calibration.py
```

Tests verify:
- Temperature scaling is applied correctly
- Optimization converges successfully
- Calibrated z-scores closer to ideal N(0,1)

## Expected Results

Well-calibrated uncertainties should have:

1. **Z-scores ≈ N(0,1)**
   - Mean ≈ 0 (unbiased)
   - Std ≈ 1 (correct scale)

2. **Good coverage**
   - ~68% of points within 1σ
   - ~95% of points within 2σ
   - ~99.7% of points within 3σ

3. **Uncertainty-error correlation**
   - High predicted std → high actual error
   - Low predicted std → low actual error

## Calibration Metrics Reporting

Training now automatically computes and displays calibration metrics:

### During Training

After each evaluation, you'll see:

```
Validation - Calibration Metrics:
  Z-scores: μ = +0.0123 (ideal: 0.0), σ = 0.9876 (ideal: 1.0)
  Coverage: 1σ = 68.5% (ideal: 68.3%), 2σ = 95.2% (ideal: 95.4%), 3σ = 99.6% (ideal: 99.7%)

Test - Calibration Metrics:
  Z-scores: μ = -0.0045 (ideal: 0.0), σ = 1.0234 (ideal: 1.0)
  Coverage: 1σ = 67.8% (ideal: 68.3%), 2σ = 94.9% (ideal: 95.4%), 3σ = 99.5% (ideal: 99.7%)
```

### Summary Table

At the end of training, a summary table compares all models:

```
================================================================================
CALIBRATION SUMMARY (Test Set)
================================================================================
Model                Z-score μ    Z-score σ   1σ Cov%   2σ Cov%   3σ Cov%
                    (ideal: 0)   (ideal: 1)   (68.3%)   (95.4%)   (99.7%)
--------------------------------------------------------------------------------
RANDOM_FOREST          -0.0045       1.0234      67.8      94.9      99.5
XGBOOST                +0.0123       0.9876      68.5      95.2      99.6
IDW                    +0.1234       1.4567      72.3      96.8      99.8
================================================================================
```

### Interpreting Results

**Z-score mean (μ)**:
- μ ≈ 0: Unbiased (predictions centered on truth)
- μ > 0: Systematic overestimation
- μ < 0: Systematic underestimation

**Z-score std (σ)**:
- σ ≈ 1: Well-calibrated (uncertainties match errors)
- σ < 1: Over-confident (uncertainties too small)
- σ > 1: Under-confident (uncertainties too large)

**Coverage**:
- Close to ideal: Well-calibrated prediction intervals
- Higher than ideal: Conservative (over-estimates uncertainty)
- Lower than ideal: Optimistic (under-estimates uncertainty)

## Implementation Details

### Numerical Stability

- Variance clamped to minimum of 1e-10 to avoid log(0)
- Temperature bounded to [0.1, 10.0] range
- Optimization uses scipy's bounded scalar minimization

### Validation Space

Calibration performed in **normalized log-space** (same as training):
- Matches ANP's calibration assessment space
- Ensures fair comparison across models

### Persistence

Temperature is stored in model object and saved via pickle:
- Automatically loaded when model is loaded
- No need to recalibrate at inference time

## Comparison with ANP

| Aspect | ANP | RF/XGBoost (Before) | RF/XGBoost (After) |
|--------|-----|---------------------|-------------------|
| **Variance Type** | Learned | Data-derived | Data-derived |
| **Calibration** | Via NLL loss (training) | None | Via temperature (post-hoc) |
| **Learning Signal** | Integrated | N/A | Separate |
| **Fairness** | Full | Partial | Improved |

Now all models have calibrated uncertainties for fair comparison!

## Files Modified

1. `baselines/models.py`: Added calibration methods
2. `train_baselines.py`: Integrated calibration into training
3. `test_calibration.py`: Test suite (new)
4. `CALIBRATION_NOTES.md`: This documentation (new)

## Next Steps

After training with calibration:

1. **Run diagnostics**: Use `diagnostics.py` to generate calibration plots
2. **Compare models**: Check if RF/XGBoost calibration improves vs ANP
3. **Iterate**: Adjust bounds or try other calibration methods if needed

## References

- **Temperature Scaling**: Guo et al. "On Calibration of Modern Neural Networks" (ICML 2017)
- **Gaussian NLL**: Standard probabilistic regression objective
- **Post-hoc Calibration**: Kuleshov et al. "Accurate Uncertainties for Deep Learning Using Calibrated Regression" (ICML 2018)
