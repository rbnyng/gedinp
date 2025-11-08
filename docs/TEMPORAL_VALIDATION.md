# Temporal Validation Guide

This guide explains how to perform temporal validation for the GEDI Neural Process model, testing whether the model generalizes to future years not seen during training.

## Overview

**Temporal validation** tests whether a model trained on historical data can accurately predict on future data. This is critical for:

1. **Operational deployment**: Real-world applications require predicting future biomass from historical training data
2. **Distribution shift**: Tests robustness to temporal changes (forest growth, disturbance, seasonal effects)
3. **Publication rigor**: Reviewers often require both spatial and temporal validation

## Workflow

### Step 1: Train on Historical Years

Train your model using only specific years (e.g., 2019-2021):

```bash
python train.py \
  --region_bbox -60 -5 -55 0 \
  --start_time 2019-01-01 \
  --end_time 2023-12-31 \
  --train_years 2019 2020 2021 \
  --output_dir ./outputs/temporal_experiment \
  --epochs 100
```

**Key arguments:**
- `--train_years 2019 2020 2021`: Only use GEDI shots from these years for training
- `--start_time` / `--end_time`: Query range (should include both train and test years)
- Model performs spatial CV on these years only

**What happens:**
1. Queries GEDI data from 2019-2023
2. Filters to only 2019-2021 for training
3. Performs spatial train/val/test split on these years
4. Saves `train_years` to `config.json` for reference

### Step 2: Evaluate on Future Years

Test the trained model on held-out future years (e.g., 2022-2023):

```bash
python evaluate_temporal.py \
  --model_dir ./outputs/temporal_experiment \
  --test_years 2022 2023 \
  --checkpoint best_r2_model.pt
```

**Key arguments:**
- `--model_dir`: Directory containing trained model
- `--test_years 2022 2023`: Years to use for temporal evaluation
- `--checkpoint`: Which checkpoint to evaluate (default: `best_r2_model.pt`)

**What happens:**
1. Loads trained model and config
2. Queries GEDI data for 2022-2023
3. Extracts embeddings for these years
4. Evaluates model performance
5. Saves results:
   - `temporal_results_years_2022_2023.json`: Metrics
   - `temporal_predictions_years_2022_2023.csv`: Predictions
   - `temporal_eval_2022_2023.png`: Visualization
   - `temporal_data_years_2022_2023.pkl`: Processed data

### Step 3: Compare Spatial vs Temporal Performance

Generate comparison plots to assess generalization gap:

```bash
python compare_spatial_temporal.py \
  --model_dir ./outputs/temporal_experiment \
  --temporal_suffix years_2022_2023
```

**What happens:**
1. Loads spatial test results (from spatial CV)
2. Loads temporal test results (from Step 2)
3. Creates comprehensive comparison:
   - Metrics bar chart (RMSE, MAE, R²)
   - Side-by-side scatter plots
   - Residual distributions
   - Uncertainty comparisons
   - Performance summary table
4. Saves: `spatial_vs_temporal_years_2022_2023.png`

## Recommended Temporal Splits

### Multi-Year Training (Recommended)

**Best for most cases:**
```
Training:   2019 + 2020 + 2021  (spatial CV within these years)
Validation: 2022                (temporal validation)
Test:       2023                (temporal test)
```

Why this is good:
- ✅ More training data
- ✅ Model learns inter-annual variation
- ✅ Two temporal holdout years (2022, 2023)

### Year-on-Year Testing

**For analyzing temporal drift:**
```
Training:   2019 + 2020
Test:       2021, 2022, 2023 (separately)
```

Plot performance vs. temporal gap to see degradation over time.

### Region-Specific Considerations

**Tropical forests** (high inter-annual stability):
- 2-year training window may suffice
- Test on 1-2 years out

**Temperate/Boreal forests** (seasonal variation, disturbances):
- Use 3+ year training window
- Test across multiple seasons/years

## Interpreting Results

### Expected Performance Gaps

Typical temporal generalization gap:
- **Small gap (Δ R² < 0.05)**: Excellent temporal generalization
- **Moderate gap (Δ R² 0.05-0.15)**: Acceptable for most applications
- **Large gap (Δ R² > 0.15)**: Investigate causes (distribution shift, overfitting)

### What Affects Temporal Generalization?

1. **Forest dynamics**: Rapid growth/disturbance → larger gap
2. **Seasonal effects**: Leaf-on/off variations
3. **Sensor drift**: GEDI calibration changes over time
4. **Embedding shift**: Foundation model trained on different years
5. **Overfitting**: Model memorizes training years

### Diagnostic Questions

If temporal performance is worse:

1. **Is it consistent across years?**
   - Test 2022 and 2023 separately
   - If both worse → systematic issue
   - If one worse → check for data quality/coverage

2. **Does uncertainty increase?**
   - Higher uncertainty on temporal test → model knows it's uncertain (good!)
   - Low uncertainty but high error → calibration issue (bad)

3. **Are errors spatially clustered?**
   - Check if specific regions/ecosystems fail
   - May need stratified temporal validation

## Example: Full Workflow

```bash
# 1. Train on 2019-2021
python train.py \
  --region_bbox -75 -15 -50 5 \
  --train_years 2019 2020 2021 \
  --output_dir ./outputs/amazon_temporal \
  --epochs 100 \
  --architecture_mode anp

# 2. Evaluate on spatial test set (from training)
python evaluate.py \
  --model_dir ./outputs/amazon_temporal \
  --checkpoint best_r2_model.pt

# 3. Evaluate on 2022 (temporal validation)
python evaluate_temporal.py \
  --model_dir ./outputs/amazon_temporal \
  --test_years 2022 \
  --checkpoint best_r2_model.pt

# 4. Evaluate on 2023 (temporal test)
python evaluate_temporal.py \
  --model_dir ./outputs/amazon_temporal \
  --test_years 2023 \
  --checkpoint best_r2_model.pt

# 5. Compare spatial vs temporal (2022)
python compare_spatial_temporal.py \
  --model_dir ./outputs/amazon_temporal \
  --temporal_suffix years_2022

# 6. Compare spatial vs temporal (2023)
python compare_spatial_temporal.py \
  --model_dir ./outputs/amazon_temporal \
  --temporal_suffix years_2023
```

## For Publication

### Reporting Temporal Validation

In your paper, include:

1. **Methods section:**
   - Training years: "We trained on GEDI data from 2019-2021..."
   - Test years: "...and evaluated on held-out years 2022-2023"
   - Spatial CV: "Within training years, we used spatial cross-validation..."

2. **Results section:**
   - Table comparing spatial vs temporal metrics
   - Scatter plots for both holdout types
   - Discuss generalization gap and implications

3. **Figures:**
   - Use `compare_spatial_temporal.py` output
   - Show side-by-side spatial/temporal predictions
   - Include uncertainty calibration for both

### Example Results Table

| Metric | Spatial Test | Temporal 2022 | Temporal 2023 |
|--------|-------------|---------------|---------------|
| RMSE   | 0.145       | 0.162         | 0.158         |
| MAE    | 0.112       | 0.125         | 0.121         |
| R²     | 0.872       | 0.841         | 0.853         |

**Interpretation:** "The model shows robust temporal generalization, with only a 3.1% and 1.9% decrease in R² for 2022 and 2023 respectively..."

## Advanced Usage

### Testing Multiple Temporal Splits

```bash
# Create a script to test all years
for year in 2020 2021 2022 2023; do
  python evaluate_temporal.py \
    --model_dir ./outputs/temporal_experiment \
    --test_years $year \
    --output_suffix year_$year
done
```

### Custom Region for Temporal Test

```bash
# Test on different region than training
python evaluate_temporal.py \
  --model_dir ./outputs/amazon_trained \
  --test_years 2022 \
  --region_bbox -80 -20 -75 -15  # Different region
```

### Combining Multiple Test Years

```bash
# Test on combined 2022-2023
python evaluate_temporal.py \
  --model_dir ./outputs/temporal_experiment \
  --test_years 2022 2023 \
  --output_suffix years_2022_2023
```

## Troubleshooting

### "Could not find timestamp column"

The GEDI data needs a timestamp. Check available columns:
```python
import pandas as pd
df = pd.read_pickle('outputs/your_model/processed_data.pkl')
print(df.columns)
```

Common timestamp columns: `time`, `date_time`, `datetime`

### "No data found for year X"

GEDI coverage varies by region and year. Try:
- Expanding region bbox
- Checking GEDI mission timeline (launched 2019)
- Using `query_bbox` to verify data availability

### Temporal results much worse than spatial

This can be normal! Consider:
1. Check for distribution shift in input features
2. Analyze which tiles/regions fail
3. Try training on more years
4. Check if embeddings are from different year than GEDI

## References

**Temporal validation best practices:**
- Roberts et al. (2017): "Cross-validation strategies for data with temporal, spatial, hierarchical, or phylogenetic structure"
- Meyer & Pebesma (2021): "Predicting into unknown space? Estimating the area of applicability of spatial prediction models"

**For GEDI-specific considerations:**
- Check GEDI mission status: https://gedi.umd.edu/
- GEDI L4A product guide for temporal coverage
