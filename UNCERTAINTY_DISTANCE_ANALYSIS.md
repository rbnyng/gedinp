# Uncertainty vs Distance to Training Data Analysis

This analysis compares how ANP (Attentive Neural Process) and XGBoost models' predicted uncertainties correlate with distance to the nearest training point.

## Research Question

**Does predicted uncertainty increase with distance from training data, and how do ANP vs XGBoost differ in this behavior?**

## Why This Matters

- **Epistemic uncertainty**: Models should be more uncertain when predicting far from training data (epistemic uncertainty = "model doesn't know")
- **Model comparison**: Different ML paradigms may handle extrapolation differently
  - **ANP**: Designed to output uncertainty directly, should naturally increase uncertainty in data-sparse regions
  - **XGBoost**: Uses quantile regression for uncertainty, may show different spatial patterns

## What the Analysis Does

1. **Loads models**: Trained ANP and XGBoost models from regional results
2. **Computes distances**: For each test point, calculates distance to nearest training point using KD-tree
3. **Extracts uncertainties**: Gets predicted uncertainties from both models
4. **Computes correlations**:
   - Pearson correlation (linear relationship)
   - Spearman correlation (monotonic relationship)
5. **Creates visualizations**:
   - Scatter plots of uncertainty vs distance
   - Binned averages with error bars
   - Density hexbin plots
   - Distribution comparisons

## Usage

### Single Region

```bash
# Analyze Maine region
python analyze_uncertainty_distance.py \
    --results_dir ./regional_results \
    --region maine \
    --output_dir ./uncertainty_distance_analysis
```

Or use the helper script:

```bash
./run_uncertainty_distance_analysis.sh maine
```

### All Regions

```bash
# Analyze all available regions
python analyze_uncertainty_distance.py \
    --results_dir ./regional_results \
    --all_regions \
    --output_dir ./uncertainty_distance_analysis
```

Or:

```bash
./run_uncertainty_distance_analysis.sh all
```

### Direct Region Path

```bash
# If results_dir points directly to a region
python analyze_uncertainty_distance.py \
    --results_dir ./regional_results/maine \
    --output_dir ./uncertainty_distance_analysis
```

## Outputs

For each region, the analysis generates:

1. **`{region}_uncertainty_distance.png`**: 6-panel comparison plot
   - Top row: ANP scatter, XGBoost scatter, binned comparison
   - Bottom row: ANP density, XGBoost density, uncertainty distributions

2. **`{region}_detailed_results.csv`**: Point-by-point data including:
   - Coordinates
   - Distance to nearest training point
   - ANP predictions and uncertainties
   - XGBoost predictions and uncertainties
   - True AGBD values

For multi-region analysis:

3. **`multi_region_summary.png`**: Cross-region comparison
   - Pearson correlations by region
   - Spearman correlations by region
   - Mean uncertainties
   - ANP vs XGBoost correlation scatter

4. **`correlation_summary.csv`**: Summary statistics table

## Expected Results

### Strong Positive Correlation (r > 0.5)
- Model properly captures epistemic uncertainty
- Uncertainty increases smoothly with distance from training data
- Good for spatial extrapolation confidence

### Weak/No Correlation (r ≈ 0)
- Uncertainty driven primarily by aleatoric factors (noise)
- Model may not distinguish between interpolation and extrapolation
- Less reliable for out-of-distribution predictions

### Negative Correlation (r < 0)
- Problematic: uncertainty decreases with distance
- Model is overconfident in extrapolation regions
- Calibration issues

## Interpretation Guide

### ANP Expected Behavior
- **Should show**: Moderate to strong positive correlation
- **Why**: Neural processes are designed to increase uncertainty in low-data regions
- **Latent path**: Provides global uncertainty that should increase with distance

### XGBoost Expected Behavior
- **May show**: Weaker or more irregular correlation
- **Why**: Uncertainty from quantile regression reflects prediction interval width, not necessarily distance-based epistemic uncertainty
- **Tree boundaries**: Uncertainty may change abruptly at decision boundaries

### Interesting Findings
1. **If ANP >> XGBoost correlation**: ANP better captures epistemic uncertainty
2. **If similar correlations**: Both models respond to data density
3. **If XGBoost >> ANP**: ANP may be overconfident or not properly calibrated

## Distance Metrics

- **Units**: Degrees (lat/lon) and kilometers (approximate conversion: 1° ≈ 111 km)
- **Calculation**: Euclidean distance in geographic coordinates
- **Method**: KD-tree for efficient nearest neighbor search

## Correlation Metrics

- **Pearson r**: Measures linear correlation (-1 to 1)
  - r = 1: Perfect positive linear relationship
  - r = 0: No linear relationship
  - r = -1: Perfect negative linear relationship

- **Spearman ρ**: Measures monotonic correlation (-1 to 1)
  - More robust to outliers
  - Detects non-linear monotonic relationships

- **P-values**: Statistical significance (typically p < 0.05 is significant)

## Requirements

The script requires:
- Trained ANP and XGBoost models in `regional_results/{region}/`
- Model directory structure:
  ```
  regional_results/
    maine/
      anp/
        seed_42/
          best_r2_model.pt
          config.json
          train_split.parquet
          test_split.parquet
      baselines/
        seed_42/
          xgboost.pkl
          config.json
  ```

## Performance Notes

- **Memory**: Loads full training set into memory for distance computation
- **GPU**: ANP inference runs on GPU if available (much faster)
- **Speed**: ~1-5 minutes per region depending on dataset size

## Example Output Interpretation

```
ANP Correlations:
  Pearson r: 0.623 (p=1.2e-145)
  Spearman ρ: 0.589 (p=3.4e-128)

XGBoost Correlations:
  Pearson r: 0.412 (p=2.1e-64)
  Spearman ρ: 0.398 (p=1.5e-59)
```

**Interpretation**:
- ANP shows stronger correlation (r=0.62) than XGBoost (r=0.41)
- Both are statistically significant (p << 0.05)
- ANP uncertainty increases more consistently with distance
- Both show moderate positive correlation, indicating epistemic uncertainty capture

## Related Scripts

- `evaluate_spatial_extrapolation.py`: Cross-region model evaluation
- `diagnostics.py`: Uncertainty calibration analysis
- `train.py`: ANP training
- `train_baselines.py`: Baseline model training

## Citation

If you use this analysis in research, please cite:
- The GEDI AGB Neural Process paper (when published)
- Neural Processes literature for ANP methodology

## Questions?

This analysis was designed to provide insights into:
1. How well models quantify epistemic uncertainty
2. Whether uncertainty estimates are trustworthy for spatial extrapolation
3. Comparative advantages of ANP vs traditional ML for uncertainty quantification

For more details on the models, see the main project README.
