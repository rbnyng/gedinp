# Pareto Frontier Visualization

This document explains how to visualize the Pareto frontier for baseline model sweeps.

## Overview

The Pareto frontier visualization helps answer the key question: **What is the tradeoff between accuracy and calibration quality?**

Instead of reading through Table 3 (a wall of numbers), the scatter plot makes it immediately clear:
- Which hyperparameter configurations achieve the best accuracy-calibration tradeoffs
- Whether Random Forest or XGBoost dominates in certain regions
- How much calibration quality we sacrifice for better accuracy (or vice versa)

## Quick Start

### 1. Run the baseline sweep (if not already done)

```bash
python sweep_baselines.py \
    --baseline_dir ./outputs_baselines \
    --output_dir ./outputs_pareto \
    --models rf xgb \
    --n_seeds 5
```

This generates `outputs_pareto/pareto_results.csv` with all hyperparameter configurations.

### 2. Generate the Pareto frontier plot

**Default (recommended):**
```bash
python plot_pareto_frontier.py \
    --input outputs_pareto/pareto_results.csv \
    --output figures/pareto_frontier.png \
    --show_pareto
```

This creates separate panels for RF and XGB, plotting:
- X-axis: Test Log RMSE (lower = better accuracy)
- Y-axis: Z-Score Std Dev (1.0 = perfect calibration)
- Reference line at z_std = 1.0
- Pareto frontier lines connecting optimal configurations

## Visualization Options

### Y-Axis Metrics

Choose what aspect of calibration to emphasize:

**1. Z-Score Standard Deviation (default, recommended)**
```bash
--y_metric z_std
```
- Ideal value: 1.0 (shown as reference line)
- Above 1.0: Overconfident (uncertainty too small)
- Below 1.0: Underconfident (uncertainty too large)
- Most interpretable for understanding calibration quality

**2. Calibration Error**
```bash
--y_metric calibration_error
```
- Measures |z_std - 1.0|
- Ideal value: 0.0
- Good for emphasizing deviation magnitude
- Can cluster if z_std varies widely

**3. Coverage at 1-Sigma**
```bash
--y_metric coverage_1sigma
```
- Percentage of predictions within ±1σ
- Ideal value: 68.3%
- Alternative view of calibration quality

### Layout Options

**Separate panels (default, recommended for paper):**
```bash
python plot_pareto_frontier.py --show_pareto
```
- Left panel: Random Forest
- Right panel: XGBoost
- Easier to see each model's frontier
- Better for comparing model families

**Combined panel:**
```bash
python plot_pareto_frontier.py --combined --show_pareto
```
- All models in one plot
- Good for dense comparison
- Can be cluttered if many points

### Scale Options

**Log scale for Y-axis:**
```bash
python plot_pareto_frontier.py --log_y
```
- Use if z_std ranges widely (e.g., 0.1 to 10)
- Spreads out clustered points
- Check the printed summary statistics to decide

**High-resolution output:**
```bash
python plot_pareto_frontier.py --dpi 600
```
- Default is 300 dpi
- Script automatically saves a 600 dpi version for papers

## Interpreting the Plot

### What to Look For

1. **Pareto Frontier** (if `--show_pareto` enabled):
   - The dashed line connects the "optimal" configurations
   - Any point not on the frontier is dominated by another configuration
   - The frontier shows the accuracy-calibration tradeoff curve

2. **Model Comparison**:
   - Does RF or XGB dominate across the frontier?
   - Are there regions where one model is clearly better?
   - Do they occupy different parts of the accuracy-calibration space?

3. **Calibration Quality**:
   - How close do points get to the reference line (z_std = 1.0)?
   - Is there a cluster of well-calibrated models?
   - Does better accuracy come at the cost of worse calibration?

4. **Hyperparameter Patterns**:
   - Check the summary statistics printed to console
   - See which configs achieve best calibration vs. best accuracy
   - Understand the tradeoff: deeper trees = better accuracy but worse calibration?

### Example Interpretations

**Scenario 1: "RF dominates"**
- RF points are consistently left and closer to z_std=1.0
- Conclusion: Use RF for this problem

**Scenario 2: "Accuracy-calibration tradeoff"**
- Points form a curve: low RMSE → high z_std deviation
- Conclusion: There's a fundamental tradeoff; choose based on application needs

**Scenario 3: "Separate niches"**
- RF good calibration but moderate accuracy
- XGB best accuracy but poor calibration
- Conclusion: Choose based on whether you prioritize uncertainty quality or predictions

## Advanced Usage

### Comparing with ANP

If you have ANP results with the same format, add them to the CSV and plot:

```bash
python plot_pareto_frontier.py \
    --input combined_results.csv \
    --output figures/anp_vs_baselines.png \
    --show_pareto
```

ANP points will appear in red.

### Multiple Visualizations for Paper

Generate a set of plots exploring different metrics:

```bash
# Main figure: z_std with separate panels
python plot_pareto_frontier.py \
    --show_pareto \
    --output figures/pareto_main.png

# Supplement: calibration error
python plot_pareto_frontier.py \
    --y_metric calibration_error \
    --show_pareto \
    --output figures/pareto_calib_error.png

# Supplement: coverage
python plot_pareto_frontier.py \
    --y_metric coverage_1sigma \
    --show_pareto \
    --output figures/pareto_coverage.png

# Combined view for talks
python plot_pareto_frontier.py \
    --combined \
    --show_pareto \
    --output figures/pareto_combined.png
```

## Output Files

The script generates:

1. **Main plot** (`pareto_frontier.png`):
   - 300 dpi by default
   - Ready for manuscript submission

2. **High-res version** (`pareto_frontier_highres.png`):
   - 600 dpi
   - Automatically generated if main plot is < 600 dpi
   - Use for final publication

3. **Console output**:
   - Summary statistics for each model
   - Range of metrics (RMSE, z_std, calibration error, etc.)
   - Best configurations for accuracy vs. calibration
   - Use to understand the data before interpreting the plot

## Integration with Paper

### Recommended Figure Caption

> **Figure X: Pareto Frontier for Baseline Model Hyperparameter Sweep.**
> Each point represents a hyperparameter configuration averaged across 5 seeds.
> X-axis shows test set log RMSE (lower is better). Y-axis shows z-score standard
> deviation, with the ideal value of 1.0 shown as a dashed horizontal line. Points
> above the line indicate overconfident models (uncertainty estimates too small),
> while points below indicate underconfident models (uncertainty estimates too large).
> The dashed frontier lines connect Pareto-optimal configurations, representing the
> accuracy-calibration tradeoff. (Left) Random Forest. (Right) XGBoost.

### Recommended Discussion Points

1. "Figure X shows the Pareto frontier for the baseline hyperparameter sweep. Random Forest
   achieves better calibration (z_std closer to 1.0) but XGBoost achieves lower test RMSE."

2. "The frontier reveals a fundamental tradeoff: configurations with the lowest RMSE tend
   to be overconfident (z_std > 1.0), while well-calibrated models (z_std ≈ 1.0) sacrifice
   some accuracy."

3. "We select the Pareto-optimal configuration that minimizes |z_std - 1.0| + 0.5 * RMSE
   for our final baseline comparison, balancing accuracy and calibration."

## Troubleshooting

**"Input file not found"**
- Run `sweep_baselines.py` first to generate the CSV

**"Plot is too cluttered"**
- Use `--combined` to reduce the number of panels
- Consider filtering the CSV to show only top-k configs per model

**"Points are clustered"**
- Try `--log_y` for log scale
- Check printed statistics to understand the range
- Consider different y_metric (calibration_error vs z_std)

**"I want to customize colors/styles"**
- Edit `plot_pareto_frontier.py` lines 72-82 for colors
- Edit lines 180-200 for plot styling
