# Pareto Frontier Analysis for GEDI Baseline Models

This directory contains tools for comprehensive hyperparameter sweep and Pareto frontier analysis of baseline models (Random Forest and XGBoost). The analysis explores the fundamental trade-offs between model accuracy, uncertainty calibration, and computational cost.

## Overview

The Pareto frontier analysis demonstrates that:

1. **Baseline models face inherent trade-offs**: For tree-based models (RF/XGBoost), improving point-estimate accuracy (by increasing tree depth/complexity) often worsens uncertainty calibration
2. **ANP occupies a superior region**: The proposed ANP model achieves high accuracy AND well-calibrated uncertainty simultaneously, without the trade-offs faced by traditional baselines
3. **The comparison is fair**: By evaluating baselines across their entire hyperparameter space, we preemptively counter the "under-tuned baseline" argument

## Key Features

- **Comprehensive hyperparameter sweeps** for Random Forest and XGBoost
- **Pareto frontier computation** to identify optimal trade-off points
- **Publication-quality visualizations** showing accuracy vs calibration trade-offs
- **Detailed summary tables** for reporting in papers
- **Fair comparison** with ANP using identical data splits

## Quick Start

### 1. Run Baseline Training (if not already done)

First, ensure you have baseline model outputs with train/val/test splits:

```bash
python train_baselines.py \
    --region_bbox -75.0 -10.0 -70.0 -5.0 \
    --output_dir ./outputs_baselines \
    --models rf xgb
```

### 2. Run Hyperparameter Sweep

Run the Pareto frontier analysis with a comprehensive hyperparameter sweep:

```bash
# Quick sweep (12 configs per model, ~30 minutes)
python analyze_pareto_frontier.py \
    --baseline_dir ./outputs_baselines \
    --output_dir ./outputs_pareto \
    --models rf xgb \
    --quick

# Full sweep (30 configs per model, ~2-3 hours)
python analyze_pareto_frontier.py \
    --baseline_dir ./outputs_baselines \
    --output_dir ./outputs_pareto \
    --models rf xgb
```

The script will:
- Load your existing train/test splits for fair comparison
- Train each hyperparameter configuration
- Evaluate on the test set
- Save intermediate results (safe to interrupt and resume)

### 3. Generate Visualizations

Create publication-quality plots from the sweep results:

```bash
python plot_pareto.py \
    --results_dir ./outputs_pareto \
    --anp_results ./outputs/results.json \
    --output_dir ./outputs_pareto/plots
```

This generates:
- `pareto_frontier_accuracy_calibration.png` - Main Pareto frontier plot
- `pareto_time_tradeoffs.png` - Training time vs performance metrics
- `pareto_summary_table.csv` - Detailed results table
- `pareto_summary_table.md` - Markdown-formatted summary

## Hyperparameter Grids

### Random Forest
- `max_depth`: [2, 3, 4, 6, 8, 10]
- `n_estimators`: [50, 100, 200, 500, 1000]
- Total: 30 configurations (quick mode: 12)

### XGBoost
- `max_depth`: [2, 3, 4, 6, 8, 10]
- `n_estimators`: [50, 100, 200, 500, 1000]
- `learning_rate`: 0.1 (fixed)
- Total: 30 configurations (quick mode: 12)

## Understanding the Plots

### Main Pareto Frontier Plot

**X-axis: Calibration Error (|1 - Z-score Std|)**
- Lower is better
- Measures how well predicted uncertainties match actual errors
- Ideal value: 0 (Z-score std = 1.0)

**Y-axis: Test Log R²**
- Higher is better
- Measures prediction accuracy in log-space (aligned with training objective)

**Point Size: Training Time**
- Larger points = longer training time

**Interpretation:**
- Points in the **lower-right corner** are ideal (high accuracy, good calibration)
- The **Pareto frontier** (starred points with dashed line) shows the performance envelope
- Points below the frontier are strictly dominated (worse on both axes)
- The **ANP model** (blue diamond) should occupy a superior position compared to baseline frontiers

### Expected Findings

Based on the hypothesis, you should observe:

1. **XGBoost/RF form a trade-off frontier**:
   - Shallow trees (depth=2-3): Better calibration, lower accuracy
   - Deep trees (depth=8-10): Better accuracy, worse calibration
   - More trees: Generally hurts calibration (especially XGBoost)

2. **ANP occupies a "magic corner"**:
   - Comparable or better accuracy than best baseline configuration
   - Significantly better calibration than baseline configurations with similar accuracy
   - Single model achieves both objectives simultaneously

3. **Training time is reasonable**:
   - ANP takes longer than simple baselines but is competitive with large ensembles
   - Performance gain justifies computational cost

## Advanced Usage

### Resume Interrupted Sweep

If the hyperparameter sweep is interrupted, resume from where it left off:

```bash
python analyze_pareto_frontier.py \
    --baseline_dir ./outputs_baselines \
    --output_dir ./outputs_pareto \
    --models rf xgb \
    --resume
```

### Analyze Specific Models Only

```bash
# Only Random Forest
python analyze_pareto_frontier.py \
    --baseline_dir ./outputs_baselines \
    --output_dir ./outputs_pareto \
    --models rf

# Only XGBoost
python analyze_pareto_frontier.py \
    --baseline_dir ./outputs_baselines \
    --output_dir ./outputs_pareto \
    --models xgb
```

### Custom ANP Results Path

If your ANP results are in a different location:

```bash
python plot_pareto.py \
    --results_dir ./outputs_pareto \
    --anp_results ./path/to/your/anp_outputs/results.json \
    --output_dir ./custom_plots_dir
```

## Output Files

### From `analyze_pareto_frontier.py`

```
outputs_pareto/
├── config.json                  # Analysis configuration
├── pareto_results.json          # Full results (all configs)
└── pareto_results.csv           # Tabular format for easy analysis
```

### From `plot_pareto.py`

```
outputs_pareto/plots/
├── pareto_frontier_accuracy_calibration.png  # Main plot
├── pareto_time_tradeoffs.png                 # Time vs metrics
├── pareto_summary_table.csv                  # Detailed table
└── pareto_summary_table.md                   # Formatted summary
```

## Interpreting Results for Papers

### Key Metrics to Report

1. **Accuracy Range**: Min/max Log R² achieved by each baseline across all configs
2. **Calibration Range**: Min/max calibration error across all configs
3. **ANP Position**: Where does ANP fall relative to baseline Pareto frontiers?
4. **Trade-off Evidence**: Do baselines show negative correlation between accuracy and calibration?

### Sample Findings Statement

> "We performed a comprehensive hyperparameter sweep for Random Forest (30 configurations) and XGBoost (30 configurations), varying tree depth and ensemble size. Figure X shows the resulting Pareto frontier for accuracy vs. uncertainty calibration. While both baselines exhibit a clear trade-off between point-estimate accuracy and calibration quality, our proposed ANP model simultaneously achieves state-of-the-art accuracy (Log R² = 0.78) and near-perfect calibration (Z-score std = 1.02, error = 0.02). This demonstrates that ANP's superiority is not due to under-tuned baselines, but rather reflects a fundamental advantage of the neural process framework for joint prediction and uncertainty quantification."

## Technical Details

### Calibration Metrics

- **Z-score**: Standardized residual = (target - prediction) / predicted_std
- **Z-score Std**: Standard deviation of z-scores (ideal: 1.0)
- **Calibration Error**: |Z-score Std - 1.0| (ideal: 0.0)
- **Coverage**: Percentage of predictions within k-sigma intervals

### Pareto Frontier Computation

The Pareto frontier is computed by finding all non-dominated points:
- A point is dominated if another point is better on both axes
- We minimize calibration error and maximize accuracy
- Points on the frontier represent optimal trade-offs

## Troubleshooting

### Out of Memory

If you encounter OOM errors with large datasets:

```bash
# Use quick mode with fewer configs
python analyze_pareto_frontier.py --quick ...
```

### Missing Dependencies

```bash
pip install matplotlib pandas numpy scipy tqdm
```

### ANP Results Not Found

Ensure your ANP results JSON contains either:
- `test_metrics` key with calibration metrics, OR
- `metrics` key with the same

The script looks for: `log_r2`, `z_std`, `calibration_error`, `coverage_1sigma`

## Citation

If you use this Pareto frontier analysis in your research, please cite:

```bibtex
@article{your_paper,
  title={Your Paper Title},
  author={Your Name},
  journal={Your Journal},
  year={2024}
}
```

## Questions?

For issues or questions about the Pareto frontier analysis, please open an issue on GitHub or contact the authors.
