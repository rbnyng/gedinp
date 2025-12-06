# Few-Shot Sensitivity Analysis

This directory contains scripts for analyzing the sensitivity of model performance to the number of few-shot training examples.

## Overview

The few-shot sensitivity experiment tests how model performance varies with different numbers of training tiles from the target region. This helps answer:

- **Sample Efficiency**: How quickly does the model adapt with more examples?
- **Optimal Shot Count**: What's the minimum number of shots needed for good performance?
- **Diminishing Returns**: At what point do additional shots provide minimal benefit?
- **Regional Variation**: Does the optimal shot count vary by region?

## Scripts

### 1. `run_fewshot_sensitivity.py`

Runs a sweep of spatial extrapolation experiments with different numbers of few-shot tiles.

**Usage:**
```bash
# Run with default shot counts (1, 3, 5, 7, 10, 12, 15)
python run_fewshot_sensitivity.py --results_dir ./regional_results

# Custom shot counts
python run_fewshot_sensitivity.py \
    --results_dir ./regional_results \
    --shot_counts 1 5 10 20 50

# More epochs for better fine-tuning
python run_fewshot_sensitivity.py \
    --results_dir ./regional_results \
    --few_shot_epochs 10 \
    --few_shot_lr 1e-4
```

**Key Arguments:**
- `--results_dir`: Directory containing regional training results (required)
- `--output_dir`: Output directory for sweep results (default: `./fewshot_sensitivity_results`)
- `--shot_counts`: List of shot counts to test (default: `1 3 5 7 10 12 15`)
- `--few_shot_epochs`: Number of fine-tuning epochs (default: 5)
- `--few_shot_lr`: Learning rate for fine-tuning (default: 1e-4)
- `--device`: Device to use (default: cuda)

**Output Structure:**
```
fewshot_sensitivity_results/
├── shots_1/
│   ├── spatial_extrapolation_results.csv
│   ├── extrapolation_*.png
│   └── ...
├── shots_3/
│   └── ...
├── shots_5/
│   └── ...
├── ...
├── fewshot_sensitivity_all_results.csv  # Aggregated results
└── sweep_metadata.json                   # Experiment configuration
```

### 2. `analyze_fewshot_sensitivity.py`

Analyzes and visualizes the results from the sensitivity sweep.

**Usage:**
```bash
python analyze_fewshot_sensitivity.py \
    --results_dir ./fewshot_sensitivity_results
```

**Key Arguments:**
- `--results_dir`: Directory containing aggregated sensitivity results (required)
- `--output_dir`: Output directory for analysis (default: `{results_dir}/analysis`)

**Generated Visualizations:**

1. **Learning Curves** (`learning_curves_*.png`)
   - Performance vs. number of shots
   - Separate plots for in-distribution and out-of-distribution
   - Grouped by train region

2. **Improvement Analysis** (`improvement_analysis.png`)
   - R² and RMSE improvement from zero-shot baseline
   - Heatmap of improvement by region and shot count
   - Distribution of optimal shot counts

3. **Region-Specific Analysis** (`region_specific_analysis.png`)
   - Detailed transfer curves for each train region
   - Shows performance to all test regions

4. **Summary Report** (`summary_report.txt`)
   - Overall performance statistics
   - Best shot count identification
   - Improvement percentages

**Generated Data:**

- `improvement_statistics.csv`: Detailed improvement metrics by shot count, split, and region
- `summary_report.txt`: Human-readable summary of findings

## Example Workflow

```bash
# Step 1: Run the sensitivity sweep (this will take a while!)
python run_fewshot_sensitivity.py \
    --results_dir ./regional_results \
    --output_dir ./fewshot_sensitivity_results \
    --shot_counts 1 3 5 7 10 12 15 \
    --few_shot_epochs 5 \
    --device cuda

# Step 2: Analyze the results
python analyze_fewshot_sensitivity.py \
    --results_dir ./fewshot_sensitivity_results

# Step 3: View the outputs
ls fewshot_sensitivity_results/analysis/
# learning_curves_log_r2.png
# learning_curves_log_rmse.png
# improvement_analysis.png
# region_specific_analysis.png
# improvement_statistics.csv
# summary_report.txt
```

## Understanding the Results

### Learning Curves

Learning curves show how performance (R², RMSE, etc.) changes with increasing shot counts. Look for:
- **Steep initial improvement**: Model learns quickly with first few examples
- **Plateau**: Point where additional shots provide minimal benefit
- **Regional differences**: Some regions may require more shots than others

### Improvement Analysis

The improvement analysis compares few-shot performance to the zero-shot baseline:
- **Positive values**: Few-shot improves over zero-shot
- **Best shot count**: Shot count with maximum average improvement
- **Regional variation**: Some region pairs benefit more from few-shot than others

### Optimal Shot Count

The "sweet spot" balances:
- **Performance gain**: Substantial improvement over zero-shot
- **Efficiency**: Minimal data collection and training time
- **Diminishing returns**: Additional shots provide little extra benefit

Typically, you'll see the biggest gains in the first 5-10 shots, with diminishing returns beyond that.

## Notes

- **Computational Cost**: Each shot count requires full evaluation across all region pairs and seeds, so the sweep can be time-consuming
- **Statistical Robustness**: Results are aggregated across multiple random seeds for reliability
- **ANP Only**: Few-shot fine-tuning only applies to ANP models; baselines (XGBoost, Regression Kriging) are not included
- **OOD Focus**: Few-shot is most relevant for out-of-distribution transfer, where zero-shot performance is typically lower

## Troubleshooting

**Issue**: "Results file not found"
- **Solution**: Make sure you've run `run_fewshot_sensitivity.py` first and it completed successfully

**Issue**: Sweep takes too long
- **Solution**: Reduce the number of shot counts or use `--shot_counts` with fewer values

**Issue**: Out of memory errors
- **Solution**: Reduce `--batch_size` or `--few_shot_batch_size` in the sweep script

**Issue**: Want to re-analyze without re-running sweep
- **Solution**: Use `--skip_evaluation` flag in `run_fewshot_sensitivity.py` to only aggregate existing results
