# Training Harness for Multi-Seed Experiments

## Overview

The `run_training_harness.py` script automates running training experiments multiple times with different random seeds to produce statistically robust results with mean and standard deviation metrics.

## Why Use the Harness?

Single-seed results can be misleading due to:
- Random weight initialization variations
- Spatial data split randomness
- Training dynamics (batch sampling, early stopping, etc.)

The harness provides:
- **Statistical robustness**: Mean ± std dev over multiple runs
- **Result stability**: Coefficient of variation to assess reliability
- **Automatic aggregation**: No manual result collection
- **Comprehensive reporting**: CSV files, plots, and text summaries

## Installation

No additional dependencies beyond those required for training:
```bash
pip install torch numpy pandas matplotlib seaborn tqdm
```

## Usage

### Basic Usage

Run Neural Process training with 5 different seeds:
```bash
python run_training_harness.py \
    --script train.py \
    --n_seeds 5 \
    --output_dir ./results/np_5seeds \
    --region_bbox -122.5 37.0 -122.0 37.5
```

Run baseline models with specific seeds:
```bash
python run_training_harness.py \
    --script train_baselines.py \
    --seeds 42 43 44 45 46 \
    --output_dir ./results/baselines_5seeds \
    --region_bbox -122.5 37.0 -122.0 37.5
```

### Advanced Usage

**Custom Neural Process configuration:**
```bash
python run_training_harness.py \
    --script train.py \
    --n_seeds 10 \
    --base_seed 100 \
    --architecture_mode anp \
    --hidden_dim 512 \
    --latent_dim 256 \
    --epochs 100 \
    --batch_size 16 \
    --lr 5e-4 \
    --weight_decay 0.01 \
    --output_dir ./results/anp_10seeds \
    --region_bbox -122.5 37.0 -122.0 37.5 \
    --start_time 2022-01-01 \
    --end_time 2022-12-31
```

**Baseline models with specific models only:**
```bash
python run_training_harness.py \
    --script train_baselines.py \
    --n_seeds 5 \
    --models rf xgb \
    --output_dir ./results/baselines_rf_xgb \
    --region_bbox -122.5 37.0 -122.0 37.5
```

## Command-Line Arguments

### Harness-Specific Arguments

| Argument | Type | Required | Description |
|----------|------|----------|-------------|
| `--script` | str | Yes | Training script to run: `train.py` or `train_baselines.py` |
| `--n_seeds` | int | * | Number of seeds to run (generates seeds starting from base_seed) |
| `--seeds` | int+ | * | Specific seeds to use (e.g., `42 43 44`) |
| `--base_seed` | int | No | Base seed for generating seed list (default: 42) |
| `--output_dir` | str | Yes | Base output directory for all runs |
| `--parallel` | flag | No | Run seeds in parallel (not yet implemented) |

\* Must specify either `--n_seeds` or `--seeds`, but not both.

### Common Training Arguments

These are passed through to the training scripts:

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--region_bbox` | float×4 | Required | Region bounding box: min_lon min_lat max_lon max_lat |
| `--start_time` | str | 2022-01-01 | Start date for GEDI data (YYYY-MM-DD) |
| `--end_time` | str | 2022-12-31 | End date for GEDI data (YYYY-MM-DD) |
| `--embedding_year` | int | 2022 | Year of GeoTessera embeddings |
| `--cache_dir` | str | ./cache | Directory for caching tiles and embeddings |

### Neural Process Specific Arguments

Used only when `--script train.py`:

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--architecture_mode` | str | anp | Architecture: `deterministic`, `latent`, `anp`, `cnp` |
| `--hidden_dim` | int | 512 | Hidden layer dimension |
| `--latent_dim` | int | 256 | Latent variable dimension |
| `--epochs` | int | 100 | Number of training epochs |
| `--batch_size` | int | 16 | Batch size |
| `--lr` | float | 5e-4 | Learning rate |
| `--weight_decay` | float | 0.01 | Weight decay for AdamW optimizer |

### Baseline Specific Arguments

Used only when `--script train_baselines.py`:

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--models` | str+ | rf xgb idw | Which baseline models to train |

## Output Files

The harness creates a comprehensive output directory:

```
output_dir/
├── harness_config.json          # Configuration used for the harness run
├── all_runs.csv                 # All individual run results
├── statistics.csv               # Aggregated statistics (mean, std, min, max, median)
├── multi_seed_results.png       # Box plots and scatter plots of all runs
├── model_comparison.png         # Comparison plot with error bars (mean ± std)
├── harness_summary.txt          # Comprehensive text report
└── seed_*/                      # Individual run directories
    ├── config.json
    ├── best_r2_model.pt
    ├── history.json
    └── ...
```

### Key Output Files

1. **harness_config.json**: Records all arguments and seeds used
2. **all_runs.csv**: Complete results table with one row per run (or per model per run for baselines)
3. **statistics.csv**: Aggregated statistics across all seeds
4. **harness_summary.txt**: Human-readable summary with:
   - Configuration details
   - Individual run results table
   - Aggregated statistics table
   - Key findings and interpretation

### Visualizations

1. **multi_seed_results.png**: Box plots showing distribution of metrics across seeds
2. **model_comparison.png**: Bar charts with error bars showing mean ± std dev
3. **performance_summary.png**: Summary visualization (Neural Process only)

## Interpreting Results

### Statistical Robustness

The harness computes key statistics:

- **Mean (μ)**: Average performance across seeds
- **Std Dev (σ)**: Variation in performance
- **Coefficient of Variation (CV)**: `(σ/μ) × 100%`
  - CV < 5%: Very stable results
  - 5% ≤ CV < 10%: Moderate variability
  - CV ≥ 10%: High variability (consider more seeds)

### Reporting Results

When reporting results in papers or documentation, use:

> Test R² Score: **0.8234 ± 0.0156** (mean ± std over 5 seeds)

This format shows both the central tendency and the uncertainty in your results.

### Comparing Models

When comparing models, check if confidence intervals overlap:
- **Non-overlapping intervals**: Likely significant difference
- **Overlapping intervals**: May not be significantly different

Example:
```
Model A: 0.82 ± 0.02  (95% CI: 0.78-0.86)
Model B: 0.79 ± 0.03  (95% CI: 0.73-0.85)
```
The intervals overlap, suggesting the difference may not be statistically significant.

## Recommended Number of Seeds

- **Quick experiments**: 3 seeds (minimum for std dev)
- **Standard experiments**: 5 seeds (good balance)
- **Publication-quality**: 10 seeds (high confidence)
- **Benchmark comparisons**: 10+ seeds (establish baselines)

## Examples

### Example 1: Quick Neural Process Evaluation

Test if your model changes improved performance:
```bash
python run_training_harness.py \
    --script train.py \
    --n_seeds 3 \
    --output_dir ./results/quick_test \
    --region_bbox -122.5 37.0 -122.0 37.5 \
    --epochs 50
```

### Example 2: Publication-Quality Baseline Comparison

Generate robust baseline results:
```bash
python run_training_harness.py \
    --script train_baselines.py \
    --n_seeds 10 \
    --models rf xgb idw \
    --output_dir ./results/baselines_final \
    --region_bbox -122.5 37.0 -122.0 37.5
```

### Example 3: Architecture Comparison with Error Bars

Compare different architectures:
```bash
for arch in cnp deterministic latent anp; do
    python run_training_harness.py \
        --script train.py \
        --n_seeds 5 \
        --architecture_mode $arch \
        --output_dir ./results/arch_comparison_${arch} \
        --region_bbox -122.5 37.0 -122.0 37.5
done

# Then aggregate results across architectures
```

## Tips and Best Practices

1. **Use consistent seeds**: For fair comparison across experiments, use the same seed set
2. **Check individual runs**: If std dev is high, inspect individual runs for outliers
3. **Save everything**: The harness preserves all individual run outputs
4. **Monitor resources**: Each seed runs sequentially; plan accordingly
5. **Document parameters**: The harness saves all config automatically

## Troubleshooting

**Q: One seed failed but others succeeded. What should I do?**

A: The harness continues with remaining seeds and reports on successful runs. Check the failed seed's output directory for error logs. This is usually acceptable as long as you have ≥3 successful runs.

**Q: Results show high variability (CV > 10%). What does this mean?**

A: High variability suggests:
- The model is sensitive to initialization/data splits
- More seeds are needed for reliable estimates
- There may be issues with training stability

**Q: How do I compare two harness runs?**

A: Load the `statistics.csv` files and compare mean values. Check if confidence intervals overlap to assess significance.

**Q: Can I resume a harness run if it crashes?**

A: Not directly. However, you can manually specify `--seeds` to run only the missing seeds, then manually aggregate results.

## Future Enhancements

Planned features:
- `--parallel` flag implementation for concurrent seed execution
- Automatic statistical significance testing (t-tests, bootstrapping)
- Resume capability for interrupted runs
- Integration with experiment tracking tools (wandb, mlflow)

## See Also

- `train.py`: Neural Process training script
- `train_baselines.py`: Baseline model training script
- `run_ablation_study.py`: Architecture ablation study
