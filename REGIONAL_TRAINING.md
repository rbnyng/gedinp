# Regional Training Guide

This guide explains how to run multi-seed training experiments across multiple geographic regions using the `run_regional_training.py` script.

## Overview

The script trains both Neural Process (ANP) and baseline models (XGBoost, Random Forest, IDW, MLP) across 5 diverse forest regions:

1. **Maine, USA** - Temperate mixed forest, northeastern USA
2. **South Tyrol, Italy** - Alpine coniferous forest, European Alps
3. **Ili, Xinjiang, China** - Arid continental forest, Central Asia
4. **Hokkaido, Japan** - Temperate deciduous/coniferous forest, northern Japan
5. **Tolima, Colombia** - Tropical montane forest, Andean region

For each region, the script:
- Runs the training harness with multiple seeds for statistical robustness
- Trains both ANP and baseline models (or just one if specified)
- Generates comprehensive comparisons across regions

## Quick Start

### Run all regions with 5 seeds (ANP + baselines)

```bash
python run_regional_training.py \
  --n_seeds 5 \
  --cache_dir ./cache \
  --output_dir ./regional_results_2022
```

### Run specific regions only

```bash
python run_regional_training.py \
  --n_seeds 5 \
  --regions maine hokkaido tolima \
  --cache_dir ./cache \
  --output_dir ./regional_results_subset
```

### Run only ANP (skip baselines)

```bash
python run_regional_training.py \
  --n_seeds 5 \
  --skip_baselines \
  --cache_dir ./cache \
  --output_dir ./regional_results_anp_only
```

### Run only baselines (skip ANP)

```bash
python run_regional_training.py \
  --n_seeds 5 \
  --skip_anp \
  --baseline_models rf xgb idw mlp-dropout \
  --cache_dir ./cache \
  --output_dir ./regional_results_baselines_only
```

### Use specific seeds

```bash
python run_regional_training.py \
  --seeds 42 43 44 45 46 \
  --cache_dir ./cache \
  --output_dir ./regional_results_specific_seeds
```

## Command Reference

The original commands for each region are:

### Maine
```bash
python train.py \
  --region_bbox -70 44 -69 45 \
  --embedding_year 2022 \
  --start_time 2022-01-01 \
  --end_time 2022-12-31 \
  --buffer_size 0.1 \
  --cache_dir ./cache \
  --output_dir ./outputs \
  --epochs 100
```

### South Tyrol, Italy
```bash
python train.py \
  --region_bbox 10.5 45.6 11.5 46.4 \
  --embedding_year 2022 \
  --start_time 2022-01-01 \
  --end_time 2022-12-31 \
  --buffer_size 0.1 \
  --batch_size 4 \
  --cache_dir ./cache \
  --output_dir ./outputs \
  --epochs 100
```

### Ili, Xinjiang
```bash
python train.py \
  --region_bbox 86 42.8 87 43.5 \
  --embedding_year 2022 \
  --start_time 2022-01-01 \
  --end_time 2022-12-31 \
  --buffer_size 0.1 \
  --cache_dir ./cache \
  --output_dir ./outputs \
  --epochs 100
```

### Hokkaido, Japan
```bash
python train.py \
  --region_bbox 143.8 43.2 144.8 43.9 \
  --embedding_year 2022 \
  --start_time 2022-01-01 \
  --end_time 2022-12-31 \
  --buffer_size 0.1 \
  --cache_dir ./cache \
  --output_dir ./outputs \
  --epochs 100
```

### Tolima, Colombia
```bash
python train.py \
  --region_bbox -75 3 -74 4 \
  --embedding_year 2022 \
  --start_time 2022-01-01 \
  --end_time 2022-12-31 \
  --buffer_size 0.1 \
  --cache_dir ./cache \
  --output_dir ./outputs \
  --epochs 100
```

## Output Structure

After running, you'll get:

```
regional_results/
├── regional_config.json                 # Experiment configuration
├── anp_regional_results.csv             # ANP results across all regions
├── baseline_regional_results.csv        # Baseline results across all regions
├── anp_regional_comparison.png          # ANP performance visualization
├── baselines_regional_comparison.png    # Baseline performance visualization
├── anp_vs_baselines.png                 # Direct comparison plot
├── regional_summary.txt                 # Comprehensive text report
│
├── maine/
│   ├── anp/
│   │   ├── harness_config.json
│   │   ├── all_runs.csv                 # All seed runs
│   │   ├── statistics.csv               # Mean ± std across seeds
│   │   ├── multi_seed_results.png
│   │   └── seed_42/, seed_43/, ...      # Individual runs
│   └── baselines/
│       ├── harness_config.json
│       ├── all_runs.csv
│       ├── statistics.csv
│       ├── multi_seed_results.png
│       └── seed_42/, seed_43/, ...
│
├── sudtirol/
│   └── ... (same structure)
│
├── ili/
│   └── ... (same structure)
│
├── hokkaido/
│   └── ... (same structure)
│
└── tolima/
    └── ... (same structure)
```

## Key Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--regions` | Which regions to train on (maine, sudtirol, ili, hokkaido, tolima, all) | all |
| `--n_seeds` | Number of seeds to run per region | 5 |
| `--seeds` | Specific seeds to use (e.g., 42 43 44) | None |
| `--skip_anp` | Skip Neural Process training | False |
| `--skip_baselines` | Skip baseline model training | False |
| `--baseline_models` | Which baselines to train | rf xgb idw mlp-dropout |
| `--cache_dir` | Cache directory | Required |
| `--output_dir` | Output directory | ./regional_results |
| `--epochs` | Training epochs for ANP | 100 |
| `--buffer_size` | Spatial CV buffer size (degrees) | 0.1 |

## Region-Specific Settings

The script automatically handles region-specific configurations:

- **South Tyrol**: Uses `batch_size=4` (smaller batches due to region size)
- **Other regions**: Use `batch_size=16` (default)

All regions use:
- Embedding year: 2022
- Time range: 2022-01-01 to 2022-12-31
- Buffer size: 0.1 degrees (~11km)

## Expected Runtime

Approximate times (will vary based on hardware and GEDI data density):

- **Per region (ANP, 5 seeds)**: 2-6 hours
- **Per region (baselines, 5 seeds)**: 1-3 hours
- **All 5 regions (ANP + baselines, 5 seeds)**: 15-45 hours

Consider running on a cluster or overnight for full experiments.

## Tips

1. **Start small**: Test with 1-2 regions and 2-3 seeds before running the full experiment
2. **Use specific seeds**: For reproducibility, specify exact seeds with `--seeds 42 43 44`
3. **Monitor cache**: The cache directory will grow significantly. Ensure sufficient disk space.
4. **Review logs**: Each harness run logs to stdout - redirect to files if needed
5. **Incremental runs**: You can run ANP first (`--skip_baselines`), review results, then run baselines

## Examples

### Test run (1 region, 2 seeds)
```bash
python run_regional_training.py \
  --regions maine \
  --n_seeds 2 \
  --cache_dir ./cache \
  --output_dir ./test_run
```

### Production run (all regions, 5 seeds)
```bash
python run_regional_training.py \
  --n_seeds 5 \
  --cache_dir ./cache \
  --output_dir ./regional_results_2022 \
  --epochs 100
```

### Baseline comparison study
```bash
python run_regional_training.py \
  --n_seeds 5 \
  --skip_anp \
  --baseline_models rf xgb mlp-dropout \
  --cache_dir ./cache \
  --output_dir ./baseline_comparison
```

## Troubleshooting

**Import errors**: Ensure all dependencies are installed:
```bash
pip install torch pandas numpy matplotlib seaborn
```

**Memory issues**: Reduce `--batch_size` or run fewer regions at once

**Cache issues**: Clear cache directory if you encounter corrupted data

**GEDI query failures**: Check network connection and retry. The script continues with remaining regions if one fails.
