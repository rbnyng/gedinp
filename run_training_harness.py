#!/usr/bin/env python3
"""
Training Harness for Multi-Seed Experiments

This script runs training multiple times with different random seeds to compute
mean and standard deviation of metrics, providing statistically robust results.

Usage examples:
  # Run Neural Process with 5 seeds
  python run_training_harness.py --script train.py --n_seeds 5 \\
      --region_bbox -122.5 37.0 -122.0 37.5

  # Run baselines with specific seeds
  python run_training_harness.py --script train_baselines.py \\
      --seeds 42 43 44 45 46 --region_bbox -122.5 37.0 -122.0 37.5
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from collections import defaultdict


def parse_args():
    parser = argparse.ArgumentParser(
        description='Run training with multiple seeds for statistical robustness',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    # Harness-specific arguments
    parser.add_argument('--script', type=str, required=True,
                        choices=['train.py', 'train_baselines.py'],
                        help='Training script to run')
    parser.add_argument('--n_seeds', type=int, default=None,
                        help='Number of seeds to run (generates seeds starting from base_seed)')
    parser.add_argument('--seeds', type=int, nargs='+', default=None,
                        help='Specific seeds to use (e.g., 42 43 44 45)')
    parser.add_argument('--base_seed', type=int, default=42,
                        help='Base seed for generating seed list (default: 42)')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Base output directory for all runs')
    parser.add_argument('--parallel', action='store_true',
                        help='Run seeds in parallel (requires sufficient resources)')

    # Common training arguments (passed through to training scripts)
    parser.add_argument('--region_bbox', type=float, nargs=4, required=True,
                        help='Region bounding box: min_lon min_lat max_lon max_lat')
    parser.add_argument('--start_time', type=str, default='2022-01-01',
                        help='Start date for GEDI data (YYYY-MM-DD)')
    parser.add_argument('--end_time', type=str, default='2022-12-31',
                        help='End date for GEDI data (YYYY-MM-DD)')
    parser.add_argument('--embedding_year', type=int, default=2022,
                        help='Year of GeoTessera embeddings')
    parser.add_argument('--cache_dir', type=str, default='./cache',
                        help='Directory for caching tiles and embeddings')
    parser.add_argument('--buffer_size', type=float, default=0.5,
                        help='Buffer size in degrees for spatial CV (~55km at 0.5 deg)')

    # Neural Process specific arguments (for train.py)
    parser.add_argument('--architecture_mode', type=str, default='anp',
                        choices=['deterministic', 'latent', 'anp', 'cnp'],
                        help='[train.py only] Architecture mode')
    parser.add_argument('--hidden_dim', type=int, default=512,
                        help='[train.py only] Hidden layer dimension')
    parser.add_argument('--latent_dim', type=int, default=256,
                        help='[train.py only] Latent variable dimension')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Batch size')
    parser.add_argument('--lr', type=float, default=5e-4,
                        help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.01,
                        help='Weight decay for AdamW optimizer')

    # Baseline specific arguments (for train_baselines.py)
    parser.add_argument('--models', type=str, nargs='+',
                        default=['rf', 'xgb', 'idw', 'lr'],
                        choices=['rf', 'xgb', 'idw', 'lr'],
                        help='[train_baselines.py only] Which baseline models to train')

    args = parser.parse_args()

    # Validate seed specification
    if args.n_seeds is None and args.seeds is None:
        parser.error('Must specify either --n_seeds or --seeds')
    if args.n_seeds is not None and args.seeds is not None:
        parser.error('Cannot specify both --n_seeds and --seeds')

    return args


def generate_seeds(args):
    """Generate list of seeds based on arguments."""
    if args.seeds is not None:
        return args.seeds
    else:
        return list(range(args.base_seed, args.base_seed + args.n_seeds))


def run_training_single_seed(script, seed, output_subdir, args):
    """Run training script with a single seed."""
    print("=" * 80)
    print(f"Running {script} with seed={seed}")
    print(f"Output: {output_subdir}")
    print("=" * 80)

    # Build command
    cmd = [
        sys.executable, script,
        '--region_bbox', *[str(x) for x in args.region_bbox],
        '--start_time', args.start_time,
        '--end_time', args.end_time,
        '--embedding_year', str(args.embedding_year),
        '--cache_dir', args.cache_dir,
        '--output_dir', str(output_subdir),
        '--seed', str(seed),
    ]

    # Add script-specific arguments
    if script == 'train.py':
        cmd.extend([
            '--architecture_mode', args.architecture_mode,
            '--hidden_dim', str(args.hidden_dim),
            '--latent_dim', str(args.latent_dim),
            '--epochs', str(args.epochs),
            '--batch_size', str(args.batch_size),
            '--lr', str(args.lr),
            '--weight_decay', str(args.weight_decay),
            '--buffer_size', str(args.buffer_size),
        ])
    elif script == 'train_baselines.py':
        cmd.extend([
            '--models', *args.models,
            '--buffer_size', str(args.buffer_size),
        ])

    # Run training
    start_time = datetime.now()
    result = subprocess.run(cmd, capture_output=True, text=True)
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()

    if result.returncode != 0:
        print(f"ERROR: Training failed for seed={seed}")
        print(f"STDOUT:\n{result.stdout}")
        print(f"STDERR:\n{result.stderr}")
        return None, duration

    print(f"Training completed successfully in {duration:.1f}s")
    return output_subdir, duration


def collect_results_neural_process(output_subdirs):
    """Collect results from Neural Process training runs."""
    results = []

    for output_dir in output_subdirs:
        if output_dir is None:
            continue

        try:
            # Load config
            with open(output_dir / 'config.json', 'r') as f:
                config = json.load(f)

            # Load best model checkpoint
            checkpoint = torch.load(
                output_dir / 'best_r2_model.pt',
                map_location='cpu',
                weights_only=False
            )

            # Extract metrics
            seed = config['seed']
            val_metrics = checkpoint.get('val_metrics', {})
            test_metrics = checkpoint.get('test_metrics', {})

            results.append({
                'seed': seed,
                'output_dir': str(output_dir),
                'val_log_rmse': val_metrics.get('log_rmse', np.nan),
                'val_log_mae': val_metrics.get('log_mae', np.nan),
                'val_log_r2': val_metrics.get('log_r2', np.nan),
                'val_linear_rmse': val_metrics.get('linear_rmse', np.nan),
                'val_linear_mae': val_metrics.get('linear_mae', np.nan),
                'val_z_mean': val_metrics.get('z_mean', np.nan),
                'val_z_std': val_metrics.get('z_std', np.nan),
                'val_coverage_1sigma': val_metrics.get('coverage_1sigma', np.nan),
                'val_coverage_2sigma': val_metrics.get('coverage_2sigma', np.nan),
                'val_coverage_3sigma': val_metrics.get('coverage_3sigma', np.nan),
                'test_log_rmse': test_metrics.get('log_rmse', np.nan),
                'test_log_mae': test_metrics.get('log_mae', np.nan),
                'test_log_r2': test_metrics.get('log_r2', np.nan),
                'test_linear_rmse': test_metrics.get('linear_rmse', np.nan),
                'test_linear_mae': test_metrics.get('linear_mae', np.nan),
                'test_z_mean': test_metrics.get('z_mean', np.nan),
                'test_z_std': test_metrics.get('z_std', np.nan),
                'test_coverage_1sigma': test_metrics.get('coverage_1sigma', np.nan),
                'test_coverage_2sigma': test_metrics.get('coverage_2sigma', np.nan),
                'test_coverage_3sigma': test_metrics.get('coverage_3sigma', np.nan),
                'epoch': checkpoint.get('epoch', -1),
                'mean_uncertainty': val_metrics.get('mean_uncertainty', np.nan),
            })

        except Exception as e:
            print(f"Warning: Failed to load results from {output_dir}: {e}")
            continue

    return pd.DataFrame(results)


def collect_results_baselines(output_subdirs):
    """Collect results from baseline training runs."""
    results = []

    for output_dir in output_subdirs:
        if output_dir is None:
            continue

        try:
            # Load config
            with open(output_dir / 'config.json', 'r') as f:
                config = json.load(f)

            # Load results
            with open(output_dir / 'results.json', 'r') as f:
                model_results = json.load(f)

            seed = config['seed']

            # Extract metrics for each model
            for model_name, metrics_dict in model_results.items():
                val_metrics = metrics_dict['val_metrics']
                test_metrics = metrics_dict['test_metrics']

                results.append({
                    'seed': seed,
                    'model': model_name,
                    'output_dir': str(output_dir),
                    'val_log_rmse': val_metrics.get('log_rmse', np.nan),
                    'val_log_mae': val_metrics.get('log_mae', np.nan),
                    'val_log_r2': val_metrics.get('log_r2', np.nan),
                    'val_linear_rmse': val_metrics.get('linear_rmse', np.nan),
                    'val_linear_mae': val_metrics.get('linear_mae', np.nan),
                    'val_z_mean': val_metrics.get('z_mean', np.nan),
                    'val_z_std': val_metrics.get('z_std', np.nan),
                    'val_coverage_1sigma': val_metrics.get('coverage_1sigma', np.nan),
                    'val_coverage_2sigma': val_metrics.get('coverage_2sigma', np.nan),
                    'val_coverage_3sigma': val_metrics.get('coverage_3sigma', np.nan),
                    'test_log_rmse': test_metrics.get('log_rmse', np.nan),
                    'test_log_mae': test_metrics.get('log_mae', np.nan),
                    'test_log_r2': test_metrics.get('log_r2', np.nan),
                    'test_linear_rmse': test_metrics.get('linear_rmse', np.nan),
                    'test_linear_mae': test_metrics.get('linear_mae', np.nan),
                    'test_z_mean': test_metrics.get('z_mean', np.nan),
                    'test_z_std': test_metrics.get('z_std', np.nan),
                    'test_coverage_1sigma': test_metrics.get('coverage_1sigma', np.nan),
                    'test_coverage_2sigma': test_metrics.get('coverage_2sigma', np.nan),
                    'test_coverage_3sigma': test_metrics.get('coverage_3sigma', np.nan),
                    'train_time': metrics_dict.get('train_time', np.nan),
                })

        except Exception as e:
            print(f"Warning: Failed to load results from {output_dir}: {e}")
            continue

    return pd.DataFrame(results)


def compute_statistics(df, group_by=None):
    """Compute mean, std, min, max for metrics."""
    metric_cols = [col for col in df.columns if any(
        metric in col for metric in ['rmse', 'mae', 'r2', 'uncertainty', 'time', 'z_mean', 'z_std', 'coverage']
    )]

    if group_by:
        grouped = df.groupby(group_by)[metric_cols]
        # Use agg to compute all statistics at once
        stats = grouped.agg(['mean', 'std', 'min', 'max', 'median'])
        # Flatten the MultiIndex columns
        stats.columns = ['_'.join(col).strip() for col in stats.columns.values]
        stats = stats.reset_index()
    else:
        # No grouping, compute statistics across all rows
        stats_dict = {}
        for col in metric_cols:
            stats_dict[f'{col}_mean'] = [df[col].mean()]
            stats_dict[f'{col}_std'] = [df[col].std()]
            stats_dict[f'{col}_min'] = [df[col].min()]
            stats_dict[f'{col}_max'] = [df[col].max()]
            stats_dict[f'{col}_median'] = [df[col].median()]
        stats = pd.DataFrame(stats_dict)

    return stats


def create_visualizations(df, output_dir, script_type):
    """Create visualization plots for multi-seed results."""
    if script_type == 'train.py':
        create_neural_process_plots(df, output_dir)
    elif script_type == 'train_baselines.py':
        create_baseline_plots(df, output_dir)


def create_neural_process_plots(df, output_dir):
    """Create plots for Neural Process results."""
    fig, axes = plt.subplots(2, 5, figsize=(24, 8))
    fig.suptitle('Neural Process Multi-Seed Results', fontsize=16, fontweight='bold')

    metrics = [
        ('log_r2', 'Log R²'),
        ('log_rmse', 'Log RMSE'),
        ('log_mae', 'Log MAE'),
        ('linear_rmse', 'Linear RMSE (Mg/ha)'),
        ('linear_mae', 'Linear MAE (Mg/ha)')
    ]
    splits = ['val', 'test']

    for idx, split in enumerate(splits):
        for jdx, (metric, label) in enumerate(metrics):
            ax = axes[idx, jdx]
            col = f'{split}_{metric}'

            if col in df.columns:
                # Box plot
                ax.boxplot([df[col].dropna()], labels=[''], widths=0.5)

                # Overlay individual points
                x = np.random.normal(1, 0.04, size=len(df))
                ax.scatter(x, df[col], alpha=0.6, s=50)

                # Add mean line
                mean_val = df[col].mean()
                ax.axhline(mean_val, color='r', linestyle='--', linewidth=2,
                          label=f'Mean: {mean_val:.4f}')

                # Add std annotation
                std_val = df[col].std()
                ax.text(0.05, 0.95, f'μ ± σ: {mean_val:.4f} ± {std_val:.4f}',
                       transform=ax.transAxes, verticalalignment='top',
                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

                ax.set_ylabel(label, fontweight='bold')
                ax.set_title(f'{split.capitalize()} {label}')
                ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / 'multi_seed_results.png', dpi=300, bbox_inches='tight')
    print(f"Saved visualization to: {output_dir / 'multi_seed_results.png'}")


def create_baseline_plots(df, output_dir):
    """Create plots for baseline model results."""
    models = df['model'].unique()
    n_models = len(models)

    fig, axes = plt.subplots(2, 5, figsize=(24, 8))
    fig.suptitle('Baseline Models Multi-Seed Results', fontsize=16, fontweight='bold')

    metrics = [
        ('log_r2', 'Log R²'),
        ('log_rmse', 'Log RMSE'),
        ('log_mae', 'Log MAE'),
        ('linear_rmse', 'Linear RMSE (Mg/ha)'),
        ('linear_mae', 'Linear MAE (Mg/ha)')
    ]
    splits = ['val', 'test']

    for idx, split in enumerate(splits):
        for jdx, (metric, label) in enumerate(metrics):
            ax = axes[idx, jdx]
            col = f'{split}_{metric}'

            if col in df.columns:
                # Prepare data for box plot
                data = [df[df['model'] == model][col].dropna() for model in models]
                positions = range(1, n_models + 1)

                # Box plot
                bp = ax.boxplot(data, positions=positions, labels=models, widths=0.5)

                # Overlay individual points
                for i, model in enumerate(models):
                    model_data = df[df['model'] == model][col].dropna()
                    x = np.random.normal(i + 1, 0.04, size=len(model_data))
                    ax.scatter(x, model_data, alpha=0.6, s=30)

                ax.set_ylabel(label, fontweight='bold')
                ax.set_title(f'{split.capitalize()} {label}')
                ax.grid(True, alpha=0.3, axis='y')
                plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

    plt.tight_layout()
    plt.savefig(output_dir / 'multi_seed_results.png', dpi=300, bbox_inches='tight')
    print(f"Saved visualization to: {output_dir / 'multi_seed_results.png'}")


def create_comparison_plot(stats_df, output_dir, script_type):
    """Create a comparison plot with error bars showing mean ± std."""
    if script_type == 'train_baselines.py' and 'model' in stats_df.columns:
        # Baseline comparison
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        fig.suptitle('Model Comparison: Mean ± Std Dev (Test Set)',
                     fontsize=16, fontweight='bold')

        metrics = [
            ('test_log_r2', 'Log R²', True),           # higher is better
            ('test_log_rmse', 'Log RMSE', False),      # lower is better
            ('test_log_mae', 'Log MAE', False),        # lower is better
            ('test_linear_rmse', 'Linear RMSE (Mg/ha)', False),  # lower is better
            ('test_linear_mae', 'Linear MAE (Mg/ha)', False),    # lower is better
        ]

        for idx, (metric, label, higher_better) in enumerate(metrics):
            ax = axes.flat[idx]

            if f'{metric}_mean' in stats_df.columns:
                models = stats_df['model']
                means = stats_df[f'{metric}_mean']
                stds = stats_df[f'{metric}_std']

                x = range(len(models))
                colors = sns.color_palette("husl", len(models))

                # Bar plot with error bars
                ax.bar(x, means, yerr=stds, capsize=5, color=colors, alpha=0.7,
                      error_kw={'linewidth': 2})

                # Add value labels
                for i, (mean, std) in enumerate(zip(means, stds)):
                    ax.text(i, mean + std + 0.02 * abs(mean),
                           f'{mean:.3f}±{std:.3f}',
                           ha='center', va='bottom', fontsize=9)

                ax.set_xticks(x)
                ax.set_xticklabels(models, rotation=45, ha='right')
                ax.set_ylabel(label, fontweight='bold')
                ax.set_title(f'{label} Comparison')
                ax.grid(True, alpha=0.3, axis='y')

                if metric == 'test_log_r2':
                    ax.axhline(y=0, color='k', linestyle='-', linewidth=0.5)

        # Hide the unused subplot (6th position in 2x3 grid)
        axes.flat[5].axis('off')

        plt.tight_layout()
        plt.savefig(output_dir / 'model_comparison.png', dpi=300, bbox_inches='tight')
        print(f"Saved comparison plot to: {output_dir / 'model_comparison.png'}")

    else:
        # Single model (Neural Process)
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        fig.suptitle('Test Set Performance: Mean ± Std Dev',
                     fontsize=16, fontweight='bold')

        metrics = [
            ('test_log_r2', 'Log R²'),
            ('test_log_rmse', 'Log RMSE'),
            ('test_log_mae', 'Log MAE'),
            ('test_linear_rmse', 'Linear RMSE (Mg/ha)'),
            ('test_linear_mae', 'Linear MAE (Mg/ha)'),
        ]

        for idx, (metric, label) in enumerate(metrics):
            ax = axes.flat[idx]
            if f'{metric}_mean' in stats_df.columns:
                mean = stats_df[f'{metric}_mean'].values[0]
                std = stats_df[f'{metric}_std'].values[0]

                ax.bar([''], [mean], yerr=[std], capsize=10,
                       color='skyblue', alpha=0.7, error_kw={'linewidth': 2})
                ax.set_ylabel(label, fontweight='bold')
                ax.set_title(label)
                ax.text(0, mean + std + 0.02 * abs(mean),
                        f'{mean:.4f} ± {std:.4f}',
                        ha='center', va='bottom', fontsize=12, fontweight='bold')
                ax.grid(True, alpha=0.3, axis='y')

        # Hide the unused subplot (6th position in 2x3 grid)
        axes.flat[5].axis('off')

        plt.tight_layout()
        plt.savefig(output_dir / 'performance_summary.png', dpi=300, bbox_inches='tight')
        print(f"Saved summary plot to: {output_dir / 'performance_summary.png'}")


def write_summary_report(df, stats_df, output_dir, script_type, args, total_duration):
    """Write a comprehensive text summary report."""
    report_path = output_dir / 'harness_summary.txt'

    with open(report_path, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("MULTI-SEED TRAINING HARNESS REPORT\n")
        f.write("=" * 80 + "\n\n")

        f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Script: {script_type}\n")
        f.write(f"Seeds: {df['seed'].tolist()}\n")
        f.write(f"Number of runs: {len(df) if 'model' not in df.columns else len(df['seed'].unique())}\n")
        f.write(f"Total duration: {total_duration:.1f} seconds ({total_duration/60:.1f} minutes)\n")
        f.write(f"Region: {args.region_bbox}\n")
        f.write(f"Time range: {args.start_time} to {args.end_time}\n\n")

        if script_type == 'train.py':
            f.write(f"Architecture: {args.architecture_mode}\n")
            f.write(f"Hidden dim: {args.hidden_dim}\n")
            f.write(f"Latent dim: {args.latent_dim}\n")
            f.write(f"Epochs: {args.epochs}\n\n")

        f.write("-" * 80 + "\n")
        f.write("INDIVIDUAL RUN RESULTS\n")
        f.write("-" * 80 + "\n\n")
        f.write(df.to_string(index=False) + "\n\n")

        f.write("-" * 80 + "\n")
        f.write("AGGREGATED STATISTICS\n")
        f.write("-" * 80 + "\n\n")
        f.write(stats_df.to_string(index=False) + "\n\n")

        if script_type == 'train_baselines.py' and 'model' in stats_df.columns:
            f.write("-" * 80 + "\n")
            f.write("MODEL RANKING (by Test Log R²)\n")
            f.write("-" * 80 + "\n\n")
            ranking = stats_df.sort_values('test_log_r2_mean', ascending=False)
            for i, row in ranking.iterrows():
                f.write(f"{i+1}. {row['model'].upper()}: "
                       f"Log R² = {row['test_log_r2_mean']:.4f} ± {row['test_log_r2_std']:.4f}\n")
            f.write("\n")

        f.write("-" * 80 + "\n")
        f.write("SUMMARY\n")
        f.write("-" * 80 + "\n\n")

        if script_type == 'train.py':
            test_log_r2_mean = stats_df['test_log_r2_mean'].values[0]
            test_log_r2_std = stats_df['test_log_r2_std'].values[0]
            test_log_rmse_mean = stats_df['test_log_rmse_mean'].values[0]
            test_log_rmse_std = stats_df['test_log_rmse_std'].values[0]
            test_log_mae_mean = stats_df['test_log_mae_mean'].values[0]
            test_log_mae_std = stats_df['test_log_mae_std'].values[0]
            test_linear_rmse_mean = stats_df['test_linear_rmse_mean'].values[0]
            test_linear_rmse_std = stats_df['test_linear_rmse_std'].values[0]
            test_linear_mae_mean = stats_df['test_linear_mae_mean'].values[0]
            test_linear_mae_std = stats_df['test_linear_mae_std'].values[0]

            f.write(f"Log-space metrics (aligned with training):\n")
            f.write(f"  Test Log R²:    {test_log_r2_mean:.4f} ± {test_log_r2_std:.4f}\n")
            f.write(f"  Test Log RMSE:  {test_log_rmse_mean:.4f} ± {test_log_rmse_std:.4f}\n")
            f.write(f"  Test Log MAE:   {test_log_mae_mean:.4f} ± {test_log_mae_std:.4f}\n\n")
            f.write(f"Linear-space metrics (Mg/ha, for interpretability):\n")
            f.write(f"  Test RMSE:      {test_linear_rmse_mean:.2f} ± {test_linear_rmse_std:.2f} Mg/ha\n")
            f.write(f"  Test MAE:       {test_linear_mae_mean:.2f} ± {test_linear_mae_std:.2f} Mg/ha\n\n")
            f.write(f"Coefficient of Variation (Log R²): {(test_log_r2_std/test_log_r2_mean)*100:.2f}%\n\n")

            # Interpret stability
            cv = (test_log_r2_std / test_log_r2_mean) * 100
            if cv < 5:
                f.write("→ Results are very stable across seeds (CV < 5%)\n")
            elif cv < 10:
                f.write("→ Results show moderate variability (5% ≤ CV < 10%)\n")
            else:
                f.write("→ Results show high variability (CV ≥ 10%). Consider more seeds.\n")

        elif script_type == 'train_baselines.py':
            f.write("Best model per metric (Test Set):\n\n")

            best_log_r2 = stats_df.loc[stats_df['test_log_r2_mean'].idxmax()]
            f.write(f"Log R²: {best_log_r2['model'].upper()} "
                   f"({best_log_r2['test_log_r2_mean']:.4f} ± {best_log_r2['test_log_r2_std']:.4f})\n")

            best_log_rmse = stats_df.loc[stats_df['test_log_rmse_mean'].idxmin()]
            f.write(f"Log RMSE: {best_log_rmse['model'].upper()} "
                   f"({best_log_rmse['test_log_rmse_mean']:.4f} ± {best_log_rmse['test_log_rmse_std']:.4f})\n")

            best_log_mae = stats_df.loc[stats_df['test_log_mae_mean'].idxmin()]
            f.write(f"Log MAE: {best_log_mae['model'].upper()} "
                   f"({best_log_mae['test_log_mae_mean']:.4f} ± {best_log_mae['test_log_mae_std']:.4f})\n")

            best_linear_rmse = stats_df.loc[stats_df['test_linear_rmse_mean'].idxmin()]
            f.write(f"Linear RMSE: {best_linear_rmse['model'].upper()} "
                   f"({best_linear_rmse['test_linear_rmse_mean']:.2f} ± {best_linear_rmse['test_linear_rmse_std']:.2f} Mg/ha)\n")

            best_linear_mae = stats_df.loc[stats_df['test_linear_mae_mean'].idxmin()]
            f.write(f"Linear MAE: {best_linear_mae['model'].upper()} "
                   f"({best_linear_mae['test_linear_mae_mean']:.2f} ± {best_linear_mae['test_linear_mae_std']:.2f} Mg/ha)\n")

        f.write("\n" + "=" * 80 + "\n")

    print(f"Saved summary report to: {report_path}")


def main():
    args = parse_args()

    # Generate seeds
    seeds = generate_seeds(args)

    # Setup output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save harness configuration
    harness_config = {
        'script': args.script,
        'seeds': seeds,
        'timestamp': datetime.now().isoformat(),
        'args': vars(args)
    }
    with open(output_dir / 'harness_config.json', 'w') as f:
        json.dump(harness_config, f, indent=2)

    print("\n" + "=" * 80)
    print("MULTI-SEED TRAINING HARNESS")
    print("=" * 80)
    print(f"Script: {args.script}")
    print(f"Seeds: {seeds}")
    print(f"Number of runs: {len(seeds)}")
    print(f"Output directory: {output_dir}")
    print("=" * 80 + "\n")

    # Run training for each seed
    output_subdirs = []
    durations = []
    start_time_total = datetime.now()

    for i, seed in enumerate(seeds, 1):
        print(f"\n{'=' * 80}")
        print(f"RUN {i}/{len(seeds)}: Seed {seed}")
        print(f"{'=' * 80}\n")

        output_subdir = output_dir / f"seed_{seed}"
        output_subdir.mkdir(parents=True, exist_ok=True)

        result_dir, duration = run_training_single_seed(
            args.script, seed, output_subdir, args
        )
        output_subdirs.append(result_dir)
        durations.append(duration)

        if result_dir is None:
            print(f"WARNING: Run with seed={seed} failed. Continuing with remaining seeds...")

    end_time_total = datetime.now()
    total_duration = (end_time_total - start_time_total).total_seconds()

    # Collect results
    print("\n" + "=" * 80)
    print("COLLECTING RESULTS")
    print("=" * 80)

    if args.script == 'train.py':
        df = collect_results_neural_process(output_subdirs)
        stats_df = compute_statistics(df)
    elif args.script == 'train_baselines.py':
        df = collect_results_baselines(output_subdirs)
        stats_df = compute_statistics(df, group_by='model')

    if df.empty:
        print("ERROR: No results collected. All training runs may have failed.")
        return

    # Save results
    df.to_csv(output_dir / 'all_runs.csv', index=False)
    stats_df.to_csv(output_dir / 'statistics.csv', index=False)

    print(f"\nCollected results from {len(df) if 'model' not in df.columns else len(df['seed'].unique())} successful runs")
    print(f"Saved individual results to: {output_dir / 'all_runs.csv'}")
    print(f"Saved statistics to: {output_dir / 'statistics.csv'}")

    # Create visualizations
    print("\n" + "=" * 80)
    print("GENERATING VISUALIZATIONS")
    print("=" * 80)

    create_visualizations(df, output_dir, args.script)
    create_comparison_plot(stats_df, output_dir, args.script)

    # Write summary report
    print("\n" + "=" * 80)
    print("GENERATING SUMMARY REPORT")
    print("=" * 80)

    write_summary_report(df, stats_df, output_dir, args.script, args, total_duration)

    # Print final summary
    print("\n" + "=" * 80)
    print("MULTI-SEED TRAINING COMPLETE")
    print("=" * 80)
    print(f"Total time: {total_duration:.1f} seconds ({total_duration/60:.1f} minutes)")
    print(f"Average time per run: {np.mean(durations):.1f} seconds")
    print(f"\nResults saved to: {output_dir}")
    print("\nGenerated files:")
    print("  - harness_config.json    # Configuration used")
    print("  - all_runs.csv          # All individual run results")
    print("  - statistics.csv        # Aggregated statistics (mean ± std)")
    print("  - multi_seed_results.png # Box plots and scatter plots")
    print("  - model_comparison.png  # Comparison with error bars")
    print("  - harness_summary.txt   # Comprehensive text report")
    print("=" * 80 + "\n")

    # Print key statistics
    print("KEY RESULTS:")
    print("-" * 80)
    if args.script == 'train.py':
        print(f"Test Log R²:        {stats_df['test_log_r2_mean'].values[0]:.4f} ± {stats_df['test_log_r2_std'].values[0]:.4f}")
        print(f"Test Log RMSE:      {stats_df['test_log_rmse_mean'].values[0]:.4f} ± {stats_df['test_log_rmse_std'].values[0]:.4f}")
        print(f"Test Log MAE:       {stats_df['test_log_mae_mean'].values[0]:.4f} ± {stats_df['test_log_mae_std'].values[0]:.4f}")
        print(f"Test Linear RMSE:   {stats_df['test_linear_rmse_mean'].values[0]:.2f} ± {stats_df['test_linear_rmse_std'].values[0]:.2f} Mg/ha")
        print(f"Test Linear MAE:    {stats_df['test_linear_mae_mean'].values[0]:.2f} ± {stats_df['test_linear_mae_std'].values[0]:.2f} Mg/ha")
    else:
        display_cols = ['model', 'test_log_r2_mean', 'test_log_r2_std', 'test_log_rmse_mean', 'test_log_rmse_std',
                        'test_log_mae_mean', 'test_log_mae_std', 'test_linear_rmse_mean', 'test_linear_rmse_std',
                        'test_linear_mae_mean', 'test_linear_mae_std']
        print(stats_df[display_cols].to_string(index=False))
    print("=" * 80 + "\n")


if __name__ == '__main__':
    main()
