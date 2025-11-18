#!/usr/bin/env python3
"""
Usage:
    # Run only specific regions
    python run_regional_training.py --n_seeds 3 --regions maine hokkaido --cache_dir ./cache
    
    # Use specific seeds
    python run_regional_training.py --seeds 42 69 9000 1618 54088 777 1314 3 67 80085 --cache_dir ./cache
    
    python run_regional_training.py --seeds 42 69 9001 1618 1 777 1314 2 67 6 --cache_dir ./cache
    
    
    
    python run_regional_training.py --seeds 42 69 12345 1618 1 --cache_dir ./cache
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
from collections import defaultdict

REGIONS = {
    'maine': {
        'name': 'Maine, USA',
        'bbox': [-70, 44, -69, 45],
        'batch_size': 16,  # Default
        'description': 'Temperate mixed forest, northeastern USA'
    },
    'sudtirol': {
        'name': 'South Tyrol, Italy',
        'bbox': [10.5, 45.6, 11.5, 46.4],
        'batch_size': 4,  # Smaller due to region size
        'description': 'Alpine coniferous forest, European Alps'
    },
    'ili': {
        'name': 'Ili, Xinjiang, China',
        'bbox': [86, 42.8, 87, 43.5],
        'batch_size': 16,
        'description': 'Arid continental forest, Central Asia'
    },
    'hokkaido': {
        'name': 'Hokkaido, Japan',
        'bbox': [143.8, 43.2, 144.8, 43.9],
        'batch_size': 16,
        'description': 'Temperate deciduous/coniferous forest, northern Japan'
    },
    'tolima': {
        'name': 'Tolima, Colombia',
        'bbox': [-75, 3, -74, 4],
        'batch_size': 16,
        'description': 'Tropical montane forest, Andean region'
    },
    'guaviare': {
        'name': 'Guaviare, Colombia',
        'bbox': [-73, 2, -72, 3],
        'batch_size': 16,
        'description': ''
    }
}


def parse_args():
    parser = argparse.ArgumentParser(
        description='Run training across multiple geographic regions',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    # Region selection
    parser.add_argument('--regions', nargs='+',
                        choices=list(REGIONS.keys()) + ['all'],
                        default='all',
                        help='Which regions to train on (default: all)')

    # Seed arguments
    parser.add_argument('--n_seeds', type=int, default=5,
                        help='Number of seeds to run per region (default: 5)')
    parser.add_argument('--seeds', type=int, nargs='+', default=None,
                        help='Specific seeds to use (e.g., 42 43 44)')
    parser.add_argument('--base_seed', type=int, default=42,
                        help='Base seed for generating seed list (default: 42)')

    # Model selection
    parser.add_argument('--skip_anp', action='store_true',
                        help='Skip Neural Process training (only run baselines)')
    parser.add_argument('--skip_baselines', action='store_true',
                        help='Skip baseline training (only run ANP)')
    parser.add_argument('--baseline_models', nargs='+',
                        default=['rf', 'xgb', 'idw', 'mlp-dropout'],
                        choices=['rf', 'xgb', 'idw', 'lr', 'mlp-dropout', 'mlp-ensemble'],
                        help='Which baseline models to train')

    # Common training arguments
    parser.add_argument('--embedding_year', type=int, default=2022,
                        help='Year of GeoTessera embeddings')
    parser.add_argument('--start_time', type=str, default='2022-01-01',
                        help='Start date for GEDI data (YYYY-MM-DD)')
    parser.add_argument('--end_time', type=str, default='2022-12-31',
                        help='End date for GEDI data (YYYY-MM-DD)')
    parser.add_argument('--buffer_size', type=float, default=0.1,
                        help='Buffer size in degrees for spatial CV (default: 0.1)')
    parser.add_argument('--cache_dir', type=str, required=True,
                        help='Directory for caching tiles and embeddings')
    parser.add_argument('--output_dir', type=str, default='./regional_results',
                        help='Base output directory')

    # ANP-specific arguments
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of epochs for ANP training')
    parser.add_argument('--hidden_dim', type=int, default=512,
                        help='Hidden dimension for ANP')
    parser.add_argument('--latent_dim', type=int, default=256,
                        help='Latent dimension for ANP')

    args = parser.parse_args()

    # Validate
    if args.skip_anp and args.skip_baselines:
        parser.error('Cannot skip both ANP and baselines')

    # Handle 'all' regions
    if args.regions == 'all' or 'all' in args.regions:
        args.regions = list(REGIONS.keys())

    return args


def generate_seeds(args):
    """Generate list of seeds based on arguments."""
    if args.seeds is not None:
        return args.seeds
    else:
        return list(range(args.base_seed, args.base_seed + args.n_seeds))


def run_harness_for_region(region_key, region_info, script, seeds, args, output_dir):
    """Run the training harness for a single region and script."""
    print("\n" + "=" * 80)
    print(f"REGION: {region_info['name']}")
    print(f"SCRIPT: {script}")
    print("=" * 80)

    # Build command
    cmd = [
        sys.executable, 'run_training_harness.py',
        '--script', script,
        '--seeds', *[str(s) for s in seeds],
        '--region_bbox', *[str(x) for x in region_info['bbox']],
        '--embedding_year', str(args.embedding_year),
        '--start_time', args.start_time,
        '--end_time', args.end_time,
        '--buffer_size', str(args.buffer_size),
        '--cache_dir', args.cache_dir,
        '--output_dir', str(output_dir),
    ]

    # Add script-specific arguments
    if script == 'train.py':
        cmd.extend([
            '--architecture_mode', 'anp',
            '--epochs', str(args.epochs),
            '--batch_size', str(region_info['batch_size']),
            '--hidden_dim', str(args.hidden_dim),
            '--latent_dim', str(args.latent_dim),
        ])
    elif script == 'train_baselines.py':
        cmd.extend([
            '--models', *args.baseline_models,
        ])

    # Run harness
    print(f"\nExecuting: {' '.join(cmd)}\n")
    start_time = datetime.now()
    result = subprocess.run(cmd)
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()

    if result.returncode != 0:
        print(f"\nERROR: Training harness failed for {region_info['name']} - {script}")
        return None, duration

    print(f"\nCompleted {region_info['name']} - {script} in {duration/60:.1f} minutes")
    return output_dir, duration


def collect_regional_results(regional_dirs, regions_info):
    """Collect and aggregate results across all regions."""
    anp_results = []
    baseline_results = []

    for region_key, dirs in regional_dirs.items():
        region_name = regions_info[region_key]['name']

        # Collect ANP results
        if dirs['anp'] is not None:
            stats_file = dirs['anp'] / 'statistics.csv'
            if stats_file.exists():
                stats = pd.read_csv(stats_file)
                stats['region'] = region_name
                stats['region_key'] = region_key
                anp_results.append(stats)

        # Collect baseline results
        if dirs['baselines'] is not None:
            stats_file = dirs['baselines'] / 'statistics.csv'
            if stats_file.exists():
                stats = pd.read_csv(stats_file)
                stats['region'] = region_name
                stats['region_key'] = region_key
                baseline_results.append(stats)

    anp_df = pd.concat(anp_results, ignore_index=True) if anp_results else pd.DataFrame()
    baseline_df = pd.concat(baseline_results, ignore_index=True) if baseline_results else pd.DataFrame()

    return anp_df, baseline_df


def create_regional_comparison_plots(anp_df, baseline_df, output_dir):
    """Create comparison plots across regions."""

    # Determine what we have
    has_anp = not anp_df.empty
    has_baselines = not baseline_df.empty

    if not has_anp and not has_baselines:
        print("No results to plot")
        return

    # Plot 1: ANP performance across regions
    if has_anp:
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        fig.suptitle('Neural Process (ANP) Performance Across Regions',
                     fontsize=16, fontweight='bold')

        metrics = [
            ('test_log_r2', 'Test Log R²', True),
            ('test_log_rmse', 'Test Log RMSE', False),
            ('test_log_mae', 'Test Log MAE', False),
            ('test_linear_rmse', 'Test Linear RMSE (Mg/ha)', False),
            ('test_linear_mae', 'Test Linear MAE (Mg/ha)', False),
        ]

        for idx, (metric, label, higher_better) in enumerate(metrics):
            ax = axes.flat[idx]

            mean_col = f'{metric}_mean'
            std_col = f'{metric}_std'

            if mean_col in anp_df.columns:
                regions = anp_df['region']
                means = anp_df[mean_col]
                stds = anp_df[std_col]

                x = range(len(regions))
                colors = sns.color_palette("husl", len(regions))

                ax.bar(x, means, yerr=stds, capsize=5, color=colors, alpha=0.7,
                      error_kw={'linewidth': 2})

                # Add value labels
                for i, (mean, std, region) in enumerate(zip(means, stds, regions)):
                    ax.text(i, mean + std + 0.02 * abs(mean),
                           f'{mean:.3f}±{std:.3f}',
                           ha='center', va='bottom', fontsize=8, rotation=45)

                ax.set_xticks(x)
                ax.set_xticklabels(regions, rotation=45, ha='right')
                ax.set_ylabel(label, fontweight='bold')
                ax.set_title(f'{label}')
                ax.grid(True, alpha=0.3, axis='y')

                if metric == 'test_log_r2':
                    ax.axhline(y=0, color='k', linestyle='-', linewidth=0.5)

        # Hide unused subplot
        axes.flat[5].axis('off')

        plt.tight_layout()
        plt.savefig(output_dir / 'anp_regional_comparison.png', dpi=300, bbox_inches='tight')
        print(f"Saved ANP comparison to: {output_dir / 'anp_regional_comparison.png'}")
        plt.close()

        # Plot calibration metrics if available
        calib_cols = [col for col in anp_df.columns if any(x in col for x in ['z_mean', 'z_std', 'coverage'])]
        if calib_cols:
            calib_metrics = []
            for col in calib_cols:
                if 'test' in col and '_mean' in col:
                    base_metric = col.replace('_mean', '')
                    std_col = f"{base_metric}_std"
                    if std_col in anp_df.columns:
                        label = base_metric.replace('test_', '').replace('_', ' ').title()
                        calib_metrics.append((base_metric, label))

            if calib_metrics:
                n_metrics = len(calib_metrics)
                fig, axes = plt.subplots(2, 3, figsize=(18, 10))
                fig.suptitle('Neural Process (ANP) Uncertainty & Calibration Metrics Across Regions',
                             fontsize=16, fontweight='bold')

                for idx, (metric, label) in enumerate(calib_metrics[:6]):  # Max 6 plots
                    ax = axes.flat[idx]

                    mean_col = f'{metric}_mean'
                    std_col = f'{metric}_std'

                    if mean_col in anp_df.columns and std_col in anp_df.columns:
                        regions = anp_df['region']
                        means = anp_df[mean_col]
                        stds = anp_df[std_col]

                        x = range(len(regions))
                        colors = sns.color_palette("husl", len(regions))

                        ax.bar(x, means, yerr=stds, capsize=5, color=colors, alpha=0.7,
                              error_kw={'linewidth': 2})

                        # Add value labels
                        for i, (mean, std, region) in enumerate(zip(means, stds, regions)):
                            ax.text(i, mean + std + 0.02 * abs(mean) if mean >= 0 else mean - std - 0.02 * abs(mean),
                                   f'{mean:.2f}±{std:.2f}',
                                   ha='center', va='bottom' if mean >= 0 else 'top',
                                   fontsize=8, rotation=45)

                        ax.set_xticks(x)
                        ax.set_xticklabels(regions, rotation=45, ha='right')
                        ax.set_ylabel(label, fontweight='bold')
                        ax.set_title(f'{label}')
                        ax.grid(True, alpha=0.3, axis='y')

                        # Add reference lines for calibration metrics
                        if 'z_mean' in metric:
                            ax.axhline(y=0, color='r', linestyle='--', linewidth=1.5, alpha=0.7, label='Ideal: 0')
                            ax.legend()
                        elif 'z_std' in metric:
                            ax.axhline(y=1, color='r', linestyle='--', linewidth=1.5, alpha=0.7, label='Ideal: 1')
                            ax.legend()
                        elif 'coverage_1sigma' in metric:
                            ax.axhline(y=68.3, color='r', linestyle='--', linewidth=1.5, alpha=0.7, label='Ideal: 68.3%')
                            ax.legend()
                        elif 'coverage_2sigma' in metric:
                            ax.axhline(y=95.4, color='r', linestyle='--', linewidth=1.5, alpha=0.7, label='Ideal: 95.4%')
                            ax.legend()
                        elif 'coverage_3sigma' in metric:
                            ax.axhline(y=99.7, color='r', linestyle='--', linewidth=1.5, alpha=0.7, label='Ideal: 99.7%')
                            ax.legend()

                # Hide unused subplots
                for idx in range(len(calib_metrics), 6):
                    axes.flat[idx].axis('off')

                plt.tight_layout()
                plt.savefig(output_dir / 'anp_calibration_comparison.png', dpi=300, bbox_inches='tight')
                print(f"Saved ANP calibration comparison to: {output_dir / 'anp_calibration_comparison.png'}")
                plt.close()

    # Plot 2: Baseline comparison across regions
    if has_baselines:
        # Create a plot for each region comparing models
        fig, axes = plt.subplots(len(baseline_df['region'].unique()), 2,
                                 figsize=(16, 6*len(baseline_df['region'].unique())))
        if len(baseline_df['region'].unique()) == 1:
            axes = axes.reshape(1, -1)

        fig.suptitle('Baseline Model Performance by Region',
                     fontsize=16, fontweight='bold')

        for idx, (region, region_df) in enumerate(baseline_df.groupby('region')):
            # R² comparison
            ax = axes[idx, 0]
            models = region_df['model']
            means = region_df['test_log_r2_mean']
            stds = region_df['test_log_r2_std']

            x = range(len(models))
            colors = sns.color_palette("Set2", len(models))

            ax.bar(x, means, yerr=stds, capsize=5, color=colors, alpha=0.7)
            ax.set_xticks(x)
            ax.set_xticklabels(models, rotation=45, ha='right')
            ax.set_ylabel('Test Log R²', fontweight='bold')
            ax.set_title(f'{region} - Log R² by Model')
            ax.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
            ax.grid(True, alpha=0.3, axis='y')

            # RMSE comparison
            ax = axes[idx, 1]
            means = region_df['test_linear_rmse_mean']
            stds = region_df['test_linear_rmse_std']

            ax.bar(x, means, yerr=stds, capsize=5, color=colors, alpha=0.7)
            ax.set_xticks(x)
            ax.set_xticklabels(models, rotation=45, ha='right')
            ax.set_ylabel('Test Linear RMSE (Mg/ha)', fontweight='bold')
            ax.set_title(f'{region} - Linear RMSE by Model')
            ax.grid(True, alpha=0.3, axis='y')

        plt.tight_layout()
        plt.savefig(output_dir / 'baselines_regional_comparison.png',
                   dpi=300, bbox_inches='tight')
        print(f"Saved baseline comparison to: {output_dir / 'baselines_regional_comparison.png'}")
        plt.close()

    # Plot 3: Combined comparison (ANP vs best baseline per region)
    if has_anp and has_baselines:
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        fig.suptitle('ANP vs Best Baseline by Region', fontsize=16, fontweight='bold')

        # Get best baseline per region (by test_log_r2)
        best_baselines = baseline_df.loc[
            baseline_df.groupby('region')['test_log_r2_mean'].idxmax()
        ]

        # Merge with ANP results
        comparison = pd.merge(
            anp_df[['region', 'test_log_r2_mean', 'test_log_r2_std',
                    'test_linear_rmse_mean', 'test_linear_rmse_std']],
            best_baselines[['region', 'model', 'test_log_r2_mean', 'test_log_r2_std',
                           'test_linear_rmse_mean', 'test_linear_rmse_std']],
            on='region',
            suffixes=('_anp', '_baseline')
        )

        # R² comparison
        ax = axes[0]
        x = np.arange(len(comparison))
        width = 0.35

        ax.bar(x - width/2, comparison['test_log_r2_mean_anp'], width,
               yerr=comparison['test_log_r2_std_anp'], label='ANP',
               alpha=0.8, capsize=5, color='steelblue')
        ax.bar(x + width/2, comparison['test_log_r2_mean_baseline'], width,
               yerr=comparison['test_log_r2_std_baseline'],
               label='Best Baseline', alpha=0.8, capsize=5, color='coral')

        ax.set_ylabel('Test Log R²', fontweight='bold')
        ax.set_title('Log R² Score Comparison')
        ax.set_xticks(x)
        ax.set_xticklabels(comparison['region'], rotation=45, ha='right')
        ax.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')

        # RMSE comparison
        ax = axes[1]
        ax.bar(x - width/2, comparison['test_linear_rmse_mean_anp'], width,
               yerr=comparison['test_linear_rmse_std_anp'], label='ANP',
               alpha=0.8, capsize=5, color='steelblue')
        ax.bar(x + width/2, comparison['test_linear_rmse_mean_baseline'], width,
               yerr=comparison['test_linear_rmse_std_baseline'],
               label='Best Baseline', alpha=0.8, capsize=5, color='coral')

        ax.set_ylabel('Test Linear RMSE (Mg/ha)', fontweight='bold')
        ax.set_title('Linear RMSE Comparison')
        ax.set_xticks(x)
        ax.set_xticklabels(comparison['region'], rotation=45, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')

        plt.tight_layout()
        plt.savefig(output_dir / 'anp_vs_baselines.png', dpi=300, bbox_inches='tight')
        print(f"Saved ANP vs baselines to: {output_dir / 'anp_vs_baselines.png'}")
        plt.close()


def write_regional_summary(anp_df, baseline_df, output_dir, args, total_duration):
    """Write comprehensive summary report."""
    report_path = output_dir / 'regional_summary.txt'

    with open(report_path, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("REGIONAL TRAINING SUMMARY\n")
        f.write("=" * 80 + "\n\n")

        f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Regions: {', '.join(args.regions)}\n")
        f.write(f"Seeds: {generate_seeds(args)}\n")
        f.write(f"Total duration: {total_duration/3600:.2f} hours\n\n")

        # ANP Results
        if not anp_df.empty:
            f.write("-" * 80 + "\n")
            f.write("NEURAL PROCESS (ANP) RESULTS\n")
            f.write("-" * 80 + "\n\n")

            display_cols = ['region', 'test_log_r2_mean', 'test_log_r2_std',
                          'test_log_rmse_mean', 'test_log_rmse_std',
                          'test_linear_rmse_mean', 'test_linear_rmse_std',
                          'test_linear_mae_mean', 'test_linear_mae_std']
            f.write(anp_df[display_cols].to_string(index=False) + "\n\n")

            # Add calibration metrics if available
            calib_cols = [col for col in anp_df.columns if any(x in col for x in ['z_mean', 'z_std', 'coverage'])]
            if calib_cols:
                f.write("-" * 80 + "\n")
                f.write("UNCERTAINTY QUANTIFICATION & CALIBRATION METRICS\n")
                f.write("-" * 80 + "\n\n")
                display_calib_cols = ['region'] + [col for col in calib_cols if 'test' in col]
                if len(display_calib_cols) > 1:  # More than just 'region'
                    f.write(anp_df[display_calib_cols].to_string(index=False) + "\n\n")

            best_region = anp_df.loc[anp_df['test_log_r2_mean'].idxmax()]
            f.write(f"Best Region: {best_region['region']}\n")
            f.write(f"  Test Log R²: {best_region['test_log_r2_mean']:.4f} ± "
                   f"{best_region['test_log_r2_std']:.4f}\n\n")

        # Baseline Results
        if not baseline_df.empty:
            f.write("-" * 80 + "\n")
            f.write("BASELINE MODEL RESULTS\n")
            f.write("-" * 80 + "\n\n")

            for region in baseline_df['region'].unique():
                region_df = baseline_df[baseline_df['region'] == region]
                f.write(f"\n{region}:\n")
                f.write("-" * 40 + "\n")

                display_cols = ['model', 'test_log_r2_mean', 'test_log_r2_std',
                              'test_linear_rmse_mean', 'test_linear_rmse_std']
                f.write(region_df[display_cols].to_string(index=False) + "\n")

                # Add calibration metrics if available
                calib_cols = [col for col in baseline_df.columns if any(x in col for x in ['z_mean', 'z_std', 'coverage'])]
                if calib_cols:
                    display_calib_cols = ['model'] + [col for col in calib_cols if 'test' in col]
                    if len(display_calib_cols) > 1 and all(col in region_df.columns for col in display_calib_cols):
                        f.write("\nCalibration metrics:\n")
                        f.write(region_df[display_calib_cols].to_string(index=False) + "\n")

                best_model = region_df.loc[region_df['test_log_r2_mean'].idxmax()]
                f.write(f"\nBest Model: {best_model['model'].upper()}\n")
                f.write(f"  Test Log R²: {best_model['test_log_r2_mean']:.4f} ± "
                       f"{best_model['test_log_r2_std']:.4f}\n")

        # Combined comparison
        if not anp_df.empty and not baseline_df.empty:
            f.write("\n" + "=" * 80 + "\n")
            f.write("ANP vs BEST BASELINE BY REGION\n")
            f.write("=" * 80 + "\n\n")

            best_baselines = baseline_df.loc[
                baseline_df.groupby('region')['test_log_r2_mean'].idxmax()
            ]

            for _, anp_row in anp_df.iterrows():
                region = anp_row['region']
                baseline_row = best_baselines[best_baselines['region'] == region].iloc[0]

                anp_r2 = anp_row['test_log_r2_mean']
                baseline_r2 = baseline_row['test_log_r2_mean']
                improvement = ((anp_r2 - baseline_r2) / abs(baseline_r2)) * 100

                f.write(f"\n{region}:\n")
                f.write(f"  ANP:           {anp_r2:.4f} ± {anp_row['test_log_r2_std']:.4f}\n")
                f.write(f"  {baseline_row['model'].upper():14s} {baseline_r2:.4f} ± "
                       f"{baseline_row['test_log_r2_std']:.4f}\n")
                f.write(f"  Improvement:   {improvement:+.2f}%\n")

        f.write("\n" + "=" * 80 + "\n")

    print(f"\nSaved regional summary to: {report_path}")


def main():
    args = parse_args()
    seeds = generate_seeds(args)

    # Setup output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save configuration
    config = {
        'timestamp': datetime.now().isoformat(),
        'regions': args.regions,
        'seeds': seeds,
        'skip_anp': args.skip_anp,
        'skip_baselines': args.skip_baselines,
        'baseline_models': args.baseline_models,
        'args': vars(args)
    }
    with open(output_dir / 'regional_config.json', 'w') as f:
        json.dump(config, f, indent=2)

    print("\n" + "=" * 80)
    print("REGIONAL TRAINING EXPERIMENT")
    print("=" * 80)
    print(f"Regions: {', '.join(args.regions)}")
    print(f"Seeds: {seeds}")
    print(f"Models: {'ANP' if not args.skip_anp else ''}"
          f"{' + ' if not args.skip_anp and not args.skip_baselines else ''}"
          f"{'Baselines' if not args.skip_baselines else ''}")
    if not args.skip_baselines:
        print(f"Baseline models: {', '.join(args.baseline_models)}")
    print(f"Output directory: {output_dir}")
    print("=" * 80 + "\n")

    # Run training for each region
    regional_dirs = {}
    start_time_total = datetime.now()

    for region_key in args.regions:
        region_info = REGIONS[region_key]
        regional_dirs[region_key] = {'anp': None, 'baselines': None}

        print("\n" + "=" * 80)
        print(f"PROCESSING REGION: {region_info['name']}")
        print(f"Description: {region_info['description']}")
        print(f"Bounding box: {region_info['bbox']}")
        print("=" * 80)

        # Run ANP
        if not args.skip_anp:
            anp_output = output_dir / region_key / 'anp'
            anp_dir, duration = run_harness_for_region(
                region_key, region_info, 'train.py', seeds, args, anp_output
            )
            regional_dirs[region_key]['anp'] = anp_dir

            if anp_dir is None:
                print(f"WARNING: ANP training failed for {region_info['name']}")

        # Run baselines
        if not args.skip_baselines:
            baseline_output = output_dir / region_key / 'baselines'
            baseline_dir, duration = run_harness_for_region(
                region_key, region_info, 'train_baselines.py', seeds, args, baseline_output
            )
            regional_dirs[region_key]['baselines'] = baseline_dir

            if baseline_dir is None:
                print(f"WARNING: Baseline training failed for {region_info['name']}")

    end_time_total = datetime.now()
    total_duration = (end_time_total - start_time_total).total_seconds()

    # Collect and aggregate results
    print("\n" + "=" * 80)
    print("COLLECTING AND AGGREGATING RESULTS")
    print("=" * 80)

    anp_df, baseline_df = collect_regional_results(regional_dirs, REGIONS)

    # Save aggregated results
    if not anp_df.empty:
        anp_df.to_csv(output_dir / 'anp_regional_results.csv', index=False)
        print(f"Saved ANP results to: {output_dir / 'anp_regional_results.csv'}")

    if not baseline_df.empty:
        baseline_df.to_csv(output_dir / 'baseline_regional_results.csv', index=False)
        print(f"Saved baseline results to: {output_dir / 'baseline_regional_results.csv'}")

    # Create visualizations
    print("\n" + "=" * 80)
    print("GENERATING REGIONAL COMPARISON PLOTS")
    print("=" * 80)

    create_regional_comparison_plots(anp_df, baseline_df, output_dir)

    # Write summary report
    print("\n" + "=" * 80)
    print("GENERATING SUMMARY REPORT")
    print("=" * 80)

    write_regional_summary(anp_df, baseline_df, output_dir, args, total_duration)

    # Print final summary
    print("\n" + "=" * 80)
    print("REGIONAL TRAINING COMPLETE")
    print("=" * 80)
    print(f"Total time: {total_duration/3600:.2f} hours")
    print(f"Results saved to: {output_dir}")
    print("\nGenerated files:")
    print("  - regional_config.json              # Experiment configuration")
    if not anp_df.empty:
        print("  - anp_regional_results.csv          # ANP results across regions")
        print("  - anp_regional_comparison.png       # ANP performance visualization")
        calib_cols = [col for col in anp_df.columns if any(x in col for x in ['z_mean', 'z_std', 'coverage'])]
        if calib_cols:
            print("  - anp_calibration_comparison.png    # ANP calibration & uncertainty metrics")
    if not baseline_df.empty:
        print("  - baseline_regional_results.csv     # Baseline results across regions")
        print("  - baselines_regional_comparison.png # Baseline performance visualization")
    if not anp_df.empty and not baseline_df.empty:
        print("  - anp_vs_baselines.png              # Direct comparison")
    print("  - regional_summary.txt              # Comprehensive text report")
    print("\nIndividual region results are in subdirectories:")
    for region_key in args.regions:
        print(f"  - {region_key}/")
        if not args.skip_anp:
            print(f"    - anp/")
        if not args.skip_baselines:
            print(f"    - baselines/")
    print("=" * 80 + "\n")

    # Print key results
    if not anp_df.empty or not baseline_df.empty:
        print("KEY RESULTS:")
        print("-" * 80)

        if not anp_df.empty:
            print("\nANP Performance by Region:")
            print(anp_df[['region', 'test_log_r2_mean', 'test_log_r2_std',
                          'test_linear_rmse_mean', 'test_linear_rmse_std']].to_string(index=False))

            # Print calibration metrics if available
            calib_cols = [col for col in anp_df.columns if any(x in col for x in ['z_mean', 'z_std', 'coverage_1sigma'])]
            if calib_cols:
                display_calib_cols = ['region'] + [col for col in calib_cols if 'test' in col]
                if len(display_calib_cols) > 1:
                    print("\nCalibration & Uncertainty Metrics:")
                    print(anp_df[display_calib_cols].to_string(index=False))

        if not baseline_df.empty:
            print("\nBest Baseline by Region:")
            best_baselines = baseline_df.loc[
                baseline_df.groupby('region')['test_log_r2_mean'].idxmax()
            ]
            print(best_baselines[['region', 'model', 'test_log_r2_mean', 'test_log_r2_std',
                                 'test_linear_rmse_mean', 'test_linear_rmse_std']].to_string(index=False))

        print("=" * 80 + "\n")


if __name__ == '__main__':
    main()
