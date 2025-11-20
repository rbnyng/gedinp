#!/usr/bin/env python3
"""
Plot Pareto frontier for baseline model sweep results.

Visualizes the tradeoff between accuracy (test log RMSE) and calibration quality (z_std).
Creates separate panels for Random Forest and XGBoost to clearly show their different
regions of the Pareto frontier.

Usage:
    python plot_pareto_frontier.py --input outputs_pareto/pareto_results.csv --output pareto_frontier.png
"""

import argparse
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description='Plot Pareto frontier for baseline sweep results')
    parser.add_argument('--input', type=str, default='outputs_pareto/pareto_results.csv',
                        help='Path to pareto_results.csv')
    parser.add_argument('--output', type=str, default='pareto_frontier.png',
                        help='Output plot filename')
    parser.add_argument('--y_metric', type=str, default='z_std',
                        choices=['z_std', 'calibration_error', 'coverage_1sigma'],
                        help='Y-axis metric: z_std (default), calibration_error, or coverage_1sigma')
    parser.add_argument('--log_y', action='store_true',
                        help='Use log scale for y-axis')
    parser.add_argument('--combined', action='store_true',
                        help='Single panel with all models (default: separate panels for RF and XGB)')
    parser.add_argument('--show_pareto', action='store_true',
                        help='Draw Pareto frontier line connecting optimal points')
    parser.add_argument('--dpi', type=int, default=300,
                        help='Output DPI (default: 300)')
    return parser.parse_args()


def compute_pareto_frontier(df, x_col, y_col, minimize_both=True):
    """
    Compute Pareto frontier points.

    Args:
        df: DataFrame with points
        x_col: Column name for x-axis (assumed to minimize)
        y_col: Column name for y-axis
        minimize_both: If True, both x and y are minimized. If False, x is minimized and y is maximized.

    Returns:
        DataFrame subset containing only Pareto-optimal points, sorted by x_col
    """
    pareto_points = []

    # Sort by x-axis (ascending for minimization)
    sorted_df = df.sort_values(x_col).reset_index(drop=True)

    if minimize_both:
        # For minimization: a point is Pareto-optimal if no other point is better in both dimensions
        best_y = float('inf')
        for idx, row in sorted_df.iterrows():
            if row[y_col] < best_y:
                pareto_points.append(idx)
                best_y = row[y_col]
    else:
        # For minimization of x and maximization of y
        best_y = float('-inf')
        for idx, row in sorted_df.iterrows():
            if row[y_col] > best_y:
                pareto_points.append(idx)
                best_y = row[y_col]

    return sorted_df.iloc[pareto_points]


def plot_pareto_frontier(df, args):
    """Create Pareto frontier visualization."""

    # Setup y-axis metric and labels
    y_col = f'{args.y_metric}_mean'
    y_label_map = {
        'z_std': 'Z-Score Std Dev (Calibration Quality)',
        'calibration_error': 'Calibration Error |z_std - 1.0|',
        'coverage_1sigma': '1-Sigma Coverage (%)'
    }
    y_label = y_label_map[args.y_metric]

    # X-axis is log_rmse (not test_log_rmse)
    x_col = 'log_rmse_mean'

    # Reference values for calibration metrics
    y_reference = {
        'z_std': 1.0,
        'calibration_error': 0.0,
        'coverage_1sigma': 68.3
    }

    # Model colors
    model_colors = {
        'rf': '#2563eb',   # Blue
        'xgb': '#16a34a',  # Green
        'anp': '#dc2626'   # Red (if present)
    }

    model_labels = {
        'rf': 'Random Forest',
        'xgb': 'XGBoost',
        'anp': 'ANP'
    }

    # Check which models are present
    models_present = df['model_type'].unique()

    # Determine whether we need calibration error minimization or z_std/coverage optimization
    minimize_both = args.y_metric in ['calibration_error']

    # Create figure
    if args.combined:
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        axes = [ax]
        model_groups = [models_present]
        titles = ['All Models']
    else:
        # Separate panels for RF and XGB
        n_panels = len([m for m in ['rf', 'xgb'] if m in models_present])
        fig, axes = plt.subplots(1, n_panels, figsize=(6*n_panels, 5), squeeze=False)
        axes = axes.flatten()
        model_groups = [[m] for m in ['rf', 'xgb'] if m in models_present]
        titles = [model_labels.get(m, m) for m in ['rf', 'xgb'] if m in models_present]

    # Plot each panel
    for ax, models, title in zip(axes, model_groups, titles):
        for model in models:
            model_df = df[df['model_type'] == model].copy()

            if len(model_df) == 0:
                continue

            # Plot all points
            ax.scatter(
                model_df[x_col],
                model_df[y_col],
                c=model_colors.get(model, '#666666'),
                label=model_labels.get(model, model),
                alpha=0.6,
                s=50,
                edgecolors='white',
                linewidths=0.5
            )

            # Plot Pareto frontier
            if args.show_pareto:
                pareto_df = compute_pareto_frontier(
                    model_df,
                    x_col,
                    y_col,
                    minimize_both=minimize_both
                )
                ax.plot(
                    pareto_df[x_col],
                    pareto_df[y_col],
                    c=model_colors.get(model, '#666666'),
                    linestyle='--',
                    linewidth=2,
                    alpha=0.8,
                    label=f'{model_labels.get(model, model)} Frontier'
                )

        # Add reference line for ideal calibration
        if args.y_metric in y_reference:
            ax.axhline(
                y_reference[args.y_metric],
                color='black',
                linestyle=':',
                linewidth=1.5,
                alpha=0.5,
                label=f'Ideal: {y_reference[args.y_metric]}'
            )

        # Labels and formatting
        ax.set_xlabel('Log RMSE (lower = better)', fontsize=11, fontweight='bold')
        ax.set_ylabel(y_label, fontsize=11, fontweight='bold')
        ax.set_title(title, fontsize=13, fontweight='bold', pad=10)

        if args.log_y:
            ax.set_yscale('log')

        ax.legend(loc='best', framealpha=0.9)
        ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    # Overall title
    fig.suptitle(
        'Pareto Frontier: Accuracy vs Calibration Quality',
        fontsize=15,
        fontweight='bold',
        y=0.98
    )

    plt.tight_layout()
    return fig


def print_summary_statistics(df):
    """Print summary statistics for the sweep results."""
    print("\n" + "="*80)
    print("PARETO SWEEP SUMMARY STATISTICS")
    print("="*80)

    for model in df['model_type'].unique():
        model_df = df[df['model_type'] == model]
        print(f"\n{model.upper()} (n={len(model_df)} configs):")
        print(f"  Log RMSE:          {model_df['log_rmse_mean'].min():.4f} - {model_df['log_rmse_mean'].max():.4f}")
        print(f"  Z-Score Std:       {model_df['z_std_mean'].min():.4f} - {model_df['z_std_mean'].max():.4f}")
        print(f"  Calibration Error: {model_df['calibration_error_mean'].min():.4f} - {model_df['calibration_error_mean'].max():.4f}")
        print(f"  Coverage 1-sigma:  {model_df['coverage_1sigma_mean'].min():.1f}% - {model_df['coverage_1sigma_mean'].max():.1f}%")
        print(f"  Train Time (s):    {model_df['train_time_mean'].min():.2f} - {model_df['train_time_mean'].max():.2f}")

        # Best calibration (z_std closest to 1.0)
        best_calib_idx = (model_df['z_std_mean'] - 1.0).abs().idxmin()
        best_calib = model_df.loc[best_calib_idx]
        print(f"\n  Best Calibration Config:")
        print(f"    max_depth={best_calib['config_max_depth']}, n_estimators={best_calib['config_n_estimators']}")
        print(f"    Log RMSE: {best_calib['log_rmse_mean']:.4f}")
        print(f"    Z-Score Std: {best_calib['z_std_mean']:.4f}")

        # Best accuracy
        best_acc_idx = model_df['log_rmse_mean'].idxmin()
        best_acc = model_df.loc[best_acc_idx]
        print(f"\n  Best Accuracy Config:")
        print(f"    max_depth={best_acc['config_max_depth']}, n_estimators={best_acc['config_n_estimators']}")
        print(f"    Log RMSE: {best_acc['log_rmse_mean']:.4f}")
        print(f"    Z-Score Std: {best_acc['z_std_mean']:.4f}")

    print("\n" + "="*80 + "\n")


def main():
    args = parse_args()

    # Load results
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: Input file not found: {input_path}")
        print(f"\nPlease run the sweep first:")
        print(f"  python sweep_baselines.py --baseline_dir ./outputs_baselines --output_dir ./outputs_pareto")
        return

    print(f"Loading results from: {input_path}")
    df = pd.read_csv(input_path)

    print(f"Loaded {len(df)} configurations")
    print(f"Models: {', '.join(df['model_type'].unique())}")

    # Print summary statistics
    print_summary_statistics(df)

    # Create plot
    print(f"\nGenerating Pareto frontier plot...")
    print(f"  Y-axis metric: {args.y_metric}")
    print(f"  Log scale: {args.log_y}")
    print(f"  Layout: {'Combined' if args.combined else 'Separate panels'}")
    print(f"  Show frontier: {args.show_pareto}")

    fig = plot_pareto_frontier(df, args)

    # Save
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=args.dpi, bbox_inches='tight')
    print(f"\nPlot saved to: {output_path}")

    # Also save a high-res version for papers
    if args.dpi < 600:
        highres_path = output_path.with_stem(f"{output_path.stem}_highres")
        fig.savefig(highres_path, dpi=600, bbox_inches='tight')
        print(f"High-res version saved to: {highres_path}")


if __name__ == '__main__':
    main()
