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
    parser.add_argument('--anp_input', type=str, default=None,
                        help='Path to ANP statistics.csv from run_training_harness.py (optional)')
    parser.add_argument('--output', type=str, default='pareto_frontier.png',
                        help='Output plot filename')
    parser.add_argument('--x_metric', type=str, default='log_rmse',
                        choices=['log_rmse', 'log_r2', 'log_mae', 'linear_rmse', 'linear_mae'],
                        help='X-axis metric: log_rmse (default), log_r2, log_mae, linear_rmse, linear_mae')
    parser.add_argument('--y_metric', type=str, default='z_std',
                        choices=['z_std', 'calibration_error', 'coverage_1sigma', 'log_z_std'],
                        help='Y-axis metric: z_std (default), log_z_std, calibration_error, or coverage_1sigma')
    parser.add_argument('--log_y', action='store_true',
                        help='Use log scale for y-axis (auto-enabled for z_std if range > 10x)')
    parser.add_argument('--combined', action='store_true',
                        help='Single panel with all models (default: separate panels for RF and XGB)')
    parser.add_argument('--show_pareto', action='store_true',
                        help='Draw Pareto frontier line connecting optimal points')
    parser.add_argument('--encode_time', type=str, default=None,
                        choices=['size', 'color', 'both'],
                        help='Encode training time as point size, color, or both')
    parser.add_argument('--plot_type', type=str, default='calibration',
                        choices=['calibration', 'efficiency', 'both'],
                        help='Plot type: calibration (accuracy vs calibration), efficiency (accuracy vs time), or both')
    parser.add_argument('--dpi', type=int, default=300,
                        help='Output DPI (default: 300)')
    return parser.parse_args()


def load_anp_results(anp_csv_path):
    """
    Load ANP results from run_training_harness.py statistics.csv and convert to baseline format.

    The statistics.csv has columns like:
    - test_log_rmse_mean, test_log_rmse_std
    - test_z_std_mean, test_z_std_std
    - test_coverage_1sigma_mean, etc.

    We need to convert to match baseline sweep format:
    - log_rmse_mean, log_rmse_std
    - z_std_mean, z_std_std
    - etc.
    """
    anp_df = pd.read_csv(anp_csv_path)

    # Map test_ prefixed columns to non-prefixed columns
    column_mapping = {}
    for col in anp_df.columns:
        if col.startswith('test_'):
            new_col = col.replace('test_', '')
            column_mapping[col] = new_col

    # Rename columns
    anp_df = anp_df.rename(columns=column_mapping)

    # Add model_type column
    anp_df['model_type'] = 'anp'

    # Add dummy config columns (ANP doesn't have hyperparameter sweep)
    anp_df['config_max_depth'] = np.nan
    anp_df['config_n_estimators'] = np.nan
    anp_df['config_learning_rate'] = np.nan

    # Check if train_time is available, if not use NaN
    if 'train_time_mean' not in anp_df.columns:
        anp_df['train_time_mean'] = np.nan
        anp_df['train_time_std'] = np.nan

    # Compute calibration_error if not present
    if 'calibration_error_mean' not in anp_df.columns and 'z_std_mean' in anp_df.columns:
        anp_df['calibration_error_mean'] = np.abs(anp_df['z_std_mean'] - 1.0)
        if 'z_std_std' in anp_df.columns:
            anp_df['calibration_error_std'] = anp_df['z_std_std']

    # Select only columns that match baseline format
    baseline_cols = [
        'model_type', 'config_max_depth', 'config_n_estimators', 'config_learning_rate',
        'train_time_mean', 'train_time_std',
        'log_rmse_mean', 'log_rmse_std',
        'log_mae_mean', 'log_mae_std',
        'log_r2_mean', 'log_r2_std',
        'z_mean_mean', 'z_mean_std',
        'z_std_mean', 'z_std_std',
        'calibration_error_mean', 'calibration_error_std',
        'coverage_1sigma_mean', 'coverage_1sigma_std',
        'coverage_2sigma_mean', 'coverage_2sigma_std',
        'coverage_3sigma_mean', 'coverage_3sigma_std',
    ]

    # Keep only columns that exist in both formats
    existing_cols = [col for col in baseline_cols if col in anp_df.columns]
    anp_df = anp_df[existing_cols]

    return anp_df


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

    # Setup x-axis metric and labels
    x_col = f'{args.x_metric}_mean'
    x_label_map = {
        'log_rmse': 'Log RMSE (lower = better)',
        'log_r2': 'Log R² (higher = better)',
        'log_mae': 'Log MAE (lower = better)',
        'linear_rmse': 'Linear RMSE (Mg/ha, lower = better)',
        'linear_mae': 'Linear MAE (Mg/ha, lower = better)',
    }
    x_label = x_label_map[args.x_metric]

    # Determine if we minimize or maximize x-axis
    x_minimize = args.x_metric != 'log_r2'  # R² is maximized, others minimized

    # Setup y-axis metric and labels
    if args.y_metric == 'log_z_std':
        # Compute log(z_std) on the fly
        df = df.copy()
        df['log_z_std_mean'] = np.log(df['z_std_mean'])
        y_col = 'log_z_std_mean'
        y_label = 'log(Z-Score Std Dev)'
        y_reference_val = 0.0  # log(1.0) = 0
    else:
        y_col = f'{args.y_metric}_mean'
        y_label_map = {
            'z_std': 'Z-Score Std Dev (Calibration Quality)',
            'calibration_error': 'Calibration Error |z_std - 1.0|',
            'coverage_1sigma': '1-Sigma Coverage (%)'
        }
        y_label = y_label_map[args.y_metric]
        y_reference = {
            'z_std': 1.0,
            'calibration_error': 0.0,
            'coverage_1sigma': 68.3
        }
        y_reference_val = y_reference.get(args.y_metric)

    # Auto-enable log scale if z_std range is wide
    use_log_y = args.log_y
    if args.y_metric == 'z_std' and not args.log_y:
        z_std_range = df['z_std_mean'].max() / df['z_std_mean'].min()
        if z_std_range > 10:
            use_log_y = True
            print(f"Auto-enabling log scale for y-axis (z_std range: {z_std_range:.1f}x)")

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
    has_anp = 'anp' in models_present

    # Determine whether we need calibration error minimization or z_std/coverage optimization
    minimize_both = args.y_metric in ['calibration_error']

    # Create figure
    if args.combined:
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        axes = [ax]
        model_groups = [models_present]
        titles = ['All Models']
    else:
        # Separate panels for RF and XGB (ANP will be overlaid on both)
        baseline_models = [m for m in ['rf', 'xgb'] if m in models_present]
        n_panels = len(baseline_models)
        if n_panels == 0:
            n_panels = 1
            baseline_models = ['anp'] if has_anp else []
        fig, axes = plt.subplots(1, n_panels, figsize=(6*n_panels, 5), squeeze=False)
        axes = axes.flatten()
        # For separate panels, show baseline model + ANP overlay on each
        model_groups = [[m, 'anp'] if has_anp and m != 'anp' else [m] for m in baseline_models]
        titles = [model_labels.get(m, m) for m in baseline_models]

    # Plot each panel
    for ax, models, title in zip(axes, model_groups, titles):
        for model in models:
            model_df = df[df['model_type'] == model].copy()

            if len(model_df) == 0:
                continue

            # ANP gets special marker and doesn't encode training time
            is_anp = (model == 'anp')
            marker = 'D' if is_anp else 'o'  # Diamond for ANP, circle for baselines
            base_size = 150 if is_anp else 50  # Larger for ANP to stand out

            # Setup point sizes and colors based on training time encoding
            # Skip time encoding for ANP if it doesn't have training time data
            has_train_time = not model_df['train_time_mean'].isna().all()

            if args.encode_time in ['size', 'both'] and has_train_time and not is_anp:
                # Scale point size by training time (log scale for better visibility)
                time_norm = np.log10(model_df['train_time_mean'] + 1)
                time_norm = (time_norm - time_norm.min()) / (time_norm.max() - time_norm.min() + 1e-8)
                sizes = 20 + time_norm * 200  # Range: 20-220
            else:
                sizes = base_size

            if args.encode_time in ['color', 'both'] and has_train_time and not is_anp:
                # Use colormap for training time
                import matplotlib.cm as cm
                time_values = model_df['train_time_mean']
                colors = cm.YlOrRd(
                    (np.log10(time_values + 1) - np.log10(time_values.min() + 1)) /
                    (np.log10(time_values.max() + 1) - np.log10(time_values.min() + 1) + 1e-8)
                )
            else:
                colors = model_colors.get(model, '#666666')

            # Plot all points
            scatter = ax.scatter(
                model_df[x_col],
                model_df[y_col],
                c=colors,
                marker=marker,
                label=model_labels.get(model, model),
                alpha=0.7 if is_anp else 0.6,
                s=sizes,
                edgecolors='black' if is_anp else 'white',
                linewidths=1.5 if is_anp else 0.5,
                zorder=10 if is_anp else 5  # ANP on top
            )

            # Plot Pareto frontier (skip for ANP since it's usually a single point)
            if args.show_pareto and not is_anp and len(model_df) > 1:
                # For R², we want to maximize, so negate for Pareto computation
                if not x_minimize:
                    model_df_pareto = model_df.copy()
                    model_df_pareto[x_col] = -model_df_pareto[x_col]
                    pareto_df = compute_pareto_frontier(
                        model_df_pareto,
                        x_col,
                        y_col,
                        minimize_both=minimize_both
                    )
                    # Negate back for plotting
                    pareto_df[x_col] = -pareto_df[x_col]
                else:
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
        if y_reference_val is not None:
            ax.axhline(
                y_reference_val,
                color='black',
                linestyle=':',
                linewidth=1.5,
                alpha=0.5,
                label=f'Ideal: {y_reference_val}'
            )

        # Labels and formatting
        ax.set_xlabel(x_label, fontsize=11, fontweight='bold')
        ax.set_ylabel(y_label, fontsize=11, fontweight='bold')
        ax.set_title(title, fontsize=13, fontweight='bold', pad=10)

        if use_log_y:
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


def plot_efficiency_frontier(df, args):
    """Create efficiency frontier visualization (accuracy vs training time)."""

    x_col = 'train_time_mean'
    y_col = 'log_rmse_mean'

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

            # Plot Pareto frontier (minimize time, minimize RMSE)
            if args.show_pareto:
                pareto_df = compute_pareto_frontier(
                    model_df,
                    x_col,
                    y_col,
                    minimize_both=True
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

        # Labels and formatting
        ax.set_xlabel('Training Time (seconds)', fontsize=11, fontweight='bold')
        ax.set_ylabel('Log RMSE (lower = better)', fontsize=11, fontweight='bold')
        ax.set_title(title, fontsize=13, fontweight='bold', pad=10)
        ax.set_xscale('log')  # Log scale for training time

        ax.legend(loc='best', framealpha=0.9)
        ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    # Overall title
    fig.suptitle(
        'Efficiency Frontier: Accuracy vs Training Time',
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
        is_anp = (model == 'anp')

        print(f"\n{model.upper()} (n={len(model_df)} {'result' if is_anp else 'configs'}):")
        print(f"  Log RMSE:          {model_df['log_rmse_mean'].min():.4f} - {model_df['log_rmse_mean'].max():.4f}")
        print(f"  Z-Score Std:       {model_df['z_std_mean'].min():.4f} - {model_df['z_std_mean'].max():.4f}")
        if 'calibration_error_mean' in model_df.columns:
            print(f"  Calibration Error: {model_df['calibration_error_mean'].min():.4f} - {model_df['calibration_error_mean'].max():.4f}")
        if 'coverage_1sigma_mean' in model_df.columns:
            print(f"  Coverage 1-sigma:  {model_df['coverage_1sigma_mean'].min():.1f}% - {model_df['coverage_1sigma_mean'].max():.1f}%")
        if not model_df['train_time_mean'].isna().all():
            print(f"  Train Time (s):    {model_df['train_time_mean'].min():.2f} - {model_df['train_time_mean'].max():.2f}")

        # Skip best config search for ANP (single point) or if insufficient data
        if is_anp or len(model_df) <= 1:
            continue

        # Best calibration (z_std closest to 1.0)
        best_calib_idx = (model_df['z_std_mean'] - 1.0).abs().idxmin()
        best_calib = model_df.loc[best_calib_idx]
        print(f"\n  Best Calibration Config:")
        if not pd.isna(best_calib['config_max_depth']):
            print(f"    max_depth={int(best_calib['config_max_depth'])}, n_estimators={int(best_calib['config_n_estimators'])}")
        print(f"    Log RMSE: {best_calib['log_rmse_mean']:.4f}")
        print(f"    Z-Score Std: {best_calib['z_std_mean']:.4f}")

        # Best accuracy
        best_acc_idx = model_df['log_rmse_mean'].idxmin()
        best_acc = model_df.loc[best_acc_idx]
        print(f"\n  Best Accuracy Config:")
        if not pd.isna(best_acc['config_max_depth']):
            print(f"    max_depth={int(best_acc['config_max_depth'])}, n_estimators={int(best_acc['config_n_estimators'])}")
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

    print(f"Loaded {len(df)} baseline configurations")

    # Load ANP results if provided
    if args.anp_input:
        anp_path = Path(args.anp_input)
        if not anp_path.exists():
            print(f"Warning: ANP input file not found: {anp_path}")
            print("Continuing with baselines only...")
        else:
            print(f"Loading ANP results from: {anp_path}")
            anp_df = load_anp_results(anp_path)
            print(f"Loaded {len(anp_df)} ANP results")

            # Combine baseline and ANP results
            df = pd.concat([df, anp_df], ignore_index=True)
            print(f"Combined total: {len(df)} configurations")

    print(f"Models: {', '.join(df['model_type'].unique())}")

    # Print summary statistics
    print_summary_statistics(df)

    # Create plot(s)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if args.plot_type == 'calibration':
        print(f"\nGenerating calibration Pareto frontier plot...")
        print(f"  X-axis metric: {args.x_metric}")
        print(f"  Y-axis metric: {args.y_metric}")
        print(f"  Layout: {'Combined' if args.combined else 'Separate panels'}")
        print(f"  Show frontier: {args.show_pareto}")
        print(f"  Encode time: {args.encode_time}")

        fig = plot_pareto_frontier(df, args)
        fig.savefig(output_path, dpi=args.dpi, bbox_inches='tight')
        print(f"\nPlot saved to: {output_path}")

        if args.dpi < 600:
            highres_path = output_path.with_stem(f"{output_path.stem}_highres")
            fig.savefig(highres_path, dpi=600, bbox_inches='tight')
            print(f"High-res version saved to: {highres_path}")

    elif args.plot_type == 'efficiency':
        print(f"\nGenerating efficiency frontier plot...")
        print(f"  Layout: {'Combined' if args.combined else 'Separate panels'}")
        print(f"  Show frontier: {args.show_pareto}")

        fig = plot_efficiency_frontier(df, args)
        fig.savefig(output_path, dpi=args.dpi, bbox_inches='tight')
        print(f"\nPlot saved to: {output_path}")

        if args.dpi < 600:
            highres_path = output_path.with_stem(f"{output_path.stem}_highres")
            fig.savefig(highres_path, dpi=600, bbox_inches='tight')
            print(f"High-res version saved to: {highres_path}")

    else:  # both
        print(f"\nGenerating both calibration and efficiency frontier plots...")

        # Calibration plot
        print(f"\n  Calibration plot:")
        print(f"    X-axis metric: {args.x_metric}")
        print(f"    Y-axis metric: {args.y_metric}")
        print(f"    Encode time: {args.encode_time}")
        fig1 = plot_pareto_frontier(df, args)
        calib_path = output_path.with_stem(f"{output_path.stem}_calibration")
        fig1.savefig(calib_path, dpi=args.dpi, bbox_inches='tight')
        print(f"    Saved to: {calib_path}")

        # Efficiency plot
        print(f"\n  Efficiency plot:")
        fig2 = plot_efficiency_frontier(df, args)
        eff_path = output_path.with_stem(f"{output_path.stem}_efficiency")
        fig2.savefig(eff_path, dpi=args.dpi, bbox_inches='tight')
        print(f"    Saved to: {eff_path}")

        if args.dpi < 600:
            fig1.savefig(calib_path.with_stem(f"{calib_path.stem}_highres"), dpi=600, bbox_inches='tight')
            fig2.savefig(eff_path.with_stem(f"{eff_path.stem}_highres"), dpi=600, bbox_inches='tight')
            print(f"\nHigh-res versions saved")


if __name__ == '__main__':
    main()
