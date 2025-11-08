"""
Compare spatial vs temporal generalization performance.

This script creates visualizations comparing model performance on:
1. Spatial holdout (test set from spatial CV)
2. Temporal holdout (future years)
"""

import argparse
import json
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def load_results(model_dir, temporal_suffix):
    """Load spatial and temporal evaluation results."""

    model_dir = Path(model_dir)

    # Load spatial test results
    spatial_results_path = model_dir / 'test_results.json'
    if spatial_results_path.exists():
        with open(spatial_results_path, 'r') as f:
            spatial_results = json.load(f)
    else:
        spatial_results = None
        print(f"Warning: Spatial test results not found at {spatial_results_path}")

    # Load spatial predictions
    spatial_preds_path = model_dir / 'test_predictions.csv'
    if spatial_preds_path.exists():
        spatial_preds = pd.read_csv(spatial_preds_path)
    else:
        spatial_preds = None
        print(f"Warning: Spatial predictions not found at {spatial_preds_path}")

    # Load temporal results
    temporal_results_path = model_dir / f'temporal_results_{temporal_suffix}.json'
    if temporal_results_path.exists():
        with open(temporal_results_path, 'r') as f:
            temporal_results = json.load(f)
    else:
        temporal_results = None
        print(f"Warning: Temporal results not found at {temporal_results_path}")

    # Load temporal predictions
    temporal_preds_path = model_dir / f'temporal_predictions_{temporal_suffix}.csv'
    if temporal_preds_path.exists():
        temporal_preds = pd.read_csv(temporal_preds_path)
    else:
        temporal_preds = None
        print(f"Warning: Temporal predictions not found at {temporal_preds_path}")

    return spatial_results, spatial_preds, temporal_results, temporal_preds


def plot_comparison(spatial_results, spatial_preds, temporal_results, temporal_preds, output_path):
    """Create comprehensive comparison plots."""

    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

    # Title
    train_years = temporal_results.get('train_years', 'unknown')
    test_years = temporal_results.get('test_years', 'unknown')
    fig.suptitle(
        f'Spatial vs Temporal Generalization\nTrained on: {train_years} | Temporal test: {test_years}',
        fontsize=16, fontweight='bold'
    )

    # 1. Metrics comparison bar chart
    ax1 = fig.add_subplot(gs[0, :])
    metrics_to_compare = ['rmse', 'mae', 'r2']
    spatial_vals = [spatial_results['metrics'].get(m, 0) for m in metrics_to_compare]
    temporal_vals = [temporal_results['metrics'].get(m, 0) for m in metrics_to_compare]

    x = np.arange(len(metrics_to_compare))
    width = 0.35

    bars1 = ax1.bar(x - width/2, spatial_vals, width, label='Spatial Holdout', alpha=0.8)
    bars2 = ax1.bar(x + width/2, temporal_vals, width, label='Temporal Holdout', alpha=0.8)

    ax1.set_ylabel('Metric Value', fontweight='bold')
    ax1.set_title('Performance Metrics Comparison', fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels([m.upper() for m in metrics_to_compare])
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis='y')

    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax1.annotate(f'{height:.3f}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=9)

    # 2. Spatial predictions scatter
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.scatter(spatial_preds['true'], spatial_preds['predicted'], alpha=0.3, s=10, c='blue')
    min_val = min(spatial_preds['true'].min(), spatial_preds['predicted'].min())
    max_val = max(spatial_preds['true'].max(), spatial_preds['predicted'].max())
    ax2.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2)
    ax2.set_xlabel('True AGBD', fontweight='bold')
    ax2.set_ylabel('Predicted AGBD', fontweight='bold')
    ax2.set_title('Spatial Holdout', fontweight='bold')
    ax2.grid(True, alpha=0.3)

    # Add R² to plot
    r2_spatial = spatial_results['metrics'].get('r2', 0)
    ax2.text(0.05, 0.95, f'R² = {r2_spatial:.4f}', transform=ax2.transAxes,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))

    # 3. Temporal predictions scatter
    ax3 = fig.add_subplot(gs[1, 1])
    ax3.scatter(temporal_preds['true'], temporal_preds['predicted'], alpha=0.3, s=10, c='orange')
    min_val = min(temporal_preds['true'].min(), temporal_preds['predicted'].min())
    max_val = max(temporal_preds['true'].max(), temporal_preds['predicted'].max())
    ax3.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2)
    ax3.set_xlabel('True AGBD', fontweight='bold')
    ax3.set_ylabel('Predicted AGBD', fontweight='bold')
    ax3.set_title('Temporal Holdout', fontweight='bold')
    ax3.grid(True, alpha=0.3)

    # Add R² to plot
    r2_temporal = temporal_results['metrics'].get('r2', 0)
    ax3.text(0.05, 0.95, f'R² = {r2_temporal:.4f}', transform=ax3.transAxes,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.5))

    # 4. Combined scatter plot
    ax4 = fig.add_subplot(gs[1, 2])
    ax4.scatter(spatial_preds['true'], spatial_preds['predicted'],
               alpha=0.2, s=8, c='blue', label='Spatial')
    ax4.scatter(temporal_preds['true'], temporal_preds['predicted'],
               alpha=0.2, s=8, c='orange', label='Temporal')
    min_val = min(spatial_preds['true'].min(), temporal_preds['true'].min())
    max_val = max(spatial_preds['true'].max(), temporal_preds['true'].max())
    ax4.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect')
    ax4.set_xlabel('True AGBD', fontweight='bold')
    ax4.set_ylabel('Predicted AGBD', fontweight='bold')
    ax4.set_title('Combined View', fontweight='bold')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    # 5. Residual distributions
    ax5 = fig.add_subplot(gs[2, 0])
    spatial_residuals = spatial_preds['residual']
    temporal_residuals = temporal_preds['residual']

    ax5.hist(spatial_residuals, bins=50, alpha=0.6, label='Spatial', color='blue', edgecolor='black')
    ax5.hist(temporal_residuals, bins=50, alpha=0.6, label='Temporal', color='orange', edgecolor='black')
    ax5.axvline(x=0, color='r', linestyle='--', linewidth=2)
    ax5.set_xlabel('Residual (Pred - True)', fontweight='bold')
    ax5.set_ylabel('Frequency', fontweight='bold')
    ax5.set_title('Residual Distributions', fontweight='bold')
    ax5.legend()
    ax5.grid(True, alpha=0.3, axis='y')

    # 6. Uncertainty comparison
    ax6 = fig.add_subplot(gs[2, 1])
    if 'uncertainty' in spatial_preds.columns and 'uncertainty' in temporal_preds.columns:
        spatial_unc = spatial_preds['uncertainty']
        temporal_unc = temporal_preds['uncertainty']

        if spatial_unc.std() > 0 and temporal_unc.std() > 0:
            data = [spatial_unc, temporal_unc]
            bp = ax6.boxplot(data, labels=['Spatial', 'Temporal'], patch_artist=True)
            bp['boxes'][0].set_facecolor('lightblue')
            bp['boxes'][1].set_facecolor('lightyellow')
            ax6.set_ylabel('Predicted Uncertainty (σ)', fontweight='bold')
            ax6.set_title('Uncertainty Distributions', fontweight='bold')
            ax6.grid(True, alpha=0.3, axis='y')
        else:
            ax6.text(0.5, 0.5, 'No uncertainty variation', ha='center', va='center',
                    transform=ax6.transAxes)
    else:
        ax6.text(0.5, 0.5, 'No uncertainty predictions', ha='center', va='center',
                transform=ax6.transAxes)
    ax6.set_title('Uncertainty Distributions', fontweight='bold')

    # 7. Error statistics table
    ax7 = fig.add_subplot(gs[2, 2])
    ax7.axis('off')

    # Calculate additional statistics
    stats_data = [
        ['Metric', 'Spatial', 'Temporal', 'Δ (T-S)'],
        ['RMSE', f"{spatial_results['metrics']['rmse']:.4f}",
         f"{temporal_results['metrics']['rmse']:.4f}",
         f"{temporal_results['metrics']['rmse'] - spatial_results['metrics']['rmse']:.4f}"],
        ['MAE', f"{spatial_results['metrics']['mae']:.4f}",
         f"{temporal_results['metrics']['mae']:.4f}",
         f"{temporal_results['metrics']['mae'] - spatial_results['metrics']['mae']:.4f}"],
        ['R²', f"{spatial_results['metrics']['r2']:.4f}",
         f"{temporal_results['metrics']['r2']:.4f}",
         f"{temporal_results['metrics']['r2'] - spatial_results['metrics']['r2']:.4f}"],
        ['N samples', f"{len(spatial_preds)}",
         f"{len(temporal_preds)}",
         f"{len(temporal_preds) - len(spatial_preds)}"],
    ]

    table = ax7.table(cellText=stats_data, cellLoc='center', loc='center',
                     colWidths=[0.25, 0.25, 0.25, 0.25])
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2)

    # Style header row
    for i in range(4):
        table[(0, i)].set_facecolor('#4CAF50')
        table[(0, i)].set_text_props(weight='bold', color='white')

    # Color code the delta column
    for i in range(1, 4):  # Skip header and N samples row
        delta_val = float(stats_data[i][3])
        if i == 3:  # R² (higher is better)
            color = 'lightgreen' if delta_val >= 0 else 'lightcoral'
        else:  # RMSE, MAE (lower is better)
            color = 'lightcoral' if delta_val >= 0 else 'lightgreen'
        table[(i, 3)].set_facecolor(color)

    ax7.set_title('Performance Summary', fontweight='bold', pad=20)

    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved comparison plot to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Compare spatial vs temporal generalization performance'
    )

    parser.add_argument('--model_dir', type=str, required=True,
                        help='Directory containing model and results')
    parser.add_argument('--temporal_suffix', type=str, required=True,
                        help='Suffix for temporal results files (e.g., "years_2022_2023")')
    parser.add_argument('--output', type=str, default=None,
                        help='Output path for comparison plot')

    args = parser.parse_args()

    model_dir = Path(args.model_dir)

    if args.output is None:
        output_path = model_dir / f'spatial_vs_temporal_{args.temporal_suffix}.png'
    else:
        output_path = Path(args.output)

    print("=" * 80)
    print("COMPARING SPATIAL VS TEMPORAL GENERALIZATION")
    print("=" * 80)
    print(f"Model directory: {model_dir}")
    print(f"Temporal suffix: {args.temporal_suffix}")
    print()

    # Load results
    print("Loading results...")
    spatial_results, spatial_preds, temporal_results, temporal_preds = load_results(
        model_dir, args.temporal_suffix
    )

    if spatial_results is None or temporal_results is None:
        print("Error: Could not load both spatial and temporal results.")
        return

    if spatial_preds is None or temporal_preds is None:
        print("Error: Could not load both spatial and temporal predictions.")
        return

    print(f"Spatial test: {len(spatial_preds)} predictions")
    print(f"Temporal test: {len(temporal_preds)} predictions")
    print()

    # Print comparison
    print("=" * 80)
    print("PERFORMANCE COMPARISON")
    print("=" * 80)
    print(f"{'Metric':<15} {'Spatial':<12} {'Temporal':<12} {'Δ (T-S)':<12}")
    print("-" * 80)

    for metric in ['rmse', 'mae', 'r2']:
        spatial_val = spatial_results['metrics'].get(metric, 0)
        temporal_val = temporal_results['metrics'].get(metric, 0)
        delta = temporal_val - spatial_val

        print(f"{metric.upper():<15} {spatial_val:<12.4f} {temporal_val:<12.4f} {delta:<12.4f}")

    print("=" * 80)

    # Generate comparison plot
    print("\nGenerating comparison plot...")
    plot_comparison(spatial_results, spatial_preds, temporal_results, temporal_preds, output_path)

    print("\n" + "=" * 80)
    print("COMPARISON COMPLETE")
    print("=" * 80)
    print(f"Comparison plot saved to: {output_path}")
    print("=" * 80)


if __name__ == '__main__':
    main()
