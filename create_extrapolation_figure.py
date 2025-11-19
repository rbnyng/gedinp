#!/usr/bin/env python3
"""
Create publication-quality figure for spatial extrapolation results.

Generates a 3-panel figure:
- Panel A: R² comparison (ANP vs XGBoost)
- Panel B: Coverage comparison (THE KEY RESULT)
- Panel C: Degradation metrics bar chart

Usage:
    python create_extrapolation_figure.py --results ./extrapolation_results
    python create_extrapolation_figure.py --results ./extrapolation_results --output figure_spatial_extrapolation.pdf
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import seaborn as sns

# Publication settings
plt.rcParams.update({
    'font.size': 10,
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'DejaVu Sans'],
    'axes.labelsize': 10,
    'axes.titlesize': 11,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.titlesize': 12,
    'pdf.fonttype': 42,  # TrueType fonts for editability
    'ps.fonttype': 42,
})

# Region definitions
REGIONS = {
    'maine': 'Maine\n(Temperate)',
    'tolima': 'Tolima\n(Tropical)',
    'hokkaido': 'Hokkaido\n(Boreal)',
    'sudtirol': 'Sudtirol\n(Alpine)'
}

REGION_ORDER = ['maine', 'tolima', 'hokkaido', 'sudtirol']
REGION_LABELS = [REGIONS[r] for r in REGION_ORDER]


def create_heatmap_axis(
    ax, matrix, title, cmap, vmin, vmax, cbar_label,
    annotate=True, fmt='.2f', highlight_diagonal=True
):
    """Create a single heatmap with consistent styling."""
    # Create heatmap
    im = ax.imshow(matrix, cmap=cmap, vmin=vmin, vmax=vmax, aspect='auto')

    # Add annotations
    if annotate:
        for i in range(len(REGION_ORDER)):
            for j in range(len(REGION_ORDER)):
                value = matrix[i, j]

                # Bold diagonal values
                if i == j and highlight_diagonal:
                    weight = 'bold'
                    color = 'black'
                else:
                    weight = 'normal'
                    # Choose text color based on background
                    color = 'white' if value < (vmin + vmax) / 2 else 'black'

                text = ax.text(
                    j, i, f'{value:{fmt}}',
                    ha='center', va='center',
                    color=color, fontweight=weight,
                    fontsize=9
                )

    # Set ticks and labels
    ax.set_xticks(np.arange(len(REGION_ORDER)))
    ax.set_yticks(np.arange(len(REGION_ORDER)))
    ax.set_xticklabels(REGION_LABELS, fontsize=9)
    ax.set_yticklabels(REGION_LABELS, fontsize=9)

    # Rotate x labels
    plt.setp(ax.get_xticklabels(), rotation=0, ha='center')

    # Grid
    ax.set_xticks(np.arange(len(REGION_ORDER)) - 0.5, minor=True)
    ax.set_yticks(np.arange(len(REGION_ORDER)) - 0.5, minor=True)
    ax.grid(which='minor', color='white', linestyle='-', linewidth=1.5)

    # Labels
    ax.set_xlabel('Test Region', fontsize=10, fontweight='bold')
    if ax.get_subplotspec().colspan.start == 0:  # Only leftmost panel
        ax.set_ylabel('Train Region', fontsize=10, fontweight='bold')

    # Title
    ax.set_title(title, fontsize=11, fontweight='bold', pad=10)

    # Colorbar
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(cbar_label, fontsize=9)
    cbar.ax.tick_params(labelsize=8)

    return im


def create_publication_figure(results_dir: Path, output_path: Path):
    """Create 3-panel publication figure."""
    # Load results
    df = pd.read_csv(results_dir / 'spatial_extrapolation_results.csv')
    df_degradation = pd.read_csv(results_dir / 'degradation_metrics.csv')

    # Create figure with 3 panels
    fig = plt.figure(figsize=(14, 4.5))

    # Use gridspec for better control
    gs = fig.add_gridspec(1, 3, wspace=0.4, left=0.06, right=0.98, bottom=0.15, top=0.88)

    # Panel A: R² comparison (ANP vs XGBoost)
    ax_r2_anp = fig.add_subplot(gs[0, 0])
    ax_r2_xgb = fig.add_subplot(gs[0, 0])  # Will create separate subplots below

    # Actually, let's use subplots within the gridspec
    # Revised: Create 2 rows of heatmaps in columns 1 and 2, and bar chart in column 3

    # Better layout: 2x3 grid
    fig = plt.figure(figsize=(15, 8))
    gs = fig.add_gridspec(2, 3, wspace=0.35, hspace=0.35, left=0.08, right=0.95, bottom=0.1, top=0.92)

    # --- Panel A: R² heatmaps (top left, top middle) ---

    # ANP R²
    ax_r2_anp = fig.add_subplot(gs[0, 0])
    df_anp = df[df['model_type'] == 'anp']
    matrix_r2_anp = df_anp.pivot(
        index='train_region', columns='test_region', values='log_r2'
    ).reindex(index=REGION_ORDER, columns=REGION_ORDER).values

    create_heatmap_axis(
        ax_r2_anp, matrix_r2_anp,
        title='ANP: R²',
        cmap='RdYlGn', vmin=0.0, vmax=1.0,
        cbar_label='R²', fmt='.2f'
    )
    ax_r2_anp.text(
        -0.15, 0.5, 'A', transform=ax_r2_anp.transAxes,
        fontsize=16, fontweight='bold', va='center'
    )

    # XGBoost R²
    ax_r2_xgb = fig.add_subplot(gs[0, 1])
    df_xgb = df[df['model_type'] == 'xgboost']
    matrix_r2_xgb = df_xgb.pivot(
        index='train_region', columns='test_region', values='log_r2'
    ).reindex(index=REGION_ORDER, columns=REGION_ORDER).values

    create_heatmap_axis(
        ax_r2_xgb, matrix_r2_xgb,
        title='XGBoost: R²',
        cmap='RdYlGn', vmin=0.0, vmax=1.0,
        cbar_label='R²', fmt='.2f', highlight_diagonal=True
    )

    # --- Panel B: Coverage heatmaps (bottom left, bottom middle) ---

    # ANP Coverage
    ax_cov_anp = fig.add_subplot(gs[1, 0])
    matrix_cov_anp = df_anp.pivot(
        index='train_region', columns='test_region', values='coverage_1sigma'
    ).reindex(index=REGION_ORDER, columns=REGION_ORDER).values

    create_heatmap_axis(
        ax_cov_anp, matrix_cov_anp,
        title='ANP: 1σ Coverage',
        cmap='RdYlGn', vmin=0.0, vmax=1.0,
        cbar_label='Coverage', fmt='.2f'
    )
    ax_cov_anp.text(
        -0.15, 0.5, 'B', transform=ax_cov_anp.transAxes,
        fontsize=16, fontweight='bold', va='center'
    )

    # Add ideal coverage reference line
    ax_cov_anp.text(
        0.5, -0.15, 'Ideal: 0.68', transform=ax_cov_anp.transAxes,
        ha='center', fontsize=9, style='italic', color='darkblue'
    )

    # XGBoost Coverage (THE KEY RESULT!)
    ax_cov_xgb = fig.add_subplot(gs[1, 1])
    matrix_cov_xgb = df_xgb.pivot(
        index='train_region', columns='test_region', values='coverage_1sigma'
    ).reindex(index=REGION_ORDER, columns=REGION_ORDER).values

    create_heatmap_axis(
        ax_cov_xgb, matrix_cov_xgb,
        title='XGBoost: 1σ Coverage',
        cmap='RdYlGn', vmin=0.0, vmax=1.0,
        cbar_label='Coverage', fmt='.2f', highlight_diagonal=True
    )

    ax_cov_xgb.text(
        0.5, -0.15, 'Ideal: 0.68', transform=ax_cov_xgb.transAxes,
        ha='center', fontsize=9, style='italic', color='darkblue'
    )

    # --- Panel C: Degradation metrics (right side, spans both rows) ---

    ax_deg = fig.add_subplot(gs[:, 2])

    # Prepare data for grouped bar chart
    metrics = ['R²', 'Coverage']
    x = np.arange(len(metrics))
    width = 0.35

    # Extract degradation percentages (convert to positive for display)
    anp_r2_drop = abs(df_degradation[df_degradation['model_type'] == 'anp']['r2_drop_pct'].values[0])
    xgb_r2_drop = abs(df_degradation[df_degradation['model_type'] == 'xgboost']['r2_drop_pct'].values[0])
    anp_cov_drop = abs(df_degradation[df_degradation['model_type'] == 'anp']['coverage_drop_pct'].values[0])
    xgb_cov_drop = abs(df_degradation[df_degradation['model_type'] == 'xgboost']['coverage_drop_pct'].values[0])

    anp_values = [anp_r2_drop, anp_cov_drop]
    xgb_values = [xgb_r2_drop, xgb_cov_drop]

    # Create bars
    bars1 = ax_deg.bar(x - width/2, anp_values, width, label='ANP',
                       color='#2ecc71', edgecolor='black', linewidth=1.2)
    bars2 = ax_deg.bar(x + width/2, xgb_values, width, label='XGBoost',
                       color='#e74c3c', edgecolor='black', linewidth=1.2)

    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax_deg.text(
                bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}%',
                ha='center', va='bottom', fontsize=9, fontweight='bold'
            )

    # Styling
    ax_deg.set_ylabel('Performance Drop\n(In-Dist → Out-of-Dist) [%]',
                      fontsize=10, fontweight='bold')
    ax_deg.set_xlabel('Metric', fontsize=10, fontweight='bold')
    ax_deg.set_title('Degradation on Out-of-Distribution Data',
                     fontsize=11, fontweight='bold', pad=10)
    ax_deg.set_xticks(x)
    ax_deg.set_xticklabels(metrics, fontsize=10)
    ax_deg.legend(loc='upper right', frameon=True, fontsize=10)
    ax_deg.grid(axis='y', alpha=0.3, linestyle='--')
    ax_deg.set_axisbelow(True)

    # Add panel label
    ax_deg.text(
        -0.15, 0.5, 'C', transform=ax_deg.transAxes,
        fontsize=16, fontweight='bold', va='center'
    )

    # Add interpretation box
    textstr = (
        'Key Finding:\n'
        f'ANP coverage drops by {anp_cov_drop:.1f}%\n'
        f'XGBoost coverage drops by {xgb_cov_drop:.1f}%\n\n'
        'ANP recognizes OOD contexts and\n'
        'increases epistemic uncertainty,\n'
        'maintaining honest UQ.'
    )
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.3)
    ax_deg.text(
        0.5, 0.3, textstr, transform=ax_deg.transAxes,
        fontsize=9, verticalalignment='top',
        bbox=props, ha='center'
    )

    # Overall title
    fig.suptitle(
        'Spatial Extrapolation: Uncertainty Quantification on Out-of-Distribution Ecosystems',
        fontsize=13, fontweight='bold', y=0.98
    )

    # Save figure
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"\nPublication figure saved to: {output_path}")

    # Also save as PDF for LaTeX
    pdf_path = output_path.with_suffix('.pdf')
    plt.savefig(pdf_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"PDF version saved to: {pdf_path}")

    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description='Create publication-quality spatial extrapolation figure'
    )
    parser.add_argument(
        '--results',
        type=str,
        default='./extrapolation_results',
        help='Directory containing spatial extrapolation results'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='./figure_spatial_extrapolation.png',
        help='Output file path (PNG or PDF)'
    )

    args = parser.parse_args()

    results_dir = Path(args.results)
    output_path = Path(args.output)

    # Verify inputs
    if not results_dir.exists():
        print(f"Error: Results directory not found: {results_dir}")
        print("\nRun evaluation first:")
        print("  python evaluate_spatial_extrapolation.py --results_dir ./regional_results")
        return

    if not (results_dir / 'spatial_extrapolation_results.csv').exists():
        print(f"Error: Results CSV not found in {results_dir}")
        print("\nRun evaluation first:")
        print("  python evaluate_spatial_extrapolation.py --results_dir ./regional_results")
        return

    # Create figure
    print("\nCreating publication figure...")
    create_publication_figure(results_dir, output_path)
    print("\nDone!")


if __name__ == '__main__':
    main()
