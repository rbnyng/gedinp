"""
Pareto Frontier Visualization for GEDI Baseline Models

This script creates publication-quality visualizations of the Pareto frontier
analysis, including:
- 2D Pareto frontier plots (Calibration Error vs Accuracy)
- Supporting plots (Time trade-offs)
- Detailed summary tables

Usage:
    python plot_pareto.py \
        --results_dir ./outputs_pareto \
        --anp_results ./outputs/results.json \
        --output_dir ./outputs_pareto/plots
"""

import argparse
import json
from pathlib import Path
from typing import List, Dict, Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.spatial import ConvexHull


def parse_args():
    parser = argparse.ArgumentParser(
        description='Generate Pareto Frontier visualizations'
    )

    parser.add_argument('--results_dir', type=str, required=True,
                        help='Directory containing pareto_results.json')
    parser.add_argument('--anp_results', type=str, default=None,
                        help='Optional: Path to ANP results JSON to include in plots')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Output directory for plots (default: results_dir/plots)')

    return parser.parse_args()


def compute_pareto_frontier(points: np.ndarray, minimize_x: bool = True, minimize_y: bool = False) -> np.ndarray:
    """
    Compute Pareto frontier from a set of 2D points.

    Args:
        points: (N, 2) array of points
        minimize_x: If True, minimize x-axis (e.g., calibration error)
        minimize_y: If True, minimize y-axis (if False, maximize, e.g., accuracy)

    Returns:
        Boolean mask indicating which points are on the Pareto frontier
    """
    # Flip signs as needed to convert to maximization problem
    points_transformed = points.copy()
    if minimize_x:
        points_transformed[:, 0] = -points_transformed[:, 0]
    if minimize_y:
        points_transformed[:, 1] = -points_transformed[:, 1]

    # Sort by first dimension
    sorted_indices = np.argsort(points_transformed[:, 0])[::-1]
    points_sorted = points_transformed[sorted_indices]

    # Find Pareto frontier
    pareto_mask = np.zeros(len(points), dtype=bool)
    max_y = -np.inf

    for i, idx in enumerate(sorted_indices):
        if points_sorted[i, 1] >= max_y:
            pareto_mask[idx] = True
            max_y = points_sorted[i, 1]

    return pareto_mask


def load_anp_results(anp_results_path: str) -> Optional[Dict]:
    """
    Load ANP results from a results JSON file.

    Returns:
        Dict with ANP metrics, or None if not found
    """
    if anp_results_path is None or not Path(anp_results_path).exists():
        return None

    with open(anp_results_path, 'r') as f:
        results = json.load(f)

    # Try to extract test metrics
    if 'test_metrics' in results:
        test_metrics = results['test_metrics']
    elif 'metrics' in results:
        test_metrics = results['metrics']
    else:
        print(f"Warning: Could not find test metrics in {anp_results_path}")
        return None

    # Extract training time if available
    train_time = results.get('train_time', None)

    anp_result = {
        'model_type': 'anp',
        'test_metrics': test_metrics,
        'train_time': train_time
    }

    return anp_result


def plot_pareto_frontier_2d(
    results: List[Dict],
    anp_result: Optional[Dict],
    output_path: Path,
    x_metric: str = 'calibration_error',
    y_metric: str = 'log_r2',
    size_metric: str = 'train_time',
    x_label: str = 'Calibration Error (|1 - Z-score Std|)',
    y_label: str = 'Test Log R²',
    title: str = 'Pareto Frontier: Accuracy vs Uncertainty Calibration'
):
    """
    Create a 2D Pareto frontier plot.

    Args:
        results: List of result dictionaries
        anp_result: Optional ANP result to include
        output_path: Path to save the plot
        x_metric: Metric for x-axis (default: calibration_error)
        y_metric: Metric for y-axis (default: log_r2)
        size_metric: Metric for point size (default: train_time)
    """
    fig, ax = plt.subplots(figsize=(12, 8))

    # Color scheme
    colors = {
        'rf': '#2ecc71',  # Green
        'xgb': '#e74c3c',  # Red
        'anp': '#3498db',  # Blue
    }

    model_names = {
        'rf': 'Random Forest',
        'xgb': 'XGBoost',
        'anp': 'ANP (Proposed)',
    }

    # Plot each model type
    for model_type in ['rf', 'xgb']:
        model_results = [r for r in results if r['model_type'] == model_type]
        if not model_results:
            continue

        # Extract metrics
        x_values = np.array([r['test_metrics'][x_metric] for r in model_results])
        y_values = np.array([r['test_metrics'][y_metric] for r in model_results])
        size_values = np.array([r[size_metric] for r in model_results])

        # Normalize sizes for visualization (20-200 point size)
        size_min, size_max = size_values.min(), size_values.max()
        if size_max > size_min:
            sizes = 20 + 180 * (size_values - size_min) / (size_max - size_min)
        else:
            sizes = np.full_like(size_values, 50)

        # Plot all points
        scatter = ax.scatter(
            x_values, y_values, s=sizes,
            c=colors[model_type], alpha=0.4,
            edgecolors='black', linewidths=0.5,
            label=model_names[model_type]
        )

        # Compute and plot Pareto frontier
        points = np.column_stack([x_values, y_values])
        pareto_mask = compute_pareto_frontier(points, minimize_x=True, minimize_y=False)

        if pareto_mask.sum() > 1:
            pareto_x = x_values[pareto_mask]
            pareto_y = y_values[pareto_mask]

            # Sort by x for plotting
            sort_idx = np.argsort(pareto_x)
            pareto_x = pareto_x[sort_idx]
            pareto_y = pareto_y[sort_idx]

            ax.plot(pareto_x, pareto_y, '--',
                   color=colors[model_type], linewidth=2, alpha=0.8)

            # Highlight Pareto-optimal points
            ax.scatter(pareto_x, pareto_y,
                      s=100, c=colors[model_type],
                      edgecolors='black', linewidths=2,
                      marker='*', zorder=10)

    # Add ANP result if available
    if anp_result is not None:
        anp_x = anp_result['test_metrics'][x_metric]
        anp_y = anp_result['test_metrics'][y_metric]

        ax.scatter(
            anp_x, anp_y, s=400,
            c=colors['anp'], alpha=1.0,
            edgecolors='black', linewidths=3,
            marker='D', zorder=15,
            label=model_names['anp']
        )

        # Add annotation
        ax.annotate(
            'ANP',
            xy=(anp_x, anp_y),
            xytext=(10, 10),
            textcoords='offset points',
            fontsize=12, fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='white', edgecolor='black', linewidth=2),
            arrowprops=dict(arrowstyle='->', lw=2)
        )

    # Formatting
    ax.set_xlabel(x_label, fontsize=14, fontweight='bold')
    ax.set_ylabel(y_label, fontsize=14, fontweight='bold')
    ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
    ax.legend(loc='best', fontsize=12, frameon=True, shadow=True)
    ax.grid(True, alpha=0.3)

    # Add ideal region annotation
    ax.axvline(x=0, color='gray', linestyle=':', linewidth=1, alpha=0.5)
    ax.text(0.02, 0.98, 'Ideal calibration →',
            transform=ax.transAxes, fontsize=10, va='top', ha='left',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def plot_time_tradeoffs(
    results: List[Dict],
    anp_result: Optional[Dict],
    output_dir: Path
):
    """
    Create plots showing trade-offs with training time.
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    colors = {
        'rf': '#2ecc71',
        'xgb': '#e74c3c',
        'anp': '#3498db',
    }

    model_names = {
        'rf': 'Random Forest',
        'xgb': 'XGBoost',
        'anp': 'ANP (Proposed)',
    }

    # Plot 1: Training Time vs Accuracy
    ax = axes[0]
    for model_type in ['rf', 'xgb']:
        model_results = [r for r in results if r['model_type'] == model_type]
        if not model_results:
            continue

        times = [r['train_time'] for r in model_results]
        accuracies = [r['test_metrics']['log_r2'] for r in model_results]

        ax.scatter(times, accuracies, s=80, c=colors[model_type],
                  alpha=0.6, edgecolors='black', linewidths=0.5,
                  label=model_names[model_type])

    if anp_result is not None and anp_result['train_time'] is not None:
        ax.scatter(
            anp_result['train_time'],
            anp_result['test_metrics']['log_r2'],
            s=300, c=colors['anp'], marker='D',
            edgecolors='black', linewidths=2,
            label=model_names['anp'], zorder=10
        )

    ax.set_xlabel('Training Time (seconds)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Test Log R²', fontsize=12, fontweight='bold')
    ax.set_title('Training Time vs Accuracy', fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xscale('log')

    # Plot 2: Training Time vs Calibration Error
    ax = axes[1]
    for model_type in ['rf', 'xgb']:
        model_results = [r for r in results if r['model_type'] == model_type]
        if not model_results:
            continue

        times = [r['train_time'] for r in model_results]
        cal_errors = [r['test_metrics']['calibration_error'] for r in model_results]

        ax.scatter(times, cal_errors, s=80, c=colors[model_type],
                  alpha=0.6, edgecolors='black', linewidths=0.5,
                  label=model_names[model_type])

    if anp_result is not None and anp_result['train_time'] is not None:
        ax.scatter(
            anp_result['train_time'],
            anp_result['test_metrics']['calibration_error'],
            s=300, c=colors['anp'], marker='D',
            edgecolors='black', linewidths=2,
            label=model_names['anp'], zorder=10
        )

    ax.set_xlabel('Training Time (seconds)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Calibration Error (|1 - Z-score Std|)', fontsize=12, fontweight='bold')
    ax.set_title('Training Time vs Calibration', fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xscale('log')

    plt.tight_layout()
    output_path = output_dir / 'pareto_time_tradeoffs.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def create_summary_table(
    results: List[Dict],
    anp_result: Optional[Dict],
    output_dir: Path
):
    """
    Create detailed summary tables in CSV and Markdown formats.
    """
    # Prepare data
    rows = []

    for result in results:
        config = result['config']
        metrics = result['test_metrics']

        row = {
            'Model': result['model_type'].upper(),
            'max_depth': config.get('max_depth', '-'),
            'n_estimators': config.get('n_estimators', '-'),
            'Test Log R²': f"{metrics['log_r2']:.4f}",
            'Z-score Std': f"{metrics['z_std']:.4f}",
            'Calibration Error': f"{metrics['calibration_error']:.4f}",
            '1σ Coverage (%)': f"{metrics['coverage_1sigma']:.1f}",
            'Train Time (s)': f"{result['train_time']:.2f}",
        }
        rows.append(row)

    # Add ANP result if available
    if anp_result is not None:
        metrics = anp_result['test_metrics']
        row = {
            'Model': 'ANP',
            'max_depth': '-',
            'n_estimators': '-',
            'Test Log R²': f"{metrics['log_r2']:.4f}",
            'Z-score Std': f"{metrics['z_std']:.4f}",
            'Calibration Error': f"{metrics['calibration_error']:.4f}",
            '1σ Coverage (%)': f"{metrics['coverage_1sigma']:.1f}",
            'Train Time (s)': f"{anp_result['train_time']:.2f}" if anp_result['train_time'] else '-',
        }
        rows.append(row)

    df = pd.DataFrame(rows)

    # Sort by model type and accuracy
    df = df.sort_values(['Model', 'Test Log R²'], ascending=[True, False])

    # Save CSV
    csv_path = output_dir / 'pareto_summary_table.csv'
    df.to_csv(csv_path, index=False)
    print(f"Saved: {csv_path}")

    # Save Markdown
    md_path = output_dir / 'pareto_summary_table.md'
    with open(md_path, 'w') as f:
        f.write("# Pareto Frontier Analysis: Summary Table\n\n")
        f.write(df.to_markdown(index=False))
        f.write("\n\n")
        f.write("**Legend:**\n")
        f.write("- **Test Log R²**: Accuracy in log-space (higher is better)\n")
        f.write("- **Z-score Std**: Calibration quality (ideal: 1.0)\n")
        f.write("- **Calibration Error**: |1 - Z-score Std| (lower is better)\n")
        f.write("- **1σ Coverage**: Empirical coverage at 1σ (ideal: 68.3%)\n")

    print(f"Saved: {md_path}")

    # Create best configurations table
    print("\n" + "=" * 80)
    print("BEST CONFIGURATIONS BY OBJECTIVE")
    print("=" * 80)

    for model_type in ['rf', 'xgb']:
        model_results = [r for r in results if r['model_type'] == model_type]
        if not model_results:
            continue

        print(f"\n{model_type.upper()}:")

        # Best accuracy
        best_acc = max(model_results, key=lambda r: r['test_metrics']['log_r2'])
        print(f"  Best Accuracy (Log R² = {best_acc['test_metrics']['log_r2']:.4f}):")
        print(f"    Config: {best_acc['config']}")
        print(f"    Calibration Error: {best_acc['test_metrics']['calibration_error']:.4f}")

        # Best calibration
        best_cal = min(model_results, key=lambda r: r['test_metrics']['calibration_error'])
        print(f"  Best Calibration (Error = {best_cal['test_metrics']['calibration_error']:.4f}):")
        print(f"    Config: {best_cal['config']}")
        print(f"    Log R²: {best_cal['test_metrics']['log_r2']:.4f}")

        # Fastest
        fastest = min(model_results, key=lambda r: r['train_time'])
        print(f"  Fastest (Time = {fastest['train_time']:.2f}s):")
        print(f"    Config: {fastest['config']}")
        print(f"    Log R²: {fastest['test_metrics']['log_r2']:.4f}")

    if anp_result is not None:
        print(f"\nANP:")
        print(f"  Log R²: {anp_result['test_metrics']['log_r2']:.4f}")
        print(f"  Calibration Error: {anp_result['test_metrics']['calibration_error']:.4f}")
        if anp_result['train_time']:
            print(f"  Training Time: {anp_result['train_time']:.2f}s")

    print("=" * 80)


def main():
    args = parse_args()

    # Setup paths
    results_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir) if args.output_dir else results_dir / 'plots'
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("PARETO FRONTIER VISUALIZATION")
    print("=" * 80)
    print(f"Results directory: {results_dir}")
    print(f"Output directory: {output_dir}")
    print()

    # Load results
    print("Loading results...")
    with open(results_dir / 'pareto_results.json', 'r') as f:
        results = json.load(f)

    print(f"Loaded {len(results)} configurations")

    # Load ANP results if provided
    anp_result = None
    if args.anp_results:
        print(f"Loading ANP results from: {args.anp_results}")
        anp_result = load_anp_results(args.anp_results)
        if anp_result:
            print("ANP results loaded successfully")
        else:
            print("Warning: Could not load ANP results")
    print()

    # Generate plots
    print("Generating visualizations...")
    print()

    # Main Pareto frontier plot
    plot_pareto_frontier_2d(
        results, anp_result,
        output_dir / 'pareto_frontier_accuracy_calibration.png'
    )

    # Time trade-off plots
    plot_time_tradeoffs(results, anp_result, output_dir)

    # Summary tables
    create_summary_table(results, anp_result, output_dir)

    print()
    print("=" * 80)
    print("VISUALIZATION COMPLETE")
    print("=" * 80)
    print(f"\nGenerated files:")
    print(f"  - {output_dir / 'pareto_frontier_accuracy_calibration.png'}")
    print(f"  - {output_dir / 'pareto_time_tradeoffs.png'}")
    print(f"  - {output_dir / 'pareto_summary_table.csv'}")
    print(f"  - {output_dir / 'pareto_summary_table.md'}")
    print("=" * 80)


if __name__ == '__main__':
    main()
