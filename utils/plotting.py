"""Plotting utilities for model evaluation."""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Optional, Union


def plot_results(
    predictions: np.ndarray,
    targets: np.ndarray,
    uncertainties: Optional[np.ndarray],
    output_dir: Union[str, Path],
    dataset_name: str = 'test',
    title: Optional[str] = None,
    filename_prefix: str = 'evaluation'
):
    """
    Create evaluation plots with 2x2 grid showing predictions, residuals, and uncertainty.

    Args:
        predictions: Model predictions
        targets: Ground truth values
        uncertainties: Optional predicted uncertainties
        output_dir: Directory to save plots
        dataset_name: Name of dataset for labeling
        title: Optional custom title (default: "Model Evaluation ({dataset_name.upper()} set)")
        filename_prefix: Prefix for output filename (default: "evaluation")
    """
    output_dir = Path(output_dir)

    # Default title if not provided
    if title is None:
        title = f'Model Evaluation ({dataset_name.upper()} set)'

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle(title, fontsize=16, fontweight='bold')

    # 1. Scatter plot with perfect prediction line
    ax = axes[0, 0]
    ax.scatter(targets, predictions, alpha=0.3, s=10)
    min_val = min(targets.min(), predictions.min())
    max_val = max(targets.max(), predictions.max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect prediction')
    ax.set_xlabel('True AGBD', fontweight='bold')
    ax.set_ylabel('Predicted AGBD', fontweight='bold')
    ax.set_title('Predictions vs Truth')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Add R² to plot
    ss_res = ((targets - predictions) ** 2).sum()
    ss_tot = ((targets - targets.mean()) ** 2).sum()
    r2 = 1 - ss_res / (ss_tot + 1e-8)
    ax.text(0.05, 0.95, f'R² = {r2:.4f}', transform=ax.transAxes,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # 2. Residual plot
    ax = axes[0, 1]
    residuals = predictions - targets
    ax.scatter(predictions, residuals, alpha=0.3, s=10)
    ax.axhline(y=0, color='r', linestyle='--', linewidth=2)
    ax.set_xlabel('Predicted AGBD', fontweight='bold')
    ax.set_ylabel('Residual (Pred - True)', fontweight='bold')
    ax.set_title('Residual Plot')
    ax.grid(True, alpha=0.3)

    # 3. Error distribution
    ax = axes[1, 0]
    ax.hist(residuals, bins=50, edgecolor='black', alpha=0.7)
    ax.axvline(x=0, color='r', linestyle='--', linewidth=2)
    ax.set_xlabel('Residual', fontweight='bold')
    ax.set_ylabel('Frequency', fontweight='bold')
    ax.set_title('Distribution of Residuals')
    ax.grid(True, alpha=0.3, axis='y')

    # Add RMSE and MAE
    rmse = np.sqrt(np.mean(residuals ** 2))
    mae = np.mean(np.abs(residuals))
    ax.text(0.05, 0.95, f'RMSE = {rmse:.4f}\nMAE = {mae:.4f}',
            transform=ax.transAxes, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # 4. Uncertainty calibration
    ax = axes[1, 1]
    if uncertainties is not None and uncertainties.std() > 0:
        # Sort by uncertainty
        sorted_indices = np.argsort(uncertainties)
        sorted_uncertainties = uncertainties[sorted_indices]
        sorted_errors = np.abs(residuals[sorted_indices])

        # Bin by uncertainty
        n_bins = 20
        bin_size = len(sorted_uncertainties) // n_bins
        bin_uncertainties = []
        bin_errors = []

        for i in range(n_bins):
            start_idx = i * bin_size
            end_idx = (i + 1) * bin_size if i < n_bins - 1 else len(sorted_uncertainties)
            bin_uncertainties.append(sorted_uncertainties[start_idx:end_idx].mean())
            bin_errors.append(sorted_errors[start_idx:end_idx].mean())

        ax.scatter(bin_uncertainties, bin_errors, s=50)
        min_val = min(min(bin_uncertainties), min(bin_errors))
        max_val = max(max(bin_uncertainties), max(bin_errors))
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect calibration')
        ax.set_xlabel('Predicted Uncertainty (σ)', fontweight='bold')
        ax.set_ylabel('Actual Error (|pred - true|)', fontweight='bold')
        ax.set_title('Uncertainty Calibration')
        ax.legend()
        ax.grid(True, alpha=0.3)
    else:
        ax.text(0.5, 0.5, 'No uncertainty predictions', ha='center', va='center',
                transform=ax.transAxes)
        ax.set_title('Uncertainty Calibration')

    plt.tight_layout()
    output_file = output_dir / f'{filename_prefix}_{dataset_name}.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved evaluation plot to: {output_file}")
