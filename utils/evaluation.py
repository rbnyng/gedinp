"""
Evaluation and visualization utilities for GEDI Neural Process models.

This module provides functions for:
- Evaluating models on datasets
- Computing metrics (RMSE, MAE, R²)
- Creating evaluation visualizations
"""

import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from pathlib import Path
from typing import Tuple, Dict, Optional, Union


def compute_metrics(
    pred: Union[np.ndarray, torch.Tensor],
    true: Union[np.ndarray, torch.Tensor],
    pred_std: Optional[Union[np.ndarray, torch.Tensor]] = None
) -> Dict[str, float]:
    """
    Compute evaluation metrics (RMSE, MAE, R²).

    Args:
        pred: Predicted values (numpy array or torch tensor)
        true: True values (numpy array or torch tensor)
        pred_std: Optional predicted standard deviations

    Returns:
        Dictionary with metrics: rmse, mae, r2, and optionally mean_uncertainty
    """
    # Convert to numpy if torch tensors
    if isinstance(pred, torch.Tensor):
        pred = pred.detach().cpu().numpy().flatten()
    if isinstance(true, torch.Tensor):
        true = true.detach().cpu().numpy().flatten()
    if pred_std is not None and isinstance(pred_std, torch.Tensor):
        pred_std = pred_std.detach().cpu().numpy().flatten()

    # Flatten arrays
    pred = pred.flatten()
    true = true.flatten()

    # RMSE
    rmse = np.sqrt(np.mean((pred - true) ** 2))

    # MAE
    mae = np.mean(np.abs(pred - true))

    # R²
    ss_res = np.sum((true - pred) ** 2)
    ss_tot = np.sum((true - np.mean(true)) ** 2)
    r2 = 1 - (ss_res / (ss_tot + 1e-8))

    metrics = {
        'rmse': rmse,
        'mae': mae,
        'r2': r2,
    }

    # Add uncertainty metric if available
    if pred_std is not None:
        if not isinstance(pred_std, np.ndarray):
            pred_std = np.array(pred_std)
        pred_std = pred_std.flatten()
        metrics['mean_uncertainty'] = pred_std.mean()

    return metrics


def evaluate_model(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    max_context_shots: int = 20000,
    max_targets_per_chunk: int = 1000
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, float]]:
    """
    Evaluate model on a dataset with memory-efficient chunking.

    Args:
        model: The model to evaluate
        dataloader: DataLoader for the dataset
        device: Device to run evaluation on
        max_context_shots: Maximum context shots to use (subsample if exceeded)
        max_targets_per_chunk: Maximum targets to process at once

    Returns:
        Tuple of (predictions, targets, uncertainties, avg_metrics)
    """
    model.eval()
    all_predictions = []
    all_targets = []
    all_uncertainties = []
    all_metrics = []

    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader, desc='Evaluating')):
            for i in range(len(batch['context_coords'])):
                context_coords = batch['context_coords'][i].to(device)
                context_embeddings = batch['context_embeddings'][i].to(device)
                context_agbd = batch['context_agbd'][i].to(device)
                target_coords = batch['target_coords'][i].to(device)
                target_embeddings = batch['target_embeddings'][i].to(device)
                target_agbd = batch['target_agbd'][i].to(device)

                if len(target_coords) == 0:
                    continue

                n_context = len(context_coords)
                n_targets = len(target_coords)

                # Subsample context if too large to avoid OOM in attention
                if n_context > max_context_shots:
                    if batch_idx == 0 and i == 0:  # Only print once
                        tqdm.write(f"Note: Subsampling context from {n_context} to {max_context_shots} shots for memory efficiency")
                    indices = torch.randperm(n_context)[:max_context_shots]
                    context_coords = context_coords[indices]
                    context_embeddings = context_embeddings[indices]
                    context_agbd = context_agbd[indices]
                    n_context = max_context_shots

                # Process targets in chunks
                tile_predictions = []
                tile_targets = []
                tile_uncertainties = []

                for chunk_start in range(0, n_targets, max_targets_per_chunk):
                    chunk_end = min(chunk_start + max_targets_per_chunk, n_targets)

                    chunk_target_coords = target_coords[chunk_start:chunk_end]
                    chunk_target_embeddings = target_embeddings[chunk_start:chunk_end]
                    chunk_target_agbd = target_agbd[chunk_start:chunk_end]

                    # Forward pass on chunk
                    pred_mean, pred_log_var, _, _ = model(
                        context_coords,
                        context_embeddings,
                        context_agbd,
                        chunk_target_coords,
                        chunk_target_embeddings,
                        training=False
                    )

                    tile_predictions.append(pred_mean.detach().cpu().numpy().flatten())
                    tile_targets.append(chunk_target_agbd.detach().cpu().numpy().flatten())

                    if pred_log_var is not None:
                        tile_uncertainties.append(
                            torch.exp(0.5 * pred_log_var).detach().cpu().numpy().flatten()
                        )
                    else:
                        tile_uncertainties.append(np.zeros_like(pred_mean.detach().cpu().numpy().flatten()))

                    # Clear GPU cache after each chunk
                    device_str = str(device) if not isinstance(device, str) else device
                    if 'cuda' in device_str:
                        torch.cuda.empty_cache()

                # Concatenate chunks
                pred_mean_np = np.concatenate(tile_predictions)
                target_np = np.concatenate(tile_targets)
                pred_std_np = np.concatenate(tile_uncertainties)

                # Store predictions
                all_predictions.extend(pred_mean_np)
                all_targets.extend(target_np)
                all_uncertainties.extend(pred_std_np)

                # Compute metrics for this tile
                metrics = compute_metrics(pred_mean_np, target_np, pred_std_np)
                all_metrics.append(metrics)

                # Clear GPU cache after each tile
                device_str = str(device) if not isinstance(device, str) else device
                if 'cuda' in device_str:
                    torch.cuda.empty_cache()

    # Convert to arrays
    predictions = np.array(all_predictions)
    targets = np.array(all_targets)
    uncertainties = np.array(all_uncertainties)

    # Aggregate metrics
    avg_metrics = {}
    if len(all_metrics) > 0:
        for key in all_metrics[0].keys():
            avg_metrics[key] = np.mean([m[key] for m in all_metrics])

    return predictions, targets, uncertainties, avg_metrics


def plot_results(
    predictions: np.ndarray,
    targets: np.ndarray,
    uncertainties: Optional[np.ndarray],
    output_dir: Path,
    dataset_name: str = 'test'
) -> None:
    """
    Create evaluation plots.

    Creates a 2x2 subplot with:
    1. Scatter plot of predictions vs targets
    2. Residual plot
    3. Distribution of residuals
    4. Uncertainty calibration plot

    Args:
        predictions: Predicted values
        targets: True values
        uncertainties: Predicted uncertainties (can be None)
        output_dir: Directory to save plots
        dataset_name: Name of dataset for plot title (default: 'test')
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle(f'Model Evaluation ({dataset_name.upper()} set)', fontsize=16, fontweight='bold')

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
    plt.savefig(output_dir / f'evaluation_{dataset_name}.png', dpi=300, bbox_inches='tight')
    print(f"Saved evaluation plot to: {output_dir / f'evaluation_{dataset_name}.png'}")
    plt.close()
