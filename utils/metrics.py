"""Evaluation metrics for model performance."""

import numpy as np
from typing import Optional, Union
try:
    import torch
except ImportError:
    torch = None


def compute_metrics(
    pred_mean: Union[np.ndarray, 'torch.Tensor'],
    target: Union[np.ndarray, 'torch.Tensor'],
    pred_std: Optional[Union[np.ndarray, 'torch.Tensor']] = None
) -> dict:
    """
    Compute evaluation metrics (RMSE, MAE, R²).

    Handles both numpy arrays and PyTorch tensors.

    Args:
        pred_mean: Predicted mean values
        target: Target/ground truth values
        pred_std: Optional predicted standard deviations for uncertainty metrics

    Returns:
        Dictionary containing:
        - rmse: Root Mean Squared Error
        - mae: Mean Absolute Error
        - r2: R² (coefficient of determination)
        - mse: Mean Squared Error
        - mean_uncertainty: Mean of predicted uncertainties (if pred_std provided)
    """
    # Convert torch tensors to numpy if needed
    if torch is not None and isinstance(pred_mean, torch.Tensor):
        pred_mean = pred_mean.detach().cpu().numpy().flatten()
    else:
        pred_mean = np.asarray(pred_mean).flatten()

    if torch is not None and isinstance(target, torch.Tensor):
        target = target.detach().cpu().numpy().flatten()
    else:
        target = np.asarray(target).flatten()

    # Compute metrics
    mse = ((pred_mean - target) ** 2).mean()
    rmse = np.sqrt(mse)
    mae = np.abs(pred_mean - target).mean()

    # R² score
    ss_res = ((target - pred_mean) ** 2).sum()
    ss_tot = ((target - target.mean()) ** 2).sum()
    r2 = 1 - ss_res / (ss_tot + 1e-8)

    metrics = {
        'rmse': rmse,
        'mae': mae,
        'r2': r2,
        'mse': mse
    }

    # Add uncertainty metric if provided
    if pred_std is not None:
        if torch is not None and isinstance(pred_std, torch.Tensor):
            pred_std = pred_std.detach().cpu().numpy().flatten()
        else:
            pred_std = np.asarray(pred_std).flatten()
        metrics['mean_uncertainty'] = pred_std.mean()

    return metrics
