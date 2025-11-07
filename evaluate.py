"""
Evaluate a trained GEDI Neural Process model on test data.
"""

import argparse
import json
from pathlib import Path
import pickle

import torch
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

from data.dataset import GEDINeuralProcessDataset, collate_neural_process
from models.neural_process import GEDINeuralProcess, compute_metrics


def evaluate_model(model, dataloader, device):
    """Evaluate model on a dataset."""
    model.eval()
    all_predictions = []
    all_targets = []
    all_uncertainties = []
    all_metrics = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc='Evaluating'):
            for i in range(len(batch['context_coords'])):
                context_coords = batch['context_coords'][i].to(device)
                context_embeddings = batch['context_embeddings'][i].to(device)
                context_agbd = batch['context_agbd'][i].to(device)
                target_coords = batch['target_coords'][i].to(device)
                target_embeddings = batch['target_embeddings'][i].to(device)
                target_agbd = batch['target_agbd'][i].to(device)

                if len(target_coords) == 0:
                    continue

                # Forward pass
                pred_mean, pred_log_var, _, _ = model(
                    context_coords,
                    context_embeddings,
                    context_agbd,
                    target_coords,
                    target_embeddings,
                    training=False
                )

                # Convert to numpy
                pred_mean_np = pred_mean.detach().cpu().numpy().flatten()
                target_np = target_agbd.detach().cpu().numpy().flatten()

                if pred_log_var is not None:
                    pred_std_np = torch.exp(0.5 * pred_log_var).detach().cpu().numpy().flatten()
                else:
                    pred_std_np = np.zeros_like(pred_mean_np)

                # Store predictions
                all_predictions.extend(pred_mean_np)
                all_targets.extend(target_np)
                all_uncertainties.extend(pred_std_np)

                # Compute metrics for this tile
                pred_std = torch.exp(0.5 * pred_log_var) if pred_log_var is not None else None
                metrics = compute_metrics(pred_mean, pred_std, target_agbd)
                all_metrics.append(metrics)

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


def plot_results(predictions, targets, uncertainties, output_dir, dataset_name='test'):
    """Create evaluation plots."""

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


def main():
    parser = argparse.ArgumentParser(description='Evaluate GEDI Neural Process model')

    parser.add_argument('--model_dir', type=str, required=True,
                        help='Directory containing trained model')
    parser.add_argument('--checkpoint', type=str, default='best_r2_model.pt',
                        help='Checkpoint filename (default: best_r2_model.pt)')
    parser.add_argument('--test_split', type=str, default=None,
                        help='Path to test split CSV (default: model_dir/test_split.csv)')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--num_workers', type=int, default=0)

    args = parser.parse_args()

    model_dir = Path(args.model_dir)

    # Load config
    with open(model_dir / 'config.json', 'r') as f:
        config = json.load(f)

    print("=" * 80)
    print("EVALUATING GEDI NEURAL PROCESS")
    print("=" * 80)
    print(f"Model directory: {model_dir}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Architecture: {config.get('architecture_mode', 'deterministic')}")
    print(f"Device: {args.device}")
    print()

    # Load test data
    if args.test_split:
        test_csv = Path(args.test_split)
    else:
        test_csv = model_dir / 'test_split.csv'

    if not test_csv.exists():
        print(f"Error: Test split not found at {test_csv}")
        return

    print(f"Loading test data from: {test_csv}")
    test_df = pd.read_csv(test_csv)

    # Convert embedding_patch from string back to array
    if 'embedding_patch' in test_df.columns and isinstance(test_df['embedding_patch'].iloc[0], str):
        test_df['embedding_patch'] = test_df['embedding_patch'].apply(
            lambda x: np.array(eval(x)) if isinstance(x, str) else x
        )

    print(f"Test set: {len(test_df)} shots across {test_df['tile_id'].nunique()} tiles")

    # Create dataset
    global_bounds = tuple(config['global_bounds'])
    test_dataset = GEDINeuralProcessDataset(
        test_df,
        min_shots_per_tile=config.get('min_shots_per_tile', 10),
        log_transform_agbd=config.get('log_transform_agbd', True),
        augment_coords=False,
        coord_noise_std=0.0,
        global_bounds=global_bounds
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_neural_process,
        num_workers=args.num_workers
    )

    # Initialize model
    print("Initializing model...")
    model = GEDINeuralProcess(
        patch_size=config.get('patch_size', 3),
        embedding_channels=128,
        embedding_feature_dim=config.get('embedding_feature_dim', 128),
        context_repr_dim=config.get('context_repr_dim', 128),
        hidden_dim=config.get('hidden_dim', 512),
        latent_dim=config.get('latent_dim', 128),
        output_uncertainty=True,
        architecture_mode=config.get('architecture_mode', 'deterministic'),
        num_attention_heads=config.get('num_attention_heads', 4)
    ).to(args.device)

    # Load checkpoint
    checkpoint_path = model_dir / args.checkpoint
    print(f"Loading checkpoint from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=args.device)
    model.load_state_dict(checkpoint['model_state_dict'])

    print(f"Checkpoint epoch: {checkpoint.get('epoch', 'unknown')}")
    if 'val_metrics' in checkpoint:
        print(f"Validation metrics:")
        for key, val in checkpoint['val_metrics'].items():
            print(f"  {key}: {val:.4f}")
    print()

    # Evaluate
    print("Evaluating on test set...")
    predictions, targets, uncertainties, metrics = evaluate_model(
        model, test_loader, args.device
    )

    # Print results
    print("\n" + "=" * 80)
    print("TEST SET RESULTS")
    print("=" * 80)
    for key, val in metrics.items():
        print(f"{key.upper()}: {val:.4f}")
    print("=" * 80)

    # Save results
    results = {
        'metrics': metrics,
        'config': config,
        'checkpoint': args.checkpoint
    }

    with open(model_dir / 'test_results.json', 'w') as f:
        json.dump(results, f, indent=2)

    # Save predictions
    results_df = pd.DataFrame({
        'true': targets,
        'predicted': predictions,
        'uncertainty': uncertainties,
        'residual': predictions - targets
    })
    results_df.to_csv(model_dir / 'test_predictions.csv', index=False)

    # Create plots
    plot_results(predictions, targets, uncertainties, model_dir, 'test')

    print("\n" + "=" * 80)
    print("EVALUATION COMPLETE")
    print("=" * 80)
    print(f"Results saved to: {model_dir}")
    print("Files:")
    print("  - test_results.json")
    print("  - test_predictions.csv")
    print("  - evaluation_test.png")
    print("=" * 80)


if __name__ == '__main__':
    main()
