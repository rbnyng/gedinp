"""
Compare CNP and baseline model performance on the test set.

This script loads trained models and evaluates them on the same test data
to determine if the CNP's context aggregation adds value over the baseline.
"""

import argparse
from pathlib import Path
import pickle
import json

import torch
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
from tqdm import tqdm

from models.neural_process import GEDINeuralProcess, neural_process_loss, compute_metrics
from models.baseline import SimpleMLPBaseline, FlatMLPBaseline, baseline_loss
from data.dataset import GEDINeuralProcessDataset, collate_neural_process
from train_baseline import GEDIBaselineDataset


def evaluate_cnp(model, dataloader, device):
    """Evaluate CNP model."""
    model.eval()
    total_loss = 0
    all_metrics = []
    n_tiles = 0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc='Evaluating CNP'):
            batch_loss = 0
            n_tiles_in_batch = 0

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
                pred_mean, pred_log_var = model(
                    context_coords,
                    context_embeddings,
                    context_agbd,
                    target_coords,
                    target_embeddings
                )

                # Compute loss
                loss = neural_process_loss(pred_mean, pred_log_var, target_agbd)

                if torch.isnan(loss) or torch.isinf(loss):
                    continue

                batch_loss += loss
                n_tiles_in_batch += 1

                # Compute metrics
                pred_std = torch.exp(0.5 * pred_log_var) if pred_log_var is not None else None
                metrics = compute_metrics(pred_mean, pred_std, target_agbd)
                all_metrics.append(metrics)

            if n_tiles_in_batch > 0:
                batch_loss = batch_loss / n_tiles_in_batch
                total_loss += batch_loss.item()
                n_tiles += n_tiles_in_batch

    avg_loss = total_loss / max(n_tiles, 1)

    # Aggregate metrics
    avg_metrics = {}
    if len(all_metrics) > 0:
        for key in all_metrics[0].keys():
            avg_metrics[key] = np.mean([m[key] for m in all_metrics])

    return avg_loss, avg_metrics


def evaluate_baseline(model, dataloader, device):
    """Evaluate baseline model."""
    model.eval()
    total_loss = 0
    all_preds = []
    all_targets = []
    all_uncertainties = []
    n_samples = 0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc='Evaluating Baseline'):
            coords = batch['coords'].to(device)
            embeddings = batch['embedding'].to(device)
            agbd = batch['agbd'].to(device)

            # Forward pass
            pred_mean, pred_log_var = model(coords, embeddings)

            # Compute loss
            loss = baseline_loss(pred_mean, pred_log_var, agbd)

            if torch.isnan(loss) or torch.isinf(loss):
                continue

            total_loss += loss.item() * len(coords)
            n_samples += len(coords)

            # Collect predictions
            all_preds.append(pred_mean)
            all_targets.append(agbd)

            if pred_log_var is not None:
                pred_std = torch.exp(0.5 * pred_log_var)
                all_uncertainties.append(pred_std)

    avg_loss = total_loss / max(n_samples, 1)

    # Compute metrics
    all_preds = torch.cat(all_preds, dim=0)
    all_targets = torch.cat(all_targets, dim=0)

    if len(all_uncertainties) > 0:
        all_uncertainties = torch.cat(all_uncertainties, dim=0)
    else:
        all_uncertainties = None

    metrics = compute_metrics(all_preds, all_uncertainties, all_targets)

    return avg_loss, metrics


def parse_args():
    parser = argparse.ArgumentParser(description='Compare CNP and baseline models')

    parser.add_argument('--cnp_output_dir', type=str, required=True,
                        help='Directory with CNP model outputs')
    parser.add_argument('--baseline_output_dir', type=str, required=True,
                        help='Directory with baseline model outputs')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Batch size for CNP evaluation')
    parser.add_argument('--baseline_batch_size', type=int, default=256,
                        help='Batch size for baseline evaluation')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Device to use')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of data loading workers')
    parser.add_argument('--output_file', type=str, default='model_comparison.json',
                        help='Output file for comparison results')

    return parser.parse_args()


def main():
    args = parse_args()

    print("=" * 80)
    print("Model Comparison: CNP vs Baseline")
    print("=" * 80)
    print(f"CNP directory: {args.cnp_output_dir}")
    print(f"Baseline directory: {args.baseline_output_dir}")
    print()

    # Load configs
    cnp_dir = Path(args.cnp_output_dir)
    baseline_dir = Path(args.baseline_output_dir)

    with open(cnp_dir / 'config.json', 'r') as f:
        cnp_config = json.load(f)

    with open(baseline_dir / 'config.json', 'r') as f:
        baseline_config = json.load(f)

    # Load test data
    print("Loading test data...")
    test_df = pd.read_csv(cnp_dir / 'test_split.csv')

    # Reconstruct embedding_patch column (it's saved as string in CSV)
    with open(cnp_dir / 'processed_data.pkl', 'rb') as f:
        processed_data = pickle.load(f)

    # Merge to get embeddings back
    test_df = test_df.merge(
        processed_data[['latitude', 'longitude', 'tile_id', 'embedding_patch']],
        on=['latitude', 'longitude', 'tile_id'],
        how='left'
    )

    print(f"Test set: {len(test_df)} shots across {test_df['tile_id'].nunique()} tiles")
    print()

    # ===== Evaluate CNP =====
    print("=" * 80)
    print("Evaluating CNP Model")
    print("=" * 80)

    # Load CNP model
    cnp_model = GEDINeuralProcess(
        patch_size=cnp_config['patch_size'],
        embedding_channels=128,
        embedding_feature_dim=cnp_config['embedding_feature_dim'],
        context_repr_dim=cnp_config['context_repr_dim'],
        hidden_dim=cnp_config['hidden_dim'],
        output_uncertainty=True,
        use_attention=cnp_config['use_attention'],
        num_attention_heads=cnp_config['num_attention_heads']
    ).to(args.device)

    # Load best model weights
    checkpoint = torch.load(cnp_dir / 'best_r2_model.pt', map_location=args.device)
    cnp_model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Loaded CNP model from epoch {checkpoint['epoch']}")

    # Create CNP test dataset
    from data.dataset import GEDINeuralProcessDataset, collate_neural_process

    cnp_test_dataset = GEDINeuralProcessDataset(
        test_df,
        min_shots_per_tile=cnp_config.get('min_shots_per_tile', 10),
        log_transform_agbd=cnp_config['log_transform_agbd'],
        augment_coords=False,
        coord_noise_std=0.0
    )

    cnp_test_loader = DataLoader(
        cnp_test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_neural_process,
        num_workers=args.num_workers
    )

    cnp_loss, cnp_metrics = evaluate_cnp(cnp_model, cnp_test_loader, args.device)

    print(f"\nCNP Test Results:")
    print(f"  Loss: {cnp_loss:.6e}")
    print(f"  RMSE: {cnp_metrics['rmse']:.4f}")
    print(f"  MAE:  {cnp_metrics['mae']:.4f}")
    print(f"  R²:   {cnp_metrics['r2']:.4f}")
    if 'mean_uncertainty' in cnp_metrics:
        print(f"  Mean Uncertainty: {cnp_metrics['mean_uncertainty']:.4f}")
    print()

    # ===== Evaluate Baseline =====
    print("=" * 80)
    print("Evaluating Baseline Model")
    print("=" * 80)

    # Load baseline model
    if baseline_config['model_type'] == 'simple_mlp':
        baseline_model = SimpleMLPBaseline(
            patch_size=baseline_config['patch_size'],
            embedding_channels=128,
            embedding_feature_dim=baseline_config['embedding_feature_dim'],
            hidden_dim=baseline_config['hidden_dim'],
            output_uncertainty=baseline_config['output_uncertainty'],
            num_hidden_layers=baseline_config['num_hidden_layers']
        ).to(args.device)
    elif baseline_config['model_type'] == 'flat_mlp':
        baseline_model = FlatMLPBaseline(
            patch_size=baseline_config['patch_size'],
            embedding_channels=128,
            hidden_dim=baseline_config['hidden_dim'],
            output_uncertainty=baseline_config['output_uncertainty'],
            num_hidden_layers=baseline_config['num_hidden_layers']
        ).to(args.device)

    # Load best model weights
    checkpoint = torch.load(baseline_dir / 'best_r2_model.pt', map_location=args.device)
    baseline_model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Loaded baseline model from epoch {checkpoint['epoch']}")

    # Create baseline test dataset
    baseline_test_dataset = GEDIBaselineDataset(
        test_df,
        log_transform_agbd=baseline_config['log_transform_agbd'],
        augment_coords=False,
        coord_noise_std=0.0
    )

    baseline_test_loader = DataLoader(
        baseline_test_dataset,
        batch_size=args.baseline_batch_size,
        shuffle=False,
        num_workers=args.num_workers
    )

    baseline_loss, baseline_metrics = evaluate_baseline(
        baseline_model, baseline_test_loader, args.device
    )

    print(f"\nBaseline Test Results:")
    print(f"  Loss: {baseline_loss:.6e}")
    print(f"  RMSE: {baseline_metrics['rmse']:.4f}")
    print(f"  MAE:  {baseline_metrics['mae']:.4f}")
    print(f"  R²:   {baseline_metrics['r2']:.4f}")
    if 'mean_uncertainty' in baseline_metrics:
        print(f"  Mean Uncertainty: {baseline_metrics['mean_uncertainty']:.4f}")
    print()

    # ===== Comparison =====
    print("=" * 80)
    print("Comparison Summary")
    print("=" * 80)

    rmse_diff = cnp_metrics['rmse'] - baseline_metrics['rmse']
    mae_diff = cnp_metrics['mae'] - baseline_metrics['mae']
    r2_diff = cnp_metrics['r2'] - baseline_metrics['r2']

    print(f"RMSE: CNP={cnp_metrics['rmse']:.4f}, Baseline={baseline_metrics['rmse']:.4f}, "
          f"Diff={rmse_diff:+.4f} ({rmse_diff/baseline_metrics['rmse']*100:+.2f}%)")
    print(f"MAE:  CNP={cnp_metrics['mae']:.4f}, Baseline={baseline_metrics['mae']:.4f}, "
          f"Diff={mae_diff:+.4f} ({mae_diff/baseline_metrics['mae']*100:+.2f}%)")
    print(f"R²:   CNP={cnp_metrics['r2']:.4f}, Baseline={baseline_metrics['r2']:.4f}, "
          f"Diff={r2_diff:+.4f}")
    print()

    if abs(r2_diff) < 0.05:  # Less than 5% difference
        print("⚠️  Performance is SIMILAR - CNP may not be adding significant value!")
    elif r2_diff > 0.05:
        print("✓ CNP shows BETTER performance - context aggregation is valuable!")
    else:
        print("⚠️  Baseline shows BETTER performance - CNP may be overfitting or poorly tuned!")
    print()

    # Save comparison results
    results = {
        'cnp': {
            'loss': float(cnp_loss),
            'rmse': float(cnp_metrics['rmse']),
            'mae': float(cnp_metrics['mae']),
            'r2': float(cnp_metrics['r2']),
        },
        'baseline': {
            'loss': float(baseline_loss),
            'rmse': float(baseline_metrics['rmse']),
            'mae': float(baseline_metrics['mae']),
            'r2': float(baseline_metrics['r2']),
        },
        'difference': {
            'rmse': float(rmse_diff),
            'rmse_pct': float(rmse_diff / baseline_metrics['rmse'] * 100),
            'mae': float(mae_diff),
            'mae_pct': float(mae_diff / baseline_metrics['mae'] * 100),
            'r2': float(r2_diff),
        }
    }

    if 'mean_uncertainty' in cnp_metrics:
        results['cnp']['mean_uncertainty'] = float(cnp_metrics['mean_uncertainty'])
    if 'mean_uncertainty' in baseline_metrics:
        results['baseline']['mean_uncertainty'] = float(baseline_metrics['mean_uncertainty'])

    with open(args.output_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"Results saved to: {args.output_file}")


if __name__ == '__main__':
    main()
