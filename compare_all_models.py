"""
Comprehensive comparison of all models: CNP, MLP baselines, and tree baselines.

This script loads all trained models and compares them on the same test set.
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

try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False


def evaluate_cnp(model, dataloader, device):
    """Evaluate CNP model."""
    model.eval()
    total_loss = 0
    all_metrics = []
    n_tiles = 0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc='Evaluating CNP', leave=False):
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

                pred_mean, pred_log_var = model(
                    context_coords,
                    context_embeddings,
                    context_agbd,
                    target_coords,
                    target_embeddings
                )

                loss = neural_process_loss(pred_mean, pred_log_var, target_agbd)

                if torch.isnan(loss) or torch.isinf(loss):
                    continue

                batch_loss += loss
                n_tiles_in_batch += 1

                pred_std = torch.exp(0.5 * pred_log_var) if pred_log_var is not None else None
                metrics = compute_metrics(pred_mean, pred_std, target_agbd)
                all_metrics.append(metrics)

            if n_tiles_in_batch > 0:
                batch_loss = batch_loss / n_tiles_in_batch
                total_loss += batch_loss.item()
                n_tiles += n_tiles_in_batch

    avg_loss = total_loss / max(n_tiles, 1)
    avg_metrics = {}
    if len(all_metrics) > 0:
        for key in all_metrics[0].keys():
            avg_metrics[key] = np.mean([m[key] for m in all_metrics])

    return avg_loss, avg_metrics


def evaluate_mlp_baseline(model, dataloader, device):
    """Evaluate MLP baseline model."""
    model.eval()
    total_loss = 0
    all_preds = []
    all_targets = []
    all_uncertainties = []
    n_samples = 0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc='Evaluating MLP', leave=False):
            coords = batch['coords'].to(device)
            embeddings = batch['embedding'].to(device)
            agbd = batch['agbd'].to(device)

            pred_mean, pred_log_var = model(coords, embeddings)
            loss = baseline_loss(pred_mean, pred_log_var, agbd)

            if torch.isnan(loss) or torch.isinf(loss):
                continue

            total_loss += loss.item() * len(coords)
            n_samples += len(coords)

            all_preds.append(pred_mean)
            all_targets.append(agbd)

            if pred_log_var is not None:
                pred_std = torch.exp(0.5 * pred_log_var)
                all_uncertainties.append(pred_std)

    avg_loss = total_loss / max(n_samples, 1)
    all_preds = torch.cat(all_preds, dim=0)
    all_targets = torch.cat(all_targets, dim=0)

    if len(all_uncertainties) > 0:
        all_uncertainties = torch.cat(all_uncertainties, dim=0)
    else:
        all_uncertainties = None

    metrics = compute_metrics(all_preds, all_uncertainties, all_targets)

    return avg_loss, metrics


def evaluate_tree_model(model_dir, test_df, model_type='xgboost'):
    """Evaluate tree-based model."""
    # Load config
    with open(model_dir / 'config.json', 'r') as f:
        config = json.load(f)

    # Load features
    features_path = model_dir / 'features.npz'
    if not features_path.exists():
        print(f"Warning: Features not found at {features_path}")
        return None, None

    data = np.load(features_path)
    X_test = data['X_test']
    y_test = data['y_test']

    # Load model
    if model_type == 'xgboost':
        if not HAS_XGBOOST:
            print("Warning: XGBoost not installed")
            return None, None

        model_path = model_dir / 'xgboost_model.json'
        if not model_path.exists():
            print(f"Warning: XGBoost model not found at {model_path}")
            return None, None

        model = xgb.XGBRegressor()
        model.load_model(str(model_path))

    elif model_type == 'random_forest':
        model_path = model_dir / 'random_forest_model.pkl'
        if not model_path.exists():
            print(f"Warning: Random Forest model not found at {model_path}")
            return None, None

        with open(model_path, 'rb') as f:
            model = pickle.load(f)

    # Predict
    y_pred = model.predict(X_test)

    # Compute metrics (using numpy directly)
    mse = ((y_pred - y_test) ** 2).mean()
    rmse = np.sqrt(mse)
    mae = np.abs(y_pred - y_test).mean()

    ss_res = ((y_test - y_pred) ** 2).sum()
    ss_tot = ((y_test - y_test.mean()) ** 2).sum()
    r2 = 1 - ss_res / (ss_tot + 1e-8)

    metrics = {
        'rmse': float(rmse),
        'mae': float(mae),
        'r2': float(r2),
        'mse': float(mse)
    }

    # Use MSE as "loss" for consistency
    loss = float(mse)

    return loss, metrics


def parse_args():
    parser = argparse.ArgumentParser(description='Compare all models')

    parser.add_argument('--cnp_dir', type=str, default=None,
                        help='Directory with CNP model outputs')
    parser.add_argument('--mlp_dir', type=str, default=None,
                        help='Directory with MLP baseline outputs')
    parser.add_argument('--tree_dir', type=str, default=None,
                        help='Directory with tree baseline outputs')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Batch size for CNP evaluation')
    parser.add_argument('--mlp_batch_size', type=int, default=256,
                        help='Batch size for MLP evaluation')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Device to use')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of data loading workers')
    parser.add_argument('--output_file', type=str, default='all_models_comparison.json',
                        help='Output file for comparison results')

    return parser.parse_args()


def main():
    args = parse_args()

    print("=" * 80)
    print("Comprehensive Model Comparison")
    print("=" * 80)

    results = {}
    test_df = None

    # ===== Evaluate CNP =====
    if args.cnp_dir:
        cnp_dir = Path(args.cnp_dir)
        print(f"\nCNP directory: {cnp_dir}")

        if (cnp_dir / 'config.json').exists():
            print("\n" + "=" * 80)
            print("Evaluating CNP Model")
            print("=" * 80)

            with open(cnp_dir / 'config.json', 'r') as f:
                cnp_config = json.load(f)

            # Load test data
            test_df = pd.read_csv(cnp_dir / 'test_split.csv')
            with open(cnp_dir / 'processed_data.pkl', 'rb') as f:
                processed_data = pickle.load(f)
            test_df = test_df.merge(
                processed_data[['latitude', 'longitude', 'tile_id', 'embedding_patch']],
                on=['latitude', 'longitude', 'tile_id'],
                how='left'
            )

            # Load model
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

            checkpoint = torch.load(cnp_dir / 'best_r2_model.pt', map_location=args.device)
            cnp_model.load_state_dict(checkpoint['model_state_dict'])
            print(f"Loaded CNP from epoch {checkpoint['epoch']}")

            # Create dataset
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

            # Evaluate
            cnp_loss, cnp_metrics = evaluate_cnp(cnp_model, cnp_test_loader, args.device)

            print(f"\nCNP Results:")
            print(f"  RMSE: {cnp_metrics['rmse']:.4f}")
            print(f"  MAE:  {cnp_metrics['mae']:.4f}")
            print(f"  R²:   {cnp_metrics['r2']:.4f}")

            results['cnp'] = {
                'loss': cnp_loss,
                **cnp_metrics
            }

    # ===== Evaluate MLP Baseline =====
    if args.mlp_dir:
        mlp_dir = Path(args.mlp_dir)
        print(f"\nMLP directory: {mlp_dir}")

        if (mlp_dir / 'config.json').exists():
            print("\n" + "=" * 80)
            print("Evaluating MLP Baseline")
            print("=" * 80)

            with open(mlp_dir / 'config.json', 'r') as f:
                mlp_config = json.load(f)

            # Load test data if not already loaded
            if test_df is None:
                test_df = pd.read_csv(mlp_dir / 'test_split.csv')
                with open(mlp_dir / 'processed_data.pkl', 'rb') as f:
                    processed_data = pickle.load(f)
                test_df = test_df.merge(
                    processed_data[['latitude', 'longitude', 'tile_id', 'embedding_patch']],
                    on=['latitude', 'longitude', 'tile_id'],
                    how='left'
                )

            # Load model
            if mlp_config['model_type'] == 'simple_mlp':
                mlp_model = SimpleMLPBaseline(
                    patch_size=mlp_config['patch_size'],
                    embedding_channels=128,
                    embedding_feature_dim=mlp_config['embedding_feature_dim'],
                    hidden_dim=mlp_config['hidden_dim'],
                    output_uncertainty=mlp_config['output_uncertainty'],
                    num_hidden_layers=mlp_config['num_hidden_layers']
                ).to(args.device)
            elif mlp_config['model_type'] == 'flat_mlp':
                mlp_model = FlatMLPBaseline(
                    patch_size=mlp_config['patch_size'],
                    embedding_channels=128,
                    hidden_dim=mlp_config['hidden_dim'],
                    output_uncertainty=mlp_config['output_uncertainty'],
                    num_hidden_layers=mlp_config['num_hidden_layers']
                ).to(args.device)

            checkpoint = torch.load(mlp_dir / 'best_r2_model.pt', map_location=args.device)
            mlp_model.load_state_dict(checkpoint['model_state_dict'])
            print(f"Loaded MLP from epoch {checkpoint['epoch']}")

            # Create dataset
            mlp_test_dataset = GEDIBaselineDataset(
                test_df,
                log_transform_agbd=mlp_config['log_transform_agbd'],
                augment_coords=False,
                coord_noise_std=0.0
            )
            mlp_test_loader = DataLoader(
                mlp_test_dataset,
                batch_size=args.mlp_batch_size,
                shuffle=False,
                num_workers=args.num_workers
            )

            # Evaluate
            mlp_loss, mlp_metrics = evaluate_mlp_baseline(mlp_model, mlp_test_loader, args.device)

            print(f"\nMLP Results:")
            print(f"  RMSE: {mlp_metrics['rmse']:.4f}")
            print(f"  MAE:  {mlp_metrics['mae']:.4f}")
            print(f"  R²:   {mlp_metrics['r2']:.4f}")

            results['mlp_baseline'] = {
                'loss': mlp_loss,
                **mlp_metrics
            }

    # ===== Evaluate Tree Models =====
    if args.tree_dir:
        tree_dir = Path(args.tree_dir)
        print(f"\nTree directory: {tree_dir}")

        # XGBoost
        if (tree_dir / 'xgboost_model.json').exists():
            print("\n" + "=" * 80)
            print("Evaluating XGBoost")
            print("=" * 80)

            # Load test data if needed
            if test_df is None:
                test_df = pd.read_csv(tree_dir / 'test_split.csv')
                with open(tree_dir / 'processed_data.pkl', 'rb') as f:
                    processed_data = pickle.load(f)
                test_df = test_df.merge(
                    processed_data[['latitude', 'longitude', 'tile_id', 'embedding_patch']],
                    on=['latitude', 'longitude', 'tile_id'],
                    how='left'
                )

            xgb_loss, xgb_metrics = evaluate_tree_model(tree_dir, test_df, 'xgboost')

            if xgb_metrics:
                print(f"\nXGBoost Results:")
                print(f"  RMSE: {xgb_metrics['rmse']:.4f}")
                print(f"  MAE:  {xgb_metrics['mae']:.4f}")
                print(f"  R²:   {xgb_metrics['r2']:.4f}")

                results['xgboost'] = {
                    'loss': xgb_loss,
                    **xgb_metrics
                }

        # Random Forest
        if (tree_dir / 'random_forest_model.pkl').exists():
            print("\n" + "=" * 80)
            print("Evaluating Random Forest")
            print("=" * 80)

            rf_loss, rf_metrics = evaluate_tree_model(tree_dir, test_df, 'random_forest')

            if rf_metrics:
                print(f"\nRandom Forest Results:")
                print(f"  RMSE: {rf_metrics['rmse']:.4f}")
                print(f"  MAE:  {rf_metrics['mae']:.4f}")
                print(f"  R²:   {rf_metrics['r2']:.4f}")

                results['random_forest'] = {
                    'loss': rf_loss,
                    **rf_metrics
                }

    # ===== Summary Comparison =====
    print("\n" + "=" * 80)
    print("Summary Comparison")
    print("=" * 80)

    if len(results) > 0:
        # Create comparison table
        print(f"\n{'Model':<20} {'RMSE':<10} {'MAE':<10} {'R²':<10}")
        print("-" * 50)

        model_order = ['cnp', 'mlp_baseline', 'xgboost', 'random_forest']
        for model_name in model_order:
            if model_name in results:
                m = results[model_name]
                print(f"{model_name:<20} {m['rmse']:<10.4f} {m['mae']:<10.4f} {m['r2']:<10.4f}")

        # Find best model by R²
        best_model = max(results.items(), key=lambda x: x[1]['r2'])
        print(f"\n✓ Best model (R²): {best_model[0]} (R² = {best_model[1]['r2']:.4f})")

        # Compute differences relative to best simple baseline
        if 'mlp_baseline' in results:
            baseline_r2 = results['mlp_baseline']['r2']
            print(f"\nPerformance vs MLP Baseline:")
            for model_name, metrics in results.items():
                if model_name != 'mlp_baseline':
                    diff = metrics['r2'] - baseline_r2
                    pct = (diff / abs(baseline_r2)) * 100 if baseline_r2 != 0 else 0
                    print(f"  {model_name}: R² diff = {diff:+.4f} ({pct:+.2f}%)")

    # Save results
    with open(args.output_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to: {args.output_file}")


if __name__ == '__main__':
    main()
