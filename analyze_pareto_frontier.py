"""
Pareto Frontier Analysis for GEDI Baseline Models

This script performs a comprehensive hyperparameter sweep for baseline models
(XGBoost and Random Forest) to explore the trade-off between:
- Model accuracy (Test Log R²)
- Uncertainty calibration (Z-score Std)
- Computational cost (Training Time)

The analysis generates Pareto frontier plots that visualize the performance
envelope of each model class, helping to demonstrate that ANP occupies a
superior region of the trade-off space.

Usage:
    python analyze_pareto_frontier.py \
        --baseline_dir ./outputs_baselines \
        --output_dir ./outputs_pareto \
        --models rf xgb \
        --quick  # Optional: run quick sweep with fewer configs
"""

import argparse
import json
import pickle
from pathlib import Path
from time import time
from typing import Dict, List, Tuple
import itertools

import numpy as np
import pandas as pd
from tqdm import tqdm

from baselines import RandomForestBaseline, XGBoostBaseline
from utils.normalization import normalize_coords, normalize_agbd, denormalize_agbd
from utils.evaluation import compute_metrics


def parse_args():
    parser = argparse.ArgumentParser(
        description='Pareto Frontier Analysis for GEDI Baseline Models'
    )

    # Input/Output
    parser.add_argument('--baseline_dir', type=str, required=True,
                        help='Directory containing baseline training outputs (for data splits)')
    parser.add_argument('--output_dir', type=str, default='./outputs_pareto',
                        help='Output directory for Pareto analysis results')
    parser.add_argument('--anp_results', type=str, default=None,
                        help='Optional: Path to ANP results JSON to include in plots')

    # Model selection
    parser.add_argument('--models', type=str, nargs='+', default=['rf', 'xgb'],
                        choices=['rf', 'xgb'],
                        help='Which models to analyze (default: rf xgb)')

    # Sweep configuration
    parser.add_argument('--quick', action='store_true',
                        help='Run quick sweep with fewer hyperparameter combinations')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')

    # Skip expensive evaluations
    parser.add_argument('--resume', action='store_true',
                        help='Resume from existing results (skip already computed configs)')

    return parser.parse_args()


def compute_calibration_metrics(predictions, targets, stds):
    """
    Compute calibration metrics for uncertainty quantification.

    Args:
        predictions: (N,) array of predictions (in normalized log space)
        targets: (N,) array of ground truth (in normalized log space)
        stds: (N,) array of predicted standard deviations

    Returns:
        dict with calibration metrics
    """
    # Compute z-scores (standardized residuals)
    z_scores = (targets - predictions) / (stds + 1e-8)

    # Z-score statistics
    z_mean = np.mean(z_scores)
    z_std = np.std(z_scores)

    # Compute empirical coverage at key confidence levels
    abs_z = np.abs(z_scores)
    coverage_1sigma = np.sum(abs_z <= 1.0) / len(z_scores) * 100
    coverage_2sigma = np.sum(abs_z <= 2.0) / len(z_scores) * 100
    coverage_3sigma = np.sum(abs_z <= 3.0) / len(z_scores) * 100

    # Calibration error (how far z_std is from ideal value of 1.0)
    calibration_error = abs(z_std - 1.0)

    return {
        'z_mean': z_mean,
        'z_std': z_std,
        'calibration_error': calibration_error,
        'coverage_1sigma': coverage_1sigma,
        'coverage_2sigma': coverage_2sigma,
        'coverage_3sigma': coverage_3sigma,
    }


def evaluate_model(model, coords, embeddings, agbd_true, agbd_scale=200.0, log_transform=True):
    """
    Evaluate baseline model and compute metrics.

    Args:
        model: Baseline model (RF or XGBoost)
        coords: Coordinates for prediction
        embeddings: Embeddings for prediction
        agbd_true: True AGBD values in linear space (Mg/ha)
        agbd_scale: AGBD scale factor (default: 200.0)
        log_transform: Whether log transform was used (default: True)

    Returns:
        metrics: Dict with metrics
    """
    # Predict (normalized, in log space)
    pred_norm, pred_std_norm = model.predict(coords, embeddings, return_std=True)

    # Normalize true values to log space for log-space metrics
    agbd_true_norm = normalize_agbd(agbd_true, agbd_scale=agbd_scale, log_transform=log_transform)

    # Compute log-space metrics
    log_metrics = compute_metrics(pred_norm, agbd_true_norm)

    # Denormalize predictions to linear space
    pred = denormalize_agbd(pred_norm, agbd_scale=agbd_scale, log_transform=log_transform)

    # Compute linear-space metrics
    linear_metrics = compute_metrics(pred, agbd_true)

    # Compute calibration metrics (in normalized log space where model predicts)
    calibration_metrics = compute_calibration_metrics(pred_norm, agbd_true_norm, pred_std_norm)

    # Combine metrics
    metrics = {
        'log_rmse': log_metrics['rmse'],
        'log_mae': log_metrics['mae'],
        'log_r2': log_metrics['r2'],
        'linear_rmse': linear_metrics['rmse'],
        'linear_mae': linear_metrics['mae'],
        **calibration_metrics
    }

    return metrics


def get_hyperparameter_grid(model_type: str, quick: bool = False) -> List[Dict]:
    """
    Get hyperparameter grid for a model type.

    Args:
        model_type: 'rf' or 'xgb'
        quick: If True, use smaller grid for quick testing

    Returns:
        List of hyperparameter dictionaries
    """
    if model_type == 'rf':
        if quick:
            # Quick sweep: ~12 configs
            max_depths = [3, 6, 10]
            n_estimators_list = [50, 100, 200, 500]
        else:
            # Full sweep: ~30 configs
            max_depths = [1, 2, 3, 4, 6, 8, 10, 20]
            n_estimators_list = [50, 100, 200, 500, 1000]

        grid = []
        for max_depth, n_estimators in itertools.product(max_depths, n_estimators_list):
            grid.append({
                'max_depth': max_depth,
                'n_estimators': n_estimators
            })

    elif model_type == 'xgb':
        if quick:
            # Quick sweep: ~12 configs
            max_depths = [3, 6, 10]
            n_estimators_list = [50, 100, 200, 500]
        else:
            # Full sweep: ~30 configs
            max_depths = [1, 2, 3, 4, 6, 8, 10, 20]
            n_estimators_list = [50, 100, 200, 500, 1000]

        grid = []
        for max_depth, n_estimators in itertools.product(max_depths, n_estimators_list):
            grid.append({
                'max_depth': max_depth,
                'n_estimators': n_estimators,
                'learning_rate': 0.1  # Keep learning rate fixed
            })

    else:
        raise ValueError(f"Unknown model type: {model_type}")

    return grid


def train_and_evaluate_config(
    model_type: str,
    config: Dict,
    train_coords: np.ndarray,
    train_embeddings: np.ndarray,
    train_agbd_norm: np.ndarray,
    test_coords: np.ndarray,
    test_embeddings: np.ndarray,
    test_agbd: np.ndarray,
    agbd_scale: float,
    log_transform: bool,
    seed: int
) -> Dict:
    """
    Train and evaluate a single hyperparameter configuration.

    Returns:
        Dict with config, metrics, and training time
    """
    # Train model
    if model_type == 'rf':
        model = RandomForestBaseline(
            n_estimators=config['n_estimators'],
            max_depth=config['max_depth'],
            random_state=seed
        )
    elif model_type == 'xgb':
        model = XGBoostBaseline(
            n_estimators=config['n_estimators'],
            max_depth=config['max_depth'],
            learning_rate=config['learning_rate'],
            random_state=seed
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    start_time = time()
    if model_type == 'xgb':
        model.fit(train_coords, train_embeddings, train_agbd_norm, fit_quantiles=True)
    else:
        model.fit(train_coords, train_embeddings, train_agbd_norm)
    train_time = time() - start_time

    # Evaluate on test set
    test_metrics = evaluate_model(
        model, test_coords, test_embeddings, test_agbd, agbd_scale, log_transform
    )

    # Return results
    result = {
        'model_type': model_type,
        'config': config,
        'train_time': train_time,
        'test_metrics': test_metrics
    }

    return result


def load_existing_results(output_dir: Path) -> List[Dict]:
    """
    Load existing results if they exist.

    Returns:
        List of result dictionaries
    """
    results_file = output_dir / 'pareto_results.json'
    if results_file.exists():
        with open(results_file, 'r') as f:
            return json.load(f)
    return []


def save_results(results: List[Dict], output_dir: Path):
    """
    Save results to JSON and CSV.
    """
    # Save JSON
    with open(output_dir / 'pareto_results.json', 'w') as f:
        json.dump(results, f, indent=2)

    # Save CSV
    rows = []
    for result in results:
        row = {
            'model_type': result['model_type'],
            'train_time': result['train_time'],
            **{f'config_{k}': v for k, v in result['config'].items()},
            **result['test_metrics']
        }
        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(output_dir / 'pareto_results.csv', index=False)

    print(f"\nResults saved to:")
    print(f"  - {output_dir / 'pareto_results.json'}")
    print(f"  - {output_dir / 'pareto_results.csv'}")


def config_already_computed(config: Dict, model_type: str, existing_results: List[Dict]) -> bool:
    """
    Check if a configuration has already been computed.
    """
    for result in existing_results:
        if result['model_type'] == model_type and result['config'] == config:
            return True
    return False


def main():
    args = parse_args()
    np.random.seed(args.seed)

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save configuration
    config_dict = vars(args)
    with open(output_dir / 'config.json', 'w') as f:
        json.dump(config_dict, f, indent=2)

    print("=" * 80)
    print("PARETO FRONTIER ANALYSIS FOR GEDI BASELINE MODELS")
    print("=" * 80)
    print(f"Models to analyze: {', '.join(args.models)}")
    print(f"Sweep mode: {'QUICK' if args.quick else 'FULL'}")
    print(f"Output directory: {output_dir}")
    print()

    # Load existing results if resuming
    existing_results = []
    if args.resume:
        existing_results = load_existing_results(output_dir)
        if existing_results:
            print(f"Resuming from {len(existing_results)} existing results")
            print()

    # Load data splits from baseline training
    baseline_dir = Path(args.baseline_dir)

    print("Step 1: Loading data splits from Parquet files...")
    # Load Parquet files which include embedding vectors
    train_df = pd.read_parquet(baseline_dir / 'train_split.parquet')
    test_df = pd.read_parquet(baseline_dir / 'test_split.parquet')

    # Convert embedding lists back to numpy arrays
    train_df['embedding_patch'] = train_df['embedding_patch'].apply(lambda x: np.array(x) if isinstance(x, list) else x)
    test_df['embedding_patch'] = test_df['embedding_patch'].apply(lambda x: np.array(x) if isinstance(x, list) else x)

    print(f"Loaded {len(train_df)} training samples, {len(test_df)} test samples with embeddings")

    # Load config for normalization parameters
    with open(baseline_dir / 'config.json', 'r') as f:
        baseline_config = json.load(f)

    agbd_scale = baseline_config['agbd_scale']
    log_transform = baseline_config['log_transform_agbd']
    global_bounds = baseline_config['global_bounds']
    buffer_size = baseline_config.get('buffer_size', 'unknown')

    print(f"AGBD normalization: scale={agbd_scale}, log_transform={log_transform}")
    print(f"Spatial split buffer: {buffer_size}° (~{float(buffer_size)*111:.0f}km)" if buffer_size != 'unknown' else "Spatial split buffer: unknown")
    print()

    # Prepare data
    print("Step 2: Preparing features...")

    train_coords = train_df[['longitude', 'latitude']].values
    # Embeddings are already converted to numpy arrays above
    train_embeddings = np.stack(train_df['embedding_patch'].values)
    train_agbd = train_df['agbd'].values
    train_agbd_norm = normalize_agbd(train_agbd, agbd_scale=agbd_scale, log_transform=log_transform)
    train_coords = normalize_coords(train_coords, global_bounds)

    test_coords = test_df[['longitude', 'latitude']].values
    test_embeddings = np.stack(test_df['embedding_patch'].values)
    test_agbd = test_df['agbd'].values
    test_coords = normalize_coords(test_coords, global_bounds)

    print(f"Train: {len(train_coords)} samples")
    print(f"Test: {len(test_coords)} samples")
    print()

    # Run hyperparameter sweep
    all_results = existing_results.copy()

    for model_type in args.models:
        print("=" * 80)
        print(f"HYPERPARAMETER SWEEP: {model_type.upper()}")
        print("=" * 80)

        # Get hyperparameter grid
        grid = get_hyperparameter_grid(model_type, quick=args.quick)
        print(f"Total configurations: {len(grid)}")
        print()

        # Train and evaluate each configuration
        for config in tqdm(grid, desc=f"Training {model_type.upper()}"):
            # Skip if already computed
            if args.resume and config_already_computed(config, model_type, existing_results):
                continue

            # Train and evaluate
            result = train_and_evaluate_config(
                model_type=model_type,
                config=config,
                train_coords=train_coords,
                train_embeddings=train_embeddings,
                train_agbd_norm=train_agbd_norm,
                test_coords=test_coords,
                test_embeddings=test_embeddings,
                test_agbd=test_agbd,
                agbd_scale=agbd_scale,
                log_transform=log_transform,
                seed=args.seed
            )

            all_results.append(result)

            # Save intermediate results
            save_results(all_results, output_dir)

    print()
    print("=" * 80)
    print("SWEEP COMPLETE")
    print("=" * 80)
    print(f"Total configurations evaluated: {len(all_results)}")
    print()

    # Generate summary statistics
    print("SUMMARY BY MODEL:")
    print("-" * 80)

    for model_type in args.models:
        model_results = [r for r in all_results if r['model_type'] == model_type]
        if not model_results:
            continue

        # Extract metrics
        log_r2_values = [r['test_metrics']['log_r2'] for r in model_results]
        cal_errors = [r['test_metrics']['calibration_error'] for r in model_results]
        train_times = [r['train_time'] for r in model_results]

        print(f"\n{model_type.upper()} ({len(model_results)} configs):")
        print(f"  Log R² range: [{min(log_r2_values):.4f}, {max(log_r2_values):.4f}]")
        print(f"  Calibration error range: [{min(cal_errors):.4f}, {max(cal_errors):.4f}]")
        print(f"  Training time range: [{min(train_times):.2f}s, {max(train_times):.2f}s]")

    print()
    print("=" * 80)
    print("Next steps:")
    print("  1. Run plot_pareto.py to generate visualizations")
    print("  2. Include ANP results with --anp_results flag")
    print("=" * 80)


if __name__ == '__main__':
    main()
