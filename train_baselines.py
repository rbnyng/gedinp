import argparse
import json
from pathlib import Path
import pickle
from time import time

import numpy as np
import pandas as pd
from tqdm import tqdm

from data.gedi import GEDIQuerier
from data.embeddings import EmbeddingExtractor
from data.spatial_cv import SpatialTileSplitter, BufferedSpatialSplitter
from baselines import RandomForestBaseline, XGBoostBaseline, LinearRegressionBaseline, IDWBaseline
from utils.normalization import normalize_coords, normalize_agbd, denormalize_agbd
from utils.evaluation import compute_metrics
from scipy.stats import norm
from utils.config import save_config, _make_serializable


def parse_args():
    parser = argparse.ArgumentParser(description='Train baseline models for GEDI AGBD prediction')

    # Data arguments
    parser.add_argument('--region_bbox', type=float, nargs=4, required=True,
                        help='Region bounding box: min_lon min_lat max_lon max_lat')
    parser.add_argument('--start_time', type=str, default='2022-01-01',
                        help='Start date for GEDI data (YYYY-MM-DD)')
    parser.add_argument('--end_time', type=str, default='2022-12-31',
                        help='End date for GEDI data (YYYY-MM-DD)')
    parser.add_argument('--embedding_year', type=int, default=2022,
                        help='Year of GeoTessera embeddings')
    parser.add_argument('--cache_dir', type=str, default='./cache',
                        help='Directory for caching tiles and embeddings')

    # Model arguments
    parser.add_argument('--patch_size', type=int, default=3,
                        help='Embedding patch size (default: 3x3)')

    # Random Forest arguments
    parser.add_argument('--rf_n_estimators', type=int, default=100,
                        help='Random Forest: number of trees')
    parser.add_argument('--rf_max_depth', type=int, default=None,
                        help='Random Forest: maximum tree depth (None=unlimited)')

    # XGBoost arguments
    parser.add_argument('--xgb_n_estimators', type=int, default=100,
                        help='XGBoost: number of boosting rounds')
    parser.add_argument('--xgb_max_depth', type=int, default=6,
                        help='XGBoost: maximum tree depth')
    parser.add_argument('--xgb_learning_rate', type=float, default=0.1,
                        help='XGBoost: learning rate')

    # IDW arguments
    parser.add_argument('--idw_power', type=float, default=2.0,
                        help='IDW: power parameter for distance weighting')
    parser.add_argument('--idw_n_neighbors', type=int, default=10,
                        help='IDW: number of nearest neighbors')

    # Training arguments
    parser.add_argument('--val_ratio', type=float, default=0.15,
                        help='Validation set ratio')
    parser.add_argument('--test_ratio', type=float, default=0.15,
                        help='Test set ratio')
    parser.add_argument('--buffer_size', type=float, default=0.5,
                        help='Buffer size in degrees for spatial CV (~55km at 0.5 deg)')
    parser.add_argument('--agbd_scale', type=float, default=200.0,
                        help='AGBD scale factor for normalization (default: 200.0 Mg/ha)')
    parser.add_argument('--log_transform_agbd', action='store_true', default=True,
                        help='Apply log transform to AGBD')

    # Output arguments
    parser.add_argument('--output_dir', type=str, default='./outputs_baselines',
                        help='Output directory for models and logs')

    # Other
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--models', type=str, nargs='+', default=['rf', 'xgb', 'lr', 'idw'],
                        choices=['rf', 'xgb', 'lr', 'idw'],
                        help='Which models to train (default: all)')

    return parser.parse_args()


def prepare_data(df, log_transform=True, agbd_scale=200.0):
    coords = df[['longitude', 'latitude']].values
    embeddings = np.stack(df['embedding_patch'].values)
    agbd = df['agbd'].values

    agbd = normalize_agbd(agbd, agbd_scale=agbd_scale, log_transform=log_transform)

    return coords, embeddings, agbd


def compute_calibration_metrics(predictions, targets, stds):
    """
    Compute calibration metrics for uncertainty quantification.

    Args:
        predictions: (N,) array of predictions (in normalized log space)
        targets: (N,) array of ground truth (in normalized log space)
        stds: (N,) array of predicted standard deviations

    Returns:
        dict with calibration metrics:
            - z_mean: mean of z-scores (ideal: 0)
            - z_std: std of z-scores (ideal: 1)
            - coverage_1sigma: empirical coverage at 1σ (ideal: 68.3%)
            - coverage_2sigma: empirical coverage at 2σ (ideal: 95.4%)
            - coverage_3sigma: empirical coverage at 3σ (ideal: 99.7%)
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

    return {
        'z_mean': z_mean,
        'z_std': z_std,
        'coverage_1sigma': coverage_1sigma,
        'coverage_2sigma': coverage_2sigma,
        'coverage_3sigma': coverage_3sigma,
    }


def print_calibration_metrics(metrics, prefix=""):
    """
    Pretty print calibration metrics.

    Args:
        metrics: Dict with calibration metrics
        prefix: Optional prefix for print statements (e.g., "Validation", "Test")
    """
    if prefix:
        prefix = f"{prefix} - "

    print(f"{prefix}Calibration Metrics:")
    print(f"  Z-scores: μ = {metrics['z_mean']:+.4f} (ideal: 0.0), σ = {metrics['z_std']:.4f} (ideal: 1.0)")
    print(f"  Coverage: 1σ = {metrics['coverage_1sigma']:.1f}% (ideal: 68.3%), "
          f"2σ = {metrics['coverage_2sigma']:.1f}% (ideal: 95.4%), "
          f"3σ = {metrics['coverage_3sigma']:.1f}% (ideal: 99.7%)")


def evaluate_model(model, coords, embeddings, agbd_true, agbd_scale=200.0, log_transform=True):
    """
    Evaluate baseline model and compute metrics in both log and linear space.

    Args:
        model: Baseline model (RF, XGBoost, or IDW)
        coords: Coordinates for prediction
        embeddings: Embeddings for prediction
        agbd_true: True AGBD values in linear space (Mg/ha)
        agbd_scale: AGBD scale factor (default: 200.0)
        log_transform: Whether log transform was used (default: True)

    Returns:
        metrics: Dict with log_rmse, log_mae, log_r2, linear_rmse, linear_mae, and calibration metrics
        pred: Predictions in linear space (Mg/ha)
        pred_std_norm: Predicted std in normalized space
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
        **calibration_metrics  # Add calibration metrics
    }

    return metrics, pred, pred_std_norm


def train_random_forest(train_coords, train_embeddings, train_agbd, args,
                        val_coords=None, val_embeddings=None, val_agbd_norm=None):
    print("\n" + "=" * 80)
    print("Training Random Forest Baseline")
    print("=" * 80)

    model = RandomForestBaseline(
        n_estimators=args.rf_n_estimators,
        max_depth=args.rf_max_depth,
        random_state=args.seed
    )

    print(f"n_estimators: {args.rf_n_estimators}")
    print(f"max_depth: {args.rf_max_depth}")

    start_time = time()
    model.fit(train_coords, train_embeddings, train_agbd)
    train_time = time() - start_time

    print(f"Training completed in {train_time:.2f} seconds")

    return model, train_time


def train_xgboost(train_coords, train_embeddings, train_agbd, args,
                  val_coords=None, val_embeddings=None, val_agbd_norm=None):
    print("\n" + "=" * 80)
    print("Training XGBoost Baseline")
    print("=" * 80)

    model = XGBoostBaseline(
        n_estimators=args.xgb_n_estimators,
        max_depth=args.xgb_max_depth,
        learning_rate=args.xgb_learning_rate,
        random_state=args.seed
    )

    print(f"n_estimators: {args.xgb_n_estimators}")
    print(f"max_depth: {args.xgb_max_depth}")
    print(f"learning_rate: {args.xgb_learning_rate}")

    start_time = time()
    model.fit(train_coords, train_embeddings, train_agbd, fit_quantiles=True)
    train_time = time() - start_time

    print(f"Training completed in {train_time:.2f} seconds")

    return model, train_time


def train_linear_regression(train_coords, train_embeddings, train_agbd, args,
                            val_coords=None, val_embeddings=None, val_agbd_norm=None):
    print("\n" + "=" * 80)
    print("Training Linear Regression Baseline")
    print("=" * 80)

    model = LinearRegressionBaseline(
        fit_intercept=True,
        random_state=args.seed
    )

    print(f"fit_intercept: True")
    print(f"uncertainty: residual-based (constant std)")

    start_time = time()
    model.fit(train_coords, train_embeddings, train_agbd, fit_quantiles=True)
    train_time = time() - start_time

    print(f"Training completed in {train_time:.2f} seconds")

    return model, train_time


def train_idw(train_coords, train_embeddings, train_agbd, args):
    print("\n" + "=" * 80)
    print("Training IDW Baseline (Spatial Only)")
    print("=" * 80)

    model = IDWBaseline(
        power=args.idw_power,
        n_neighbors=args.idw_n_neighbors
    )

    print(f"power: {args.idw_power}")
    print(f"n_neighbors: {args.idw_n_neighbors}")
    print("Note: IDW ignores embeddings and uses only spatial coordinates")

    start_time = time()
    model.fit(train_coords, train_embeddings, train_agbd)
    train_time = time() - start_time

    print(f"Training completed in {train_time:.2f} seconds")

    return model, train_time


def main():
    args = parse_args()
    np.random.seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    save_config(vars(args), output_dir / 'config.json')

    print("=" * 80)
    print("GEDI Baseline Models Training")
    print("=" * 80)
    print(f"Models to train: {', '.join(args.models)}")
    print(f"Region: {args.region_bbox}")
    print(f"Output: {output_dir}")
    print()

    print("Step 1: Querying GEDI data...")
    querier = GEDIQuerier(cache_dir=args.cache_dir)
    gedi_df = querier.query_region_tiles(
        region_bbox=args.region_bbox,
        tile_size=0.1,
        start_time=args.start_time,
        end_time=args.end_time,
        max_agbd=500.0  # Cap at 500 Mg/ha to remove unrealistic outliers (e.g., 3000+)
    )
    print(f"Retrieved {len(gedi_df)} GEDI shots across {gedi_df['tile_id'].nunique()} tiles")
    print()

    if len(gedi_df) == 0:
        print("No GEDI data found in region. Exiting.")
        return

    print("Step 2: Extracting GeoTessera embeddings...")
    extractor = EmbeddingExtractor(
        year=args.embedding_year,
        patch_size=args.patch_size,
        cache_dir=args.cache_dir
    )
    gedi_df = extractor.extract_patches_batch(gedi_df, verbose=True)
    print()

    gedi_df = gedi_df[gedi_df['embedding_patch'].notna()]
    print(f"Retained {len(gedi_df)} shots with valid embeddings")
    print()

    with open(output_dir / 'processed_data.pkl', 'wb') as f:
        pickle.dump(gedi_df, f)

    # Step 3: Spatial split
    print("Step 3: Creating spatial train/val/test split...")
    print(f"Using BufferedSpatialSplitter with buffer_size={args.buffer_size}° (~{args.buffer_size*111:.0f}km)")
    splitter = BufferedSpatialSplitter(
        gedi_df,
        buffer_size=args.buffer_size,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        random_state=args.seed
    )
    train_df, val_df, test_df = splitter.split()
    print()

    train_df.to_csv(output_dir / 'train_split.csv', index=False)
    val_df.to_csv(output_dir / 'val_split.csv', index=False)
    test_df.to_csv(output_dir / 'test_split.csv', index=False)

    global_bounds = (
        train_df['longitude'].min(),
        train_df['latitude'].min(),
        train_df['longitude'].max(),
        train_df['latitude'].max()
    )
    print(f"Global bounds: lon [{global_bounds[0]:.4f}, {global_bounds[2]:.4f}], "
          f"lat [{global_bounds[1]:.4f}, {global_bounds[3]:.4f}]")
    print()

    config = vars(args)
    config['global_bounds'] = list(global_bounds)
    save_config(config, output_dir / 'config.json')

    print("Step 4: Preparing data for baseline models...")

    train_coords, train_embeddings, train_agbd_norm = prepare_data(
        train_df, log_transform=args.log_transform_agbd, agbd_scale=args.agbd_scale
    )
    val_coords, val_embeddings, val_agbd_norm = prepare_data(
        val_df, log_transform=args.log_transform_agbd, agbd_scale=args.agbd_scale
    )
    test_coords, test_embeddings, test_agbd_norm = prepare_data(
        test_df, log_transform=args.log_transform_agbd, agbd_scale=args.agbd_scale
    )

    train_coords = normalize_coords(train_coords, global_bounds)
    val_coords = normalize_coords(val_coords, global_bounds)
    test_coords = normalize_coords(test_coords, global_bounds)

    train_agbd = train_df['agbd'].values
    val_agbd = val_df['agbd'].values
    test_agbd = test_df['agbd'].values

    print(f"Training set: {len(train_coords)} shots")
    print(f"Validation set: {len(val_coords)} shots")
    print(f"Test set: {len(test_coords)} shots")
    print(f"Feature dimensions: coords={train_coords.shape[1]}, embeddings={train_embeddings.shape[1:]}")
    print()

    results = {}

    if 'rf' in args.models:
        model_rf, train_time = train_random_forest(
            train_coords, train_embeddings, train_agbd_norm, args,
            val_coords, val_embeddings, val_agbd_norm
        )

        print("\nEvaluating on validation set...")
        val_metrics, val_pred, _ = evaluate_model(
            model_rf, val_coords, val_embeddings, val_agbd, args.agbd_scale, args.log_transform_agbd
        )
        print(f"Validation - Log R²: {val_metrics['log_r2']:.4f}, Log RMSE: {val_metrics['log_rmse']:.4f}, Log MAE: {val_metrics['log_mae']:.4f}")
        print(f"             Linear RMSE: {val_metrics['linear_rmse']:.2f} Mg/ha, Linear MAE: {val_metrics['linear_mae']:.2f} Mg/ha")
        print_calibration_metrics(val_metrics, prefix="Validation")

        print("\nEvaluating on test set...")
        test_metrics, test_pred, _ = evaluate_model(
            model_rf, test_coords, test_embeddings, test_agbd, args.agbd_scale, args.log_transform_agbd
        )
        print(f"Test - Log R²: {test_metrics['log_r2']:.4f}, Log RMSE: {test_metrics['log_rmse']:.4f}, Log MAE: {test_metrics['log_mae']:.4f}")
        print(f"       Linear RMSE: {test_metrics['linear_rmse']:.2f} Mg/ha, Linear MAE: {test_metrics['linear_mae']:.2f} Mg/ha")
        print_calibration_metrics(test_metrics, prefix="Test")

        results['random_forest'] = {
            'train_time': train_time,
            'val_metrics': val_metrics,
            'test_metrics': test_metrics
        }

        with open(output_dir / 'random_forest.pkl', 'wb') as f:
            pickle.dump(model_rf, f)

    if 'xgb' in args.models:
        model_xgb, train_time = train_xgboost(
            train_coords, train_embeddings, train_agbd_norm, args,
            val_coords, val_embeddings, val_agbd_norm
        )

        print("\nEvaluating on validation set...")
        val_metrics, val_pred, _ = evaluate_model(
            model_xgb, val_coords, val_embeddings, val_agbd, args.agbd_scale, args.log_transform_agbd
        )
        print(f"Validation - Log R²: {val_metrics['log_r2']:.4f}, Log RMSE: {val_metrics['log_rmse']:.4f}, Log MAE: {val_metrics['log_mae']:.4f}")
        print(f"             Linear RMSE: {val_metrics['linear_rmse']:.2f} Mg/ha, Linear MAE: {val_metrics['linear_mae']:.2f} Mg/ha")
        print_calibration_metrics(val_metrics, prefix="Validation")

        print("\nEvaluating on test set...")
        test_metrics, test_pred, _ = evaluate_model(
            model_xgb, test_coords, test_embeddings, test_agbd, args.agbd_scale, args.log_transform_agbd
        )
        print(f"Test - Log R²: {test_metrics['log_r2']:.4f}, Log RMSE: {test_metrics['log_rmse']:.4f}, Log MAE: {test_metrics['log_mae']:.4f}")
        print(f"       Linear RMSE: {test_metrics['linear_rmse']:.2f} Mg/ha, Linear MAE: {test_metrics['linear_mae']:.2f} Mg/ha")
        print_calibration_metrics(test_metrics, prefix="Test")

        results['xgboost'] = {
            'train_time': train_time,
            'val_metrics': val_metrics,
            'test_metrics': test_metrics
        }

        with open(output_dir / 'xgboost.pkl', 'wb') as f:
            pickle.dump(model_xgb, f)

    if 'lr' in args.models:
        model_lr, train_time = train_linear_regression(
            train_coords, train_embeddings, train_agbd_norm, args,
            val_coords, val_embeddings, val_agbd_norm
        )

        print("\nEvaluating on validation set...")
        val_metrics, val_pred, _ = evaluate_model(
            model_lr, val_coords, val_embeddings, val_agbd, args.agbd_scale, args.log_transform_agbd
        )
        print(f"Validation - Log R²: {val_metrics['log_r2']:.4f}, Log RMSE: {val_metrics['log_rmse']:.4f}, Log MAE: {val_metrics['log_mae']:.4f}")
        print(f"             Linear RMSE: {val_metrics['linear_rmse']:.2f} Mg/ha, Linear MAE: {val_metrics['linear_mae']:.2f} Mg/ha")
        print_calibration_metrics(val_metrics, prefix="Validation")

        print("\nEvaluating on test set...")
        test_metrics, test_pred, _ = evaluate_model(
            model_lr, test_coords, test_embeddings, test_agbd, args.agbd_scale, args.log_transform_agbd
        )
        print(f"Test - Log R²: {test_metrics['log_r2']:.4f}, Log RMSE: {test_metrics['log_rmse']:.4f}, Log MAE: {test_metrics['log_mae']:.4f}")
        print(f"       Linear RMSE: {test_metrics['linear_rmse']:.2f} Mg/ha, Linear MAE: {test_metrics['linear_mae']:.2f} Mg/ha")
        print_calibration_metrics(test_metrics, prefix="Test")

        results['linear_regression'] = {
            'train_time': train_time,
            'val_metrics': val_metrics,
            'test_metrics': test_metrics
        }

        with open(output_dir / 'linear_regression.pkl', 'wb') as f:
            pickle.dump(model_lr, f)

    if 'idw' in args.models:
        model_idw, train_time = train_idw(
            train_coords, train_embeddings, train_agbd_norm, args
        )

        print("\nEvaluating on validation set...")
        val_metrics, val_pred, _ = evaluate_model(
            model_idw, val_coords, val_embeddings, val_agbd, args.agbd_scale, args.log_transform_agbd
        )
        print(f"Validation - Log R²: {val_metrics['log_r2']:.4f}, Log RMSE: {val_metrics['log_rmse']:.4f}, Log MAE: {val_metrics['log_mae']:.4f}")
        print(f"             Linear RMSE: {val_metrics['linear_rmse']:.2f} Mg/ha, Linear MAE: {val_metrics['linear_mae']:.2f} Mg/ha")
        print_calibration_metrics(val_metrics, prefix="Validation")

        print("\nEvaluating on test set...")
        test_metrics, test_pred, _ = evaluate_model(
            model_idw, test_coords, test_embeddings, test_agbd, args.agbd_scale, args.log_transform_agbd
        )
        print(f"Test - Log R²: {test_metrics['log_r2']:.4f}, Log RMSE: {test_metrics['log_rmse']:.4f}, Log MAE: {test_metrics['log_mae']:.4f}")
        print(f"       Linear RMSE: {test_metrics['linear_rmse']:.2f} Mg/ha, Linear MAE: {test_metrics['linear_mae']:.2f} Mg/ha")
        print_calibration_metrics(test_metrics, prefix="Test")

        results['idw'] = {
            'train_time': train_time,
            'val_metrics': val_metrics,
            'test_metrics': test_metrics
        }

        with open(output_dir / 'idw.pkl', 'wb') as f:
            pickle.dump(model_idw, f)

    print("\n" + "=" * 80)
    print("SUMMARY OF RESULTS")
    print("=" * 80)

    summary_table = []
    for model_name, model_results in results.items():
        row = {
            'Model': model_name.upper(),
            'Train Time (s)': f"{model_results['train_time']:.2f}",
            'Val Log R²': f"{model_results['val_metrics']['log_r2']:.4f}",
            'Val Log RMSE': f"{model_results['val_metrics']['log_rmse']:.4f}",
            'Val Log MAE': f"{model_results['val_metrics']['log_mae']:.4f}",
            'Val RMSE (Mg/ha)': f"{model_results['val_metrics']['linear_rmse']:.2f}",
            'Val MAE (Mg/ha)': f"{model_results['val_metrics']['linear_mae']:.2f}",
            'Test Log R²': f"{model_results['test_metrics']['log_r2']:.4f}",
            'Test Log RMSE': f"{model_results['test_metrics']['log_rmse']:.4f}",
            'Test Log MAE': f"{model_results['test_metrics']['log_mae']:.4f}",
            'Test RMSE (Mg/ha)': f"{model_results['test_metrics']['linear_rmse']:.2f}",
            'Test MAE (Mg/ha)': f"{model_results['test_metrics']['linear_mae']:.2f}",
        }
        summary_table.append(row)

    if summary_table:
        df_summary = pd.DataFrame(summary_table)
        print(df_summary.to_string(index=False))

    # Print calibration summary
    print("\n" + "=" * 80)
    print("CALIBRATION SUMMARY (Test Set)")
    print("=" * 80)
    print(f"{'Model':<20} {'Z-score μ':>12} {'Z-score σ':>12} {'1σ Cov%':>10} {'2σ Cov%':>10} {'3σ Cov%':>10}")
    print(f"{'':20} {'(ideal: 0)':>12} {'(ideal: 1)':>12} {'(68.3%)':>10} {'(95.4%)':>10} {'(99.7%)':>10}")
    print("-" * 80)
    for model_name, model_results in results.items():
        m = model_results['test_metrics']
        print(f"{model_name.upper():<20} {m['z_mean']:>+12.4f} {m['z_std']:>12.4f} "
              f"{m['coverage_1sigma']:>10.1f} {m['coverage_2sigma']:>10.1f} {m['coverage_3sigma']:>10.1f}")

    print("=" * 80)

    with open(output_dir / 'results.json', 'w') as f:
        json.dump(_make_serializable(results), f, indent=2)

    print(f"\nResults saved to: {output_dir}")
    print("Files:")
    print("  - config.json")
    print("  - results.json")
    for model in args.models:
        if model == 'rf':
            print("  - random_forest.pkl")
        elif model == 'xgb':
            print("  - xgboost.pkl")
        elif model == 'lr':
            print("  - linear_regression.pkl")
        elif model == 'idw':
            print("  - idw.pkl")
    print("=" * 80)

if __name__ == '__main__':
    main()
