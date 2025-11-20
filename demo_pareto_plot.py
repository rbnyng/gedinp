#!/usr/bin/env python3
"""
Generate synthetic Pareto sweep data and create visualization demo.

This script creates realistic synthetic data that mimics the output of sweep_baselines.py,
allowing you to test the visualization without running a full hyperparameter sweep.
"""

import numpy as np
import pandas as pd
from pathlib import Path


def generate_synthetic_sweep_data(n_seeds=5):
    """
    Generate synthetic Pareto sweep data that mimics real baseline results.

    Creates realistic patterns:
    - Deeper trees → lower RMSE but worse calibration
    - More estimators → slightly better accuracy, longer training
    - XGBoost generally more accurate than RF but worse calibrated
    """

    results = []

    # Random Forest configurations
    rf_depths = [1, 2, 3, 4, 6, 8, 10, 20]
    rf_estimators = [50, 100, 200, 500, 1000]

    for depth in rf_depths:
        for n_est in rf_estimators:
            # Accuracy improves with depth (but with diminishing returns)
            base_rmse = 0.45 - 0.15 * (1 - np.exp(-depth / 5))
            # Add noise based on n_estimators (more estimators = slightly better)
            rmse = base_rmse - 0.02 * np.log10(n_est / 50) + np.random.normal(0, 0.01)

            # Calibration degrades with depth (overconfidence)
            # Shallow models are underconfident, deep models overconfident
            z_std = 0.7 + 0.08 * depth + np.random.normal(0, 0.05)

            # Training time increases with depth and estimators
            train_time = (depth * n_est / 50) * np.random.uniform(0.8, 1.2)

            # Compute derived metrics
            calibration_error = abs(z_std - 1.0)
            z_mean = np.random.normal(0, 0.1)

            # Coverage (should match z_std - if overconfident, coverage is low)
            if z_std > 1.0:  # overconfident
                coverage_1sigma = 68.3 / z_std + np.random.normal(0, 2)
            else:  # underconfident
                coverage_1sigma = 68.3 * z_std + np.random.normal(0, 2)
            coverage_1sigma = np.clip(coverage_1sigma, 40, 90)

            coverage_2sigma = min(95.4, coverage_1sigma + 20 + np.random.normal(0, 2))
            coverage_3sigma = min(99.7, coverage_2sigma + 3 + np.random.normal(0, 1))

            results.append({
                'model_type': 'rf',
                'config_max_depth': depth,
                'config_n_estimators': n_est,
                'n_seeds': n_seeds,
                'train_time_mean': train_time,
                'train_time_std': train_time * 0.1,
                'log_rmse_mean': rmse,
                'log_rmse_std': rmse * 0.05,
                'log_r2_mean': 0.6 + (0.45 - rmse) * 2,  # R2 inversely related to RMSE
                'log_r2_std': 0.02,
                'z_mean_mean': z_mean,
                'z_mean_std': 0.05,
                'z_std_mean': z_std,
                'z_std_std': 0.08,
                'calibration_error_mean': calibration_error,
                'calibration_error_std': 0.03,
                'coverage_1sigma_mean': coverage_1sigma,
                'coverage_1sigma_std': 2.0,
                'coverage_2sigma_mean': coverage_2sigma,
                'coverage_2sigma_std': 1.5,
                'coverage_3sigma_mean': coverage_3sigma,
                'coverage_3sigma_std': 0.5,
            })

    # XGBoost configurations - generally better accuracy, worse calibration
    xgb_depths = [1, 2, 3, 4, 6, 8, 10, 20]
    xgb_estimators = [50, 100, 200, 500, 1000]

    for depth in xgb_depths:
        for n_est in xgb_estimators:
            # XGBoost is more accurate than RF
            base_rmse = 0.40 - 0.18 * (1 - np.exp(-depth / 4))
            rmse = base_rmse - 0.025 * np.log10(n_est / 50) + np.random.normal(0, 0.01)

            # XGBoost tends to be more overconfident
            z_std = 0.8 + 0.12 * depth + np.random.normal(0, 0.06)

            # Training time (XGBoost is slower)
            train_time = (depth * n_est / 40) * np.random.uniform(0.9, 1.3)

            calibration_error = abs(z_std - 1.0)
            z_mean = np.random.normal(0, 0.1)

            if z_std > 1.0:
                coverage_1sigma = 68.3 / z_std + np.random.normal(0, 2)
            else:
                coverage_1sigma = 68.3 * z_std + np.random.normal(0, 2)
            coverage_1sigma = np.clip(coverage_1sigma, 40, 90)

            coverage_2sigma = min(95.4, coverage_1sigma + 20 + np.random.normal(0, 2))
            coverage_3sigma = min(99.7, coverage_2sigma + 3 + np.random.normal(0, 1))

            results.append({
                'model_type': 'xgb',
                'config_max_depth': depth,
                'config_n_estimators': n_est,
                'config_learning_rate': 0.1,
                'n_seeds': n_seeds,
                'train_time_mean': train_time,
                'train_time_std': train_time * 0.12,
                'log_rmse_mean': rmse,
                'log_rmse_std': rmse * 0.05,
                'log_r2_mean': 0.65 + (0.40 - rmse) * 2.2,
                'log_r2_std': 0.02,
                'z_mean_mean': z_mean,
                'z_mean_std': 0.05,
                'z_std_mean': z_std,
                'z_std_std': 0.09,
                'calibration_error_mean': calibration_error,
                'calibration_error_std': 0.03,
                'coverage_1sigma_mean': coverage_1sigma,
                'coverage_1sigma_std': 2.0,
                'coverage_2sigma_mean': coverage_2sigma,
                'coverage_2sigma_std': 1.5,
                'coverage_3sigma_mean': coverage_3sigma,
                'coverage_3sigma_std': 0.5,
            })

    return pd.DataFrame(results)


def main():
    """Generate synthetic data and create demo plots."""

    print("Generating synthetic Pareto sweep data...")
    df = generate_synthetic_sweep_data(n_seeds=5)

    # Save synthetic data
    output_dir = Path('demo_outputs')
    output_dir.mkdir(exist_ok=True)

    csv_path = output_dir / 'synthetic_pareto_results.csv'
    df.to_csv(csv_path, index=False)
    print(f"Saved synthetic data to: {csv_path}")
    print(f"  Total configurations: {len(df)}")
    print(f"  RF: {len(df[df['model_type'] == 'rf'])}")
    print(f"  XGB: {len(df[df['model_type'] == 'xgb'])}")

    # Generate various demo plots
    import subprocess

    print("\nGenerating demo visualizations...")

    demos = [
        {
            'name': 'Main plot (z_std, separate panels)',
            'args': ['--show_pareto'],
            'output': 'demo_pareto_main.png'
        },
        {
            'name': 'Combined plot',
            'args': ['--combined', '--show_pareto'],
            'output': 'demo_pareto_combined.png'
        },
        {
            'name': 'Calibration error metric',
            'args': ['--y_metric', 'calibration_error', '--show_pareto'],
            'output': 'demo_pareto_calib_error.png'
        },
        {
            'name': 'Coverage metric',
            'args': ['--y_metric', 'coverage_1sigma', '--show_pareto'],
            'output': 'demo_pareto_coverage.png'
        },
        {
            'name': 'Log scale Y-axis',
            'args': ['--log_y', '--show_pareto'],
            'output': 'demo_pareto_log.png'
        },
    ]

    for demo in demos:
        print(f"\n  Creating: {demo['name']}")
        cmd = [
            'python', 'plot_pareto_frontier.py',
            '--input', str(csv_path),
            '--output', str(output_dir / demo['output']),
            '--dpi', '150'  # Lower DPI for demo
        ] + demo['args']

        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0:
            print(f"    ✓ Saved to: {output_dir / demo['output']}")
        else:
            print(f"    ✗ Error: {result.stderr}")

    print("\n" + "="*80)
    print("Demo complete! Check the demo_outputs/ directory for visualizations.")
    print("="*80)


if __name__ == '__main__':
    main()
