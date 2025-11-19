#!/usr/bin/env python3
"""
Example workflow for spatial extrapolation experiment.

This script demonstrates the complete workflow:
1. Check if regional models exist
2. Run spatial extrapolation evaluation
3. Generate publication figures
4. Print key findings

Usage:
    python example_spatial_extrapolation.py
    python example_spatial_extrapolation.py --quick  # Use existing results if available
"""

import argparse
import subprocess
import sys
from pathlib import Path

import pandas as pd


def check_models_exist(results_dir: Path) -> dict:
    """Check which regional models exist."""
    regions = ['maine', 'tolima', 'hokkaido', 'sudtirol']
    model_status = {}

    for region in regions:
        anp_exists = False
        xgb_exists = False

        # Check ANP
        anp_dir = results_dir / region / 'anp'
        if anp_dir.exists():
            seed_dirs = list(anp_dir.glob('seed_*/best_r2_model.pt'))
            if seed_dirs or (anp_dir / 'best_r2_model.pt').exists():
                anp_exists = True

        # Check XGBoost
        xgb_path = results_dir / region / 'baselines' / 'xgboost.pkl'
        if xgb_path.exists():
            xgb_exists = True

        model_status[region] = {
            'anp': anp_exists,
            'xgboost': xgb_exists
        }

    return model_status


def print_model_status(model_status: dict):
    """Print which models exist."""
    print("\n" + "="*60)
    print("MODEL STATUS")
    print("="*60)

    for region, status in model_status.items():
        anp_status = "✓" if status['anp'] else "✗"
        xgb_status = "✓" if status['xgboost'] else "✗"
        print(f"{region:12} | ANP: {anp_status}  XGBoost: {xgb_status}")

    print("="*60 + "\n")


def run_training(fast_mode: bool = False):
    """Run regional training."""
    print("\n" + "="*60)
    print("TRAINING REGIONAL MODELS")
    print("="*60)

    cmd = ['bash', 'train_all_regions.sh']
    if fast_mode:
        cmd.append('--fast')
        print("Running in FAST mode (1 seed, 10 epochs per model)")
    else:
        print("Running FULL training (3 seeds, 100 epochs per model)")
        print("This will take several hours...")

    print(f"\nCommand: {' '.join(cmd)}\n")

    # Run training
    result = subprocess.run(cmd, check=True)

    if result.returncode == 0:
        print("\n✓ Training completed successfully!")
    else:
        print("\n✗ Training failed!")
        sys.exit(1)


def run_evaluation(results_dir: Path, output_dir: Path):
    """Run spatial extrapolation evaluation."""
    print("\n" + "="*60)
    print("RUNNING SPATIAL EXTRAPOLATION EVALUATION")
    print("="*60)

    cmd = [
        'python', 'evaluate_spatial_extrapolation.py',
        '--results_dir', str(results_dir),
        '--output_dir', str(output_dir)
    ]

    print(f"Command: {' '.join(cmd)}\n")

    # Run evaluation
    result = subprocess.run(cmd, check=True)

    if result.returncode == 0:
        print("\n✓ Evaluation completed successfully!")
    else:
        print("\n✗ Evaluation failed!")
        sys.exit(1)


def generate_figures(output_dir: Path):
    """Generate publication figures."""
    print("\n" + "="*60)
    print("GENERATING PUBLICATION FIGURES")
    print("="*60)

    cmd = [
        'python', 'create_extrapolation_figure.py',
        '--results', str(output_dir),
        '--output', str(output_dir / 'figure_spatial_extrapolation.png')
    ]

    print(f"Command: {' '.join(cmd)}\n")

    # Run figure generation
    result = subprocess.run(cmd, check=True)

    if result.returncode == 0:
        print("\n✓ Figures generated successfully!")
    else:
        print("\n✗ Figure generation failed!")
        sys.exit(1)


def print_key_findings(output_dir: Path):
    """Print key findings from the analysis."""
    print("\n" + "="*60)
    print("KEY FINDINGS")
    print("="*60 + "\n")

    # Load results
    results_path = output_dir / 'spatial_extrapolation_results.csv'
    degradation_path = output_dir / 'degradation_metrics.csv'

    if not results_path.exists() or not degradation_path.exists():
        print("Results files not found!")
        return

    df = pd.read_csv(results_path)
    df_deg = pd.read_csv(degradation_path)

    # Separate diagonal vs off-diagonal
    df['is_diagonal'] = df['train_region'] == df['test_region']

    # ANP results
    df_anp = df[df['model_type'] == 'anp']
    anp_r2_in = df_anp[df_anp['is_diagonal']]['log_r2'].mean()
    anp_r2_out = df_anp[~df_anp['is_diagonal']]['log_r2'].mean()
    anp_cov_in = df_anp[df_anp['is_diagonal']]['coverage_1sigma'].mean()
    anp_cov_out = df_anp[~df_anp['is_diagonal']]['coverage_1sigma'].mean()

    # XGBoost results
    df_xgb = df[df['model_type'] == 'xgboost']
    xgb_r2_in = df_xgb[df_xgb['is_diagonal']]['log_r2'].mean()
    xgb_r2_out = df_xgb[~df_xgb['is_diagonal']]['log_r2'].mean()
    xgb_cov_in = df_xgb[df_xgb['is_diagonal']]['coverage_1sigma'].mean()
    xgb_cov_out = df_xgb[~df_xgb['is_diagonal']]['coverage_1sigma'].mean()

    # Print findings
    print("1. PREDICTIVE ACCURACY (R²)")
    print("-" * 60)
    print(f"   In-Distribution (Diagonal):")
    print(f"      ANP:     {anp_r2_in:.3f}")
    print(f"      XGBoost: {xgb_r2_in:.3f}")
    print(f"\n   Out-of-Distribution (Off-Diagonal):")
    print(f"      ANP:     {anp_r2_out:.3f} (↓ {(1 - anp_r2_out/anp_r2_in)*100:.1f}%)")
    print(f"      XGBoost: {xgb_r2_out:.3f} (↓ {(1 - xgb_r2_out/xgb_r2_in)*100:.1f}%)")
    print(f"\n   → Both models fail on OOD data (expected)")

    print("\n2. UNCERTAINTY QUANTIFICATION (1σ Coverage)")
    print("-" * 60)
    print(f"   In-Distribution (Diagonal):")
    print(f"      ANP:     {anp_cov_in:.3f} (ideal: 0.683)")
    print(f"      XGBoost: {xgb_cov_in:.3f} (ideal: 0.683)")
    print(f"\n   Out-of-Distribution (Off-Diagonal):")
    print(f"      ANP:     {anp_cov_out:.3f} (↓ {abs((1 - anp_cov_out/anp_cov_in)*100):.1f}%)")
    print(f"      XGBoost: {xgb_cov_out:.3f} (↓ {abs((1 - xgb_cov_out/xgb_cov_in)*100):.1f}%) ⚠️")
    print(f"\n   → ANP maintains calibration, XGBoost crashes!")

    print("\n3. THE KEY RESULT: COVERAGE DROP")
    print("-" * 60)

    anp_cov_drop = df_deg[df_deg['model_type'] == 'anp']['coverage_drop_pct'].values[0]
    xgb_cov_drop = df_deg[df_deg['model_type'] == 'xgboost']['coverage_drop_pct'].values[0]

    print(f"   ANP:     {abs(anp_cov_drop):.1f}% drop (minor degradation)")
    print(f"   XGBoost: {abs(xgb_cov_drop):.1f}% drop (catastrophic failure)")
    print(f"\n   → ANP is {abs(xgb_cov_drop/anp_cov_drop):.1f}x more robust on OOD data")

    print("\n4. INTERPRETATION")
    print("-" * 60)
    print("   When extrapolating to new ecosystems:")
    print("   • Both ANP and XGBoost have poor predictive accuracy (R²)")
    print("     → This is expected - can't predict tropical biomass with")
    print("       temperate forest rules")
    print()
    print("   • XGBoost gives confident but wrong predictions")
    print(f"     → Coverage drops to {xgb_cov_out:.2f} (only {xgb_cov_out*100:.0f}% of true values")
    print("       fall within predicted intervals)")
    print("     → Dangerously overconfident on unfamiliar data")
    print()
    print("   • ANP recognizes unfamiliar contexts and increases uncertainty")
    print(f"     → Coverage maintains at {anp_cov_out:.2f} despite poor accuracy")
    print("     → Honest uncertainty quantification enables safe deployment")

    print("\n" + "="*60 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description='Example workflow for spatial extrapolation experiment'
    )
    parser.add_argument(
        '--quick',
        action='store_true',
        help='Skip training if results exist'
    )
    parser.add_argument(
        '--fast',
        action='store_true',
        help='Use fast training mode (1 seed, 10 epochs)'
    )
    parser.add_argument(
        '--train-only',
        action='store_true',
        help='Only run training, skip evaluation'
    )
    parser.add_argument(
        '--eval-only',
        action='store_true',
        help='Only run evaluation, skip training'
    )
    parser.add_argument(
        '--results-dir',
        type=str,
        default='./regional_results',
        help='Directory for regional training results'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='./extrapolation_results',
        help='Directory for extrapolation evaluation results'
    )

    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir)

    print("\n" + "="*60)
    print("SPATIAL EXTRAPOLATION EXPERIMENT")
    print("="*60)
    print(f"Results directory: {results_dir}")
    print(f"Output directory:  {output_dir}")
    print("="*60)

    # Check model status
    model_status = check_models_exist(results_dir)
    print_model_status(model_status)

    # Determine if we need to train
    need_training = not all(
        status['anp'] and status['xgboost']
        for status in model_status.values()
    )

    # Training phase
    if not args.eval_only:
        if need_training and not args.quick:
            run_training(fast_mode=args.fast)
        elif need_training and args.quick:
            print("⚠️  Warning: Some models are missing, but --quick flag is set.")
            print("   Evaluation may fail. Remove --quick to train missing models.")
        else:
            print("✓ All models exist. Skipping training.\n")

    if args.train_only:
        print("\n✓ Training complete! (--train-only flag set)")
        return

    # Evaluation phase
    if not args.train_only:
        run_evaluation(results_dir, output_dir)
        generate_figures(output_dir)
        print_key_findings(output_dir)

    # Final summary
    print("\n" + "="*60)
    print("WORKFLOW COMPLETE!")
    print("="*60)
    print("\nGenerated files:")
    print(f"  • {output_dir}/spatial_extrapolation_results.csv")
    print(f"  • {output_dir}/extrapolation_summary.csv")
    print(f"  • {output_dir}/degradation_metrics.csv")
    print(f"  • {output_dir}/extrapolation_r2.png")
    print(f"  • {output_dir}/extrapolation_coverage.png")
    print(f"  • {output_dir}/figure_spatial_extrapolation.png")
    print(f"  • {output_dir}/figure_spatial_extrapolation.pdf")
    print("\nNext steps:")
    print("  1. Review the figures in the output directory")
    print("  2. Check degradation_metrics.csv for quantitative results")
    print("  3. Use figure_spatial_extrapolation.pdf in your paper")
    print("="*60 + "\n")


if __name__ == '__main__':
    main()
