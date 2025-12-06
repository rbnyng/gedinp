"""
Few-shot sensitivity analysis: Test the effect of varying the number of few-shot training tiles.

This script runs multiple spatial extrapolation experiments with different numbers of
few-shot tiles (1, 3, 5, 7, 10, 12, 15) to understand:
- Sample efficiency of the model
- Optimal number of shots for practical deployment
- Learning curves for different train/test region combinations

Usage:
    # Run sensitivity sweep with default settings
    python run_fewshot_sensitivity.py --results_dir ./regional_results

    # Custom shot counts and parameters
    python run_fewshot_sensitivity.py --results_dir ./regional_results \
        --shot_counts 1 3 5 10 20 --few_shot_epochs 10
"""

import argparse
import json
import logging
import subprocess
import sys
from pathlib import Path
from typing import List

import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def run_evaluation(
    results_dir: Path,
    output_dir: Path,
    few_shot_tiles: int,
    few_shot_epochs: int = 5,
    few_shot_lr: float = 1e-4,
    models: List[str] = ['anp'],
    batch_size: int = 32,
    context_ratio: float = 0.5,
    device: str = 'cuda',
    seed: int = 42
) -> bool:
    """Run a single evaluation with specified number of few-shot tiles."""

    cmd = [
        sys.executable,
        'evaluate_spatial_extrapolation.py',
        '--results_dir', str(results_dir),
        '--output_dir', str(output_dir),
        '--few_shot_tiles', str(few_shot_tiles),
        '--few_shot_epochs', str(few_shot_epochs),
        '--few_shot_lr', str(few_shot_lr),
        '--include_zero_shot',  # Always include zero-shot for comparison
        '--batch_size', str(batch_size),
        '--context_ratio', str(context_ratio),
        '--device', device,
        '--seed', str(seed),
        '--models'
    ] + models

    logger.info(f"\n{'='*80}")
    logger.info(f"Running evaluation with {few_shot_tiles} few-shot tiles")
    logger.info(f"Command: {' '.join(cmd)}")
    logger.info(f"{'='*80}\n")

    try:
        result = subprocess.run(
            cmd,
            check=True,
            cwd=Path(__file__).parent,
            capture_output=False
        )
        logger.info(f"✓ Successfully completed evaluation for {few_shot_tiles} tiles")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"✗ Failed evaluation for {few_shot_tiles} tiles: {e}")
        return False


def aggregate_results(
    sweep_output_dir: Path,
    shot_counts: List[int]
) -> pd.DataFrame:
    """Aggregate results from all sweep runs."""

    all_results = []

    for n_shots in shot_counts:
        result_dir = sweep_output_dir / f'shots_{n_shots}'
        result_file = result_dir / 'spatial_extrapolation_results.csv'

        if not result_file.exists():
            logger.warning(f"Results not found for {n_shots} shots: {result_file}")
            continue

        df = pd.read_csv(result_file)
        df['n_shots'] = n_shots
        all_results.append(df)
        logger.info(f"Loaded results for {n_shots} shots: {len(df)} rows")

    if not all_results:
        logger.error("No results found to aggregate!")
        return pd.DataFrame()

    combined_df = pd.concat(all_results, ignore_index=True)

    # Save aggregated results
    output_file = sweep_output_dir / 'fewshot_sensitivity_all_results.csv'
    combined_df.to_csv(output_file, index=False)
    logger.info(f"\nSaved aggregated results to {output_file}")
    logger.info(f"Total rows: {len(combined_df)}")

    return combined_df


def create_metadata(
    sweep_output_dir: Path,
    shot_counts: List[int],
    args: argparse.Namespace
):
    """Save metadata about the sweep."""

    metadata = {
        'shot_counts': shot_counts,
        'few_shot_epochs': args.few_shot_epochs,
        'few_shot_lr': args.few_shot_lr,
        'models': args.models,
        'batch_size': args.batch_size,
        'context_ratio': args.context_ratio,
        'device': args.device,
        'seed': args.seed,
        'results_dir': str(args.results_dir)
    }

    metadata_file = sweep_output_dir / 'sweep_metadata.json'
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)

    logger.info(f"Saved sweep metadata to {metadata_file}")


def parse_args():
    parser = argparse.ArgumentParser(
        description='Few-shot sensitivity sweep for spatial extrapolation',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Run with default shot counts (1, 3, 5, 7, 10, 12, 15)
    python run_fewshot_sensitivity.py --results_dir ./regional_results

    # Custom shot counts
    python run_fewshot_sensitivity.py --results_dir ./regional_results \\
        --shot_counts 1 5 10 20 50

    # More epochs for better fine-tuning
    python run_fewshot_sensitivity.py --results_dir ./regional_results \\
        --few_shot_epochs 10
        """
    )

    parser.add_argument(
        '--results_dir',
        type=str,
        required=True,
        help='Directory containing regional training results'
    )

    parser.add_argument(
        '--output_dir',
        type=str,
        default='./fewshot_sensitivity_results',
        help='Output directory for sweep results (default: ./fewshot_sensitivity_results)'
    )

    parser.add_argument(
        '--shot_counts',
        nargs='+',
        type=int,
        default=[1, 3, 5, 7, 10, 12, 15],
        help='List of shot counts to test (default: 1 3 5 7 10 12 15)'
    )

    parser.add_argument(
        '--few_shot_epochs',
        type=int,
        default=5,
        help='Number of epochs for few-shot fine-tuning (default: 5)'
    )

    parser.add_argument(
        '--few_shot_lr',
        type=float,
        default=1e-4,
        help='Learning rate for few-shot fine-tuning (default: 1e-4)'
    )

    parser.add_argument(
        '--models',
        nargs='+',
        choices=['anp', 'xgboost', 'regression_kriging'],
        default=['anp'],
        help='Model types to evaluate (default: anp only, since baselines do not support few-shot)'
    )

    parser.add_argument(
        '--batch_size',
        type=int,
        default=32,
        help='Batch size for evaluation (default: 32)'
    )

    parser.add_argument(
        '--context_ratio',
        type=float,
        default=0.5,
        help='Ratio of context/target split within each tile (default: 0.5)'
    )

    parser.add_argument(
        '--device',
        type=str,
        default='cuda',
        choices=['cuda', 'cpu'],
        help='Device to use (default: cuda)'
    )

    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed (default: 42)'
    )

    parser.add_argument(
        '--skip_evaluation',
        action='store_true',
        help='Skip running evaluations and only aggregate existing results'
    )

    return parser.parse_args()


def main():
    args = parse_args()

    results_dir = Path(args.results_dir)
    sweep_output_dir = Path(args.output_dir)
    sweep_output_dir.mkdir(parents=True, exist_ok=True)

    shot_counts = sorted(args.shot_counts)

    logger.info("\n" + "="*80)
    logger.info("FEW-SHOT SENSITIVITY SWEEP")
    logger.info("="*80)
    logger.info(f"Results directory: {results_dir}")
    logger.info(f"Sweep output directory: {sweep_output_dir}")
    logger.info(f"Shot counts: {shot_counts}")
    logger.info(f"Few-shot epochs: {args.few_shot_epochs}")
    logger.info(f"Few-shot learning rate: {args.few_shot_lr}")
    logger.info(f"Models: {args.models}")
    logger.info(f"Device: {args.device}")
    logger.info("="*80 + "\n")

    # Run evaluations for each shot count
    if not args.skip_evaluation:
        success_count = 0
        for n_shots in shot_counts:
            output_dir = sweep_output_dir / f'shots_{n_shots}'

            success = run_evaluation(
                results_dir=results_dir,
                output_dir=output_dir,
                few_shot_tiles=n_shots,
                few_shot_epochs=args.few_shot_epochs,
                few_shot_lr=args.few_shot_lr,
                models=args.models,
                batch_size=args.batch_size,
                context_ratio=args.context_ratio,
                device=args.device,
                seed=args.seed
            )

            if success:
                success_count += 1

        logger.info("\n" + "="*80)
        logger.info(f"Completed {success_count}/{len(shot_counts)} evaluations successfully")
        logger.info("="*80 + "\n")
    else:
        logger.info("Skipping evaluations (--skip_evaluation flag set)")

    # Aggregate results
    logger.info("Aggregating results from all runs...")
    combined_df = aggregate_results(sweep_output_dir, shot_counts)

    if len(combined_df) > 0:
        # Create metadata
        create_metadata(sweep_output_dir, shot_counts, args)

        # Print summary
        logger.info("\n" + "="*80)
        logger.info("SWEEP SUMMARY")
        logger.info("="*80)

        for n_shots in shot_counts:
            df_shots = combined_df[combined_df['n_shots'] == n_shots]
            if len(df_shots) > 0:
                logger.info(f"\n{n_shots} shots: {len(df_shots)} result rows")

        logger.info("\n" + "="*80)
        logger.info(f"All results saved to: {sweep_output_dir}")
        logger.info("="*80)
        logger.info("\nNext steps:")
        logger.info(f"  Run analysis: python analyze_fewshot_sensitivity.py --results_dir {sweep_output_dir}")
        logger.info("="*80 + "\n")
    else:
        logger.error("No results were aggregated. Check logs for errors.")
        return 1

    return 0


if __name__ == '__main__':
    sys.exit(main())
