#!/usr/bin/env python3
"""
Spatial Extrapolation Cross-Evaluation

Tests the "Holy Grail" of Neural Processes: graceful handling of out-of-distribution data.

Creates a 4x4 matrix where models trained on region A are tested on region B's test set.
Compares ANP vs XGBoost to demonstrate that:
- Both models fail on predictive accuracy (R²) when extrapolating
- XGBoost UQ crashes (confident but wrong predictions → low coverage)
- ANP UQ saves the day (recognizes OOD data → high epistemic uncertainty → maintains coverage)

Usage:
    python evaluate_spatial_extrapolation.py --results_dir ./regional_results
    python evaluate_spatial_extrapolation.py --results_dir ./regional_results --models anp xgboost
    python evaluate_spatial_extrapolation.py --results_dir ./regional_results --output_dir ./extrapolation_analysis
"""

import argparse
import json
import logging
import pickle
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from data.dataset import GEDINeuralProcessDataset, collate_neural_process
from models.neural_process import GEDINeuralProcess
from baselines.models import XGBoostBaseline
from utils.evaluation import compute_metrics, compute_calibration_metrics
from utils.config import load_config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Region definitions (must match run_regional_training.py)
REGIONS = {
    'maine': 'Maine (Temperate)',
    'tolima': 'Tolima (Tropical)',
    'hokkaido': 'Hokkaido (Boreal)',
    'sudtirol': 'Sudtirol (Alpine)'
}

REGION_ORDER = ['maine', 'tolima', 'hokkaido', 'sudtirol']


class SpatialExtrapolationEvaluator:
    """Evaluate models trained on one region against test sets from all regions."""

    def __init__(
        self,
        results_dir: Path,
        output_dir: Path,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
        num_context: int = 100,
        batch_size: int = 32
    ):
        self.results_dir = Path(results_dir)
        self.output_dir = Path(output_dir)
        self.device = device
        self.num_context = num_context
        self.batch_size = batch_size

        self.output_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Device: {self.device}")
        logger.info(f"Results directory: {self.results_dir}")
        logger.info(f"Output directory: {self.output_dir}")

    def load_anp_model(self, region: str) -> Optional[Tuple[GEDINeuralProcess, dict]]:
        """Load ANP model for a region."""
        # Try to find best model across seeds
        region_dir = self.results_dir / region / 'anp'

        if not region_dir.exists():
            logger.warning(f"ANP directory not found for {region}: {region_dir}")
            return None

        # Look for seed directories or direct checkpoint
        seed_dirs = list(region_dir.glob('seed_*'))

        if seed_dirs:
            # Use first seed (or best seed if we track that)
            model_dir = seed_dirs[0]
            logger.info(f"Loading ANP model from seed directory: {model_dir}")
        else:
            model_dir = region_dir

        checkpoint_path = model_dir / 'best_r2_model.pt'
        config_path = model_dir / 'config.json'

        if not checkpoint_path.exists():
            logger.warning(f"Checkpoint not found: {checkpoint_path}")
            return None

        if not config_path.exists():
            logger.warning(f"Config not found: {config_path}")
            return None

        # Load config
        with open(config_path, 'r') as f:
            config = json.load(f)

        # Initialize model
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
        )

        # Load weights
        checkpoint = torch.load(checkpoint_path, map_location=self.device,weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(self.device)
        model.eval()

        logger.info(f"Loaded ANP model for {region} from {checkpoint_path}")
        return model, config

    def load_xgboost_model(self, region: str) -> Optional[Tuple[XGBoostBaseline, dict]]:
        """Load XGBoost model for a region."""
        region_dir = self.results_dir / region / 'baselines'

        if not region_dir.exists():
            logger.warning(f"Baselines directory not found for {region}: {region_dir}")
            return None

        model_path = region_dir / 'xgboost.pkl'
        config_path = region_dir / 'config.json'

        if not model_path.exists():
            logger.warning(f"XGBoost model not found: {model_path}")
            return None

        if not config_path.exists():
            logger.warning(f"Config not found: {config_path}")
            return None

        # Load config
        with open(config_path, 'r') as f:
            config = json.load(f)

        # Load model
        with open(model_path, 'rb') as f:
            model = pickle.load(f)

        logger.info(f"Loaded XGBoost model for {region} from {model_path}")
        return model, config

    def load_test_data(self, region: str) -> Optional[DataLoader]:
        """Load test dataset for a region."""
        # Try to find test split in region directory
        region_dir = self.results_dir / region / 'anp'

        if not region_dir.exists():
            logger.warning(f"Region directory not found: {region_dir}")
            return None

        # Check seed directories first
        seed_dirs = list(region_dir.glob('seed_*'))
        if seed_dirs:
            test_split_path = seed_dirs[0] / 'test_split.parquet'
        else:
            test_split_path = region_dir / 'test_split.parquet'

        if not test_split_path.exists():
            logger.warning(f"Test split not found: {test_split_path}")
            return None

        # Load parquet
        df = pd.read_parquet(test_split_path)

        # Create dataset
        dataset = GEDINeuralProcessDataset(
            df=df,
            max_num_context=self.num_context,
            max_num_target=1000,  # Use all test points as targets
            normalize_coords=True
        )

        # Create dataloader
        dataloader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=collate_neural_process,
            num_workers=4
        )

        logger.info(f"Loaded test data for {region}: {len(dataset)} tiles")
        return dataloader

    def evaluate_anp_on_region(
        self,
        model: GEDINeuralProcess,
        dataloader: DataLoader,
        train_region: str,
        test_region: str
    ) -> dict:
        """Evaluate ANP model on a region's test set."""
        all_preds = []
        all_targets = []
        all_uncertainties = []

        logger.info(f"Evaluating ANP {train_region} → {test_region}")

        with torch.no_grad():
            for batch in tqdm(dataloader, desc=f"ANP {train_region}→{test_region}"):
                # Process each tile in the batch (batch is list of tiles)
                for i in range(len(batch['context_coords'])):
                    # Get data for this tile
                    context_coords = batch['context_coords'][i].to(self.device)
                    context_embeddings = batch['context_embeddings'][i].to(self.device)
                    context_agbd = batch['context_agbd'][i].to(self.device)
                    target_coords = batch['target_coords'][i].to(self.device)
                    target_embeddings = batch['target_embeddings'][i].to(self.device)
                    target_agbd = batch['target_agbd'][i].to(self.device)

                    # Forward pass
                    pred_mean, pred_log_var, _, _ = model(
                        context_coords,
                        context_embeddings,
                        context_agbd,
                        target_coords,
                        target_embeddings,
                        training=False
                    )

                    # Convert log_var to std
                    if pred_log_var is not None:
                        pred_std = torch.exp(0.5 * pred_log_var)
                    else:
                        pred_std = torch.zeros_like(pred_mean)

                    # Collect predictions
                    all_preds.append(pred_mean.cpu().numpy())
                    all_targets.append(target_agbd.cpu().numpy())
                    all_uncertainties.append(pred_std.cpu().numpy())

        # Concatenate
        preds = np.concatenate(all_preds, axis=0).squeeze()
        targets = np.concatenate(all_targets, axis=0).squeeze()
        uncertainties = np.concatenate(all_uncertainties, axis=0).squeeze()

        # Compute metrics
        metrics = compute_metrics(preds, targets, uncertainties)

        # Compute calibration metrics
        calib_metrics = compute_calibration_metrics(preds, targets, uncertainties)

        # Combine metrics
        results = {
            **metrics,
            **calib_metrics,
            'train_region': train_region,
            'test_region': test_region,
            'model_type': 'anp',
            'num_predictions': len(preds)
        }

        return results

    def evaluate_xgboost_on_region(
        self,
        model: XGBoostBaseline,
        dataloader: DataLoader,
        train_region: str,
        test_region: str
    ) -> dict:
        """Evaluate XGBoost model on a region's test set."""
        all_preds = []
        all_targets = []
        all_uncertainties = []

        logger.info(f"Evaluating XGBoost {train_region} → {test_region}")

        for batch in tqdm(dataloader, desc=f"XGB {train_region}→{test_region}"):
            # Process each tile in the batch (batch is list of tiles)
            for i in range(len(batch['target_coords'])):
                # Get data for this tile (only need target points for XGBoost)
                target_coords = batch['target_coords'][i].cpu().numpy()
                target_embeddings = batch['target_embeddings'][i].cpu().numpy()
                target_agbd = batch['target_agbd'][i].cpu().numpy()

                # Predict using XGBoost
                preds, uncertainties = model.predict(
                    target_coords,
                    target_embeddings,
                    return_std=True
                )

                all_preds.append(preds)
                all_targets.append(target_agbd.squeeze())
                all_uncertainties.append(uncertainties)

        # Concatenate
        preds = np.concatenate(all_preds, axis=0)
        targets = np.concatenate(all_targets, axis=0)
        uncertainties = np.concatenate(all_uncertainties, axis=0)

        # Compute metrics
        metrics = compute_metrics(preds, targets, uncertainties)

        # Compute calibration metrics
        calib_metrics = compute_calibration_metrics(preds, targets, uncertainties)

        # Combine metrics
        results = {
            **metrics,
            **calib_metrics,
            'train_region': train_region,
            'test_region': test_region,
            'model_type': 'xgboost',
            'num_predictions': len(preds)
        }

        return results

    def run_cross_evaluation(self, model_types: List[str] = ['anp', 'xgboost']) -> pd.DataFrame:
        """Run cross-evaluation matrix for specified model types."""
        results = []

        for model_type in model_types:
            logger.info(f"\n{'='*60}")
            logger.info(f"Evaluating {model_type.upper()} models")
            logger.info(f"{'='*60}\n")

            # Load all models
            models = {}
            for region in REGION_ORDER:
                if model_type == 'anp':
                    model_data = self.load_anp_model(region)
                elif model_type == 'xgboost':
                    model_data = self.load_xgboost_model(region)
                else:
                    logger.error(f"Unknown model type: {model_type}")
                    continue

                if model_data is not None:
                    models[region] = model_data

            # Load all test datasets
            test_loaders = {}
            for region in REGION_ORDER:
                loader = self.load_test_data(region)
                if loader is not None:
                    test_loaders[region] = loader

            # Cross-evaluation: train_region × test_region
            for train_region in REGION_ORDER:
                if train_region not in models:
                    logger.warning(f"Skipping {train_region} (model not found)")
                    continue

                model, config = models[train_region]

                for test_region in REGION_ORDER:
                    if test_region not in test_loaders:
                        logger.warning(f"Skipping {test_region} (test data not found)")
                        continue

                    test_loader = test_loaders[test_region]

                    # Evaluate
                    if model_type == 'anp':
                        result = self.evaluate_anp_on_region(
                            model, test_loader, train_region, test_region
                        )
                    elif model_type == 'xgboost':
                        result = self.evaluate_xgboost_on_region(
                            model, test_loader, train_region, test_region
                        )

                    results.append(result)

        # Convert to DataFrame
        df = pd.DataFrame(results)

        # Save results
        output_path = self.output_dir / 'spatial_extrapolation_results.csv'
        df.to_csv(output_path, index=False)
        logger.info(f"Saved results to {output_path}")

        return df

    def create_heatmap(
        self,
        df: pd.DataFrame,
        model_type: str,
        metric: str,
        metric_name: str,
        cmap: str = 'RdYlGn',
        vmin: Optional[float] = None,
        vmax: Optional[float] = None,
        reverse_cmap: bool = False,
        ax: Optional[plt.Axes] = None
    ) -> plt.Axes:
        """Create a heatmap for a specific model type and metric."""
        # Filter data
        df_model = df[df['model_type'] == model_type]

        # Pivot to matrix
        matrix = df_model.pivot(
            index='train_region',
            columns='test_region',
            values=metric
        )

        # Reorder to match REGION_ORDER
        matrix = matrix.reindex(index=REGION_ORDER, columns=REGION_ORDER)

        # Reverse colormap if needed (e.g., for RMSE where lower is better)
        if reverse_cmap:
            cmap = cmap + '_r'

        # Create heatmap
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 7))

        sns.heatmap(
            matrix,
            annot=True,
            fmt='.3f',
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            cbar_kws={'label': metric_name},
            ax=ax,
            linewidths=0.5,
            linecolor='gray'
        )

        # Labels
        ax.set_xlabel('Test Region', fontsize=12)
        ax.set_ylabel('Train Region', fontsize=12)
        ax.set_title(f'{model_type.upper()}: {metric_name}', fontsize=14, fontweight='bold')

        # Format tick labels
        ax.set_xticklabels([REGIONS[r] for r in REGION_ORDER], rotation=45, ha='right')
        ax.set_yticklabels([REGIONS[r] for r in REGION_ORDER], rotation=0)

        return ax

    def create_comparison_heatmap(
        self,
        df: pd.DataFrame,
        metric: str,
        metric_name: str,
        cmap: str = 'RdYlGn',
        reverse_cmap: bool = False
    ) -> plt.Figure:
        """Create side-by-side comparison heatmaps for ANP and XGBoost."""
        fig, axes = plt.subplots(1, 2, figsize=(16, 7))

        # Determine shared vmin/vmax for fair comparison
        vmin = df[metric].min()
        vmax = df[metric].max()

        # ANP heatmap
        self.create_heatmap(
            df, 'anp', metric, metric_name,
            cmap=cmap, vmin=vmin, vmax=vmax, reverse_cmap=reverse_cmap,
            ax=axes[0]
        )

        # XGBoost heatmap
        self.create_heatmap(
            df, 'xgboost', metric, metric_name,
            cmap=cmap, vmin=vmin, vmax=vmax, reverse_cmap=reverse_cmap,
            ax=axes[1]
        )

        plt.suptitle(f'Spatial Extrapolation: {metric_name}', fontsize=16, fontweight='bold', y=0.98)
        plt.tight_layout(rect=[0, 0, 1, 0.96])

        return fig

    def visualize_results(self, df: pd.DataFrame):
        """Create comprehensive visualizations of spatial extrapolation results."""
        logger.info("\nCreating visualizations...")

        # 1. R² comparison
        fig = self.create_comparison_heatmap(
            df, 'log_r2', 'Log R²', cmap='RdYlGn', reverse_cmap=False
        )
        fig.savefig(self.output_dir / 'extrapolation_r2.png', dpi=300, bbox_inches='tight')
        plt.close(fig)
        logger.info("Saved R² comparison")

        # 2. RMSE comparison
        fig = self.create_comparison_heatmap(
            df, 'log_rmse', 'Log RMSE', cmap='RdYlGn', reverse_cmap=True
        )
        fig.savefig(self.output_dir / 'extrapolation_rmse.png', dpi=300, bbox_inches='tight')
        plt.close(fig)
        logger.info("Saved RMSE comparison")

        # 3. Coverage comparison (THE KEY PLOT!)
        if 'coverage_1sigma' in df.columns:
            fig = self.create_comparison_heatmap(
                df, 'coverage_1sigma', '1σ Coverage', cmap='RdYlGn', reverse_cmap=False
            )
            # Add reference line at ideal coverage (0.68)
            for ax in fig.axes[:2]:
                ax.text(
                    0.5, -0.15, 'Ideal: 0.683',
                    transform=ax.transAxes,
                    ha='center', fontsize=10, style='italic'
                )
            fig.savefig(self.output_dir / 'extrapolation_coverage.png', dpi=300, bbox_inches='tight')
            plt.close(fig)
            logger.info("Saved coverage comparison")

        # 4. Mean uncertainty comparison
        if 'mean_uncertainty' in df.columns:
            fig = self.create_comparison_heatmap(
                df, 'mean_uncertainty', 'Mean Uncertainty (σ)', cmap='YlOrRd', reverse_cmap=False
            )
            fig.savefig(self.output_dir / 'extrapolation_uncertainty.png', dpi=300, bbox_inches='tight')
            plt.close(fig)
            logger.info("Saved uncertainty comparison")

        # 5. Summary statistics table
        self.create_summary_table(df)

    def create_summary_table(self, df: pd.DataFrame):
        """Create summary statistics table."""
        logger.info("\nCreating summary statistics...")

        # Separate diagonal (in-distribution) from off-diagonal (out-of-distribution)
        df['is_diagonal'] = df['train_region'] == df['test_region']
        df['split'] = df['is_diagonal'].map({True: 'In-Distribution', False: 'Out-of-Distribution'})

        # Compute summary statistics
        summary = df.groupby(['model_type', 'split']).agg({
            'log_r2': ['mean', 'std'],
            'log_rmse': ['mean', 'std'],
            'coverage_1sigma': ['mean', 'std'],
            'mean_uncertainty': ['mean', 'std']
        }).round(3)

        # Save to CSV
        output_path = self.output_dir / 'extrapolation_summary.csv'
        summary.to_csv(output_path)
        logger.info(f"Saved summary to {output_path}")

        # Print to console
        print("\n" + "="*80)
        print("SPATIAL EXTRAPOLATION SUMMARY")
        print("="*80)
        print(summary)
        print("="*80)

        # Compute degradation metrics
        self.compute_degradation_metrics(df)

    def compute_degradation_metrics(self, df: pd.DataFrame):
        """Compute how much performance degrades from in-dist to out-of-dist."""
        logger.info("\nComputing degradation metrics...")

        results = []

        for model_type in df['model_type'].unique():
            df_model = df[df['model_type'] == model_type]

            # In-distribution (diagonal)
            df_in = df_model[df_model['is_diagonal']]

            # Out-of-distribution (off-diagonal)
            df_out = df_model[~df_model['is_diagonal']]

            # Compute degradation
            degradation = {
                'model_type': model_type,
                'r2_in_dist': df_in['log_r2'].mean(),
                'r2_out_dist': df_out['log_r2'].mean(),
                'r2_drop': df_in['log_r2'].mean() - df_out['log_r2'].mean(),
                'r2_drop_pct': ((df_in['log_r2'].mean() - df_out['log_r2'].mean()) / df_in['log_r2'].mean() * 100),
                'coverage_in_dist': df_in['coverage_1sigma'].mean(),
                'coverage_out_dist': df_out['coverage_1sigma'].mean(),
                'coverage_drop': df_in['coverage_1sigma'].mean() - df_out['coverage_1sigma'].mean(),
                'coverage_drop_pct': ((df_in['coverage_1sigma'].mean() - df_out['coverage_1sigma'].mean()) / df_in['coverage_1sigma'].mean() * 100)
            }

            results.append(degradation)

        df_degradation = pd.DataFrame(results)

        # Save
        output_path = self.output_dir / 'degradation_metrics.csv'
        df_degradation.to_csv(output_path, index=False)
        logger.info(f"Saved degradation metrics to {output_path}")

        # Print
        print("\n" + "="*80)
        print("DEGRADATION METRICS (In-Distribution → Out-of-Distribution)")
        print("="*80)
        print(df_degradation.to_string(index=False))
        print("="*80)


def parse_args():
    parser = argparse.ArgumentParser(
        description='Spatial extrapolation cross-evaluation',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Evaluate both ANP and XGBoost
    python evaluate_spatial_extrapolation.py --results_dir ./regional_results

    # Evaluate only ANP
    python evaluate_spatial_extrapolation.py --results_dir ./regional_results --models anp

    # Custom output directory
    python evaluate_spatial_extrapolation.py \\
        --results_dir ./regional_results \\
        --output_dir ./extrapolation_analysis
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
        default='./extrapolation_results',
        help='Output directory for results (default: ./extrapolation_results)'
    )

    parser.add_argument(
        '--models',
        nargs='+',
        choices=['anp', 'xgboost'],
        default=['anp', 'xgboost'],
        help='Model types to evaluate (default: anp xgboost)'
    )

    parser.add_argument(
        '--num_context',
        type=int,
        default=100,
        help='Number of context points for ANP (default: 100)'
    )

    parser.add_argument(
        '--batch_size',
        type=int,
        default=32,
        help='Batch size for evaluation (default: 32)'
    )

    parser.add_argument(
        '--device',
        type=str,
        default='cuda' if torch.cuda.is_available() else 'cpu',
        choices=['cuda', 'cpu'],
        help='Device to use (default: auto-detect)'
    )

    return parser.parse_args()


def main():
    args = parse_args()

    # Create evaluator
    evaluator = SpatialExtrapolationEvaluator(
        results_dir=Path(args.results_dir),
        output_dir=Path(args.output_dir),
        device=args.device,
        num_context=args.num_context,
        batch_size=args.batch_size
    )

    # Run cross-evaluation
    logger.info("\nStarting spatial extrapolation cross-evaluation...")
    logger.info(f"Model types: {args.models}")
    logger.info(f"Regions: {list(REGIONS.values())}\n")

    df_results = evaluator.run_cross_evaluation(model_types=args.models)

    # Create visualizations
    evaluator.visualize_results(df_results)

    logger.info("\n" + "="*80)
    logger.info("EVALUATION COMPLETE!")
    logger.info(f"Results saved to: {evaluator.output_dir}")
    logger.info("="*80 + "\n")


if __name__ == '__main__':
    main()
