#!/usr/bin/env python3
"""
Analyze correlation between predicted uncertainty and distance to nearest training point.

Compares ANP vs XGBoost uncertainty behaviors as a function of distance from training data.

Usage:
    python analyze_uncertainty_distance.py --results_dir ./regional_results/maine --output_dir ./uncertainty_distance_analysis

    # Analyze specific region
    python analyze_uncertainty_distance.py --results_dir ./regional_results --region maine

    # Analyze all regions
    python analyze_uncertainty_distance.py --results_dir ./regional_results --all_regions
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
from scipy.spatial import cKDTree
from scipy.stats import pearsonr, spearmanr
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))

from data.dataset import GEDINeuralProcessDataset, collate_neural_process
from models.neural_process import GEDINeuralProcess
from baselines.models import XGBoostBaseline
from utils.normalization import normalize_coords

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

REGIONS = {
    'maine': 'Maine (Temperate)',
    'tolima': 'Tolima (Tropical)',
    'hokkaido': 'Hokkaido (Boreal)',
    'sudtirol': 'Sudtirol (Alpine)',
    'guaviare': 'Guaviare (Tropical)'
}


class UncertaintyDistanceAnalyzer:
    def __init__(
        self,
        results_dir: Path,
        output_dir: Path,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ):
        self.results_dir = Path(results_dir)
        self.output_dir = Path(output_dir)
        self.device = device

        self.output_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Device: {self.device}")
        logger.info(f"Results directory: {self.results_dir}")
        logger.info(f"Output directory: {self.output_dir}")

    def load_anp_model(self, region_dir: Path) -> Optional[Tuple[GEDINeuralProcess, dict]]:
        """Load ANP model from region directory."""
        anp_dir = region_dir / 'anp'

        if not anp_dir.exists():
            logger.warning(f"ANP directory not found: {anp_dir}")
            return None

        # Look for seed directories
        seed_dirs = list(anp_dir.glob('seed_*'))
        if seed_dirs:
            model_dir = seed_dirs[0]  # Use first seed
            logger.info(f"Loading ANP model from: {model_dir}")
        else:
            model_dir = anp_dir

        checkpoint_path = model_dir / 'best_r2_model.pt'
        config_path = model_dir / 'config.json'

        if not checkpoint_path.exists() or not config_path.exists():
            logger.warning(f"ANP checkpoint or config not found in {model_dir}")
            return None

        with open(config_path, 'r') as f:
            config = json.load(f)

        model = GEDINeuralProcess(
            patch_size=config.get('patch_size', 3),
            embedding_channels=128,
            embedding_feature_dim=config.get('embedding_feature_dim', 128),
            context_repr_dim=config.get('context_repr_dim', 128),
            hidden_dim=config.get('hidden_dim', 512),
            latent_dim=config.get('latent_dim', 128),
            output_uncertainty=True,
            architecture_mode=config.get('architecture_mode', 'anp'),
            num_attention_heads=config.get('num_attention_heads', 4)
        )

        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(self.device)
        model.eval()

        logger.info(f"Loaded ANP model (epoch {checkpoint.get('epoch', 'unknown')})")
        return model, config, model_dir

    def load_xgboost_model(self, region_dir: Path) -> Optional[Tuple[XGBoostBaseline, dict]]:
        """Load XGBoost model from region directory."""
        baseline_dir = region_dir / 'baselines'

        if not baseline_dir.exists():
            logger.warning(f"Baselines directory not found: {baseline_dir}")
            return None

        # Look for seed directories
        seed_dirs = list(baseline_dir.glob('seed_*'))
        if seed_dirs:
            model_dir = seed_dirs[0]  # Use first seed
            logger.info(f"Loading XGBoost model from: {model_dir}")
        else:
            model_dir = baseline_dir

        model_path = model_dir / 'xgboost.pkl'
        config_path = model_dir / 'config.json'

        if not model_path.exists() or not config_path.exists():
            logger.warning(f"XGBoost model or config not found in {model_dir}")
            return None

        with open(config_path, 'r') as f:
            config = json.load(f)

        with open(model_path, 'rb') as f:
            model = pickle.load(f)

        logger.info(f"Loaded XGBoost model")
        return model, config, model_dir

    def load_data_splits(self, model_dir: Path, patch_size: int = 3, embedding_channels: int = 128) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]:
        """Load training and test data splits."""
        train_split_path = model_dir / 'train_split.parquet'
        test_split_path = model_dir / 'test_split.parquet'

        if not train_split_path.exists() or not test_split_path.exists():
            logger.warning(f"Data splits not found in {model_dir}")
            return None, None

        train_df = pd.read_parquet(train_split_path)
        test_df = pd.read_parquet(test_split_path)

        # Reshape embeddings from flat arrays
        def reshape_embedding(flat_list):
            if flat_list is None:
                return None
            arr = np.array(flat_list)
            return arr.reshape(patch_size, patch_size, embedding_channels)

        train_df['embedding_patch'] = train_df['embedding_patch'].apply(reshape_embedding)
        test_df['embedding_patch'] = test_df['embedding_patch'].apply(reshape_embedding)

        # Filter out any None embeddings
        train_df = train_df[train_df['embedding_patch'].notna()].copy()
        test_df = test_df[test_df['embedding_patch'].notna()].copy()

        logger.info(f"Loaded {len(train_df)} training points, {len(test_df)} test points")
        return train_df, test_df

    def compute_nearest_distances(self, train_coords: np.ndarray, test_coords: np.ndarray) -> np.ndarray:
        """
        Compute distance from each test point to nearest training point.

        Args:
            train_coords: (N_train, 2) array of [lon, lat]
            test_coords: (N_test, 2) array of [lon, lat]

        Returns:
            distances: (N_test,) array of distances to nearest training point
        """
        logger.info(f"Computing nearest distances for {len(test_coords)} test points...")

        # Build KD-tree for efficient nearest neighbor search
        tree = cKDTree(train_coords)

        # Query nearest neighbor for each test point
        distances, indices = tree.query(test_coords, k=1)

        logger.info(f"Distance range: [{distances.min():.6f}, {distances.max():.6f}] degrees")
        logger.info(f"Mean distance: {distances.mean():.6f} degrees (~{distances.mean() * 111:.1f} km)")

        return distances

    def get_anp_predictions(
        self,
        model: GEDINeuralProcess,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        config: dict
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Get ANP predictions and uncertainties for test data."""
        logger.info("Running ANP predictions on test data...")

        global_bounds = config.get('global_bounds', None)

        # Prepare context (all training data)
        context_coords = train_df[['longitude', 'latitude']].values
        context_coords_norm = normalize_coords(context_coords, global_bounds)
        context_embeddings = np.stack(train_df['embedding_patch'].values)
        context_agbd = train_df['agbd'].values[:, None]

        # Normalize AGBD
        agbd_scale = config.get('agbd_scale', 200.0)
        log_transform = config.get('log_transform_agbd', True)
        if log_transform:
            context_agbd_norm = np.log1p(context_agbd) / np.log1p(agbd_scale)
        else:
            context_agbd_norm = context_agbd / agbd_scale

        # Convert to tensors
        context_coords_t = torch.from_numpy(context_coords_norm).float().to(self.device)
        context_embeddings_t = torch.from_numpy(context_embeddings).float().to(self.device)
        context_agbd_t = torch.from_numpy(context_agbd_norm).float().to(self.device)

        # Prepare test data
        test_coords = test_df[['longitude', 'latitude']].values
        test_coords_norm = normalize_coords(test_coords, global_bounds)
        test_embeddings = np.stack(test_df['embedding_patch'].values)

        all_predictions = []
        all_uncertainties = []

        batch_size = 512
        n_batches = (len(test_df) + batch_size - 1) // batch_size

        with torch.no_grad():
            for i in tqdm(range(n_batches), desc="ANP inference"):
                start_idx = i * batch_size
                end_idx = min((i + 1) * batch_size, len(test_df))

                batch_coords = test_coords_norm[start_idx:end_idx]
                batch_embeddings = test_embeddings[start_idx:end_idx]

                batch_coords_t = torch.from_numpy(batch_coords).float().to(self.device)
                batch_embeddings_t = torch.from_numpy(batch_embeddings).float().to(self.device)

                # Run forward pass
                pred_mean, pred_log_var, _, _ = model(
                    context_coords_t,
                    context_embeddings_t,
                    context_agbd_t,
                    batch_coords_t,
                    batch_embeddings_t,
                    training=False
                )

                # Convert log-variance to std
                if pred_log_var is not None:
                    pred_std = torch.exp(0.5 * pred_log_var)
                else:
                    pred_std = torch.zeros_like(pred_mean)

                all_predictions.append(pred_mean.cpu().numpy().flatten())
                all_uncertainties.append(pred_std.cpu().numpy().flatten())

        predictions = np.concatenate(all_predictions)
        uncertainties = np.concatenate(all_uncertainties)

        logger.info(f"ANP uncertainty range: [{uncertainties.min():.4f}, {uncertainties.max():.4f}]")
        logger.info(f"ANP mean uncertainty: {uncertainties.mean():.4f}")

        return predictions, uncertainties

    def get_xgboost_predictions(
        self,
        model: XGBoostBaseline,
        test_df: pd.DataFrame,
        config: dict
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Get XGBoost predictions and uncertainties for test data."""
        logger.info("Running XGBoost predictions on test data...")

        global_bounds = config.get('global_bounds', None)

        test_coords = test_df[['longitude', 'latitude']].values
        test_coords_norm = normalize_coords(test_coords, global_bounds)
        test_embeddings = np.stack(test_df['embedding_patch'].values)

        predictions, uncertainties = model.predict(
            test_coords_norm,
            test_embeddings,
            return_std=True
        )

        logger.info(f"XGBoost uncertainty range: [{uncertainties.min():.4f}, {uncertainties.max():.4f}]")
        logger.info(f"XGBoost mean uncertainty: {uncertainties.mean():.4f}")

        return predictions, uncertainties

    def compute_correlations(self, distances: np.ndarray, uncertainties: np.ndarray) -> Dict[str, float]:
        """Compute correlation metrics between distance and uncertainty."""
        pearson_r, pearson_p = pearsonr(distances, uncertainties)
        spearman_r, spearman_p = spearmanr(distances, uncertainties)

        return {
            'pearson_r': pearson_r,
            'pearson_p': pearson_p,
            'spearman_r': spearman_r,
            'spearman_p': spearman_p
        }

    def create_comparison_plot(
        self,
        anp_distances: np.ndarray,
        anp_uncertainties: np.ndarray,
        xgb_distances: np.ndarray,
        xgb_uncertainties: np.ndarray,
        anp_corr: Dict[str, float],
        xgb_corr: Dict[str, float],
        region_name: str,
        output_path: Path
    ):
        """Create comprehensive comparison visualization."""
        fig = plt.figure(figsize=(16, 10))
        gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)

        # Convert distances to km for better readability
        anp_distances_km = anp_distances * 111  # approximate conversion
        xgb_distances_km = xgb_distances * 111

        # 1. ANP Scatter plot
        ax1 = fig.add_subplot(gs[0, 0])
        scatter1 = ax1.scatter(
            anp_distances_km,
            anp_uncertainties,
            c=anp_uncertainties,
            cmap='viridis',
            alpha=0.3,
            s=10,
            edgecolors='none'
        )
        ax1.set_xlabel('Distance to Nearest Training Point (km)', fontsize=11)
        ax1.set_ylabel('Predicted Uncertainty (σ)', fontsize=11)
        ax1.set_title(f'ANP\nPearson r={anp_corr["pearson_r"]:.3f}, Spearman ρ={anp_corr["spearman_r"]:.3f}',
                     fontsize=12, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        plt.colorbar(scatter1, ax=ax1, label='Uncertainty')

        # 2. XGBoost Scatter plot
        ax2 = fig.add_subplot(gs[0, 1])
        scatter2 = ax2.scatter(
            xgb_distances_km,
            xgb_uncertainties,
            c=xgb_uncertainties,
            cmap='viridis',
            alpha=0.3,
            s=10,
            edgecolors='none'
        )
        ax2.set_xlabel('Distance to Nearest Training Point (km)', fontsize=11)
        ax2.set_ylabel('Predicted Uncertainty (σ)', fontsize=11)
        ax2.set_title(f'XGBoost\nPearson r={xgb_corr["pearson_r"]:.3f}, Spearman ρ={xgb_corr["spearman_r"]:.3f}',
                     fontsize=12, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        plt.colorbar(scatter2, ax=ax2, label='Uncertainty')

        # 3. Binned comparison
        ax3 = fig.add_subplot(gs[0, 2])
        n_bins = 20

        # Bin by distance for both models
        anp_bins = pd.qcut(anp_distances_km, q=n_bins, duplicates='drop')
        xgb_bins = pd.qcut(xgb_distances_km, q=n_bins, duplicates='drop')

        anp_binned = pd.DataFrame({
            'distance': anp_distances_km,
            'uncertainty': anp_uncertainties,
            'bin': anp_bins
        }).groupby('bin').agg({
            'distance': 'mean',
            'uncertainty': ['mean', 'std']
        })

        xgb_binned = pd.DataFrame({
            'distance': xgb_distances_km,
            'uncertainty': xgb_uncertainties,
            'bin': xgb_bins
        }).groupby('bin').agg({
            'distance': 'mean',
            'uncertainty': ['mean', 'std']
        })

        ax3.errorbar(
            anp_binned[('distance', 'mean')],
            anp_binned[('uncertainty', 'mean')],
            yerr=anp_binned[('uncertainty', 'std')],
            fmt='o-',
            label='ANP',
            capsize=3,
            linewidth=2,
            markersize=6,
            alpha=0.7
        )

        ax3.errorbar(
            xgb_binned[('distance', 'mean')],
            xgb_binned[('uncertainty', 'mean')],
            yerr=xgb_binned[('uncertainty', 'std')],
            fmt='s-',
            label='XGBoost',
            capsize=3,
            linewidth=2,
            markersize=6,
            alpha=0.7
        )

        ax3.set_xlabel('Distance to Nearest Training Point (km)', fontsize=11)
        ax3.set_ylabel('Mean Uncertainty (σ)', fontsize=11)
        ax3.set_title('Binned Comparison', fontsize=12, fontweight='bold')
        ax3.legend(fontsize=10)
        ax3.grid(True, alpha=0.3)

        # 4. ANP Hexbin
        ax4 = fig.add_subplot(gs[1, 0])
        hexbin1 = ax4.hexbin(
            anp_distances_km,
            anp_uncertainties,
            gridsize=30,
            cmap='YlOrRd',
            mincnt=1,
            reduce_C_function=np.median
        )
        ax4.set_xlabel('Distance to Nearest Training Point (km)', fontsize=11)
        ax4.set_ylabel('Predicted Uncertainty (σ)', fontsize=11)
        ax4.set_title('ANP Density', fontsize=12, fontweight='bold')
        plt.colorbar(hexbin1, ax=ax4, label='Count')

        # 5. XGBoost Hexbin
        ax5 = fig.add_subplot(gs[1, 1])
        hexbin2 = ax5.hexbin(
            xgb_distances_km,
            xgb_uncertainties,
            gridsize=30,
            cmap='YlOrRd',
            mincnt=1,
            reduce_C_function=np.median
        )
        ax5.set_xlabel('Distance to Nearest Training Point (km)', fontsize=11)
        ax5.set_ylabel('Predicted Uncertainty (σ)', fontsize=11)
        ax5.set_title('XGBoost Density', fontsize=12, fontweight='bold')
        plt.colorbar(hexbin2, ax=ax5, label='Count')

        # 6. Distribution comparison
        ax6 = fig.add_subplot(gs[1, 2])

        # Normalize uncertainties for fair comparison
        anp_norm = (anp_uncertainties - anp_uncertainties.mean()) / anp_uncertainties.std()
        xgb_norm = (xgb_uncertainties - xgb_uncertainties.mean()) / xgb_uncertainties.std()

        ax6.hist(anp_norm, bins=50, alpha=0.5, label='ANP (normalized)', density=True, color='blue')
        ax6.hist(xgb_norm, bins=50, alpha=0.5, label='XGBoost (normalized)', density=True, color='orange')
        ax6.set_xlabel('Normalized Uncertainty', fontsize=11)
        ax6.set_ylabel('Density', fontsize=11)
        ax6.set_title('Uncertainty Distributions', fontsize=12, fontweight='bold')
        ax6.legend(fontsize=10)
        ax6.grid(True, alpha=0.3, axis='y')

        plt.suptitle(
            f'Uncertainty vs Distance to Training Data: {region_name}',
            fontsize=14,
            fontweight='bold',
            y=0.995
        )

        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved comparison plot to {output_path}")
        plt.close()

    def analyze_region(self, region_dir: Path, region_name: str) -> Optional[Dict]:
        """Run complete analysis for a single region."""
        logger.info(f"\n{'='*80}")
        logger.info(f"ANALYZING REGION: {region_name}")
        logger.info(f"{'='*80}")

        # Load models
        anp_result = self.load_anp_model(region_dir)
        xgb_result = self.load_xgboost_model(region_dir)

        if anp_result is None or xgb_result is None:
            logger.error("Could not load both models. Skipping region.")
            return None

        anp_model, anp_config, anp_dir = anp_result
        xgb_model, xgb_config, xgb_dir = xgb_result

        # Load data (use ANP directory for data splits)
        train_df, test_df = self.load_data_splits(anp_dir)

        if train_df is None or test_df is None:
            logger.error("Could not load data splits. Skipping region.")
            return None

        # Compute distances
        train_coords = train_df[['longitude', 'latitude']].values
        test_coords = test_df[['longitude', 'latitude']].values
        distances = self.compute_nearest_distances(train_coords, test_coords)

        # Get predictions
        anp_preds, anp_uncertainties = self.get_anp_predictions(
            anp_model, train_df, test_df, anp_config
        )
        xgb_preds, xgb_uncertainties = self.get_xgboost_predictions(
            xgb_model, test_df, xgb_config
        )

        # Compute correlations
        anp_corr = self.compute_correlations(distances, anp_uncertainties)
        xgb_corr = self.compute_correlations(distances, xgb_uncertainties)

        logger.info(f"\nANP Correlations:")
        logger.info(f"  Pearson r: {anp_corr['pearson_r']:.4f} (p={anp_corr['pearson_p']:.2e})")
        logger.info(f"  Spearman ρ: {anp_corr['spearman_r']:.4f} (p={anp_corr['spearman_p']:.2e})")

        logger.info(f"\nXGBoost Correlations:")
        logger.info(f"  Pearson r: {xgb_corr['pearson_r']:.4f} (p={xgb_corr['pearson_p']:.2e})")
        logger.info(f"  Spearman ρ: {xgb_corr['spearman_r']:.4f} (p={xgb_corr['spearman_p']:.2e})")

        # Create visualization
        output_path = self.output_dir / f'{region_name}_uncertainty_distance.png'
        self.create_comparison_plot(
            distances, anp_uncertainties,
            distances, xgb_uncertainties,
            anp_corr, xgb_corr,
            REGIONS.get(region_name, region_name),
            output_path
        )

        # Save detailed results
        results_df = pd.DataFrame({
            'longitude': test_coords[:, 0],
            'latitude': test_coords[:, 1],
            'distance_to_train': distances,
            'distance_to_train_km': distances * 111,
            'anp_prediction': anp_preds,
            'anp_uncertainty': anp_uncertainties,
            'xgb_prediction': xgb_preds,
            'xgb_uncertainty': xgb_uncertainties,
            'true_agbd': test_df['agbd'].values
        })

        results_path = self.output_dir / f'{region_name}_detailed_results.csv'
        results_df.to_csv(results_path, index=False)
        logger.info(f"Saved detailed results to {results_path}")

        # Return summary
        return {
            'region': region_name,
            'n_train': len(train_df),
            'n_test': len(test_df),
            'anp_pearson_r': anp_corr['pearson_r'],
            'anp_pearson_p': anp_corr['pearson_p'],
            'anp_spearman_r': anp_corr['spearman_r'],
            'anp_spearman_p': anp_corr['spearman_p'],
            'xgb_pearson_r': xgb_corr['pearson_r'],
            'xgb_pearson_p': xgb_corr['pearson_p'],
            'xgb_spearman_r': xgb_corr['spearman_r'],
            'xgb_spearman_p': xgb_corr['spearman_p'],
            'mean_distance_km': distances.mean() * 111,
            'max_distance_km': distances.max() * 111,
            'anp_mean_uncertainty': anp_uncertainties.mean(),
            'xgb_mean_uncertainty': xgb_uncertainties.mean()
        }

    def create_multi_region_summary(self, all_results: List[Dict]):
        """Create summary visualization across all regions."""
        if not all_results:
            logger.warning("No results to summarize")
            return

        df = pd.DataFrame(all_results)

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # 1. Pearson correlation comparison
        ax = axes[0, 0]
        x = np.arange(len(df))
        width = 0.35
        ax.bar(x - width/2, df['anp_pearson_r'], width, label='ANP', alpha=0.8)
        ax.bar(x + width/2, df['xgb_pearson_r'], width, label='XGBoost', alpha=0.8)
        ax.set_xlabel('Region', fontsize=11)
        ax.set_ylabel('Pearson Correlation (r)', fontsize=11)
        ax.set_title('Pearson Correlation: Uncertainty vs Distance', fontsize=12, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(df['region'], rotation=45, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        ax.axhline(y=0, color='k', linestyle='-', linewidth=0.5)

        # 2. Spearman correlation comparison
        ax = axes[0, 1]
        ax.bar(x - width/2, df['anp_spearman_r'], width, label='ANP', alpha=0.8)
        ax.bar(x + width/2, df['xgb_spearman_r'], width, label='XGBoost', alpha=0.8)
        ax.set_xlabel('Region', fontsize=11)
        ax.set_ylabel('Spearman Correlation (ρ)', fontsize=11)
        ax.set_title('Spearman Correlation: Uncertainty vs Distance', fontsize=12, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(df['region'], rotation=45, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        ax.axhline(y=0, color='k', linestyle='-', linewidth=0.5)

        # 3. Mean uncertainty comparison
        ax = axes[1, 0]
        ax.bar(x - width/2, df['anp_mean_uncertainty'], width, label='ANP', alpha=0.8)
        ax.bar(x + width/2, df['xgb_mean_uncertainty'], width, label='XGBoost', alpha=0.8)
        ax.set_xlabel('Region', fontsize=11)
        ax.set_ylabel('Mean Uncertainty (σ)', fontsize=11)
        ax.set_title('Mean Predicted Uncertainty', fontsize=12, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(df['region'], rotation=45, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')

        # 4. Correlation scatter (ANP vs XGB)
        ax = axes[1, 1]
        ax.scatter(df['anp_pearson_r'], df['xgb_pearson_r'], s=100, alpha=0.7)
        for idx, row in df.iterrows():
            ax.annotate(row['region'], (row['anp_pearson_r'], row['xgb_pearson_r']),
                       fontsize=8, ha='right', va='bottom')

        # Add diagonal line
        min_val = min(df['anp_pearson_r'].min(), df['xgb_pearson_r'].min())
        max_val = max(df['anp_pearson_r'].max(), df['xgb_pearson_r'].max())
        ax.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.3, label='y=x')

        ax.set_xlabel('ANP Pearson r', fontsize=11)
        ax.set_ylabel('XGBoost Pearson r', fontsize=11)
        ax.set_title('Correlation Comparison', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend()

        plt.suptitle('Multi-Region Summary: Uncertainty vs Distance Analysis',
                    fontsize=14, fontweight='bold')
        plt.tight_layout()

        output_path = self.output_dir / 'multi_region_summary.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved multi-region summary to {output_path}")
        plt.close()

        # Save summary table
        summary_path = self.output_dir / 'correlation_summary.csv'
        df.to_csv(summary_path, index=False)
        logger.info(f"Saved correlation summary to {summary_path}")

        # Print summary
        print("\n" + "="*80)
        print("CORRELATION SUMMARY")
        print("="*80)
        print(df[['region', 'anp_pearson_r', 'xgb_pearson_r', 'anp_spearman_r', 'xgb_spearman_r']].to_string(index=False))
        print("="*80)


def parse_args():
    parser = argparse.ArgumentParser(
        description='Analyze uncertainty vs distance to training data (ANP vs XGBoost)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    parser.add_argument(
        '--results_dir',
        type=str,
        required=True,
        help='Directory containing regional results (e.g., ./regional_results or ./regional_results/maine)'
    )

    parser.add_argument(
        '--output_dir',
        type=str,
        default='./uncertainty_distance_analysis',
        help='Output directory for analysis results'
    )

    parser.add_argument(
        '--region',
        type=str,
        choices=list(REGIONS.keys()),
        help='Specific region to analyze (if results_dir points to regional_results parent)'
    )

    parser.add_argument(
        '--all_regions',
        action='store_true',
        help='Analyze all available regions in results_dir'
    )

    parser.add_argument(
        '--device',
        type=str,
        default='cuda' if torch.cuda.is_available() else 'cpu',
        choices=['cuda', 'cpu'],
        help='Device to use for inference'
    )

    return parser.parse_args()


def main():
    args = parse_args()

    results_dir = Path(args.results_dir)

    if not results_dir.exists():
        logger.error(f"Results directory not found: {results_dir}")
        return

    analyzer = UncertaintyDistanceAnalyzer(
        results_dir=results_dir,
        output_dir=Path(args.output_dir),
        device=args.device
    )

    all_results = []

    if args.all_regions:
        # Analyze all regions found in results_dir
        for region_key in REGIONS.keys():
            region_dir = results_dir / region_key
            if region_dir.exists():
                result = analyzer.analyze_region(region_dir, region_key)
                if result:
                    all_results.append(result)
    elif args.region:
        # Analyze specific region
        region_dir = results_dir / args.region
        if not region_dir.exists():
            logger.error(f"Region directory not found: {region_dir}")
            return
        result = analyzer.analyze_region(region_dir, args.region)
        if result:
            all_results.append(result)
    else:
        # Assume results_dir points directly to a region
        region_name = results_dir.name
        result = analyzer.analyze_region(results_dir, region_name)
        if result:
            all_results.append(result)

    # Create multi-region summary if multiple regions analyzed
    if len(all_results) > 1:
        analyzer.create_multi_region_summary(all_results)

    logger.info(f"\n{'='*80}")
    logger.info("ANALYSIS COMPLETE")
    logger.info(f"Results saved to: {analyzer.output_dir}")
    logger.info(f"{'='*80}\n")


if __name__ == '__main__':
    main()
