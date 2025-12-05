"""
Usage:
    # Zero-shot only (default)
    python evaluate_spatial_extrapolation.py --results_dir ./regional_results --output_dir ./extrapolation_analysis

    # Few-shot with 10 training tiles from target region
    python evaluate_spatial_extrapolation.py --results_dir ./regional_results --few_shot_tiles 10 --few_shot_epochs 5

    # Both zero-shot and few-shot
    python evaluate_spatial_extrapolation.py --results_dir ./regional_results --few_shot_tiles 10 --include_zero_shot
"""

import argparse
import copy
import json
import logging
import pickle
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import random
import os
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))

from data.dataset import GEDINeuralProcessDataset, collate_neural_process
from models.neural_process import GEDINeuralProcess, kl_divergence_gaussian
from baselines.models import XGBoostBaseline, RegressionKrigingBaseline
from utils.evaluation import compute_metrics, compute_calibration_metrics
from utils.config import load_config

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

REGIONS = {
    'maine': 'Maine (Temperate)',
    'tolima': 'Tolima (Tropical)',
    'hokkaido': 'Hokkaido (Boreal)',
    'sudtirol': 'Sudtirol (Alpine)'
}

REGION_ORDER = ['maine', 'tolima', 'hokkaido', 'sudtirol']


def create_centered_diverging_colormap(name='GreenRed'):
    colors = ['#d73027', '#fee08b', '#1a9850', '#fee08b', '#d73027']  # Red → Yellow → Green → Yellow → Red
    n_bins = 256
    cmap = LinearSegmentedColormap.from_list(name, colors, N=n_bins)
    return cmap


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    logger.info(f"Random seed set to {seed} for reproducibility")


class SpatialExtrapolationEvaluator:
    def __init__(
        self,
        results_dir: Path,
        output_dir: Path,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
        context_ratio: float = 0.5,
        batch_size: int = 32,
        few_shot_tiles: Optional[int] = None,
        few_shot_epochs: int = 5,
        few_shot_lr: float = 1e-4,
        few_shot_batch_size: Optional[int] = None,
        include_zero_shot: bool = True
    ):
        self.results_dir = Path(results_dir)
        self.output_dir = Path(output_dir)
        self.device = device
        self.context_ratio = context_ratio
        self.batch_size = batch_size
        self.few_shot_tiles = few_shot_tiles
        self.few_shot_epochs = few_shot_epochs
        self.few_shot_lr = few_shot_lr
        self.few_shot_batch_size = few_shot_batch_size if few_shot_batch_size is not None else max(1, batch_size // 4)
        self.include_zero_shot = include_zero_shot

        self.output_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Device: {self.device}")
        logger.info(f"Context ratio: {context_ratio*100:.0f}% (for within-tile context/target split)")
        if few_shot_tiles is not None:
            logger.info(f"Few-shot fine-tuning: {few_shot_tiles} tiles, {few_shot_epochs} epochs, lr={few_shot_lr}, batch_size={self.few_shot_batch_size}")
            if include_zero_shot:
                logger.info("Will also evaluate zero-shot performance for comparison")
        else:
            logger.info("Zero-shot evaluation only (no few-shot fine-tuning)")
        logger.info(f"Results directory: {self.results_dir}")
        logger.info(f"Output directory: {self.output_dir}")

    def load_anp_model(self, region: str) -> Optional[List[Tuple[GEDINeuralProcess, dict, str]]]:
        region_dir = self.results_dir / region / 'anp'

        if not region_dir.exists():
            logger.warning(f"ANP directory not found for {region}: {region_dir}")
            return None

        seed_dirs = sorted(list(region_dir.glob('seed_*')))

        if not seed_dirs:
            model_dir = region_dir
            checkpoint_path = model_dir / 'best_r2_model.pt'
            config_path = model_dir / 'config.json'

            if not checkpoint_path.exists() or not config_path.exists():
                logger.warning(f"No seed directories and no model in {region_dir}")
                return None

            with open(config_path, 'r') as f:
                config = json.load(f)

            model = self._load_single_anp_model(checkpoint_path, config)
            logger.info(f"Loaded ANP model for {region} from {checkpoint_path}")
            return [(model, config, 'default')]

        models = []
        for model_dir in seed_dirs:
            seed_id = model_dir.name
            checkpoint_path = model_dir / 'best_r2_model.pt'
            config_path = model_dir / 'config.json'

            if not checkpoint_path.exists():
                logger.warning(f"Checkpoint not found: {checkpoint_path}")
                continue

            if not config_path.exists():
                logger.warning(f"Config not found: {config_path}")
                continue

            with open(config_path, 'r') as f:
                config = json.load(f)

            model = self._load_single_anp_model(checkpoint_path, config)
            models.append((model, config, seed_id))
            logger.info(f"Loaded ANP model for {region} from {seed_id}")

        if not models:
            logger.warning(f"No valid ANP models found for {region}")
            return None

        logger.info(f"Loaded {len(models)} ANP model seeds for {region}")
        return models

    def _load_single_anp_model(self, checkpoint_path: Path, config: dict) -> GEDINeuralProcess:
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

        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()

        return model

    def load_xgboost_model(self, region: str) -> Optional[List[Tuple[XGBoostBaseline, dict, str]]]:
        region_dir = self.results_dir / region / 'baselines'

        if not region_dir.exists():
            logger.warning(f"Baselines directory not found for {region}: {region_dir}")
            return None

        seed_dirs = sorted(list(region_dir.glob('seed_*')))

        if not seed_dirs:
            model_dir = region_dir
            model_path = model_dir / 'xgboost.pkl'
            config_path = model_dir / 'config.json'

            if not model_path.exists() or not config_path.exists():
                logger.warning(f"No seed directories and no model in {region_dir}")
                return None

            with open(config_path, 'r') as f:
                config = json.load(f)

            with open(model_path, 'rb') as f:
                model = pickle.load(f)

            logger.info(f"Loaded XGBoost model for {region} from {model_path}")
            return [(model, config, 'default')]

        models = []
        for model_dir in seed_dirs:
            seed_id = model_dir.name
            model_path = model_dir / 'xgboost.pkl'
            config_path = model_dir / 'config.json'

            if not model_path.exists():
                logger.warning(f"XGBoost model not found: {model_path}")
                continue

            if not config_path.exists():
                logger.warning(f"Config not found: {config_path}")
                continue

            with open(config_path, 'r') as f:
                config = json.load(f)

            with open(model_path, 'rb') as f:
                model = pickle.load(f)

            models.append((model, config, seed_id))
            logger.info(f"Loaded XGBoost model for {region} from {seed_id}")

        if not models:
            logger.warning(f"No valid XGBoost models found for {region}")
            return None

        logger.info(f"Loaded {len(models)} XGBoost model seeds for {region}")
        return models

    def load_regression_kriging_model(self, region: str) -> Optional[List[Tuple[RegressionKrigingBaseline, dict, str]]]:
        region_dir = self.results_dir / region / 'baselines'

        if not region_dir.exists():
            logger.warning(f"Baselines directory not found for {region}: {region_dir}")
            return None

        seed_dirs = sorted(list(region_dir.glob('seed_*')))

        if not seed_dirs:
            model_dir = region_dir
            model_path = model_dir / 'regression_kriging.pkl'
            config_path = model_dir / 'config.json'

            if not model_path.exists() or not config_path.exists():
                logger.warning(f"No seed directories and no model in {region_dir}")
                return None

            with open(config_path, 'r') as f:
                config = json.load(f)

            with open(model_path, 'rb') as f:
                model = pickle.load(f)

            logger.info(f"Loaded Regression Kriging model for {region} from {model_path}")
            return [(model, config, 'default')]

        models = []
        for model_dir in seed_dirs:
            seed_id = model_dir.name
            model_path = model_dir / 'regression_kriging.pkl'
            config_path = model_dir / 'config.json'

            if not model_path.exists():
                logger.warning(f"Regression Kriging model not found: {model_path}")
                continue

            if not config_path.exists():
                logger.warning(f"Config not found: {config_path}")
                continue

            with open(config_path, 'r') as f:
                config = json.load(f)

            with open(model_path, 'rb') as f:
                model = pickle.load(f)

            models.append((model, config, seed_id))
            logger.info(f"Loaded Regression Kriging model for {region} from {seed_id}")

        if not models:
            logger.warning(f"No valid Regression Kriging models found for {region}")
            return None

        logger.info(f"Loaded {len(models)} Regression Kriging model seeds for {region}")
        return models

    def load_target_region_data(
        self,
        region: str,
        seed_id: Optional[str] = None,
        split_for_fewshot: bool = False
    ) -> Optional[Dict[str, DataLoader]]:
        region_dir = self.results_dir / region / 'anp'

        if not region_dir.exists():
            logger.warning(f"Region directory not found: {region_dir}")
            return None

        # Determine paths based on seed_id
        if seed_id is not None and seed_id != 'default':
            test_split_path = region_dir / seed_id / 'test_split.parquet'
            train_split_path = region_dir / seed_id / 'train_split.parquet'
            config_path = region_dir / seed_id / 'config.json'
        else:
            # Fallback: load from first seed or root directory
            seed_dirs = list(region_dir.glob('seed_*'))
            if seed_dirs:
                test_split_path = seed_dirs[0] / 'test_split.parquet'
                train_split_path = seed_dirs[0] / 'train_split.parquet'
                config_path = seed_dirs[0] / 'config.json'
            else:
                test_split_path = region_dir / 'test_split.parquet'
                train_split_path = region_dir / 'train_split.parquet'
                config_path = region_dir / 'config.json'

        if not test_split_path.exists():
            logger.warning(f"Test split not found: {test_split_path}")
            return None

        df_test = pd.read_parquet(test_split_path)

        patch_size = 3
        embedding_channels = 128

        def reshape_embedding(flat_list):
            if flat_list is None:
                return None
            arr = np.array(flat_list)
            return arr.reshape(patch_size, patch_size, embedding_channels)

        df_test['embedding_patch'] = df_test['embedding_patch'].apply(reshape_embedding)

        global_bounds = None
        if config_path.exists():
            with open(config_path, 'r') as f:
                config = json.load(f)
                global_bounds = config.get('global_bounds', None)

        context_ratio = self.context_ratio

        result = {}

        # Load train split for few-shot fine-tuning
        if split_for_fewshot and self.few_shot_tiles is not None and self.few_shot_tiles > 0:
            if not train_split_path.exists():
                logger.warning(f"Train split not found for few-shot: {train_split_path}. Using zero-shot only.")
                split_for_fewshot = False
            else:
                df_train = pd.read_parquet(train_split_path)
                df_train['embedding_patch'] = df_train['embedding_patch'].apply(reshape_embedding)

                unique_train_tiles = df_train['tile_id'].unique()
                n_train_tiles = len(unique_train_tiles)

                if n_train_tiles < self.few_shot_tiles:
                    logger.warning(f"Not enough training tiles in {region} ({n_train_tiles}) for few-shot "
                                 f"(requested {self.few_shot_tiles} tiles). Using all available training tiles.")
                    n_tiles_to_use = n_train_tiles
                else:
                    n_tiles_to_use = self.few_shot_tiles

                sampled_train_tiles = np.random.choice(unique_train_tiles, size=n_tiles_to_use, replace=False)
                df_train_fewshot = df_train[df_train['tile_id'].isin(sampled_train_tiles)]

                # training dataset for few-shot fine-tuning
                train_dataset = GEDINeuralProcessDataset(
                    data_df=df_train_fewshot,
                    min_shots_per_tile=2,
                    context_ratio_range=(context_ratio, context_ratio),
                    normalize_coords=True,
                    augment_coords=True,
                    coord_noise_std=0.01,
                    global_bounds=global_bounds
                )

                train_loader = DataLoader(
                    train_dataset,
                    batch_size=self.few_shot_batch_size,
                    shuffle=True,
                    collate_fn=collate_neural_process,
                    num_workers=4
                )

                result['train'] = train_loader

                seed_info = f" (seed: {seed_id})" if seed_id else ""
                logger.info(f"Loaded {region}{seed_info} for few-shot: {len(train_dataset)} train tiles (from training split)")

        # create test dataset from test split
        test_dataset = GEDINeuralProcessDataset(
            data_df=df_test,
            min_shots_per_tile=2,
            context_ratio_range=(context_ratio, context_ratio),
            normalize_coords=True,
            augment_coords=False,
            coord_noise_std=0.0,
            global_bounds=global_bounds
        )

        test_loader = DataLoader(
            test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=collate_neural_process,
            num_workers=4
        )

        result['test'] = test_loader

        seed_info = f" (seed: {seed_id})" if seed_id else ""
        if 'train' not in result:
            logger.info(f"Loaded {region}{seed_info}: {len(test_dataset)} test tiles")
        else:
            logger.info(f"Loaded {region}{seed_info}: {len(test_dataset)} test tiles (for evaluation)")

        return result

    def fine_tune_anp_model(
        self,
        model: GEDINeuralProcess,
        train_loader: DataLoader,
        train_region: str,
        test_region: str
    ) -> GEDINeuralProcess:
        original_device = next(model.parameters()).device
        model_cpu = model.cpu()
        fine_tuned_model = copy.deepcopy(model_cpu)

        model.to(original_device)
        fine_tuned_model.to(self.device)
        fine_tuned_model.train()

        optimizer = torch.optim.Adam(fine_tuned_model.parameters(), lr=self.few_shot_lr)

        logger.info(f"Fine-tuning ANP {train_region} → {test_region} for {self.few_shot_epochs} epochs")

        for epoch in range(self.few_shot_epochs):
            epoch_loss = 0.0
            n_batches = 0

            for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{self.few_shot_epochs}", leave=False):
                optimizer.zero_grad()
                tile_losses = []
                n_tiles = len(batch['context_coords'])

                # losses from all tiles in the batch
                for i in range(n_tiles):
                    context_coords = batch['context_coords'][i].to(self.device)
                    context_embeddings = batch['context_embeddings'][i].to(self.device)
                    context_agbd = batch['context_agbd'][i].to(self.device)
                    target_coords = batch['target_coords'][i].to(self.device)
                    target_embeddings = batch['target_embeddings'][i].to(self.device)
                    target_agbd = batch['target_agbd'][i].to(self.device)

                    pred_mean, pred_log_var, z_mu_context, z_log_sigma_context, z_mu_all, z_log_sigma_all = fine_tuned_model(
                        context_coords,
                        context_embeddings,
                        context_agbd,
                        target_coords,
                        target_embeddings,
                        query_agbd=target_agbd,
                        training=True
                    )

                    recon_loss = torch.nn.functional.mse_loss(pred_mean, target_agbd, reduction='mean')

                    # Compute KL divergence if latent path is used
                    if z_mu_context is not None and z_log_sigma_context is not None:
                        if z_mu_all is not None and z_log_sigma_all is not None:
                            # ANP: KL[p(z|C,T) || q(z|C)]
                            kl_loss = kl_divergence_gaussian(
                                z_mu_context, z_log_sigma_context,
                                z_mu_all, z_log_sigma_all
                            )
                        else:
                            # Latent-only: KL[q(z|C) || N(0,1)]
                            kl_loss = kl_divergence_gaussian(z_mu_context, z_log_sigma_context)
                        tile_loss = recon_loss + 0.01 * kl_loss
                    else:
                        tile_loss = recon_loss

                    tile_losses.append(tile_loss)

                # Average loss across tiles in the batch
                batch_loss = torch.stack(tile_losses).mean()

                # Backprop and update once per batch
                batch_loss.backward()
                optimizer.step()

                epoch_loss += batch_loss.item()
                n_batches += 1

            avg_loss = epoch_loss / n_batches if n_batches > 0 else 0.0
            logger.info(f"  Epoch {epoch+1}: avg_loss = {avg_loss:.4f}")

        fine_tuned_model.eval()
        logger.info(f"Fine-tuning complete for {train_region} → {test_region}")

        return fine_tuned_model

    def evaluate_anp_on_region(
        self,
        model: GEDINeuralProcess,
        dataloader: DataLoader,
        train_region: str,
        test_region: str
    ) -> dict:
        all_preds = []
        all_targets = []
        all_uncertainties = []

        logger.info(f"Evaluating ANP {train_region} → {test_region}")

        with torch.no_grad():
            for batch in tqdm(dataloader, desc=f"ANP {train_region}→{test_region}"):
                # each tile in the batch (batch is list of tiles)
                for i in range(len(batch['context_coords'])):
                    context_coords = batch['context_coords'][i].to(self.device)
                    context_embeddings = batch['context_embeddings'][i].to(self.device)
                    context_agbd = batch['context_agbd'][i].to(self.device)
                    target_coords = batch['target_coords'][i].to(self.device)
                    target_embeddings = batch['target_embeddings'][i].to(self.device)
                    target_agbd = batch['target_agbd'][i].to(self.device)

                    pred_mean, pred_log_var, _, _, _, _ = model(
                        context_coords,
                        context_embeddings,
                        context_agbd,
                        target_coords,
                        target_embeddings,
                        query_agbd=None,
                        training=False
                    )

                    if pred_log_var is not None:
                        pred_std = torch.exp(0.5 * pred_log_var)
                    else:
                        pred_std = torch.zeros_like(pred_mean)

                    all_preds.append(pred_mean.cpu().numpy())
                    all_targets.append(target_agbd.cpu().numpy())
                    all_uncertainties.append(pred_std.cpu().numpy())

        preds = np.concatenate(all_preds, axis=0).squeeze()
        targets = np.concatenate(all_targets, axis=0).squeeze()
        uncertainties = np.concatenate(all_uncertainties, axis=0).squeeze()

        metrics = compute_metrics(preds, targets, uncertainties)

        calib_metrics = compute_calibration_metrics(preds, targets, uncertainties)

        # Rename z_mean and z_std to match expected column names
        if 'z_mean' in calib_metrics:
            calib_metrics['z_score_mean'] = calib_metrics.pop('z_mean')
        if 'z_std' in calib_metrics:
            calib_metrics['z_score_std'] = calib_metrics.pop('z_std')

        log_metrics = {
            'log_rmse': metrics['rmse'],
            'log_mae': metrics['mae'],
            'log_r2': metrics['r2'],
        }
        if 'mean_uncertainty' in metrics:
            log_metrics['mean_uncertainty'] = metrics['mean_uncertainty']

        results = {
            **log_metrics,
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
        all_preds = []
        all_targets = []
        all_uncertainties = []

        logger.info(f"Evaluating XGBoost {train_region} → {test_region}")

        for batch in tqdm(dataloader, desc=f"XGB {train_region}→{test_region}"):
            for i in range(len(batch['target_coords'])):
                target_coords = batch['target_coords'][i].cpu().numpy()
                target_embeddings = batch['target_embeddings'][i].cpu().numpy()
                target_agbd = batch['target_agbd'][i].cpu().numpy()

                preds, uncertainties = model.predict(
                    target_coords,
                    target_embeddings,
                    return_std=True
                )

                all_preds.append(preds)
                all_targets.append(target_agbd.squeeze())
                all_uncertainties.append(uncertainties)

        preds = np.concatenate(all_preds, axis=0)
        targets = np.concatenate(all_targets, axis=0)
        uncertainties = np.concatenate(all_uncertainties, axis=0)

        metrics = compute_metrics(preds, targets, uncertainties)

        calib_metrics = compute_calibration_metrics(preds, targets, uncertainties)

        if 'z_mean' in calib_metrics:
            calib_metrics['z_score_mean'] = calib_metrics.pop('z_mean')
        if 'z_std' in calib_metrics:
            calib_metrics['z_score_std'] = calib_metrics.pop('z_std')

        log_metrics = {
            'log_rmse': metrics['rmse'],
            'log_mae': metrics['mae'],
            'log_r2': metrics['r2'],
        }
        if 'mean_uncertainty' in metrics:
            log_metrics['mean_uncertainty'] = metrics['mean_uncertainty']

        results = {
            **log_metrics,
            **calib_metrics,
            'train_region': train_region,
            'test_region': test_region,
            'model_type': 'xgboost',
            'num_predictions': len(preds)
        }

        return results

    def evaluate_regression_kriging_on_region(
        self,
        model: RegressionKrigingBaseline,
        dataloader: DataLoader,
        train_region: str,
        test_region: str
    ) -> dict:
        all_preds = []
        all_targets = []
        all_uncertainties = []

        logger.info(f"Evaluating Regression Kriging {train_region} → {test_region}")

        for batch in tqdm(dataloader, desc=f"RK {train_region}→{test_region}"):
            for i in range(len(batch['target_coords'])):
                target_coords = batch['target_coords'][i].cpu().numpy()
                target_embeddings = batch['target_embeddings'][i].cpu().numpy()
                target_agbd = batch['target_agbd'][i].cpu().numpy()

                preds, uncertainties = model.predict(
                    target_coords,
                    target_embeddings,
                    return_std=True
                )

                all_preds.append(preds)
                all_targets.append(target_agbd.squeeze())
                all_uncertainties.append(uncertainties)

        preds = np.concatenate(all_preds, axis=0)
        targets = np.concatenate(all_targets, axis=0)
        uncertainties = np.concatenate(all_uncertainties, axis=0)

        metrics = compute_metrics(preds, targets, uncertainties)

        calib_metrics = compute_calibration_metrics(preds, targets, uncertainties)

        if 'z_mean' in calib_metrics:
            calib_metrics['z_score_mean'] = calib_metrics.pop('z_mean')
        if 'z_std' in calib_metrics:
            calib_metrics['z_score_std'] = calib_metrics.pop('z_std')

        log_metrics = {
            'log_rmse': metrics['rmse'],
            'log_mae': metrics['mae'],
            'log_r2': metrics['r2'],
        }
        if 'mean_uncertainty' in metrics:
            log_metrics['mean_uncertainty'] = metrics['mean_uncertainty']

        results = {
            **log_metrics,
            **calib_metrics,
            'train_region': train_region,
            'test_region': test_region,
            'model_type': 'regression_kriging',
            'num_predictions': len(preds)
        }

        return results

    def run_cross_evaluation(self, model_types: List[str] = ['anp', 'xgboost']) -> pd.DataFrame:
        results = []

        for model_type in model_types:
            logger.info(f"\n{'='*60}")
            logger.info(f"Evaluating {model_type.upper()} models")
            logger.info(f"{'='*60}\n")

            models = {}
            for region in REGION_ORDER:
                if model_type == 'anp':
                    model_data = self.load_anp_model(region)
                elif model_type == 'xgboost':
                    model_data = self.load_xgboost_model(region)
                elif model_type == 'regression_kriging':
                    model_data = self.load_regression_kriging_model(region)
                else:
                    logger.error(f"Unknown model type: {model_type}")
                    continue

                if model_data is not None:
                    models[region] = model_data

            # Evaluate: train_region x test_region x seed x transfer_type
            for train_region in REGION_ORDER:
                if train_region not in models:
                    logger.warning(f"Skipping {train_region} (model not found)")
                    continue

                model_seeds = models[train_region]

                for test_region in REGION_ORDER:
                    do_fewshot = (self.few_shot_tiles is not None and
                                 self.few_shot_tiles > 0 and
                                 train_region != test_region)  # Only few-shot for OOD
                    do_zeroshot = (not do_fewshot) or self.include_zero_shot

                    transfer_types = []
                    if do_zeroshot:
                        transfer_types.append('zero-shot')
                    if do_fewshot:
                        transfer_types.append('few-shot')

                    for transfer_type in transfer_types:
                        seed_results = []

                        for model, config, seed_id in model_seeds:
                            if model_type == 'anp':
                                model.to(self.device)

                            split_for_fewshot = (transfer_type == 'few-shot')
                            data_loaders = self.load_target_region_data(
                                test_region,
                                seed_id=seed_id,
                                split_for_fewshot=split_for_fewshot
                            )

                            if data_loaders is None or 'test' not in data_loaders:
                                logger.warning(f"Skipping {test_region} seed {seed_id} (data not found)")
                                if model_type == 'anp':
                                    model.cpu()
                                continue

                            eval_model = model
                            if transfer_type == 'few-shot' and model_type == 'anp':
                                if 'train' in data_loaders:
                                    eval_model = self.fine_tune_anp_model(
                                        model,
                                        data_loaders['train'],
                                        train_region,
                                        test_region
                                    )
                                else:
                                    logger.warning(f"No training data for few-shot in {test_region}, skipping")
                                    if model_type == 'anp':
                                        model.cpu()
                                    continue

                            if model_type == 'anp':
                                result = self.evaluate_anp_on_region(
                                    eval_model,
                                    data_loaders['test'],
                                    train_region,
                                    test_region
                                )
                            elif model_type == 'xgboost':
                                if transfer_type == 'few-shot':
                                    continue
                                result = self.evaluate_xgboost_on_region(
                                    eval_model,
                                    data_loaders['test'],
                                    train_region,
                                    test_region
                                )
                            elif model_type == 'regression_kriging':
                                if transfer_type == 'few-shot':
                                    continue
                                result = self.evaluate_regression_kriging_on_region(
                                    eval_model,
                                    data_loaders['test'],
                                    train_region,
                                    test_region
                                )

                            result['seed_id'] = seed_id
                            result['transfer_type'] = transfer_type
                            seed_results.append(result)
                            results.append(result)

                            if transfer_type == 'few-shot' and model_type == 'anp' and eval_model is not model:
                                del eval_model

                            if model_type == 'anp':
                                model.cpu()
                            if torch.cuda.is_available():
                                torch.cuda.empty_cache()

                        if len(seed_results) > 1:
                            aggregated = self._aggregate_seed_results(
                                seed_results, train_region, test_region, model_type, transfer_type
                            )
                            results.append(aggregated)

        df = pd.DataFrame(results)

        output_path = self.output_dir / 'spatial_extrapolation_results.csv'
        df.to_csv(output_path, index=False)
        logger.info(f"Saved results to {output_path}")

        return df

    def _aggregate_seed_results(
        self,
        seed_results: List[dict],
        train_region: str,
        test_region: str,
        model_type: str,
        transfer_type: str = 'zero-shot'
    ) -> dict:
        metric_keys = ['log_rmse', 'log_mae', 'log_r2', 'mean_uncertainty',
                       'coverage_1sigma', 'coverage_2sigma', 'coverage_3sigma',
                       'z_score_mean', 'z_score_std']

        aggregated = {
            'train_region': train_region,
            'test_region': test_region,
            'model_type': model_type,
            'transfer_type': transfer_type,
            'seed_id': 'mean',
            'num_seeds': len(seed_results)
        }

        for key in metric_keys:
            values = [r[key] for r in seed_results if key in r and r[key] is not None]
            if values:
                aggregated[key] = np.mean(values)
                aggregated[f'{key}_std'] = np.std(values)
            else:
                aggregated[key] = None
                aggregated[f'{key}_std'] = None

        aggregated['num_predictions'] = np.mean([r['num_predictions'] for r in seed_results])

        return aggregated

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
        ax: Optional[plt.Axes] = None,
        show_std: bool = True,
        center: Optional[float] = None,
        simple_title: bool = False
    ) -> plt.Axes:
        df_model = df[df['model_type'] == model_type]
        if 'seed_id' in df_model.columns:
            df_mean = df_model[df_model['seed_id'] == 'mean']
            if len(df_mean) == 0:
                df_mean = df_model
        else:
            df_mean = df_model

        matrix = df_mean.pivot(
            index='train_region',
            columns='test_region',
            values=metric
        )

        matrix = matrix.reindex(index=REGION_ORDER, columns=REGION_ORDER)

        if reverse_cmap:
            cmap = cmap + '_r'

        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 7))

        annot_matrix = matrix.copy()
        std_col = f'{metric}_std'
        if show_std and std_col in df_mean.columns:
            std_matrix = df_mean.pivot(
                index='train_region',
                columns='test_region',
                values=std_col
            ).reindex(index=REGION_ORDER, columns=REGION_ORDER)

            annot_matrix = matrix.copy().astype(object)
            for i, train_reg in enumerate(REGION_ORDER):
                for j, test_reg in enumerate(REGION_ORDER):
                    mean_val = matrix.loc[train_reg, test_reg]
                    std_val = std_matrix.loc[train_reg, test_reg]
                    if pd.notna(mean_val) and pd.notna(std_val):
                        annot_matrix.loc[train_reg, test_reg] = f'{mean_val:.3f}\n±{std_val:.3f}'
                    elif pd.notna(mean_val):
                        annot_matrix.loc[train_reg, test_reg] = f'{mean_val:.3f}'

            heatmap_kwargs = {
                'annot': annot_matrix,
                'fmt': '',
                'cmap': cmap,
                'vmin': vmin,
                'vmax': vmax,
                'cbar_kws': {'label': metric_name},
                'ax': ax,
                'linewidths': 0.5,
                'linecolor': 'gray'
            }
            if center is not None:
                heatmap_kwargs['center'] = center

            sns.heatmap(matrix, **heatmap_kwargs)
        else:
            heatmap_kwargs = {
                'annot': True,
                'fmt': '.3f',
                'cmap': cmap,
                'vmin': vmin,
                'vmax': vmax,
                'cbar_kws': {'label': metric_name},
                'ax': ax,
                'linewidths': 0.5,
                'linecolor': 'gray'
            }
            if center is not None:
                heatmap_kwargs['center'] = center

            sns.heatmap(matrix, **heatmap_kwargs)

        ax.set_xlabel('Test Region', fontsize=12)
        ax.set_ylabel('Train Region', fontsize=12)

        if simple_title:
            title = model_type.upper() if model_type != 'xgboost' else 'XGBoost'
        elif metric_name in ['Zero-Shot', 'Few-Shot']:
            title = model_type.upper() if model_type != 'xgboost' else 'XGBoost'
        else:
            model_label = model_type.upper() if model_type != 'xgboost' else 'XGBoost'
            title = f'{model_label}: {metric_name}'

        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xticklabels([REGIONS[r] for r in REGION_ORDER], rotation=45, ha='right')
        ax.set_yticklabels([REGIONS[r] for r in REGION_ORDER], rotation=0)

        return ax

    def create_comparison_heatmap(
        self,
        df: pd.DataFrame,
        metric: str,
        metric_name: str,
        cmap: str = 'RdYlGn',
        reverse_cmap: bool = False,
        center: Optional[float] = None,
        vmin: Optional[float] = None,
        vmax: Optional[float] = None
    ) -> plt.Figure:
        fig, axes = plt.subplots(1, 2, figsize=(16, 7))

        if vmin is None or vmax is None:
            if 'seed_id' in df.columns:
                df_mean = df[df['seed_id'] == 'mean']
                if len(df_mean) == 0:
                    df_mean = df
            else:
                df_mean = df

            if center is not None:
                vmax_abs = max(abs(df_mean[metric].min() - center), abs(df_mean[metric].max() - center))
                vmin = center - vmax_abs
                vmax = center + vmax_abs
                if metric == 'log_r2':
                    vmin = max(vmin, -1.0)
                    vmax = min(vmax, 1.0)
            else:
                vmin = df_mean[metric].min()
                vmax = df_mean[metric].max()

        self.create_heatmap(
            df, 'anp', metric, metric_name,
            cmap=cmap, vmin=vmin, vmax=vmax, reverse_cmap=reverse_cmap,
            ax=axes[0], center=center, simple_title=True
        )

        self.create_heatmap(
            df, 'xgboost', metric, metric_name,
            cmap=cmap, vmin=vmin, vmax=vmax, reverse_cmap=reverse_cmap,
            ax=axes[1], center=center, simple_title=True
        )

        plt.suptitle(f'Spatial Extrapolation: {metric_name}', fontsize=16, fontweight='bold', y=0.98)
        plt.tight_layout(rect=[0, 0, 1, 0.96])

        return fig

    def create_2x2_comparison_grid(
        self,
        df: pd.DataFrame,
        metrics: List[Tuple[str, str, str, bool]],
        title: str = 'Spatial Extrapolation Results',
        center_values: Optional[List[Optional[float]]] = None
    ) -> plt.Figure:
        if len(metrics) != 2:
            raise ValueError("Must provide exactly 2 metrics for 2x2 grid")

        if center_values is None:
            center_values = [None, None]

        fig, axes = plt.subplots(2, 2, figsize=(16, 14))

        if 'seed_id' in df.columns:
            df_mean = df[df['seed_id'] == 'mean']
            if len(df_mean) == 0:
                df_mean = df
        else:
            df_mean = df

        # heatmaps for each metric
        for i, ((metric, metric_name, cmap, reverse_cmap), center) in enumerate(zip(metrics, center_values)):
            # vmin/vmax for this metric
            if center is not None:
                # diverging colormap centered at a value
                vmax_abs = max(abs(df_mean[metric].min() - center), abs(df_mean[metric].max() - center))
                vmin = center - vmax_abs
                vmax = center + vmax_abs
                if metric == 'log_r2':
                    vmin = max(vmin, -1.0)
                    vmax = min(vmax, 1.0)
            else:
                vmin = df_mean[metric].min()
                vmax = df_mean[metric].max()

            # ANP heatmap (left column)
            self.create_heatmap(
                df, 'anp', metric, metric_name,
                cmap=cmap, vmin=vmin, vmax=vmax, reverse_cmap=reverse_cmap,
                ax=axes[i, 0], center=center
            )

            # XGBoost heatmap (right column)
            self.create_heatmap(
                df, 'xgboost', metric, metric_name,
                cmap=cmap, vmin=vmin, vmax=vmax, reverse_cmap=reverse_cmap,
                ax=axes[i, 1], center=center
            )

        plt.suptitle(title, fontsize=18, fontweight='bold', y=0.995)
        plt.tight_layout(rect=[0, 0, 1, 0.99])

        return fig

    def visualize_results(self, df: pd.DataFrame):
        logger.info("\nCreating visualizations...")

        has_transfer_type = 'transfer_type' in df.columns
        if has_transfer_type:
            transfer_types = df['transfer_type'].unique()
            has_zeroshot = 'zero-shot' in transfer_types
            has_fewshot = 'few-shot' in transfer_types
        else:
            has_zeroshot = True
            has_fewshot = False
            df = df.copy()
            df['transfer_type'] = 'zero-shot'

        if 'seed_id' in df.columns:
            df_mean = df[df['seed_id'] == 'mean']
            if len(df_mean) == 0:
                df_mean = df
        else:
            df_mean = df

        r2_max_abs = max(abs(df_mean['log_r2'].min()), abs(df_mean['log_r2'].max()))
        shared_r2_vmin, shared_r2_vmax = max(-r2_max_abs, -1.0), min(r2_max_abs, 1.0)

        shared_rmse_vmin = df_mean['log_rmse'].min()
        shared_rmse_vmax = df_mean['log_rmse'].max()

        shared_z_std_vmin, shared_z_std_vmax = None, None
        if 'z_score_std' in df_mean.columns and df_mean['z_score_std'].notna().any():
            shared_z_std_vmin, shared_z_std_vmax = -1.0, 3.0

        shared_cov1_vmin, shared_cov1_vmax = None, None
        if 'coverage_1sigma' in df_mean.columns and df_mean['coverage_1sigma'].notna().any():
            shared_cov1_vmin = 0.0
            shared_cov1_vmax = df_mean['coverage_1sigma'].max()

        if has_zeroshot and has_fewshot:
            logger.info("Creating zero-shot vs few-shot comparisons...")

            fig, axes = plt.subplots(2, 2, figsize=(16, 14))

            df_zero = df[df['transfer_type'] == 'zero-shot']
            df_few = df[df['transfer_type'] == 'few-shot']

            self.create_heatmap(df_zero, 'anp', 'log_r2', 'Zero-Shot',
                               cmap='RdYlGn', vmin=shared_r2_vmin, vmax=shared_r2_vmax,
                               ax=axes[0, 0], center=0.0, simple_title=True)
            self.create_heatmap(df_few, 'anp', 'log_r2', 'Few-Shot',
                               cmap='RdYlGn', vmin=shared_r2_vmin, vmax=shared_r2_vmax,
                               ax=axes[0, 1], center=0.0, simple_title=True)

            self.create_heatmap(df_zero, 'anp', 'log_rmse', 'Zero-Shot',
                               cmap='RdYlGn_r', vmin=shared_rmse_vmin, vmax=shared_rmse_vmax,
                               ax=axes[1, 0], simple_title=True)
            self.create_heatmap(df_few, 'anp', 'log_rmse', 'Few-Shot',
                               cmap='RdYlGn_r', vmin=shared_rmse_vmin, vmax=shared_rmse_vmax,
                               ax=axes[1, 1], simple_title=True)

            plt.suptitle('Zero-Shot vs Few-Shot Transfer', fontsize=18, fontweight='bold', y=0.995)
            plt.tight_layout(rect=[0, 0, 1, 0.99])
            fig.savefig(self.output_dir / 'extrapolation_zeroshot_vs_fewshot.png', dpi=750, bbox_inches='tight')
            plt.close(fig)
            logger.info("Saved zero-shot vs few-shot comparison")

            if 'z_score_std' in df.columns and df['z_score_std'].notna().any():
                fig, axes = plt.subplots(1, 2, figsize=(16, 7))

                df_zero_calib = df_zero[df_zero['z_score_std'].notna()].copy()
                df_few_calib = df_few[df_few['z_score_std'].notna()].copy()

                if len(df_zero_calib) == 0 or len(df_few_calib) == 0:
                    logger.warning("Skipping z-score std comparison: insufficient data")
                    plt.close(fig)
                else:
                    custom_cmap = create_centered_diverging_colormap()

                    self.create_heatmap(df_zero_calib, 'anp', 'z_score_std', 'Zero-Shot',
                                       cmap=custom_cmap, vmin=shared_z_std_vmin, vmax=shared_z_std_vmax,
                                       ax=axes[0], center=1.0, simple_title=True)
                    self.create_heatmap(df_few_calib, 'anp', 'z_score_std', 'Few-Shot',
                                       cmap=custom_cmap, vmin=shared_z_std_vmin, vmax=shared_z_std_vmax,
                                       ax=axes[1], center=1.0, simple_title=True)

                    plt.suptitle('Zero-Shot vs Few-Shot: Uncertainty Calibration (Z-Score Std, 1.0 = Perfect)',
                               fontsize=18, fontweight='bold', y=0.98)
                    plt.tight_layout(rect=[0, 0, 1, 0.96])
                    fig.savefig(self.output_dir / 'extrapolation_zeroshot_vs_fewshot_calibration.png', dpi=750, bbox_inches='tight')
                    plt.close(fig)
                    logger.info("Saved zero-shot vs few-shot calibration comparison")
            else:
                logger.warning("Skipping z-score std comparison: metric not available in results")

            if 'coverage_1sigma' in df.columns and df['coverage_1sigma'].notna().any():
                fig, axes = plt.subplots(1, 2, figsize=(16, 7))

                self.create_heatmap(df_zero, 'anp', 'coverage_1sigma', 'Zero-Shot',
                                   cmap='RdYlGn', vmin=shared_cov1_vmin, vmax=shared_cov1_vmax,
                                   ax=axes[0], simple_title=True)
                self.create_heatmap(df_few, 'anp', 'coverage_1sigma', 'Few-Shot',
                                   cmap='RdYlGn', vmin=shared_cov1_vmin, vmax=shared_cov1_vmax,
                                   ax=axes[1], simple_title=True)

                plt.suptitle('Zero-Shot vs Few-Shot: 1-Sigma Coverage (68.3% = Ideal)',
                           fontsize=18, fontweight='bold', y=0.98)
                plt.tight_layout(rect=[0, 0, 1, 0.96])
                fig.savefig(self.output_dir / 'extrapolation_zeroshot_vs_fewshot_coverage.png', dpi=750, bbox_inches='tight')
                plt.close(fig)
                logger.info("Saved zero-shot vs few-shot coverage comparison")

            # zero-shot (ANP vs XGBoost) across 3 metrics
            logger.info("Creating 3x2 zero-shot comprehensive comparison...")
            fig, axes = plt.subplots(3, 2, figsize=(16, 21))

            custom_cmap = create_centered_diverging_colormap()

            # Row 0: 1-sigma coverage
            self.create_heatmap(df_zero, 'anp', 'coverage_1sigma', '1-Sigma Coverage',
                               cmap='RdYlGn', vmin=shared_cov1_vmin, vmax=shared_cov1_vmax,
                               ax=axes[0, 0], simple_title=True)
            self.create_heatmap(df_zero, 'xgboost', 'coverage_1sigma', '1-Sigma Coverage',
                               cmap='RdYlGn', vmin=shared_cov1_vmin, vmax=shared_cov1_vmax,
                               ax=axes[0, 1], simple_title=True)

            # Row 1: Z-score std (calibration)
            if 'z_score_std' in df_zero.columns and df_zero['z_score_std'].notna().any():
                df_zero_calib = df_zero[df_zero['z_score_std'].notna()].copy()
                self.create_heatmap(df_zero_calib, 'anp', 'z_score_std', 'Z-Score Std (1.0 = Perfect)',
                                   cmap=custom_cmap, vmin=shared_z_std_vmin, vmax=shared_z_std_vmax,
                                   ax=axes[1, 0], center=1.0, simple_title=True)
                self.create_heatmap(df_zero_calib, 'xgboost', 'z_score_std', 'Z-Score Std (1.0 = Perfect)',
                                   cmap=custom_cmap, vmin=shared_z_std_vmin, vmax=shared_z_std_vmax,
                                   ax=axes[1, 1], center=1.0, simple_title=True)

            # Row 2: R²
            self.create_heatmap(df_zero, 'anp', 'log_r2', 'Log R²',
                               cmap='RdYlGn', vmin=shared_r2_vmin, vmax=shared_r2_vmax,
                               ax=axes[2, 0], center=0.0, simple_title=True)
            self.create_heatmap(df_zero, 'xgboost', 'log_r2', 'Log R²',
                               cmap='RdYlGn', vmin=shared_r2_vmin, vmax=shared_r2_vmax,
                               ax=axes[2, 1], center=0.0, simple_title=True)

            plt.suptitle('Zero-Shot Transfer Comparison', fontsize=18, fontweight='bold', y=0.995)
            plt.tight_layout(rect=[0, 0, 1, 0.99])
            fig.savefig(self.output_dir / 'extrapolation_zeroshot_3x2_comprehensive.png', dpi=750, bbox_inches='tight')
            plt.close(fig)
            logger.info("Saved 3x2 zero-shot comprehensive comparison")

            # few-shot (ANP only) across 3 metrics
            logger.info("Creating 3x1 few-shot ANP comparison...")
            fig, axes = plt.subplots(3, 1, figsize=(8, 21))

            # Row 0: 1-sigma coverage
            self.create_heatmap(df_few, 'anp', 'coverage_1sigma', '1-Sigma Coverage',
                               cmap='RdYlGn', vmin=shared_cov1_vmin, vmax=shared_cov1_vmax,
                               ax=axes[0], simple_title=False)

            # Row 1: Z-score std (calibration)
            if 'z_score_std' in df_few.columns and df_few['z_score_std'].notna().any():
                df_few_calib = df_few[df_few['z_score_std'].notna()].copy()
                self.create_heatmap(df_few_calib, 'anp', 'z_score_std', 'Z-Score Std (1.0 = Perfect)',
                                   cmap=custom_cmap, vmin=shared_z_std_vmin, vmax=shared_z_std_vmax,
                                   ax=axes[1], center=1.0, simple_title=False)

            # Row 2: R²
            self.create_heatmap(df_few, 'anp', 'log_r2', 'Log R²',
                               cmap='RdYlGn', vmin=shared_r2_vmin, vmax=shared_r2_vmax,
                               ax=axes[2], center=0.0, simple_title=False)

            plt.suptitle('ANP Few-Shot Fine Tuning Performance', fontsize=18, fontweight='bold', y=0.995)
            plt.tight_layout(rect=[0, 0, 1, 0.99])
            fig.savefig(self.output_dir / 'extrapolation_fewshot_3x1_anp.png', dpi=750, bbox_inches='tight')
            plt.close(fig)
            logger.info("Saved 3x1 few-shot ANP comparison")

        # visualizations for each transfer type
        for transfer_type in (['zero-shot'] if has_zeroshot else []) + (['few-shot'] if has_fewshot else []):
            df_transfer = df[df['transfer_type'] == transfer_type]
            suffix = f"_{transfer_type.replace('-', '')}"

            fig = self.create_comparison_heatmap(
                df_transfer, 'log_r2', f'Log R² ({transfer_type.title()})',
                cmap='RdYlGn', reverse_cmap=False, center=0.0,
                vmin=shared_r2_vmin, vmax=shared_r2_vmax
            )
            fig.savefig(self.output_dir / f'extrapolation_r2{suffix}.png', dpi=750, bbox_inches='tight')
            plt.close(fig)
            logger.info(f"Saved R² comparison ({transfer_type})")

            fig = self.create_comparison_heatmap(
                df_transfer, 'log_rmse', f'Log RMSE ({transfer_type.title()})',
                cmap='RdYlGn', reverse_cmap=True,
                vmin=shared_rmse_vmin, vmax=shared_rmse_vmax
            )
            fig.savefig(self.output_dir / f'extrapolation_rmse{suffix}.png', dpi=750, bbox_inches='tight')
            plt.close(fig)
            logger.info(f"Saved RMSE comparison ({transfer_type})")

            if 'z_score_std' in df_transfer.columns and shared_z_std_vmin is not None:
                # custom colormap with green at center (1.0) and red at extremes
                custom_cmap = create_centered_diverging_colormap()

                fig = self.create_comparison_heatmap(
                    df_transfer, 'z_score_std', f'Z-Score Std ({transfer_type.title()})',
                    cmap=custom_cmap, reverse_cmap=False, center=1.0,
                    vmin=shared_z_std_vmin, vmax=shared_z_std_vmax
                )
                fig.savefig(self.output_dir / f'extrapolation_calibration{suffix}.png', dpi=750, bbox_inches='tight')
                plt.close(fig)
                logger.info(f"Saved calibration comparison ({transfer_type})")

            if 'coverage_1sigma' in df_transfer.columns and shared_cov1_vmin is not None:
                fig = self.create_comparison_heatmap(
                    df_transfer, 'coverage_1sigma', f'1-Sigma Coverage ({transfer_type.title()})',
                    cmap='RdYlGn', reverse_cmap=False,
                    vmin=shared_cov1_vmin, vmax=shared_cov1_vmax
                )
                fig.savefig(self.output_dir / f'extrapolation_coverage_1sigma{suffix}.png', dpi=750, bbox_inches='tight')
                plt.close(fig)
                logger.info(f"Saved 1-sigma coverage comparison ({transfer_type})")

        self.create_summary_table(df)

    def create_summary_table(self, df: pd.DataFrame):
        logger.info("\nCreating summary statistics...")

        # aggregated (mean) results
        if 'seed_id' in df.columns:
            df_mean = df[df['seed_id'] == 'mean'].copy()
            if len(df_mean) == 0:
                df_mean = df.copy()
        else:
            df_mean = df.copy()

        df_mean['is_diagonal'] = df_mean['train_region'] == df_mean['test_region']
        df_mean['split'] = df_mean['is_diagonal'].map({True: 'In-Distribution', False: 'Out-of-Distribution'})

        group_cols = ['model_type', 'split']
        if 'transfer_type' in df_mean.columns:
            group_cols.insert(1, 'transfer_type')

        # For mean results, we already have std columns, so just report means
        agg_dict = {
            'log_r2': ['mean'],
            'log_rmse': ['mean'],
            'coverage_1sigma': ['mean'],
            'mean_uncertainty': ['mean']
        }
        # z_score_std if available
        if 'z_score_std' in df_mean.columns:
            agg_dict['z_score_std'] = ['mean']

        summary = df_mean.groupby(group_cols).agg(agg_dict).round(3)

        output_path = self.output_dir / 'extrapolation_summary.csv'
        summary.to_csv(output_path)
        logger.info(f"Saved summary to {output_path}")

        print("\n" + "="*80)
        print("SPATIAL EXTRAPOLATION SUMMARY (Aggregated across seeds)")
        print("="*80)
        print(summary)
        print("="*80)

        if 'log_r2_std' in df_mean.columns:
            print("\nSeed-level variability (mean std across train-test pairs):")
            for model_type in df_mean['model_type'].unique():
                df_model = df_mean[df_mean['model_type'] == model_type]
                for split in ['In-Distribution', 'Out-of-Distribution']:
                    df_split = df_model[df_model['split'] == split]
                    r2_std = df_split['log_r2_std'].mean()
                    rmse_std = df_split['log_rmse_std'].mean()
                    print(f"  {model_type} - {split}:")
                    print(f"    R² std: {r2_std:.4f}")
                    print(f"    RMSE std: {rmse_std:.4f}")
            print("="*80)

        self.compute_degradation_metrics(df_mean)

    def compute_degradation_metrics(self, df: pd.DataFrame):
        logger.info("\nComputing degradation metrics...")

        results = []

        group_cols = ['model_type']
        if 'transfer_type' in df.columns:
            group_cols.append('transfer_type')
            groups = df.groupby(group_cols)
        else:
            groups = [(mt, df[df['model_type'] == mt]) for mt in df['model_type'].unique()]
            groups = [((mt,), g) for mt, g in groups]

        for group_key, df_group in groups:
            if isinstance(group_key, tuple):
                if len(group_key) == 2:
                    model_type, transfer_type = group_key
                else:
                    model_type = group_key[0]
                    transfer_type = None
            else:
                model_type = group_key
                transfer_type = None

            df_in = df_group[df_group['is_diagonal']]
            df_out = df_group[~df_group['is_diagonal']]

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

            if transfer_type is not None:
                degradation['transfer_type'] = transfer_type

            results.append(degradation)

        df_degradation = pd.DataFrame(results)

        output_path = self.output_dir / 'degradation_metrics.csv'
        df_degradation.to_csv(output_path, index=False)
        logger.info(f"Saved degradation metrics to {output_path}")

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
        choices=['anp', 'xgboost', 'regression_kriging'],
        default=['anp', 'xgboost'],
        help='Model types to evaluate (default: anp xgboost)'
    )

    parser.add_argument(
        '--context_ratio',
        type=float,
        default=0.5,
        help='Ratio of context/target split within each tile (default: 0.5)'
    )

    parser.add_argument(
        '--few_shot_tiles',
        type=int,
        default=None,
        help='Number of tiles from target region for few-shot fine-tuning. If not specified, only zero-shot is performed.'
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
        '--few_shot_batch_size',
        type=int,
        default=None,
        help='Batch size for few-shot fine-tuning. If not specified, uses batch_size // 4 for memory efficiency'
    )

    parser.add_argument(
        '--include_zero_shot',
        action='store_true',
        default=False,
        help='If set along with --few_shot_tiles, evaluate both zero-shot and few-shot'
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

    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed (default: 42)'
    )

    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)

    evaluator = SpatialExtrapolationEvaluator(
        results_dir=Path(args.results_dir),
        output_dir=Path(args.output_dir),
        device=args.device,
        context_ratio=args.context_ratio,
        batch_size=args.batch_size,
        few_shot_tiles=args.few_shot_tiles,
        few_shot_epochs=args.few_shot_epochs,
        few_shot_lr=args.few_shot_lr,
        few_shot_batch_size=args.few_shot_batch_size,
        include_zero_shot=args.include_zero_shot
    )

    logger.info("\nStarting spatial extrapolation cross-evaluation...")
    logger.info(f"Model types: {args.models}")
    logger.info(f"Regions: {list(REGIONS.values())}\n")

    df_results = evaluator.run_cross_evaluation(model_types=args.models)

    evaluator.visualize_results(df_results)

    logger.info("\n" + "="*80)
    logger.info("EVALUATION COMPLETE")
    logger.info(f"Results saved to: {evaluator.output_dir}")
    logger.info("="*80 + "\n")


if __name__ == '__main__':
    main()
