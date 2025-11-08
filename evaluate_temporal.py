import argparse
import json
from pathlib import Path
import pickle

import torch
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

from data.gedi import GEDIQuerier
from data.embeddings import EmbeddingExtractor
from data.dataset import GEDINeuralProcessDataset, collate_neural_process
from models.neural_process import GEDINeuralProcess, compute_metrics

def convert_to_serializable(obj):
    if isinstance(obj, dict):
        return {key: convert_to_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_serializable(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(convert_to_serializable(item) for item in obj)
    elif isinstance(obj, (np.integer, np.int32, np.int64)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.bool_):
        return bool(obj)
    else:
        return obj
        
def evaluate_model(model, dataloader, device):
    """Evaluate model on a dataset."""
    model.eval()
    all_predictions = []
    all_targets = []
    all_uncertainties = []
    all_metrics = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc='Evaluating'):
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
                pred_mean, pred_log_var, _, _ = model(
                    context_coords,
                    context_embeddings,
                    context_agbd,
                    target_coords,
                    target_embeddings,
                    training=False
                )

                # Convert to numpy
                pred_mean_np = pred_mean.detach().cpu().numpy().flatten()
                target_np = target_agbd.detach().cpu().numpy().flatten()

                if pred_log_var is not None:
                    pred_std_np = torch.exp(0.5 * pred_log_var).detach().cpu().numpy().flatten()
                else:
                    pred_std_np = np.zeros_like(pred_mean_np)

                # Store predictions
                all_predictions.extend(pred_mean_np)
                all_targets.extend(target_np)
                all_uncertainties.extend(pred_std_np)

                # Compute metrics for this tile
                pred_std = torch.exp(0.5 * pred_log_var) if pred_log_var is not None else None
                metrics = compute_metrics(pred_mean, pred_std, target_agbd)
                all_metrics.append(metrics)

    # Convert to arrays
    predictions = np.array(all_predictions)
    targets = np.array(all_targets)
    uncertainties = np.array(all_uncertainties)

    # Aggregate metrics
    avg_metrics = {}
    if len(all_metrics) > 0:
        for key in all_metrics[0].keys():
            avg_metrics[key] = np.mean([m[key] for m in all_metrics])

    return predictions, targets, uncertainties, avg_metrics


def plot_results(predictions, targets, uncertainties, output_dir, dataset_name='temporal'):

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle(f'Temporal Validation: {dataset_name}', fontsize=16, fontweight='bold')

    ax = axes[0, 0]
    ax.scatter(targets, predictions, alpha=0.3, s=10)
    min_val = min(targets.min(), predictions.min())
    max_val = max(targets.max(), predictions.max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect prediction')
    ax.set_xlabel('True AGBD', fontweight='bold')
    ax.set_ylabel('Predicted AGBD', fontweight='bold')
    ax.set_title('Predictions vs Truth')
    ax.legend()
    ax.grid(True, alpha=0.3)

    ss_res = ((targets - predictions) ** 2).sum()
    ss_tot = ((targets - targets.mean()) ** 2).sum()
    r2 = 1 - ss_res / (ss_tot + 1e-8)
    ax.text(0.05, 0.95, f'R² = {r2:.4f}', transform=ax.transAxes,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    ax = axes[0, 1]
    residuals = predictions - targets
    ax.scatter(predictions, residuals, alpha=0.3, s=10)
    ax.axhline(y=0, color='r', linestyle='--', linewidth=2)
    ax.set_xlabel('Predicted AGBD', fontweight='bold')
    ax.set_ylabel('Residual (Pred - True)', fontweight='bold')
    ax.set_title('Residual Plot')
    ax.grid(True, alpha=0.3)

    ax = axes[1, 0]
    ax.hist(residuals, bins=50, edgecolor='black', alpha=0.7)
    ax.axvline(x=0, color='r', linestyle='--', linewidth=2)
    ax.set_xlabel('Residual', fontweight='bold')
    ax.set_ylabel('Frequency', fontweight='bold')
    ax.set_title('Distribution of Residuals')
    ax.grid(True, alpha=0.3, axis='y')

    rmse = np.sqrt(np.mean(residuals ** 2))
    mae = np.mean(np.abs(residuals))
    ax.text(0.05, 0.95, f'RMSE = {rmse:.4f}\nMAE = {mae:.4f}',
            transform=ax.transAxes, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    ax = axes[1, 1]
    if uncertainties is not None and uncertainties.std() > 0:
        sorted_indices = np.argsort(uncertainties)
        sorted_uncertainties = uncertainties[sorted_indices]
        sorted_errors = np.abs(residuals[sorted_indices])

        n_bins = 20
        bin_size = len(sorted_uncertainties) // n_bins
        bin_uncertainties = []
        bin_errors = []

        for i in range(n_bins):
            start_idx = i * bin_size
            end_idx = (i + 1) * bin_size if i < n_bins - 1 else len(sorted_uncertainties)
            bin_uncertainties.append(sorted_uncertainties[start_idx:end_idx].mean())
            bin_errors.append(sorted_errors[start_idx:end_idx].mean())

        ax.scatter(bin_uncertainties, bin_errors, s=50)
        min_val = min(min(bin_uncertainties), min(bin_errors))
        max_val = max(max(bin_uncertainties), max(bin_errors))
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect calibration')
        ax.set_xlabel('Predicted Uncertainty (σ)', fontweight='bold')
        ax.set_ylabel('Actual Error (|pred - true|)', fontweight='bold')
        ax.set_title('Uncertainty Calibration')
        ax.legend()
        ax.grid(True, alpha=0.3)
    else:
        ax.text(0.5, 0.5, 'No uncertainty predictions', ha='center', va='center',
                transform=ax.transAxes)
        ax.set_title('Uncertainty Calibration')

    plt.tight_layout()
    plt.savefig(output_dir / f'temporal_eval_{dataset_name}.png', dpi=300, bbox_inches='tight')
    print(f"Saved evaluation plot to: {output_dir / f'temporal_eval_{dataset_name}.png'}")


def main():
    parser = argparse.ArgumentParser(
        description='Evaluate GEDI Neural Process model on temporal holdout'
    )

    # Model arguments
    parser.add_argument('--model_dir', type=str, required=True,
                        help='Directory containing trained model')
    parser.add_argument('--checkpoint', type=str, default='best_r2_model.pt',
                        help='Checkpoint filename (default: best_r2_model.pt)')

    # Temporal holdout arguments
    parser.add_argument('--test_years', type=int, nargs='+', required=True,
                        help='Years to use for temporal evaluation (e.g., 2022 2023)')
    parser.add_argument('--region_bbox', type=float, nargs=4, default=None,
                        help='Region bounding box (default: use same as training)')
    parser.add_argument('--cache_dir', type=str, default='./cache',
                        help='Cache directory for embeddings')

    # Evaluation arguments
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--output_suffix', type=str, default=None,
                        help='Suffix for output files (default: years_YYYY_YYYY)')

    args = parser.parse_args()

    model_dir = Path(args.model_dir)

    # Load config
    print("=" * 80)
    print("TEMPORAL VALIDATION - GEDI NEURAL PROCESS")
    print("=" * 80)

    with open(model_dir / 'config.json', 'r') as f:
        config = json.load(f)

    print(f"Model directory: {model_dir}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Architecture: {config.get('architecture_mode', 'deterministic')}")

    # Display training years if available
    if 'train_years' in config and config['train_years'] is not None:
        print(f"Model trained on years: {config['train_years']}")
    else:
        print(f"Model trained on: {config.get('start_time', 'unknown')} to {config.get('end_time', 'unknown')}")

    print(f"Evaluating on years: {args.test_years}")
    print(f"Device: {args.device}")
    print()

    if args.region_bbox is None:
        args.region_bbox = config['region_bbox']
        print(f"Using training region bbox: {args.region_bbox}")

    if args.output_suffix is None:
        years_str = '_'.join(map(str, args.test_years))
        output_suffix = f"years_{years_str}"
    else:
        output_suffix = args.output_suffix

    print("\n" + "=" * 80)
    print("Step 1: Querying GEDI data for temporal holdout...")
    print("=" * 80)

    querier = GEDIQuerier()

    all_gedi_dfs = []
    for year in args.test_years:
        print(f"\nQuerying year {year}...")
        year_df = querier.query_region_tiles(
            region_bbox=args.region_bbox,
            tile_size=0.1,
            start_time=f"{year}-01-01",
            end_time=f"{year}-12-31"
        )

        if len(year_df) > 0:
            year_df['year'] = year
            all_gedi_dfs.append(year_df)
            print(f"  Retrieved {len(year_df)} shots from {year}")
        else:
            print(f"  No data found for {year}")

    if len(all_gedi_dfs) == 0:
        print("Error: No GEDI data found for specified years. Exiting.")
        return

    gedi_df = pd.concat(all_gedi_dfs, ignore_index=True)
    print(f"\nTotal: {len(gedi_df)} GEDI shots across {gedi_df['tile_id'].nunique()} tiles")
    print(f"Shots per year: {dict(gedi_df['year'].value_counts().sort_index())}")

    print("\n" + "=" * 80)
    print("Step 2: Extracting GeoTessera embeddings...")
    print("=" * 80)

    extractor = EmbeddingExtractor(
        year=config.get('embedding_year', 2024),
        patch_size=config.get('patch_size', 3),
        cache_dir=args.cache_dir
    )
    gedi_df = extractor.extract_patches_batch(gedi_df, verbose=True)

    gedi_df = gedi_df[gedi_df['embedding_patch'].notna()]
    print(f"Retained {len(gedi_df)} shots with valid embeddings")

    temporal_data_path = model_dir / f'temporal_data_{output_suffix}.pkl'
    with open(temporal_data_path, 'wb') as f:
        pickle.dump(gedi_df, f)
    print(f"Saved temporal data to: {temporal_data_path}")

    print("\n" + "=" * 80)
    print("Step 3: Creating evaluation dataset...")
    print("=" * 80)

    global_bounds = tuple(config['global_bounds'])
    print(f"Using global bounds from training: {global_bounds}")

    eval_dataset = GEDINeuralProcessDataset(
        gedi_df,
        min_shots_per_tile=config.get('min_shots_per_tile', 10),
        log_transform_agbd=config.get('log_transform_agbd', True),
        augment_coords=False,  # No augmentation for evaluation
        coord_noise_std=0.0,
        global_bounds=global_bounds
    )

    eval_loader = DataLoader(
        eval_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_neural_process,
        num_workers=args.num_workers
    )

    print("\n" + "=" * 80)
    print("Step 4: Loading trained model...")
    print("=" * 80)

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
    ).to(args.device)

    # Load checkpoint
    checkpoint_path = model_dir / args.checkpoint
    print(f"Loading checkpoint from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=args.device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])

    print(f"Checkpoint epoch: {checkpoint.get('epoch', 'unknown')}")
    if 'val_metrics' in checkpoint:
        print(f"Validation metrics (from training):")
        for key, val in checkpoint['val_metrics'].items():
            print(f"  {key}: {val:.4f}")

    print("\n" + "=" * 80)
    print("Step 5: Evaluating on temporal holdout...")
    print("=" * 80)

    predictions, targets, uncertainties, metrics = evaluate_model(
        model, eval_loader, args.device
    )

    print("\n" + "=" * 80)
    print(f"TEMPORAL VALIDATION RESULTS (Years: {args.test_years})")
    print("=" * 80)
    for key, val in metrics.items():
        print(f"{key.upper()}: {val:.4f}")
    print("=" * 80)

    print("\nSaving results...")

    results = {
        'metrics': metrics,
        'test_years': args.test_years,
        'train_years': config.get('train_years', None),
        'config': config,
        'checkpoint': args.checkpoint,
        'n_shots': len(predictions),
        'n_tiles': gedi_df['tile_id'].nunique()
    }

    results_path = model_dir / f'temporal_results_{output_suffix}.json'
    with open(results_path, 'w') as f:
        json.dump(convert_to_serializable(results), f, indent=2)
    print(f"Saved metrics to: {results_path}")

    results_df = pd.DataFrame({
        'true': targets,
        'predicted': predictions,
        'uncertainty': uncertainties,
        'residual': predictions - targets
    })
    predictions_path = model_dir / f'temporal_predictions_{output_suffix}.csv'
    results_df.to_csv(predictions_path, index=False)
    print(f"Saved predictions to: {predictions_path}")

    years_label = '_'.join(map(str, args.test_years))
    plot_results(predictions, targets, uncertainties, model_dir, years_label)

    print("\n" + "=" * 80)
    print("TEMPORAL VALIDATION COMPLETE")
    print("=" * 80)
    print(f"Results saved to: {model_dir}")
    print("Files:")
    print(f"  - temporal_results_{output_suffix}.json")
    print(f"  - temporal_predictions_{output_suffix}.csv")
    print(f"  - temporal_eval_{years_label}.png")
    print(f"  - temporal_data_{output_suffix}.pkl")
    print("=" * 80)


if __name__ == '__main__':
    main()
