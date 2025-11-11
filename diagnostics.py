import json
import pickle
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Rectangle
import seaborn as sns
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from scipy.stats import norm, probplot

from data.dataset import GEDINeuralProcessDataset, collate_neural_process
from utils.normalization import denormalize_agbd, denormalize_std
from utils.config import load_config
from utils.model import load_model_from_checkpoint


def plot_learning_curves(history_path, output_path):
    """
    Plot training and validation loss curves.

    Args:
        history_path: Path to history.json
        output_path: Path to save the plot
    """
    with open(history_path, 'r') as f:
        history = json.load(f)

    train_losses = history['train_losses']
    val_losses = history['val_losses']
    epochs = range(1, len(train_losses) + 1)

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(epochs, train_losses, 'b-', label='Training Loss', linewidth=2)
    ax.plot(epochs, val_losses, 'r-', label='Validation Loss', linewidth=2)

    # Mark best epoch
    best_epoch = np.argmin(val_losses) + 1
    best_loss = val_losses[best_epoch - 1]
    ax.axvline(x=best_epoch, color='g', linestyle='--', alpha=0.5, label=f'Best Epoch ({best_epoch})')
    ax.plot(best_epoch, best_loss, 'g*', markersize=15)

    ax.set_xlabel('Epoch', fontsize=12, fontweight='bold')
    ax.set_ylabel('Loss', fontsize=12, fontweight='bold')
    ax.set_title('Training History', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"✓ Saved learning curves to: {output_path}")


def plot_spatial_splits(train_csv, val_csv, test_csv, output_path):
    """
    Visualize spatial distribution of train/val/test GEDI shots.

    Args:
        train_csv: Path to train_split.csv
        val_csv: Path to val_split.csv
        test_csv: Path to test_split.csv
        output_path: Path to save the plot
    """
    # Load splits
    train_df = pd.read_csv(train_csv)
    val_df = pd.read_csv(val_csv)
    test_df = pd.read_csv(test_csv)

    # Get tile statistics
    def get_tile_info(df, split_name):
        tiles = []
        for tile_id in df['tile_id'].unique():
            tile_data = df[df['tile_id'] == tile_id]
            tiles.append({
                'tile_id': tile_id,
                'split': split_name,
                'n_shots': len(tile_data)
            })
        return tiles

    train_tiles = get_tile_info(train_df, 'train')
    val_tiles = get_tile_info(val_df, 'val')
    test_tiles = get_tile_info(test_df, 'test')

    # Create figure
    fig = plt.figure(figsize=(14, 6))
    gs = gridspec.GridSpec(1, 2, width_ratios=[2, 1])

    # Left panel: Spatial map - scatter plot of GEDI shots
    ax1 = plt.subplot(gs[0])

    colors = {'train': '#1f77b4', 'val': '#ff7f0e', 'test': '#2ca02c'}

    # Plot each split as scatter points
    for df, split_name in [(train_df, 'train'), (val_df, 'val'), (test_df, 'test')]:
        ax1.scatter(df['longitude'], df['latitude'],
                   c=colors[split_name],
                   alpha=0.5,
                   s=10,
                   label=split_name.upper(),
                   edgecolors='none')

    ax1.set_xlabel('Longitude', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Latitude', fontsize=12, fontweight='bold')
    ax1.set_title('Spatial Distribution of GEDI Shots', fontsize=14, fontweight='bold')
    ax1.set_aspect('equal', adjustable='box')
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='best', fontsize=10)

    # Right panel: Statistics
    ax2 = plt.subplot(gs[1])
    ax2.axis('off')

    # Create statistics table
    stats_text = "Dataset Statistics\n" + "=" * 30 + "\n\n"

    stats = [
        ('TRAIN', len(train_tiles), len(train_df)),
        ('VAL', len(val_tiles), len(val_df)),
        ('TEST', len(test_tiles), len(test_df))
    ]

    for split, n_tiles, n_shots in stats:
        stats_text += f"{split}:\n"
        stats_text += f"  Tiles: {n_tiles}\n"
        stats_text += f"  Shots: {n_shots:,}\n"
        stats_text += f"  Shots/tile: {n_shots/n_tiles:.1f}\n\n"

    total_tiles = len(train_tiles) + len(val_tiles) + len(test_tiles)
    total_shots = len(train_df) + len(val_df) + len(test_df)

    stats_text += f"TOTAL:\n"
    stats_text += f"  Tiles: {total_tiles}\n"
    stats_text += f"  Shots: {total_shots:,}\n\n"

    stats_text += "Split Ratios:\n"
    stats_text += f"  Train: {len(train_tiles)/total_tiles*100:.1f}%\n"
    stats_text += f"  Val:   {len(val_tiles)/total_tiles*100:.1f}%\n"
    stats_text += f"  Test:  {len(test_tiles)/total_tiles*100:.1f}%\n"

    ax2.text(0.1, 0.95, stats_text, transform=ax2.transAxes,
             fontsize=10, verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"✓ Saved spatial splits visualization to: {output_path}")


def plot_sample_predictions(model, dataset, device, n_samples=5, output_path=None, agbd_scale=200.0):
    """
    Plot sample predictions with uncertainty bands.

    Args:
        model: Trained GEDINeuralProcess model
        dataset: GEDINeuralProcessDataset
        device: torch device
        n_samples: Number of sample tiles to plot
        output_path: Path to save the plot
        agbd_scale: AGBD scale factor for denormalization (default: 200.0)
    """
    model.eval()

    # Randomly sample tiles
    indices = np.random.choice(len(dataset), min(n_samples, len(dataset)), replace=False)

    fig, axes = plt.subplots(n_samples, 1, figsize=(12, 4*n_samples))
    if n_samples == 1:
        axes = [axes]

    with torch.no_grad():
        for idx, tile_idx in enumerate(indices):
            sample = dataset[tile_idx]

            # Move to device (no unsqueeze needed - model expects (n_points, ...) directly)
            context_coords = sample['context_coords'].to(device)
            context_embeddings = sample['context_embeddings'].to(device)
            context_agbd = sample['context_agbd'].to(device)
            target_coords = sample['target_coords'].to(device)
            target_embeddings = sample['target_embeddings'].to(device)
            target_agbd = sample['target_agbd'].to(device)

            # Forward pass
            pred_mean, pred_log_var, _, _ = model(
                context_coords,
                context_embeddings,
                context_agbd,
                target_coords,
                target_embeddings,
                training=False
            )

            # Convert to numpy (normalized values)
            pred_mean_norm = pred_mean.squeeze().cpu().numpy()
            target_norm = target_agbd.squeeze().cpu().numpy()

            if pred_log_var is not None:
                pred_std_norm = torch.exp(0.5 * pred_log_var).squeeze().cpu().numpy()
            else:
                pred_std_norm = np.zeros_like(pred_mean_norm)

            # Denormalize from log scale to linear AGBD (Mg/ha)
            pred_mean_np = denormalize_agbd(pred_mean_norm, agbd_scale)
            target_np = denormalize_agbd(target_norm, agbd_scale)
            pred_std_np = denormalize_std(pred_std_norm, pred_mean_norm, agbd_scale)

            # Sort by target value for better visualization
            sort_idx = np.argsort(target_np)
            target_sorted = target_np[sort_idx]
            pred_sorted = pred_mean_np[sort_idx]
            std_sorted = pred_std_np[sort_idx]

            # Plot
            ax = axes[idx]
            x = np.arange(len(target_sorted))

            # Plot predictions and ground truth
            ax.plot(x, target_sorted, 'o-', color='black', label='Ground Truth',
                   linewidth=2, markersize=6, alpha=0.7)
            ax.plot(x, pred_sorted, 's-', color='red', label='Prediction (mean)',
                   linewidth=2, markersize=6, alpha=0.7)

            # Add uncertainty bands (1σ and 2σ)
            if pred_std_np.std() > 0:
                ax.fill_between(x, pred_sorted - std_sorted, pred_sorted + std_sorted,
                               alpha=0.3, color='red', label='±1σ (68%)')
                ax.fill_between(x, pred_sorted - 2*std_sorted, pred_sorted + 2*std_sorted,
                               alpha=0.15, color='red', label='±2σ (95%)')

            # Calculate metrics for this tile in log space (normalized)
            log_rmse = np.sqrt(np.mean((pred_mean_norm - target_norm) ** 2))
            log_mae = np.mean(np.abs(pred_mean_norm - target_norm))
            log_r2 = 1 - np.sum((target_norm - pred_mean_norm) ** 2) / np.sum((target_norm - target_norm.mean()) ** 2)

            # Check calibration: what % of points fall within 1σ and 2σ (in log space)
            within_1sigma = np.sum(np.abs(target_norm - pred_mean_norm) <= pred_std_norm) / len(target_norm) * 100
            within_2sigma = np.sum(np.abs(target_norm - pred_mean_norm) <= 2*pred_std_norm) / len(target_norm) * 100

            ax.set_xlabel('Sample Index (sorted by ground truth)', fontsize=10, fontweight='bold')
            ax.set_ylabel('AGBD (Mg/ha)', fontsize=10, fontweight='bold')
            ax.set_title(f'Tile {tile_idx} | Log RMSE: {log_rmse:.3f}, Log MAE: {log_mae:.3f}, Log R²: {log_r2:.3f} | '
                        f'Coverage: {within_1sigma:.0f}% (1σ), {within_2sigma:.0f}% (2σ)',
                        fontsize=11)
            ax.legend(loc='best', fontsize=9)
            ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"✓ Saved sample predictions to: {output_path}")


def plot_uncertainty_calibration(model, dataset, device, output_path, agbd_scale=200.0):
    """
    Analyze uncertainty calibration: where does ground truth fall in predicted distribution?

    Args:
        model: Trained GEDINeuralProcess model
        dataset: GEDINeuralProcessDataset
        device: torch device
        output_path: Path to save the plot
        agbd_scale: AGBD scale factor for denormalization (default: 200.0)
    """
    model.eval()

    # Collect predictions from entire dataset
    all_pred_means = []
    all_pred_stds = []
    all_targets = []

    dataloader = DataLoader(dataset, batch_size=1, shuffle=False,
                           collate_fn=lambda x: x[0])

    with torch.no_grad():
        for sample in tqdm(dataloader, desc='Computing calibration'):
            # Move to device (no unsqueeze needed - model expects (n_points, ...) directly)
            context_coords = sample['context_coords'].to(device)
            context_embeddings = sample['context_embeddings'].to(device)
            context_agbd = sample['context_agbd'].to(device)
            target_coords = sample['target_coords'].to(device)
            target_embeddings = sample['target_embeddings'].to(device)
            target_agbd = sample['target_agbd'].to(device)

            if len(target_coords) == 0:
                continue

            pred_mean, pred_log_var, _, _ = model(
                context_coords,
                context_embeddings,
                context_agbd,
                target_coords,
                target_embeddings,
                training=False
            )

            # Convert to numpy (keep normalized/log-scale values for calibration)
            pred_mean_norm = pred_mean.squeeze().cpu().numpy()
            target_norm = target_agbd.squeeze().cpu().numpy()

            if pred_log_var is not None:
                pred_std_norm = torch.exp(0.5 * pred_log_var).squeeze().cpu().numpy()
            else:
                pred_std_norm = np.zeros_like(pred_mean_norm)

            # Use normalized values directly for calibration analysis
            # (calibration should be assessed in the space where the model predicts)
            all_pred_means.extend(pred_mean_norm)
            all_pred_stds.extend(pred_std_norm)
            all_targets.extend(target_norm)

    all_pred_means = np.array(all_pred_means)
    all_pred_stds = np.array(all_pred_stds)
    all_targets = np.array(all_targets)

    # Calculate standardized residuals (z-scores)
    z_scores = (all_targets - all_pred_means) / (all_pred_stds + 1e-8)

    # Create figure with 4 panels
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Uncertainty Calibration Analysis', fontsize=16, fontweight='bold')

    # Panel 1: Z-score distribution
    ax = axes[0, 0]
    ax.hist(z_scores, bins=50, density=True, alpha=0.7, edgecolor='black', label='Observed')

    # Overlay ideal N(0,1) distribution
    x = np.linspace(-4, 4, 100)
    ax.plot(x, 1/np.sqrt(2*np.pi) * np.exp(-0.5*x**2), 'r-', linewidth=2, label='Ideal N(0,1)')

    ax.axvline(x=0, color='green', linestyle='--', alpha=0.5)
    ax.set_xlabel('Standardized Residual (z-score)', fontsize=11, fontweight='bold')
    ax.set_ylabel('Density', fontsize=11, fontweight='bold')
    ax.set_title('Distribution of Standardized Residuals', fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # Add statistics
    z_mean = np.mean(z_scores)
    z_std = np.std(z_scores)
    ax.text(0.05, 0.95, f'Mean: {z_mean:.3f}\nStd: {z_std:.3f}',
            transform=ax.transAxes, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # Panel 2: Coverage plot
    ax = axes[0, 1]

    # Calculate empirical coverage at different confidence levels
    confidence_levels = np.linspace(0, 3, 30)
    empirical_coverage = []
    theoretical_coverage = []

    for level in confidence_levels:
        # Empirical: what % of points fall within level*sigma
        coverage = np.sum(np.abs(all_targets - all_pred_means) <= level * all_pred_stds) / len(all_targets)
        empirical_coverage.append(coverage * 100)

        # Theoretical: for normal distribution
        theoretical_coverage.append(2 * norm.cdf(level) * 100 - 100)

    ax.plot(confidence_levels, empirical_coverage, 'o-', linewidth=2, markersize=5, label='Empirical')
    ax.plot(confidence_levels, theoretical_coverage, 'r--', linewidth=2, label='Ideal (Normal)')

    # Mark key points
    for level, name in [(1, '68%'), (2, '95%'), (3, '99.7%')]:
        idx = np.argmin(np.abs(confidence_levels - level))
        ax.axvline(x=level, color='gray', linestyle=':', alpha=0.5)
        ax.axhline(y=theoretical_coverage[idx], color='gray', linestyle=':', alpha=0.5)

    ax.set_xlabel('Confidence Level (×σ)', fontsize=11, fontweight='bold')
    ax.set_ylabel('Coverage (%)', fontsize=11, fontweight='bold')
    ax.set_title('Prediction Interval Coverage', fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 3)
    ax.set_ylim(0, 100)

    # Panel 3: Absolute error vs uncertainty
    ax = axes[1, 0]

    abs_errors = np.abs(all_targets - all_pred_means)

    # Bin by uncertainty
    n_bins = 20
    sorted_idx = np.argsort(all_pred_stds)
    sorted_stds = all_pred_stds[sorted_idx]
    sorted_errors = abs_errors[sorted_idx]

    bin_size = len(sorted_stds) // n_bins
    bin_stds = []
    bin_errors = []

    for i in range(n_bins):
        start = i * bin_size
        end = (i + 1) * bin_size if i < n_bins - 1 else len(sorted_stds)
        bin_stds.append(sorted_stds[start:end].mean())
        bin_errors.append(sorted_errors[start:end].mean())

    ax.scatter(bin_stds, bin_errors, s=80, alpha=0.7, edgecolors='black')

    # Perfect calibration line
    min_val = 0
    max_val = max(max(bin_stds), max(bin_errors))
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Calibration')

    ax.set_xlabel('Predicted Uncertainty (σ) [normalized log scale]', fontsize=11, fontweight='bold')
    ax.set_ylabel('Actual Error (|pred - true|) [normalized log scale]', fontsize=11, fontweight='bold')
    ax.set_title('Uncertainty vs Error (Log Scale)', fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # Panel 4: Q-Q plot
    ax = axes[1, 1]

    probplot(z_scores, dist="norm", plot=ax)
    ax.set_xlabel('Theoretical Quantiles', fontsize=11, fontweight='bold')
    ax.set_ylabel('Sample Quantiles', fontsize=11, fontweight='bold')
    ax.set_title('Q-Q Plot', fontsize=12)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"✓ Saved uncertainty calibration to: {output_path}")

    # Print summary statistics
    print("\nCalibration Summary (in normalized log space):")
    print(f"  Mean z-score: {z_mean:.3f} (ideal: 0.0)")
    print(f"  Std z-score:  {z_std:.3f} (ideal: 1.0)")

    within_1sigma = np.sum(np.abs(z_scores) <= 1) / len(z_scores) * 100
    within_2sigma = np.sum(np.abs(z_scores) <= 2) / len(z_scores) * 100
    within_3sigma = np.sum(np.abs(z_scores) <= 3) / len(z_scores) * 100

    print(f"  Within 1σ: {within_1sigma:.1f}% (ideal: 68.3%)")
    print(f"  Within 2σ: {within_2sigma:.1f}% (ideal: 95.4%)")
    print(f"  Within 3σ: {within_3sigma:.1f}% (ideal: 99.7%)")


def generate_all_diagnostics(model_dir, device='cpu', n_sample_plots=5):
    """
    Generate all post-training diagnostics.

    Args:
        model_dir: Directory containing trained model and outputs
        device: Device for model evaluation
        n_sample_plots: Number of sample prediction plots to generate
    """
    model_dir = Path(model_dir)

    print("\n" + "=" * 80)
    print("GENERATING POST-TRAINING DIAGNOSTICS")
    print("=" * 80)
    print(f"Model directory: {model_dir}")
    print()

    # Load config
    config = load_config(model_dir / 'config.json')

    # 1. Learning curves
    print("[1/4] Generating learning curves...")
    if (model_dir / 'history.json').exists():
        plot_learning_curves(
            model_dir / 'history.json',
            model_dir / 'diagnostics_learning_curves.png'
        )
    else:
        print("  ⚠ history.json not found, skipping")

    # 2. Spatial splits
    print("\n[2/4] Visualizing spatial tile splits...")
    if all((model_dir / f'{split}_split.csv').exists() for split in ['train', 'val', 'test']):
        plot_spatial_splits(
            model_dir / 'train_split.csv',
            model_dir / 'val_split.csv',
            model_dir / 'test_split.csv',
            model_dir / 'diagnostics_spatial_splits.png'
        )
    else:
        print("  ⚠ Split CSV files not found, skipping")

    # 3. Sample predictions (on validation set)
    print("\n[3/4] Generating sample predictions...")

    # Load validation data and model (used by both sections 3 and 4)
    model = None
    val_dataset = None

    if (model_dir / 'val_split.csv').exists() and (model_dir / 'best_r2_model.pt').exists():
        # Load validation data from pickle (preserves numpy arrays properly)
        # The CSV files contain truncated string representations that can't be parsed
        if (model_dir / 'processed_data.pkl').exists():
            with open(model_dir / 'processed_data.pkl', 'rb') as f:
                full_data = pickle.load(f)

            # Load val split CSV just to get the tile IDs
            val_split_info = pd.read_csv(model_dir / 'val_split.csv')
            val_tile_ids = val_split_info['tile_id'].unique()

            # Filter full data to validation tiles
            val_df = full_data[full_data['tile_id'].isin(val_tile_ids)].copy()

            global_bounds = tuple(config['global_bounds'])
            val_dataset = GEDINeuralProcessDataset(
                val_df,
                min_shots_per_tile=config.get('min_shots_per_tile', 10),
                log_transform_agbd=config.get('log_transform_agbd', True),
                augment_coords=False,
                coord_noise_std=0.0,
                global_bounds=global_bounds
            )

            # Load model
            model, checkpoint, checkpoint_path = load_model_from_checkpoint(
                model_dir, device
            )
        else:
            print("  ⚠ processed_data.pkl not found, cannot load validation data")
    else:
        print("  ⚠ Required files not found, skipping")

    # Generate sample predictions if model and dataset are loaded
    if model is not None and val_dataset is not None:
        agbd_scale = config.get('agbd_scale', 200.0)
        plot_sample_predictions(
            model, val_dataset, device, n_samples=n_sample_plots,
            output_path=model_dir / 'diagnostics_sample_predictions.png',
            agbd_scale=agbd_scale
        )

    # 4. Uncertainty calibration
    print("\n[4/4] Analyzing uncertainty calibration...")
    if model is not None and val_dataset is not None:
        agbd_scale = config.get('agbd_scale', 200.0)
        plot_uncertainty_calibration(
            model, val_dataset, device,
            output_path=model_dir / 'diagnostics_uncertainty_calibration.png',
            agbd_scale=agbd_scale
        )
    else:
        print("  ⚠ Required files not found, skipping")

    print("\n" + "=" * 80)
    print("DIAGNOSTICS COMPLETE")
    print("=" * 80)
    print("Generated files:")
    print("  - diagnostics_learning_curves.png")
    print("  - diagnostics_spatial_splits.png")
    print("  - diagnostics_sample_predictions.png")
    print("  - diagnostics_uncertainty_calibration.png")
    print("=" * 80)
    print()


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Generate post-training diagnostics')
    parser.add_argument('--model_dir', type=str, required=True,
                       help='Directory containing trained model')
    parser.add_argument('--device', type=str,
                       default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--n_samples', type=int, default=5,
                       help='Number of sample prediction plots')

    args = parser.parse_args()

    generate_all_diagnostics(args.model_dir, args.device, args.n_samples)
