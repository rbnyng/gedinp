"""
Analyze and visualize training dynamics to diagnose overfitting.
"""

import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


def plot_training_curves(history_path, save_path=None):
    """
    Plot training curves to diagnose overfitting.

    Args:
        history_path: Path to history.json
        save_path: Optional path to save figure
    """
    with open(history_path, 'r') as f:
        history = json.load(f)

    epochs = np.arange(1, len(history['train_loss']) + 1)

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Training Dynamics Analysis', fontsize=16, fontweight='bold')

    # Plot 1: Loss curves
    ax = axes[0, 0]
    ax.plot(epochs, history['train_loss'], label='Train Loss', linewidth=2)
    ax.plot(epochs, history['val_loss'], label='Val Loss', linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Loss Curves')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')  # Log scale for better visibility

    # Add annotation for overfitting
    if len(history['train_loss']) > 10:
        train_final = history['train_loss'][-1]
        val_final = history['val_loss'][-1]
        train_min = min(history['train_loss'])
        val_min = min(history['val_loss'])

        if train_final < train_min * 1.1 and val_final > val_min * 1.5:
            ax.text(0.5, 0.95, '⚠ OVERFITTING DETECTED',
                   transform=ax.transAxes,
                   ha='center', va='top',
                   bbox=dict(boxstyle='round', facecolor='red', alpha=0.3),
                   fontsize=12, fontweight='bold')

    # Plot 2: RMSE
    ax = axes[0, 1]
    ax.plot(epochs, history['val_rmse'], label='Val RMSE', linewidth=2, color='orange')
    ax.axhline(min(history['val_rmse']), color='green', linestyle='--',
              label=f'Best: {min(history["val_rmse"]):.4f}')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('RMSE')
    ax.set_title('Validation RMSE')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Mark best epoch
    best_rmse_epoch = np.argmin(history['val_rmse']) + 1
    ax.axvline(best_rmse_epoch, color='green', linestyle=':', alpha=0.5)
    ax.text(best_rmse_epoch, ax.get_ylim()[1], f'Best\nEpoch {best_rmse_epoch}',
           ha='center', va='top', fontsize=10)

    # Plot 3: R²
    ax = axes[1, 0]
    ax.plot(epochs, history['val_r2'], label='Val R²', linewidth=2, color='purple')
    ax.axhline(max(history['val_r2']), color='green', linestyle='--',
              label=f'Best: {max(history["val_r2"]):.4f}')
    ax.axhline(0, color='red', linestyle=':', alpha=0.5, label='Baseline')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('R²')
    ax.set_title('Validation R²')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Mark best epoch
    best_r2_epoch = np.argmax(history['val_r2']) + 1
    ax.axvline(best_r2_epoch, color='green', linestyle=':', alpha=0.5)
    ax.text(best_r2_epoch, ax.get_ylim()[1], f'Best\nEpoch {best_r2_epoch}',
           ha='center', va='top', fontsize=10)

    # Add shading for negative R² region
    ax.fill_between(epochs, ax.get_ylim()[0], 0, alpha=0.2, color='red')

    # Plot 4: Learning rate (if available)
    ax = axes[1, 1]
    if 'lr' in history and len(history['lr']) > 0:
        ax.plot(epochs, history['lr'], linewidth=2, color='brown')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Learning Rate')
        ax.set_title('Learning Rate Schedule')
        ax.set_yscale('log')
        ax.grid(True, alpha=0.3)
    else:
        # If no LR history, show MAE
        ax.plot(epochs, history['val_mae'], label='Val MAE', linewidth=2, color='teal')
        ax.axhline(min(history['val_mae']), color='green', linestyle='--',
                  label=f'Best: {min(history["val_mae"]):.4f}')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('MAE')
        ax.set_title('Validation MAE')
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved plot to {save_path}")
    else:
        plt.show()

    return fig


def diagnose_training(history_path):
    """
    Provide diagnostic analysis of training.

    Args:
        history_path: Path to history.json
    """
    with open(history_path, 'r') as f:
        history = json.load(f)

    print("\n" + "=" * 80)
    print("TRAINING DIAGNOSTIC REPORT")
    print("=" * 80)

    # Basic stats
    n_epochs = len(history['train_loss'])
    print(f"\nTotal epochs trained: {n_epochs}")

    # Loss analysis
    train_loss_start = history['train_loss'][0]
    train_loss_end = history['train_loss'][-1]
    train_loss_min = min(history['train_loss'])

    val_loss_start = history['val_loss'][0]
    val_loss_end = history['val_loss'][-1]
    val_loss_min = min(history['val_loss'])

    print(f"\n📊 LOSS ANALYSIS:")
    print(f"   Train loss: {train_loss_start:.6e} → {train_loss_end:.6e} (min: {train_loss_min:.6e})")
    print(f"   Val loss:   {val_loss_start:.6e} → {val_loss_end:.6e} (min: {val_loss_min:.6e})")

    # Check for overfitting
    print(f"\n🔍 OVERFITTING CHECK:")

    # Criterion 1: Train loss goes to near-zero while val loss is high
    if train_loss_end < 1e-4 and val_loss_end > 0.01:
        print("   ❌ SEVERE: Training loss near zero, but validation loss high")
        print("      → Model is memorizing training data")

    # Criterion 2: Train loss much lower than val loss
    ratio = val_loss_end / (train_loss_end + 1e-10)
    if ratio > 10:
        print(f"   ❌ Val/Train loss ratio: {ratio:.1f}x (should be ~1-2x)")
        print("      → Large generalization gap")
    elif ratio > 3:
        print(f"   ⚠  Val/Train loss ratio: {ratio:.1f}x (moderate overfitting)")
    else:
        print(f"   ✓  Val/Train loss ratio: {ratio:.1f}x (acceptable)")

    # Criterion 3: Val loss increasing while train loss decreasing
    if len(history['train_loss']) > 20:
        train_trend = history['train_loss'][-1] - history['train_loss'][-20]
        val_trend = history['val_loss'][-1] - history['val_loss'][-20]

        if train_trend < 0 and val_trend > 0:
            print("   ❌ Training loss decreasing while validation increasing")
            print("      → Classic overfitting pattern")

    # R² analysis
    r2_values = history['val_r2']
    best_r2 = max(r2_values)
    best_r2_epoch = np.argmax(r2_values) + 1
    final_r2 = r2_values[-1]

    print(f"\n📈 R² ANALYSIS:")
    print(f"   Best R²:  {best_r2:.4f} (epoch {best_r2_epoch})")
    print(f"   Final R²: {final_r2:.4f} (epoch {n_epochs})")

    if final_r2 < 0:
        print("   ❌ CRITICAL: Negative R² (worse than mean prediction)")
    elif final_r2 < 0.2:
        print("   ⚠  WARNING: Very low R² (poor predictive power)")
    elif final_r2 < best_r2 * 0.8:
        print(f"   ⚠  WARNING: R² degraded {(1-final_r2/best_r2)*100:.1f}% from best")

    if best_r2_epoch < n_epochs * 0.5:
        print(f"   ⚠  Best epoch was at {best_r2_epoch}/{n_epochs} (early in training)")
        print("      → Should have stopped earlier!")

    # RMSE analysis
    rmse_values = history['val_rmse']
    best_rmse = min(rmse_values)
    best_rmse_epoch = np.argmin(rmse_values) + 1
    final_rmse = rmse_values[-1]

    print(f"\n📉 RMSE ANALYSIS:")
    print(f"   Best RMSE:  {best_rmse:.4f} (epoch {best_rmse_epoch})")
    print(f"   Final RMSE: {final_rmse:.4f} (epoch {n_epochs})")

    if final_rmse > best_rmse * 1.1:
        print(f"   ⚠  WARNING: RMSE increased {(final_rmse/best_rmse-1)*100:.1f}% from best")

    # Recommendations
    print(f"\n💡 RECOMMENDATIONS:")

    if train_loss_end < 1e-4:
        print("   1. ⚠  Training loss too low - reduce model capacity or add regularization")
        print("      • Add weight decay (L2 regularization)")
        print("      • Add dropout to encoder/decoder")
        print("      • Reduce model size (hidden_dim, layers)")

    if best_r2_epoch < n_epochs * 0.6:
        print(f"   2. ⚠  Best model was at epoch {best_r2_epoch} - use early stopping")
        print(f"      • Set patience={min(15, max(5, (n_epochs-best_r2_epoch)//2))}")
        print("      • Monitor validation R² for stopping criterion")

    if ratio > 5:
        print("   3. ⚠  Large train/val gap - model may be too complex")
        print("      • Increase training data (more tiles)")
        print("      • Use data augmentation")
        print("      • Simplify model architecture")

    if best_r2 < 0.3:
        print("   4. ⚠  Low overall R² - model may need architectural changes")
        print("      • Check if data preprocessing is correct")
        print("      • Try different model architectures")
        print("      • Verify embedding quality")

    print("\n" + "=" * 80)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Analyze training dynamics')
    parser.add_argument('--history', type=str, required=True,
                       help='Path to history.json')
    parser.add_argument('--save_plot', type=str, default=None,
                       help='Path to save plot (optional)')

    args = parser.parse_args()

    # Run diagnostics
    diagnose_training(args.history)

    # Create plots
    plot_training_curves(args.history, args.save_plot)
