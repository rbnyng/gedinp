import argparse
import json
import subprocess
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from datetime import datetime
import torch
from utils.config import load_config

def run_training(base_args, architecture_mode, output_subdir):
    print("=" * 80)
    print(f"Training: {architecture_mode}")
    print("=" * 80)

    cmd = [
        'python', 'train.py',
        '--region_bbox', *[str(x) for x in base_args['region_bbox']],
        '--start_time', base_args['start_time'],
        '--end_time', base_args['end_time'],
        '--embedding_year', str(base_args['embedding_year']),
        '--cache_dir', base_args['cache_dir'],
        '--architecture_mode', architecture_mode,
        '--hidden_dim', str(base_args['hidden_dim']),
        '--latent_dim', str(base_args['latent_dim']),
        '--epochs', str(base_args['epochs']),
        '--batch_size', str(base_args['batch_size']),
        '--lr', str(base_args['lr']),
        '--output_dir', str(output_subdir),
        '--seed', str(base_args['seed']),
        '--kl_weight_max', str(base_args['kl_weight_max']),
        '--kl_warmup_epochs', str(base_args['kl_warmup_epochs'])
    ]

    result = subprocess.run(cmd, capture_output=False)

    if result.returncode != 0:
        print(f"Warning: Training failed for {architecture_mode}")
        return None

    try:
        config = load_config(output_subdir / 'config.json')

        checkpoint = torch.load(output_subdir / 'best_r2_model.pt',
                               map_location='cpu', weights_only=False)

        return {
            'architecture': architecture_mode,
            'val_metrics': checkpoint.get('val_metrics', {}),
            'test_metrics': checkpoint.get('test_metrics', {}),
            'epoch': checkpoint.get('epoch', -1),
            'r2': checkpoint.get('r2', float('-inf')),
            'output_dir': str(output_subdir)
        }
    except Exception as e:
        print(f"Error loading results for {architecture_mode}: {e}")
        return None


def compare_results(results, output_dir):
    rows = []
    for res in results:
        if res is None:
            continue
        row = {
            'Architecture': res['architecture'],
            'Val R²': res['r2'],
            'Val RMSE': res['val_metrics'].get('rmse', np.nan),
            'Val MAE': res['val_metrics'].get('mae', np.nan),
            'Test R²': res['test_metrics'].get('r2', np.nan),
            'Test RMSE': res['test_metrics'].get('rmse', np.nan),
            'Test MAE': res['test_metrics'].get('mae', np.nan),
            'Epochs': res['epoch'],
            'Mean Uncertainty': res['val_metrics'].get('mean_uncertainty', np.nan)
        }
        rows.append(row)

    df = pd.DataFrame(rows)

    # Sort by Test R² (descending) - use test performance as primary metric
    df = df.sort_values('Test R²', ascending=False)

    df.to_csv(output_dir / 'ablation_results.csv', index=False)

    print("\n" + "=" * 80)
    print("ABLATION STUDY RESULTS")
    print("=" * 80)
    print(df.to_string(index=False))
    print("=" * 80)

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('Architecture Ablation Study - Validation vs Test Performance', fontsize=16, fontweight='bold')

    # R² comparison
    ax = axes[0, 0]
    x = np.arange(len(df))
    width = 0.35
    ax.bar(x - width/2, df['Val R²'], width, label='Validation', alpha=0.8)
    ax.bar(x + width/2, df['Test R²'], width, label='Test', alpha=0.8)
    ax.set_ylabel('R² Score', fontweight='bold')
    ax.set_title('R² Score by Architecture')
    ax.set_xticks(x)
    ax.set_xticklabels(df['Architecture'])
    ax.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

    # RMSE comparison
    ax = axes[0, 1]
    ax.bar(x - width/2, df['Val RMSE'], width, label='Validation', alpha=0.8)
    ax.bar(x + width/2, df['Test RMSE'], width, label='Test', alpha=0.8)
    ax.set_ylabel('RMSE', fontweight='bold')
    ax.set_title('RMSE by Architecture')
    ax.set_xticks(x)
    ax.set_xticklabels(df['Architecture'])
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

    # MAE comparison
    ax = axes[0, 2]
    ax.bar(x - width/2, df['Val MAE'], width, label='Validation', alpha=0.8)
    ax.bar(x + width/2, df['Test MAE'], width, label='Test', alpha=0.8)
    ax.set_ylabel('MAE', fontweight='bold')
    ax.set_title('MAE by Architecture')
    ax.set_xticks(x)
    ax.set_xticklabels(df['Architecture'])
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

    # Test R² only (for clarity)
    ax = axes[1, 0]
    colors = sns.color_palette("husl", len(df))
    ax.bar(df['Architecture'], df['Test R²'], color=colors)
    ax.set_ylabel('Test R² Score', fontweight='bold')
    ax.set_title('Test R² Score (Primary Metric)')
    ax.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
    ax.grid(True, alpha=0.3)
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

    # Validation vs Test R² scatter
    ax = axes[1, 1]
    ax.scatter(df['Val R²'], df['Test R²'], s=100, alpha=0.6, c=colors)
    for i, arch in enumerate(df['Architecture']):
        ax.annotate(arch, (df['Val R²'].iloc[i], df['Test R²'].iloc[i]),
                   fontsize=8, ha='right', va='bottom')
    lims = [min(df['Val R²'].min(), df['Test R²'].min()) - 0.05,
            max(df['Val R²'].max(), df['Test R²'].max()) + 0.05]
    ax.plot(lims, lims, 'k--', alpha=0.5, label='Val = Test')
    ax.set_xlabel('Validation R²', fontweight='bold')
    ax.set_ylabel('Test R²', fontweight='bold')
    ax.set_title('Validation vs Test R²')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Uncertainty comparison
    ax = axes[1, 2]
    ax.bar(df['Architecture'], df['Mean Uncertainty'], color=colors)
    ax.set_ylabel('Mean Predicted Uncertainty', fontweight='bold')
    ax.set_title('Predicted Uncertainty by Architecture')
    ax.grid(True, alpha=0.3)
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

    plt.tight_layout()
    plt.savefig(output_dir / 'ablation_comparison.png', dpi=300, bbox_inches='tight')
    print(f"\nSaved comparison plot to: {output_dir / 'ablation_comparison.png'}")

    # Parameter count comparison
    param_counts = {}
    for res in results:
        if res is None:
            continue
        config_path = Path(res['output_dir']) / 'config.json'
        config = load_config(config_path)

        # Estimate parameters (rough calculation)
        mode = res['architecture']
        hidden = config['hidden_dim']
        context = config['context_repr_dim']
        latent = config.get('latent_dim', 128)

        # Rough parameter estimates
        base_params = 500000  # Encoders + decoder base
        attention_params = 0
        latent_params = 0

        if mode in ['deterministic', 'anp']:
            attention_params = context * context * 4  # Attention
        if mode in ['latent', 'anp']:
            latent_params = hidden * latent * 2  # Latent encoder

        total = base_params + attention_params + latent_params
        param_counts[mode] = total

    param_df = pd.DataFrame([
        {'Architecture': k, 'Parameters': v}
        for k, v in param_counts.items()
    ])
    param_df.to_csv(output_dir / 'parameter_counts.csv', index=False)

    with open(output_dir / 'ablation_summary.txt', 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("ABLATION STUDY SUMMARY\n")
        f.write("=" * 80 + "\n\n")

        f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        f.write("Architecture Variants Tested:\n")
        f.write("1. CNP: Conditional Neural Process (mean pooling baseline)\n")
        f.write("2. Deterministic: Attention-based aggregation only\n")
        f.write("3. Latent: Stochastic latent path only\n")
        f.write("4. ANP: Full Attentive Neural Process (attention + latent)\n\n")

        f.write("-" * 80 + "\n")
        f.write("RESULTS\n")
        f.write("-" * 80 + "\n\n")
        f.write(df.to_string(index=False) + "\n\n")

        best = df.iloc[0]
        f.write("-" * 80 + "\n")
        f.write("BEST ARCHITECTURE (by Test R²)\n")
        f.write("-" * 80 + "\n")
        f.write(f"Architecture: {best['Architecture']}\n\n")
        f.write(f"Validation Performance:\n")
        f.write(f"  R² Score: {best['Val R²']:.4f}\n")
        f.write(f"  RMSE: {best['Val RMSE']:.4f}\n")
        f.write(f"  MAE: {best['Val MAE']:.4f}\n\n")
        f.write(f"Test Performance:\n")
        f.write(f"  R² Score: {best['Test R²']:.4f}\n")
        f.write(f"  RMSE: {best['Test RMSE']:.4f}\n")
        f.write(f"  MAE: {best['Test MAE']:.4f}\n\n")

        cnp_r2 = df[df['Architecture'] == 'cnp']['Test R²'].values[0] if 'cnp' in df['Architecture'].values else None
        det_r2 = df[df['Architecture'] == 'deterministic']['Test R²'].values[0] if 'deterministic' in df['Architecture'].values else None
        lat_r2 = df[df['Architecture'] == 'latent']['Test R²'].values[0] if 'latent' in df['Architecture'].values else None
        anp_r2 = df[df['Architecture'] == 'anp']['Test R²'].values[0] if 'anp' in df['Architecture'].values else None

        if cnp_r2 and det_r2:
            improvement = ((det_r2 - cnp_r2) / abs(cnp_r2)) * 100
            f.write(f"\n1. Attention vs Mean Pooling:\n")
            f.write(f"   Deterministic (attention) vs CNP (mean pooling)\n")
            f.write(f"   R² improvement: {improvement:+.2f}%\n")

        if det_r2 and lat_r2:
            f.write(f"\n2. Deterministic vs Stochastic:\n")
            f.write(f"   Deterministic: R² = {det_r2:.4f}\n")
            f.write(f"   Latent: R² = {lat_r2:.4f}\n")
            if det_r2 > lat_r2:
                f.write(f"   → Attention is more effective than latent path alone\n")
            else:
                f.write(f"   → Latent path is more effective than attention alone\n")

        if det_r2 and anp_r2:
            improvement = ((anp_r2 - det_r2) / abs(det_r2)) * 100
            f.write(f"\n3. Adding Latent Path to Attention:\n")
            f.write(f"   Improvement: {improvement:+.2f}%\n")
            if improvement > 1:
                f.write(f"   → Latent path provides significant value\n")
            elif improvement > 0:
                f.write(f"   → Latent path provides marginal improvement\n")
            else:
                f.write(f"   → Latent path does not help (deterministic task)\n")

    print(f"\nSaved summary to: {output_dir / 'ablation_summary.txt'}")


def main():
    parser = argparse.ArgumentParser(description='Run ablation study')

    # Data arguments
    parser.add_argument('--region_bbox', type=float, nargs=4, required=True,
                        help='Region bounding box: min_lon min_lat max_lon max_lat')
    parser.add_argument('--start_time', type=str, default='2019-01-01')
    parser.add_argument('--end_time', type=str, default='2023-12-31')
    parser.add_argument('--embedding_year', type=int, default=2024)
    parser.add_argument('--cache_dir', type=str, default='./cache')

    # Training arguments
    parser.add_argument('--hidden_dim', type=int, default=512)
    parser.add_argument('--latent_dim', type=int, default=128)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--kl_weight_max', type=float, default=1.0)
    parser.add_argument('--kl_warmup_epochs', type=int, default=10)

    # Output
    parser.add_argument('--output_dir', type=str, default='./ablation_study')

    # Which architectures to run
    parser.add_argument('--architectures', nargs='+',
                        default=['cnp', 'deterministic', 'latent', 'anp'],
                        choices=['cnp', 'deterministic', 'latent', 'anp'],
                        help='Which architectures to test')

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir / 'ablation_config.json', 'w') as f:
        json.dump(vars(args), f, indent=2)

    print("=" * 80)
    print("ABLATION STUDY")
    print("=" * 80)
    print(f"Output directory: {output_dir}")
    print(f"Architectures to test: {args.architectures}")
    print("=" * 80)

    base_args = {
        'region_bbox': args.region_bbox,
        'start_time': args.start_time,
        'end_time': args.end_time,
        'embedding_year': args.embedding_year,
        'cache_dir': args.cache_dir,
        'hidden_dim': args.hidden_dim,
        'latent_dim': args.latent_dim,
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'lr': args.lr,
        'seed': args.seed,
        'kl_weight_max': args.kl_weight_max,
        'kl_warmup_epochs': args.kl_warmup_epochs
    }

    results = []
    for arch in args.architectures:
        output_subdir = output_dir / arch
        output_subdir.mkdir(parents=True, exist_ok=True)

        result = run_training(base_args, arch, output_subdir)
        results.append(result)

    compare_results(results, output_dir)

    print("\n" + "=" * 80)
    print("ABLATION STUDY COMPLETE")
    print("=" * 80)
    print(f"Results saved to: {output_dir}")
    print("Files:")
    print(f"  - ablation_results.csv")
    print(f"  - ablation_comparison.png")
    print(f"  - ablation_summary.txt")
    print("=" * 80)


if __name__ == '__main__':
    main()
