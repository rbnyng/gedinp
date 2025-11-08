"""
Evaluate a trained GEDI Neural Process model on test data.
"""

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

from data.dataset import GEDINeuralProcessDataset, collate_neural_process
from models.neural_process import GEDINeuralProcess
from utils.evaluation import evaluate_model, plot_results, compute_metrics


def main():
    parser = argparse.ArgumentParser(description='Evaluate GEDI Neural Process model')

    parser.add_argument('--model_dir', type=str, required=True,
                        help='Directory containing trained model')
    parser.add_argument('--checkpoint', type=str, default='best_r2_model.pt',
                        help='Checkpoint filename (default: best_r2_model.pt)')
    parser.add_argument('--test_split', type=str, default=None,
                        help='Path to test split CSV (default: model_dir/test_split.csv)')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--num_workers', type=int, default=0)

    args = parser.parse_args()

    model_dir = Path(args.model_dir)

    # Load config
    with open(model_dir / 'config.json', 'r') as f:
        config = json.load(f)

    print("=" * 80)
    print("EVALUATING GEDI NEURAL PROCESS")
    print("=" * 80)
    print(f"Model directory: {model_dir}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Architecture: {config.get('architecture_mode', 'deterministic')}")
    print(f"Device: {args.device}")
    print()

    # Load test data - use processed pickle file for embeddings
    processed_pkl = model_dir / 'processed_data.pkl'
    if not processed_pkl.exists():
        print(f"Error: Processed data not found at {processed_pkl}")
        print("The processed_data.pkl file contains embeddings and is required for evaluation.")
        return

    print(f"Loading processed data from: {processed_pkl}")
    with open(processed_pkl, 'rb') as f:
        full_df = pickle.load(f)

    # Load test split indices
    if args.test_split:
        test_csv = Path(args.test_split)
    else:
        test_csv = model_dir / 'test_split.csv'

    if not test_csv.exists():
        print(f"Error: Test split not found at {test_csv}")
        return

    print(f"Loading test split from: {test_csv}")
    test_split_df = pd.read_csv(test_csv)

    # Merge to get test data with embeddings
    # Use tile_id and shot indices to match rows
    if 'shot_number' in full_df.columns and 'shot_number' in test_split_df.columns:
        merge_cols = ['tile_id', 'shot_number']
    else:
        # Fallback to using lat/lon for matching
        merge_cols = ['tile_id', 'latitude', 'longitude']

    test_df = full_df.merge(
        test_split_df[merge_cols].drop_duplicates(),
        on=merge_cols,
        how='inner'
    )

    print(f"Test set: {len(test_df)} shots across {test_df['tile_id'].nunique()} tiles")

    # Create dataset
    global_bounds = tuple(config['global_bounds'])
    test_dataset = GEDINeuralProcessDataset(
        test_df,
        min_shots_per_tile=config.get('min_shots_per_tile', 10),
        log_transform_agbd=config.get('log_transform_agbd', True),
        augment_coords=False,
        coord_noise_std=0.0,
        global_bounds=global_bounds
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_neural_process,
        num_workers=args.num_workers
    )

    # Initialize model
    print("Initializing model...")
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
    checkpoint = torch.load(checkpoint_path, map_location=args.device)
    model.load_state_dict(checkpoint['model_state_dict'])

    print(f"Checkpoint epoch: {checkpoint.get('epoch', 'unknown')}")
    if 'val_metrics' in checkpoint:
        print(f"Validation metrics:")
        for key, val in checkpoint['val_metrics'].items():
            print(f"  {key}: {val:.4f}")
    print()

    # Evaluate
    print("Evaluating on test set...")
    predictions, targets, uncertainties, metrics = evaluate_model(
        model, test_loader, args.device
    )

    # Print results
    print("\n" + "=" * 80)
    print("TEST SET RESULTS")
    print("=" * 80)
    for key, val in metrics.items():
        print(f"{key.upper()}: {val:.4f}")
    print("=" * 80)

    # Save results
    results = {
        'metrics': metrics,
        'config': config,
        'checkpoint': args.checkpoint
    }

    with open(model_dir / 'test_results.json', 'w') as f:
        json.dump(results, f, indent=2)

    # Save predictions
    results_df = pd.DataFrame({
        'true': targets,
        'predicted': predictions,
        'uncertainty': uncertainties,
        'residual': predictions - targets
    })
    results_df.to_csv(model_dir / 'test_predictions.csv', index=False)

    # Create plots
    plot_results(predictions, targets, uncertainties, model_dir, 'test')

    print("\n" + "=" * 80)
    print("EVALUATION COMPLETE")
    print("=" * 80)
    print(f"Results saved to: {model_dir}")
    print("Files:")
    print("  - test_results.json")
    print("  - test_predictions.csv")
    print("  - evaluation_test.png")
    print("=" * 80)


if __name__ == '__main__':
    main()
