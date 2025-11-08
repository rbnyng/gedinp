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
from utils.evaluation import evaluate_model, plot_results, compute_metrics
from utils.config import load_config, get_global_bounds
from utils.model import load_model_from_checkpoint


def main():
    parser = argparse.ArgumentParser(description='Evaluate GEDI Neural Process model')

    parser.add_argument('--model_dir', type=str, required=True,
                        help='Directory containing trained model')
    parser.add_argument('--checkpoint', type=str, default='best_r2_model.pt',
                        help='Checkpoint filename (default: best_r2_model.pt)')
    parser.add_argument('--test_split', type=str, default=None,
                        help='Path to test split CSV (default: model_dir/test_split.csv)')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--batch_size', type=int, default=1,
                        help='Batch size (default: 1 for memory safety)')
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--max_context_shots', type=int, default=20000,
                        help='Maximum context shots per tile (subsample if exceeded)')
    parser.add_argument('--max_targets_per_chunk', type=int, default=1000,
                        help='Maximum target shots to process at once')

    args = parser.parse_args()

    model_dir = Path(args.model_dir)

    # Load config
    config = load_config(model_dir / 'config.json')

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
    global_bounds = get_global_bounds(config)
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

    # Initialize and load model
    print("Initializing model...")
    model, checkpoint, checkpoint_path = load_model_from_checkpoint(
        model_dir, args.device, args.checkpoint
    )

    print(f"Checkpoint epoch: {checkpoint.get('epoch', 'unknown')}")
    if 'val_metrics' in checkpoint:
        print(f"Validation metrics:")
        for key, val in checkpoint['val_metrics'].items():
            print(f"  {key}: {val:.4f}")
    print()

    # Evaluate
    print("Evaluating on test set...")
    print(f"Memory settings: max_context={args.max_context_shots}, max_targets_per_chunk={args.max_targets_per_chunk}")
    predictions, targets, uncertainties, metrics = evaluate_model(
        model, test_loader, args.device,
        max_context_shots=args.max_context_shots,
        max_targets_per_chunk=args.max_targets_per_chunk
    )

    # Print results
    print("\n" + "=" * 80)
    print("TEST SET RESULTS")
    print("=" * 80)
    for key, val in metrics.items():
        print(f"{key.upper()}: {val:.4f}")
    print("=" * 80)

    # Save results (convert numpy types to Python native types for JSON)
    results = {
        'metrics': {k: float(v) for k, v in metrics.items()},
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
