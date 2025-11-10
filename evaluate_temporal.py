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
from utils.evaluation import evaluate_model, plot_results, compute_metrics
from utils.config import load_config, save_config, get_global_bounds
from utils.model import load_model_from_checkpoint
from utils.normalization import denormalize_agbd


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

    config = load_config(model_dir / 'config.json')

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

    querier = GEDIQuerier(cache_dir=args.cache_dir)

    all_gedi_dfs = []
    for year in args.test_years:
        print(f"\nQuerying year {year}...")
        year_df = querier.query_region_tiles(
            region_bbox=args.region_bbox,
            tile_size=0.1,
            start_time=f"{year}-01-01",
            end_time=f"{year}-12-31",
            max_agbd=500.0  # Cap at 500 Mg/ha to remove unrealistic outliers
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

    global_bounds = get_global_bounds(config)
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

    model, checkpoint, checkpoint_path = load_model_from_checkpoint(
        model_dir, args.device, args.checkpoint
    )

    print(f"Checkpoint epoch: {checkpoint.get('epoch', 'unknown')}")
    if 'val_metrics' in checkpoint:
        print(f"Validation metrics (from training):")
        for key, val in checkpoint['val_metrics'].items():
            print(f"  {key}: {val:.4f}")

    print("\n" + "=" * 80)
    print("Step 5: Evaluating on temporal holdout...")
    print("=" * 80)

    predictions, targets, uncertainties, metrics = evaluate_model(
        model, eval_loader, args.device,
        agbd_scale=config.get('agbd_scale', 200.0),
        log_transform_agbd=config.get('log_transform_agbd', True),
        denormalize_for_reporting=False  # Parameter deprecated but kept for compatibility
    )

    print("\n" + "=" * 80)
    print(f"TEMPORAL VALIDATION RESULTS (Years: {args.test_years})")
    print("=" * 80)
    print("\nLog-space metrics (aligned with training):")
    print(f"  Log R²:    {metrics.get('log_r2', 0):.4f}")
    print(f"  Log RMSE:  {metrics.get('log_rmse', 0):.4f}")
    print(f"  Log MAE:   {metrics.get('log_mae', 0):.4f}")
    print("\nLinear-space metrics (Mg/ha, for interpretability):")
    print(f"  RMSE:      {metrics.get('linear_rmse', 0):.2f} Mg/ha")
    print(f"  MAE:       {metrics.get('linear_mae', 0):.2f} Mg/ha")
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
    save_config(results, results_path)
    print(f"Saved metrics to: {results_path}")

    # Denormalize predictions and targets for plotting and saving (convert to Mg/ha)
    predictions_linear = denormalize_agbd(
        predictions,
        agbd_scale=config.get('agbd_scale', 200.0),
        log_transform=config.get('log_transform_agbd', True)
    )
    targets_linear = denormalize_agbd(
        targets,
        agbd_scale=config.get('agbd_scale', 200.0),
        log_transform=config.get('log_transform_agbd', True)
    )

    results_df = pd.DataFrame({
        'true': targets_linear,
        'predicted': predictions_linear,
        'uncertainty': uncertainties,
        'residual': predictions_linear - targets_linear
    })
    predictions_path = model_dir / f'temporal_predictions_{output_suffix}.csv'
    results_df.to_csv(predictions_path, index=False)
    print(f"Saved predictions to: {predictions_path}")

    years_label = '_'.join(map(str, args.test_years))
    plot_results(predictions_linear, targets_linear, uncertainties, model_dir, years_label)

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
