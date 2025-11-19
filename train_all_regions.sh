#!/bin/bash
# Quick-start script to train all regional models for spatial extrapolation experiment
#
# This script trains both ANP and XGBoost models on all 4 regions:
# - Maine (Temperate)
# - Tolima (Tropical)
# - Hokkaido (Boreal)
# - Sudtirol (Alpine)
#
# Usage:
#   bash train_all_regions.sh
#   bash train_all_regions.sh --fast  # Quick test with fewer epochs

set -e  # Exit on error

# Parse arguments
FAST_MODE=false
if [[ "$1" == "--fast" ]]; then
    FAST_MODE=true
    echo "Running in FAST mode (fewer epochs for testing)"
fi

# Configuration
OUTPUT_DIR="./regional_results"
NUM_SEEDS=3  # Number of random seeds for robustness

if [ "$FAST_MODE" = true ]; then
    NUM_EPOCHS=10
    NUM_SEEDS=1
else
    NUM_EPOCHS=100
fi

echo "=========================================="
echo "SPATIAL EXTRAPOLATION: REGIONAL TRAINING"
echo "=========================================="
echo "Output directory: $OUTPUT_DIR"
echo "Number of seeds: $NUM_SEEDS"
echo "Epochs per model: $NUM_EPOCHS"
echo ""

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Define regions
REGIONS=("maine" "tolima" "hokkaido" "sudtirol")

# Train all regions
for REGION in "${REGIONS[@]}"; do
    echo ""
    echo "=========================================="
    echo "Training region: $REGION"
    echo "=========================================="

    # Run regional training script
    python run_regional_training.py \
        --regions "$REGION" \
        --output_dir "$OUTPUT_DIR" \
        --num_seeds "$NUM_SEEDS" \
        --epochs "$NUM_EPOCHS" \
        --models anp xgboost \
        --device cuda

    echo "Completed training for $REGION"
done

echo ""
echo "=========================================="
echo "TRAINING COMPLETE!"
echo "=========================================="
echo "All regional models trained."
echo ""
echo "Next step: Run spatial extrapolation evaluation"
echo "  python evaluate_spatial_extrapolation.py --results_dir $OUTPUT_DIR"
echo ""
