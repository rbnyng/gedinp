#!/bin/bash
#
# Helper script to run uncertainty vs distance analysis
#
# Usage examples:
#   ./run_uncertainty_distance_analysis.sh maine
#   ./run_uncertainty_distance_analysis.sh all
#

set -e

RESULTS_DIR="${RESULTS_DIR:-./regional_results}"
OUTPUT_DIR="${OUTPUT_DIR:-./uncertainty_distance_analysis}"
DEVICE="${DEVICE:-cuda}"

if [ $# -eq 0 ]; then
    echo "Usage: $0 <region|all>"
    echo ""
    echo "Available regions:"
    echo "  maine      - Maine (Temperate)"
    echo "  tolima     - Tolima (Tropical)"
    echo "  hokkaido   - Hokkaido (Boreal)"
    echo "  sudtirol   - Sudtirol (Alpine)"
    echo "  guaviare   - Guaviare (Tropical)"
    echo "  all        - Analyze all regions"
    echo ""
    echo "Environment variables:"
    echo "  RESULTS_DIR  - Path to regional results (default: ./regional_results)"
    echo "  OUTPUT_DIR   - Path to output directory (default: ./uncertainty_distance_analysis)"
    echo "  DEVICE       - Device to use (default: cuda)"
    exit 1
fi

REGION=$1

echo "========================================"
echo "Uncertainty vs Distance Analysis"
echo "========================================"
echo "Results dir: $RESULTS_DIR"
echo "Output dir:  $OUTPUT_DIR"
echo "Device:      $DEVICE"
echo "========================================"

if [ "$REGION" = "all" ]; then
    echo "Analyzing all regions..."
    python analyze_uncertainty_distance.py \
        --results_dir "$RESULTS_DIR" \
        --output_dir "$OUTPUT_DIR" \
        --all_regions \
        --device "$DEVICE"
else
    echo "Analyzing region: $REGION"
    python analyze_uncertainty_distance.py \
        --results_dir "$RESULTS_DIR" \
        --output_dir "$OUTPUT_DIR" \
        --region "$REGION" \
        --device "$DEVICE"
fi

echo ""
echo "========================================"
echo "Analysis complete!"
echo "Results saved to: $OUTPUT_DIR"
echo "========================================"
