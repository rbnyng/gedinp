#!/bin/bash
#
# Example Workflow for Pareto Frontier Analysis
#
# This script demonstrates the complete workflow from baseline training
# to Pareto frontier visualization.
#
# Usage:
#   bash example_pareto_workflow.sh
#
# Prerequisites:
#   - Python environment with all dependencies installed
#   - Access to GEDI data for the specified region
#

set -e  # Exit on error

# Configuration
REGION_BBOX="-75.0 -10.0 -70.0 -5.0"  # Example: Amazon region
BASELINE_DIR="./outputs_baselines_example"
PARETO_DIR="./outputs_pareto_example"
ANP_DIR="./outputs_anp_example"  # Optional: if you have ANP results

echo "=========================================="
echo "Pareto Frontier Analysis - Example Workflow"
echo "=========================================="
echo ""

# Step 1: Train baseline models (if not already done)
if [ ! -d "$BASELINE_DIR" ]; then
    echo "Step 1: Training baseline models..."
    echo "This will take approximately 15-30 minutes"
    echo ""

    python train_baselines.py \
        --region_bbox $REGION_BBOX \
        --output_dir $BASELINE_DIR \
        --models rf xgb \
        --rf_n_estimators 100 \
        --rf_max_depth 6 \
        --xgb_n_estimators 100 \
        --xgb_max_depth 6

    echo ""
    echo "✓ Baseline training complete"
else
    echo "✓ Found existing baseline outputs at $BASELINE_DIR"
fi

echo ""
echo "=========================================="

# Step 2: Run hyperparameter sweep (quick mode)
echo "Step 2: Running hyperparameter sweep (quick mode)..."
echo "This will take approximately 30-60 minutes"
echo ""

python analyze_pareto_frontier.py \
    --baseline_dir $BASELINE_DIR \
    --output_dir $PARETO_DIR \
    --models rf xgb \
    --quick

echo ""
echo "✓ Hyperparameter sweep complete"
echo ""
echo "=========================================="

# Step 3: Generate visualizations
echo "Step 3: Generating visualizations..."
echo ""

# Check if ANP results exist
if [ -f "$ANP_DIR/results.json" ]; then
    echo "Found ANP results at $ANP_DIR/results.json"
    python plot_pareto.py \
        --results_dir $PARETO_DIR \
        --anp_results $ANP_DIR/results.json \
        --output_dir $PARETO_DIR/plots
else
    echo "No ANP results found (optional)"
    python plot_pareto.py \
        --results_dir $PARETO_DIR \
        --output_dir $PARETO_DIR/plots
fi

echo ""
echo "✓ Visualization complete"
echo ""
echo "=========================================="
echo "WORKFLOW COMPLETE"
echo "=========================================="
echo ""
echo "Generated files:"
echo "  Results:"
echo "    - $PARETO_DIR/pareto_results.json"
echo "    - $PARETO_DIR/pareto_results.csv"
echo ""
echo "  Plots:"
echo "    - $PARETO_DIR/plots/pareto_frontier_accuracy_calibration.png"
echo "    - $PARETO_DIR/plots/pareto_time_tradeoffs.png"
echo "    - $PARETO_DIR/plots/pareto_summary_table.csv"
echo "    - $PARETO_DIR/plots/pareto_summary_table.md"
echo ""
echo "=========================================="
echo "Next steps:"
echo "  1. Review the plots in $PARETO_DIR/plots/"
echo "  2. Check the summary table for best configurations"
echo "  3. Include plots and findings in your paper"
echo ""
echo "For full sweep (not quick mode), run:"
echo "  python analyze_pareto_frontier.py \\"
echo "      --baseline_dir $BASELINE_DIR \\"
echo "      --output_dir $PARETO_DIR \\"
echo "      --models rf xgb"
echo "=========================================="
