#!/bin/bash
# Simple test script to verify caching mechanism works

echo "================================================================================"
echo "CACHE TESTING SCRIPT"
echo "================================================================================"
echo ""
echo "This script will:"
echo "1. Run preprocessing for a small region and seed 42"
echo "2. Check if cache was created"
echo "3. Run train.py which should load from cache"
echo "4. Verify cache was actually used"
echo ""
echo "================================================================================"
echo ""

# Use a very small region for testing (small part of Maine)
REGION="-70.0 44.0 -69.9 44.1"
SEED=42
CACHE_DIR="./cache_test"

echo "Test parameters:"
echo "  Region: $REGION"
echo "  Seed: $SEED"
echo "  Cache dir: $CACHE_DIR"
echo ""

# Clean up any existing test cache
echo "Cleaning up old test cache..."
rm -rf "$CACHE_DIR/preprocessed"
echo ""

# Step 1: Run preprocessing
echo "================================================================================"
echo "STEP 1: Running preprocessing script"
echo "================================================================================"
python preprocess_data.py \
    --region_bbox $REGION \
    --seed $SEED \
    --cache_dir "$CACHE_DIR" \
    --start_time 2022-01-01 \
    --end_time 2022-01-31 \
    --embedding_year 2022

if [ $? -ne 0 ]; then
    echo ""
    echo "ERROR: Preprocessing failed!"
    exit 1
fi

echo ""
echo "Preprocessing completed. Checking for cache..."
echo ""

# Step 2: Check if cache was created
CACHE_PATH=$(find "$CACHE_DIR/preprocessed" -type d -name "*seed_42*" 2>/dev/null | head -n 1)

if [ -z "$CACHE_PATH" ]; then
    echo "ERROR: Cache directory not found!"
    echo "Looking in: $CACHE_DIR/preprocessed"
    ls -la "$CACHE_DIR/preprocessed" 2>/dev/null || echo "Directory doesn't exist"
    exit 1
fi

echo "✓ Cache found at: $CACHE_PATH"
echo ""
echo "Cache contents:"
ls -lh "$CACHE_PATH"
echo ""

# Step 3: Run train.py (should load from cache)
echo "================================================================================"
echo "STEP 2: Running train.py (should load from cache)"
echo "================================================================================"
echo ""
echo "Watch for 'LOADING FROM PREPROCESSED CACHE' message..."
echo ""

python train.py \
    --region_bbox $REGION \
    --seed $SEED \
    --cache_dir "$CACHE_DIR" \
    --start_time 2022-01-01 \
    --end_time 2022-01-31 \
    --embedding_year 2022 \
    --output_dir "$CACHE_DIR/test_output" \
    --epochs 1 \
    --device cpu 2>&1 | grep -E "(CACHE|cache|Checking for cached|Step 1:|Step 2:|Step 3:)" | head -20

echo ""
echo "================================================================================"
echo "Test complete!"
echo "================================================================================"
echo ""
echo "If you saw 'LOADING FROM PREPROCESSED CACHE', the cache is working!"
echo "If you saw 'Step 1: Querying GEDI data...', the cache is NOT working."
echo ""
