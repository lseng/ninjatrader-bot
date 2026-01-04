#!/bin/bash
# RunPod Training Script
#
# Deploy this script on RunPod to run distributed training.
# Supports single-node and multi-node configurations.
#
# Environment Variables (set in RunPod):
#   DATA_URL        - URL to download data (e.g., S3, GCS, or direct link)
#   STORAGE_URL     - PostgreSQL URL for distributed Optuna (optional)
#   STUDY_NAME      - Name for the Optuna study
#   N_TRIALS        - Number of optimization trials (default: 500)
#   TIMEOUT_HOURS   - Max training time in hours (default: 24)

set -e

echo "=== NinjaTrader ML Strategy Training ==="
echo "Started at: $(date)"

# Configuration
DATA_URL=${DATA_URL:-""}
STORAGE_URL=${STORAGE_URL:-""}
STUDY_NAME=${STUDY_NAME:-"nt_strategy_$(date +%Y%m%d_%H%M%S)"}
N_TRIALS=${N_TRIALS:-500}
TIMEOUT_HOURS=${TIMEOUT_HOURS:-24}
TIMEOUT_SECONDS=$((TIMEOUT_HOURS * 3600))

# Get number of CPUs
N_CPUS=$(nproc)
echo "CPUs available: $N_CPUS"

# Setup
WORK_DIR="/workspace/ninjatrader-bot"
DATA_DIR="$WORK_DIR/data/historical"
OUTPUT_DIR="$WORK_DIR/models/$(date +%Y%m%d_%H%M%S)"

mkdir -p "$DATA_DIR"
mkdir -p "$OUTPUT_DIR"

# Clone repo if not present
if [ ! -d "$WORK_DIR" ]; then
    echo "Cloning repository..."
    git clone https://github.com/YOUR_USERNAME/ninjatrader-bot.git "$WORK_DIR"
fi

cd "$WORK_DIR"

# Install dependencies
echo "Installing dependencies..."
pip install -q -r requirements.txt
pip install -q optuna psycopg2-binary  # For distributed training

# Download data if URL provided
if [ -n "$DATA_URL" ]; then
    echo "Downloading data from $DATA_URL..."
    DATA_FILE="$DATA_DIR/training_data.parquet"

    if [[ "$DATA_URL" == s3://* ]]; then
        aws s3 cp "$DATA_URL" "$DATA_FILE"
    elif [[ "$DATA_URL" == gs://* ]]; then
        gsutil cp "$DATA_URL" "$DATA_FILE"
    else
        curl -L -o "$DATA_FILE" "$DATA_URL"
    fi
else
    # Look for existing data
    DATA_FILE=$(find "$DATA_DIR" -name "*.parquet" | head -n 1)
    if [ -z "$DATA_FILE" ]; then
        echo "ERROR: No data file found and DATA_URL not set"
        exit 1
    fi
fi

echo "Using data file: $DATA_FILE"

# Build training command
TRAIN_CMD="python scripts/parallel_train.py"
TRAIN_CMD="$TRAIN_CMD --data '$DATA_FILE'"
TRAIN_CMD="$TRAIN_CMD --n-trials $N_TRIALS"
TRAIN_CMD="$TRAIN_CMD --n-jobs $N_CPUS"
TRAIN_CMD="$TRAIN_CMD --timeout $TIMEOUT_SECONDS"
TRAIN_CMD="$TRAIN_CMD --output '$OUTPUT_DIR'"

if [ -n "$STORAGE_URL" ]; then
    TRAIN_CMD="$TRAIN_CMD --storage '$STORAGE_URL'"
    TRAIN_CMD="$TRAIN_CMD --study-name '$STUDY_NAME'"
    TRAIN_CMD="$TRAIN_CMD --resume"  # Always resume in distributed mode
fi

echo "=== Starting Training ==="
echo "Command: $TRAIN_CMD"
echo "Output directory: $OUTPUT_DIR"
echo ""

# Run training
eval $TRAIN_CMD

# Check results
if [ -f "$OUTPUT_DIR/validation_results.json" ]; then
    echo ""
    echo "=== Training Complete ==="
    echo "Results:"
    cat "$OUTPUT_DIR/validation_results.json"

    # Upload results if storage configured
    if [ -n "$DATA_URL" ] && [[ "$DATA_URL" == s3://* ]]; then
        RESULTS_URL="${DATA_URL%/*}/results/$(basename $OUTPUT_DIR)"
        echo "Uploading results to $RESULTS_URL"
        aws s3 cp --recursive "$OUTPUT_DIR" "$RESULTS_URL/"
    fi
fi

echo ""
echo "Finished at: $(date)"
