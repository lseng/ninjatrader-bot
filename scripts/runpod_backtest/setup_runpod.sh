#!/bin/bash
# RunPod Setup Script
# Run this on the RunPod instance to set up the backtesting environment

set -e

echo "=========================================="
echo "RunPod Backtesting Setup"
echo "=========================================="

# Create workspace
mkdir -p /workspace/data
mkdir -p /workspace/scripts
mkdir -p /workspace/results

# Install dependencies
echo "Installing Python packages..."
pip install pandas numpy scikit-learn joblib pyarrow fastparquet --quiet

# Verify
echo ""
echo "Python packages installed:"
pip list | grep -E "pandas|numpy|scikit-learn|joblib|pyarrow"

echo ""
echo "CPU count: $(nproc)"
echo "Memory: $(free -h | grep Mem | awk '{print $2}')"

echo ""
echo "=========================================="
echo "Setup complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "1. Upload data: scp MES_1s_2years.parquet root@IP:/workspace/data/"
echo "2. Upload scripts: scp -r scripts/runpod_backtest/* root@IP:/workspace/scripts/"
echo "3. Run: cd /workspace/scripts && python orchestrator.py"
