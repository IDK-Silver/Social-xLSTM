#!/bin/bash
# Quick Social Pooling Training Script for Development
# 快速 Social Pooling 訓練腳本（開發用）

set -e  # Exit on any error

echo "🚀 Starting Social Pooling Development Training..."

# Check if conda environment is activated
if [[ "$CONDA_DEFAULT_ENV" != "social_xlstm" ]]; then
    echo "❌ Please activate the social_xlstm conda environment first:"
    echo "   conda activate social_xlstm"
    exit 1
fi

# Configuration
DATA_H5="blob/dataset/pre-processed/h5/traffic_features_dev.h5"
CONFIG_FILE="cfgs/snakemake/dev.yaml"

echo "📁 Checking data availability..."

# Step 1: Generate H5 data if it doesn't exist
if [ ! -f "$DATA_H5" ]; then
    echo "🔄 H5 data file not found. Generating from raw data..."
    echo "   This may take a few minutes..."
    
    # Use Snakemake to generate just the H5 file
    snakemake create_h5_file --configfile "$CONFIG_FILE" --cores 4
    
    if [ ! -f "$DATA_H5" ]; then
        echo "❌ Failed to generate H5 data file. Please check raw data availability."
        exit 1
    fi
    echo "✅ H5 data file generated successfully!"
else
    echo "✅ H5 data file exists: $DATA_H5"
fi

# Step 2: Run Social Pooling Training
echo "🎯 Starting Social Pooling training..."
echo "   - Using spatial-aware pooling with radius 2.0m"
echo "   - Dev configuration: fast training for quick testing"

python scripts/train_distributed_social_xlstm.py \
    --enable_spatial_pooling \
    --spatial_radius 2.0 \
    --batch_size 8 \
    --epochs 5 \
    --sequence_length 10 \
    --prediction_length 3 \
    --hidden_size 64 \
    --num_blocks 2 \
    --experiment_name "dev_social_pooling_$(date +%Y%m%d_%H%M%S)" \
    --save_dir "logs/dev_experiments" \
    --fast_dev_run

echo "🎉 Social Pooling training completed!"
echo "📊 Check results in: logs/dev_experiments/"
echo "🔍 For full training, remove --fast_dev_run and increase --epochs"