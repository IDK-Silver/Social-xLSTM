# Without Social Pooling Training Scripts

This directory contains training scripts for LSTM/xLSTM models **without** Social Pooling mechanism.

## Purpose
- Test pure LSTM/xLSTM performance without spatial interactions
- Establish performance metrics for individual VD predictions
- Compare against Social-xLSTM models when available

## Scripts
- `train_single_vd.py` - Train single VD LSTM model
- `train_multi_vd.py` - Train multiple independent VD LSTM models
- `common.py` - Shared utility functions

## Usage
```bash
# Single VD training
python scripts/train/without_social_pooling/train_single_vd.py --model_type lstm

# Multi VD independent training
python scripts/train/without_social_pooling/train_multi_vd.py --model_type lstm
```

## Note
These scripts do NOT implement Social Pooling - each VD is processed independently.