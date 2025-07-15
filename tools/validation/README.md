# Validation Scripts

This directory contains testing and validation utilities for the Social-xLSTM project.

## Purpose
Scripts in this directory are used for:
- Model training validation and testing
- Overfitting detection and analysis
- Data splitting strategy validation
- Performance verification and benchmarking

## Scripts

### `training_validation.py`
**Purpose**: Minimal training test for overfitting verification
- Runs minimal training sessions to quickly test model behavior
- Validates overfitting fixes without full training infrastructure
- Creates training curve visualizations
- Provides rapid feedback on model configuration changes

**Usage**:
```bash
python scripts/validation/training_validation.py
```

**Key Features**:
- Configurable through YAML files
- Supports both LSTM and xLSTM models
- Generates training plots and metrics
- Quick validation (8-10 epochs)

### `overfitting_validation.py`
**Purpose**: Comprehensive overfitting fix effectiveness testing
- Tests overfitting fixes by running training comparisons
- Analyzes training results to check overfitting improvement
- Creates comparison plots showing before/after overfitting
- Provides detailed assessment of fix effectiveness

**Usage**:
```bash
python scripts/validation/overfitting_validation.py
```

**Key Features**:
- Historical comparison with original overfitting ratios
- Comprehensive training result analysis
- Automated assessment with clear recommendations
- Comparison visualizations

### `temporal_split_validation.py`
**Purpose**: Temporal data splitting strategy validation
- Tests temporal splitting functionality
- Compares temporal vs random splitting approaches
- Validates split quality and distribution consistency
- Creates splitting method comparison visualizations

**Usage**:
```bash
python scripts/validation/temporal_split_validation.py
```

**Key Features**:
- Temporal splitter integration testing
- Distribution difference analysis
- Quality validation with configurable thresholds
- Improvement measurement and reporting

## Configuration
- Most scripts use configuration files from `cfgs/` directory
- Default configurations are in `cfgs/fixed/` for validation purposes
- Development configurations available in `cfgs/snakemake/dev.yaml`

## Output Locations
- Validation results are typically saved to `blob/debug/` or `blob/experiments/dev/`
- Training plots and comparison charts saved as PNG files
- Results and metrics saved as JSON files
- Detailed logs available in console output

## Dependencies
- All scripts use the `social_xlstm` package
- PyTorch and PyTorch Lightning for model training
- Common dependencies: numpy, matplotlib, h5py, yaml
- Scripts automatically add `src/` to Python path for imports

## Integration with Training Pipeline
These validation scripts work with:
- Main training scripts in `scripts/train/`
- Configuration files in `cfgs/`
- Model implementations in `src/social_xlstm/models/`
- Dataset utilities in `src/social_xlstm/dataset/`

## Quality Thresholds
- **Overfitting Ratio**: < 5 (Good), < 10 (Acceptable), > 10 (Poor)
- **Distribution Difference**: < 15% (Good), < 30% (Acceptable)
- **Training Completion**: All validation scripts have timeout protections

## Related Documentation
- [ADR-0500: Scripts Directory Reorganization](../../docs/adr/0500-scripts-directory-reorganization.md)
- [Training System Guide](../../docs/guides/trainer_usage_guide.md)
- [LSTM Usage Guide](../../docs/guides/lstm_usage_guide.md)
- [Known Errors Documentation](../../docs/technical/known_errors.md)

## Troubleshooting
- If validation fails, check configuration file paths
- Ensure HDF5 datasets are available in `blob/dataset/pre-processed/h5/`
- For import errors, verify `social_xlstm` package is installed with `pip install -e .`
- Check CUDA availability for GPU-based validation