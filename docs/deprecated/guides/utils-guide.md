# Utility Scripts

This directory contains general-purpose utilities and helper scripts for the Social-xLSTM project.

## Purpose
Scripts in this directory are used for:
- File structure inspection and analysis
- Development and maintenance utilities
- Visualization and reporting helpers

## Scripts

### Core Utilities

#### `h5_structure_inspector.py`
**Purpose**: Quick HDF5 file structure inspection
- Checks H5 file structure and contents
- Displays dataset shapes and data types
- Inspects metadata and group organization
- Useful for debugging data loading issues

**Usage**:
```bash
python scripts/utils/h5_structure_inspector.py
```


### Integration Tools

### Development Tools

#### `run_all_plots.py`
**Purpose**: Generate all training plots
- Automated plot generation for experiments
- Configurable for dev/production environments
- Timeout protection and error handling
- Comprehensive visualization suite

**Usage**:
```bash
python scripts/utils/run_all_plots.py --config dev --timeout 120
```

#### `generate_training_plots.py`
**Purpose**: Training visualization generation
- Creates training curve plots
- Loss progression visualization
- Model performance analysis charts
- Customizable plotting options

#### `generate_training_report.py`
**Purpose**: Comprehensive training report generation
- Automated report creation from training results
- Statistical analysis and summaries
- Performance metrics compilation
- HTML/PDF report generation

#### `quality_check.py`
**Purpose**: Data and model quality assessment
- Comprehensive quality validation
- Automated quality scoring
- Issue identification and reporting
- Quality trend analysis

### Legacy and Specialized Tools


#### `plot_vd_point.py`
**Purpose**: Plot Vehicle Detector (VD) coordinates
- Visualizes VD locations and spatial relationships
- Coordinate system validation
- Spatial analysis and mapping
- VD network visualization

#### `test_integration.py`
**Purpose**: Integration testing for major components
- Tests data stabilization integration
- Validates diagnostic system integration
- Configuration generation testing
- End-to-end integration validation

## Configuration
- Most utilities use configuration files from `cfgs/` directory
- Development configurations in `cfgs/snakemake/dev.yaml`
- Production configurations in `cfgs/snakemake/default.yaml`

## Output Locations
- Diagnostic results: `blob/debug/` or `blob/analysis/`
- Generated configurations: `cfgs/fixed/` or specified output directory
- Plots and visualizations: `blob/plots/` or specified directory
- Reports: `blob/reports/` or specified directory

## Dependencies
- All scripts use the `social_xlstm` package
- Common dependencies: numpy, matplotlib, h5py, yaml, pandas
- Some scripts require PyTorch for model-related utilities
- Scripts automatically add `src/` to Python path for imports

## Integration with Main Codebase
These utilities complement integrated functions in:
- `src/social_xlstm/dataset/storage/h5_converter.py` - Data stability
- `src/social_xlstm/evaluation/evaluator.py` - Diagnostics
- `src/social_xlstm/dataset/core/processor.py` - Configuration generation
- `src/social_xlstm/utils/spatial_coords.py` - Coordinate utilities

## Usage Patterns

### Quick Development Check
```bash
# Use CLAUDE.md for project overview and quick start commands
cat CLAUDE.md
```

### Data Quality Pipeline
```bash
# Use integrated functionality from main codebase
python -c "from social_xlstm.dataset.storage.h5_converter import TrafficFeatureExtractor; TrafficFeatureExtractor.validate_dataset_quality('path/to/dataset')"
```

## Related Documentation
- [ADR-0500: Scripts Directory Reorganization](../../docs/adr/0500-scripts-directory-reorganization.md)
- [Quick Start Guide](../../docs/QUICK_START.md)
- [Trainer Usage Guide](../../docs/guides/trainer_usage_guide.md)
- [Project Status](../../docs/reports/project_status.md)

## Troubleshooting
- For import errors, ensure `social_xlstm` package is installed: `pip install -e .`
- Check file paths are absolute when specified
- Verify HDF5 datasets exist before running analysis tools
- Use `--help` flag with any script for detailed usage information