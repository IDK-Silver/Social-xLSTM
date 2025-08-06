# Analysis Scripts

This directory contains data analysis and inspection tools for the Social-xLSTM project.

## Purpose
Scripts in this directory are used for:
- Data quality assessment and analysis
- Temporal pattern investigation
- HDF5 dataset inspection and analysis
- Statistical analysis of traffic data

## Scripts

### `temporal_pattern_analysis.py`
**Purpose**: Deep analysis of temporal patterns in traffic data
- Investigates why temporal splitting doesn't improve distribution consistency
- Analyzes data completeness over time
- Detects temporal drift and systematic patterns
- Creates temporal analysis visualizations

**Usage**:
```bash
python scripts/analysis/temporal_pattern_analysis.py
```

### `h5_data_analysis.py`
**Purpose**: Analysis of HDF5 formatted traffic data
- Analyzes actual H5 data format and structure
- Performs data quality analysis for each VD (Vehicle Detector)
- Checks for missing data, outliers, and data leakage
- Creates data quality plots

**Usage**:
```bash
python scripts/analysis/h5_data_analysis.py
```

### `data_quality_analysis.py`
**Purpose**: Data cleaning and temporal quality analysis
- Implements temporal data quality validation
- Performs comprehensive data health checks
- Creates quality assessment reports
- Identifies data quality issues and suggests fixes

**Usage**:
```bash
python scripts/analysis/data_quality_analysis.py
```

## Output Locations
- Analysis results are typically saved to `blob/debug/` or `blob/analysis/`
- Plots and visualizations are saved as PNG files
- Quality reports are saved as JSON or text files

## Dependencies
- All scripts use the `social_xlstm` package
- Common dependencies: numpy, matplotlib, h5py, pandas
- Scripts automatically add `src/` to Python path for imports

## Related Documentation
- [ADR-0500: Scripts Directory Reorganization](../../docs/adr/0500-scripts-directory-reorganization.md)
- [Project Overview](../../docs/overview/project_overview.md)
- [Data Analysis Guide](../../docs/guides/data_analysis_guide.md)

## Integration with Main Codebase
These analysis tools complement the integrated analysis functions in:
- `src/social_xlstm/dataset/storage/h5_converter.py` - Dataset quality validation
- `src/social_xlstm/evaluation/evaluator.py` - Comprehensive diagnostics
- `src/social_xlstm/dataset/core/processor.py` - Data processing analysis