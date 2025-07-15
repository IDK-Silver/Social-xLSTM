# Tools Directory

This directory contains supporting scripts for development, analysis, diagnostics, and data inspection. These tools are not part of the core `social_xlstm` library or the main training/evaluation pipelines. They are meant to be run manually by developers as needed.

## Directory Structure

- **`config/`**: Configuration generation and management tools
  - Scripts for creating optimized configurations
  - Tools for managing experimental settings
  
- **`analysis/`**: Data analysis and exploration tools
  - Scripts for analyzing training data quality
  - Tools for exploring temporal patterns
  - Performance analysis utilities
  
- **`diagnostics/`**: Diagnostic and inspection tools
  - Data structure inspectors (e.g., HDF5 file analysis)
  - System health check utilities
  - Debug and troubleshooting tools
  
- **`validation/`**: Validation and testing utilities
  - Data validation scripts
  - Model validation tools
  - Integration testing utilities

## Usage Guidelines

### Running Tools
All tools should be run from the project root directory:
```bash
# Example usage
python tools/config/config_generator.py --help
python tools/diagnostics/h5_structure_inspector.py --input data.h5
```

### Adding New Tools
When adding new tools:
1. Choose the appropriate subdirectory based on the tool's purpose
2. Follow the existing naming conventions
3. Add appropriate documentation and help text
4. Test the tool thoroughly before committing

### Tool Categories

**Configuration Tools (`config/`)**: Tools that generate, modify, or validate configuration files for experiments.

**Analysis Tools (`analysis/`)**: Tools that analyze data, results, or system behavior. These are typically used for research and exploration.

**Diagnostic Tools (`diagnostics/`)**: Tools that inspect, debug, or troubleshoot system components. Used for maintenance and problem-solving.

**Validation Tools (`validation/`)**: Tools that validate data integrity, model correctness, or system behavior. Used for quality assurance.

## Design Principles

1. **Self-contained**: Each tool should be runnable independently
2. **Clear purpose**: Each tool should have a specific, well-defined function
3. **Good documentation**: Include help text and usage examples
4. **Error handling**: Provide meaningful error messages
5. **Logging**: Use appropriate logging for debugging and monitoring

## Maintenance

- Review tools periodically for relevance and functionality
- Remove or archive tools that are no longer needed
- Update documentation when tools are modified
- Ensure tools remain compatible with the current codebase

---

Generated as part of the Social-xLSTM project cleanup initiative.