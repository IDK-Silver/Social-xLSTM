"""
DEPRECATED: visualization module

This module is deprecated and will be removed in future versions.
Please use the new metrics system instead:

RECOMMENDED ALTERNATIVE:
- Use `social_xlstm.metrics.plotter.TrainingMetricsPlotter` for basic visualization
- Use CLI tool: `scripts/utils/generate_metrics_plots.py`

Author: Social-xLSTM Project Team
License: MIT
"""

import warnings

warnings.warn(
    "The 'visualization' module is deprecated. "
    "Use 'social_xlstm.metrics.plotter' for new implementations.",
    DeprecationWarning,
    stacklevel=2
)

from .training_visualizer import TrainingVisualizer

__all__ = ["TrainingVisualizer"]