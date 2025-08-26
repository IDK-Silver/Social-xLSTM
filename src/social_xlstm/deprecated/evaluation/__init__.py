"""
DEPRECATED: evaluation module

This module is deprecated and will be removed in future versions.
Please use the new metrics system instead:

RECOMMENDED ALTERNATIVE:
- Use `social_xlstm.metrics.writer.TrainingMetricsWriter` for metrics recording
- Use `social_xlstm.metrics.plotter.TrainingMetricsPlotter` for visualization

Author: Social-xLSTM Project Team  
License: MIT
"""

import warnings

warnings.warn(
    "The 'evaluation' module is deprecated. "
    "Use 'social_xlstm.metrics' for new implementations.",
    DeprecationWarning,
    stacklevel=2
)

from .evaluator import ModelEvaluator, DatasetDiagnostics

__all__ = ["ModelEvaluator", "DatasetDiagnostics"]