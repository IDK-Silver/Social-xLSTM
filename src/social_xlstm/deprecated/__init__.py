"""
DEPRECATED MODULES

This directory contains deprecated modules that are kept for backward compatibility.
These modules are no longer actively maintained and will be removed in future versions.

RECOMMENDED ALTERNATIVES:
- For metrics recording: Use `social_xlstm.metrics.writer.TrainingMetricsWriter`
- For visualization: Use `social_xlstm.metrics.plotter.TrainingMetricsPlotter`

DEPRECATION STATUS:
- evaluation/ - Deprecated since v1.0, use metrics.writer instead
- visualization/ - Deprecated since v1.0, use metrics.plotter instead

Author: Social-xLSTM Project Team
License: MIT
"""

import warnings

warnings.warn(
    "The 'deprecated' module contains outdated components. "
    "Please use 'social_xlstm.metrics' for new implementations.",
    DeprecationWarning,
    stacklevel=2
)