"""
Lightweight training metrics recording and plotting utilities.

Designed specifically for basic MAE, MSE, RMSE, R² metrics with data persistence
and visualization capabilities. Follows YAGNI principles to avoid over-engineering.
"""

from .writer import TrainingMetricsWriter
from .plotter import TrainingMetricsPlotter

__all__ = [
    "TrainingMetricsWriter",
    "TrainingMetricsPlotter"
]