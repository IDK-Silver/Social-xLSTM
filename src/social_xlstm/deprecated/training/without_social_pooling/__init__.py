"""
Training implementations without Social Pooling.

This module contains specialized trainers for models that do not use Social Pooling:
- Single VD training: Independent single vehicle detector training
- Multi VD training: Independent multiple vehicle detector training
- Independent Multi VD training: Baseline training without spatial interactions

These trainers are used for establishing baseline performance and for scenarios
where spatial relationships between vehicle detectors are not needed.

Author: Social-xLSTM Project Team
License: MIT
"""

from .single_vd_trainer import SingleVDTrainer, SingleVDTrainingConfig, create_single_vd_trainer
from .multi_vd_trainer import (
    MultiVDTrainer, 
    MultiVDTrainingConfig, 
    IndependentMultiVDTrainer,
    create_multi_vd_trainer,
    create_independent_multi_vd_trainer
)

__all__ = [
    # Single VD training
    'SingleVDTrainer',
    'SingleVDTrainingConfig',
    'create_single_vd_trainer',
    
    # Multi VD training
    'MultiVDTrainer',
    'MultiVDTrainingConfig',
    'IndependentMultiVDTrainer',
    'create_multi_vd_trainer',
    'create_independent_multi_vd_trainer',
]