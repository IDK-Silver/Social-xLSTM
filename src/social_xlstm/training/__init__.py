"""
Social-xLSTM Training Module

This module provides comprehensive training implementations for Social-xLSTM models.
The training system is organized into two main categories:

1. without_social_pooling: Baseline trainers for independent VD training
   - Single VD training
   - Multi VD training (independent)
   - Used for establishing baseline performance

2. with_social_pooling: Advanced trainers with spatial interactions (TODO)
   - Social Single VD training
   - Social Multi VD training
   - Social-xLSTM training
   - Used for the main Social-xLSTM functionality

Author: Social-xLSTM Project Team
License: MIT
"""

from .base import BaseTrainer, TrainingConfig

# Import baseline trainers (without Social Pooling)
from .without_social_pooling import (
    SingleVDTrainer, 
    SingleVDTrainingConfig, 
    create_single_vd_trainer,
    MultiVDTrainer, 
    MultiVDTrainingConfig, 
    IndependentMultiVDTrainer,
    create_multi_vd_trainer,
    create_independent_multi_vd_trainer
)

# TODO: Import Social Pooling trainers when implemented
# from .with_social_pooling import (
#     SocialSingleVDTrainer,
#     SocialSingleVDTrainingConfig,
#     SocialMultiVDTrainer,
#     SocialMultiVDTrainingConfig,
#     SocialXLSTMTrainer,
#     SocialXLSTMTrainingConfig
# )

__all__ = [
    # Base classes
    'BaseTrainer',
    'TrainingConfig',
    
    # Without Social Pooling (baseline trainers)
    'SingleVDTrainer',
    'SingleVDTrainingConfig',
    'create_single_vd_trainer',
    'MultiVDTrainer',
    'MultiVDTrainingConfig',
    'IndependentMultiVDTrainer',
    'create_multi_vd_trainer',
    'create_independent_multi_vd_trainer',
    
    # TODO: With Social Pooling (advanced trainers) - to be implemented
    # 'SocialSingleVDTrainer',
    # 'SocialSingleVDTrainingConfig',
    # 'SocialMultiVDTrainer',
    # 'SocialMultiVDTrainingConfig',
    # 'SocialXLSTMTrainer',
    # 'SocialXLSTMTrainingConfig',
]

# Legacy trainer has been removed - specialized trainers are now the standard

# Version info
__version__ = "2.1.0"
__trainer_version__ = "modular"

print(f"Social-xLSTM Training Module v{__version__} loaded")
print(f"Architecture: {__trainer_version__} trainers")
print("Available trainers:")
print("  Without Social Pooling: SingleVDTrainer, MultiVDTrainer, IndependentMultiVDTrainer")
print("  With Social Pooling: TODO - to be implemented")