"""
Training implementations with Social Pooling.

This module will contain specialized trainers for models that use Social Pooling:
- Social Single VD training: Single VD with spatial context from nearby VDs
- Social Multi VD training: Multi VD with spatial pooling mechanisms
- Social xLSTM training: Combined Social Pooling + xLSTM architecture

These trainers will implement the core Social-xLSTM functionality with
spatial-temporal modeling of vehicle interactions.

Author: Social-xLSTM Project Team
License: MIT

TODO: Implement Social Pooling trainers based on ADR-0100 decisions
"""

# TODO: Implement these trainers
# from .social_single_vd_trainer import SocialSingleVDTrainer, SocialSingleVDTrainingConfig
# from .social_multi_vd_trainer import SocialMultiVDTrainer, SocialMultiVDTrainingConfig
# from .social_xlstm_trainer import SocialXLSTMTrainer, SocialXLSTMTrainingConfig

__all__ = [
    # TODO: Add Social Pooling trainers here
    # 'SocialSingleVDTrainer',
    # 'SocialSingleVDTrainingConfig',
    # 'SocialMultiVDTrainer', 
    # 'SocialMultiVDTrainingConfig',
    # 'SocialXLSTMTrainer',
    # 'SocialXLSTMTrainingConfig',
]

# Placeholder for future implementation
def _placeholder():
    """Placeholder function indicating future Social Pooling implementation."""
    raise NotImplementedError(
        "Social Pooling trainers not yet implemented. "
        "See ADR-0100 and ADR-0101 for implementation roadmap."
    )