"""PyTorch Lightning data module for traffic data."""

import pytorch_lightning as pl
from torch.utils.data import DataLoader
from typing import Dict, Any

from ..config import TrafficDatasetConfig
from .timeseries import TrafficTimeSeries


class TrafficDataModule(pl.LightningDataModule):
    """PyTorch Lightning data module for traffic data."""
    
    def __init__(self, config: TrafficDatasetConfig):
        super().__init__()
        self.config = config
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        self.shared_scaler = None
    
    def setup(self, stage: str = None):
        """Setup datasets."""
        if stage == 'fit' or stage is None:
            self.train_dataset = TrafficTimeSeries(self.config, split='train')
            self.shared_scaler = self.train_dataset.get_scaler()
            
            # Create validation dataset with shared scaler
            self.val_dataset = TrafficTimeSeries(
                self.config, 
                split='val', 
                scaler=self.shared_scaler
            )
        
        if stage == 'test' or stage is None:
            # Create test dataset with shared scaler
            self.test_dataset = TrafficTimeSeries(
                self.config, 
                split='test', 
                scaler=self.shared_scaler
            )
    
    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=self.config.num_workers,
            pin_memory=self.config.pin_memory,
            drop_last=True
        )
    
    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
            pin_memory=self.config.pin_memory,
            drop_last=False
        )
    
    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
            pin_memory=self.config.pin_memory,
            drop_last=False
        )
    
    def get_data_info(self) -> Dict[str, Any]:
        """Get dataset information."""
        if self.train_dataset is None:
            self.setup('fit')
        
        return {
            'num_vds': len(self.train_dataset.selected_vdids),
            'num_features': len(self.train_dataset.selected_features),
            'time_feat_dim': self.train_dataset.time_features.shape[1],
            'sequence_length': self.config.sequence_length,
            'prediction_length': self.config.prediction_length,
            'vdids': self.train_dataset.selected_vdids,
            'features': self.train_dataset.selected_features,
            'scaler': self.shared_scaler
        }