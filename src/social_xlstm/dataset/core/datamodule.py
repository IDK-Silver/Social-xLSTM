"""PyTorch Lightning data module for traffic data."""

import pytorch_lightning as pl
from torch.utils.data import DataLoader
from typing import Dict, Any, List, Optional
import logging

from ..config import TrafficDatasetConfig
from .timeseries import TrafficTimeSeries
from .collators import create_collate_fn
from social_xlstm.utils.convert_coords import mercator_projection

logger = logging.getLogger(__name__)


class TrafficDataModule(pl.LightningDataModule):
    """
    PyTorch Lightning data module for traffic data.
    
    Supports both centralized [B, T, N, F] and distributed {"VD_ID": [B, T, F]} 
    batch formats based on config.batch_format.
    """
    
    def __init__(self, config: TrafficDatasetConfig):
        super().__init__()
        self.config = config
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        self.shared_scaler = None
        
        # Distributed format support
        self._collate_fn = None
        self.vd_ids: Optional[List[str]] = None
        self.num_features: Optional[int] = None
    
    def setup(self, stage: str = None):
        """Setup datasets and prepare collate function."""
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
        
        # Prepare distributed collate function if needed
        if self.config.is_distributed and self._collate_fn is None:
            self._prepare_distributed_collate()
    
    def _prepare_distributed_collate(self):
        """Prepare distributed collate function with required metadata."""
        if not self.train_dataset:
            raise RuntimeError("Train dataset must be setup before preparing distributed collate")
        
        # Extract metadata from dataset
        data_info = self.get_data_info()
        self.vd_ids = data_info['vdids']
        self.num_features = data_info['num_features']
        
        if not self.vd_ids or len(self.vd_ids) == 0:
            raise ValueError("No VD IDs found in dataset for distributed format")
        
        # Build static XY positions per VD from metadata (if available)
        vd_positions_ll = {}
        for vd in self.vd_ids:
            info = self.train_dataset.reader.get_vd_info(vd)
            if info and 'position_lat' in info and 'position_lon' in info:
                try:
                    vd_positions_ll[vd] = (float(info['position_lat']), float(info['position_lon']))
                except Exception:
                    continue

        vd_positions_xy = None
        if vd_positions_ll:
            # Use mean lat/lon as projection origin (distance invariant to origin offset)
            lats = [v[0] for v in vd_positions_ll.values()]
            lons = [v[1] for v in vd_positions_ll.values()]
            lat_origin = sum(lats) / len(lats)
            lon_origin = sum(lons) / len(lons)

            vd_positions_xy = {}
            for vd, (lat, lon) in vd_positions_ll.items():
                try:
                    x, y = mercator_projection(lat, lon, lat_origin, lon_origin)
                    vd_positions_xy[vd] = (x, y)
                except Exception:
                    # Skip if conversion fails
                    pass

            logger.info(
                f"Computed XY positions for {len(vd_positions_xy)}/{len(self.vd_ids)} VDs using Mercator projection"
            )

        # Create distributed collate function (with optional positions)
        self._collate_fn = create_collate_fn(
            batch_format='distributed',
            vd_ids=self.vd_ids,
            num_features=self.num_features,
            sequence_length=self.config.sequence_length,
            prediction_length=self.config.prediction_length,
            vd_positions_xy=vd_positions_xy
        )
        
        logger.info(
            f"Prepared distributed collate function for {len(self.vd_ids)} VDs, "
            f"{self.num_features} features"
        )
    
    def _make_dataloader(self, dataset, batch_size: int, shuffle: bool, drop_last: bool) -> DataLoader:
        """
        Unified DataLoader creation with conditional collate function.
        
        Args:
            dataset: Dataset instance
            batch_size: Batch size
            shuffle: Whether to shuffle data
            drop_last: Whether to drop last incomplete batch
            
        Returns:
            Configured DataLoader
        """
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=self.config.num_workers,
            pin_memory=self.config.pin_memory,
            drop_last=drop_last,
            collate_fn=self._collate_fn,  # None for centralized, DistributedCollator for distributed
            persistent_workers=self.config.num_workers > 0  # Improve multi-worker performance
        )
    
    def train_dataloader(self) -> DataLoader:
        """Create training dataloader with appropriate batch format."""
        return self._make_dataloader(
            dataset=self.train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            drop_last=True
        )
    
    def val_dataloader(self) -> DataLoader:
        """Create validation dataloader with appropriate batch format."""
        return self._make_dataloader(
            dataset=self.val_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            drop_last=False
        )
    
    def test_dataloader(self) -> DataLoader:
        """Create test dataloader with appropriate batch format."""
        return self._make_dataloader(
            dataset=self.test_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            drop_last=False
        )
    
    def get_data_info(self) -> Dict[str, Any]:
        """Get dataset information with batch format details."""
        if self.train_dataset is None:
            self.setup('fit')
        
        base_info = {
            'num_vds': len(self.train_dataset.selected_vdids),
            'num_features': len(self.train_dataset.selected_features),
            'time_feat_dim': self.train_dataset.time_features.shape[1],
            'sequence_length': self.config.sequence_length,
            'prediction_length': self.config.prediction_length,
            'vdids': self.train_dataset.selected_vdids,
            'features': self.train_dataset.selected_features,
            'scaler': self.shared_scaler
        }
        
        # Add batch format information
        base_info.update({
            'batch_format': self.config.batch_format,
            'is_distributed': self.config.is_distributed
        })
        
        # Add distributed-specific information if applicable
        if self.config.is_distributed:
            base_info.update({
                'distributed_format': True,
                'vd_tensor_shape': (self.config.batch_size, self.config.sequence_length, base_info['num_features'])
            })
        
        return base_info
