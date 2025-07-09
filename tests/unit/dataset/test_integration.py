"""Integration tests for refactored dataset module."""

import pytest
import tempfile
import h5py
import numpy as np
from pathlib import Path
from unittest.mock import patch, MagicMock

from social_xlstm.dataset.config import TrafficDatasetConfig
from social_xlstm.dataset.core.timeseries import TrafficTimeSeries
from social_xlstm.dataset.core.datamodule import TrafficDataModule


class TestDatasetIntegration:
    """Integration tests for the refactored dataset module."""
    
    def create_test_hdf5(self, filepath):
        """Create a test HDF5 file with realistic data."""
        with h5py.File(filepath, 'w') as f:
            # Create metadata
            metadata = f.create_group('metadata')
            vdids = ['VD001', 'VD002', 'VD003']
            feature_names = ['avg_speed', 'total_volume', 'avg_occupancy', 'speed_std', 'lane_count']
            timestamps = [f'2023-01-01T{12 + i//60:02d}:{i%60:02d}:00' for i in range(200)]
            
            metadata.create_dataset('vdids', data=np.array(vdids, dtype=h5py.string_dtype()))
            metadata.create_dataset('feature_names', data=np.array(feature_names, dtype=h5py.string_dtype()))
            metadata.create_dataset('timestamps', data=np.array(timestamps, dtype=h5py.string_dtype()))
            
            # Create VD info
            vd_info = metadata.create_group('vd_info')
            for i, vdid in enumerate(vdids):
                vd_group = vd_info.create_group(vdid)
                vd_group.attrs['position_lon'] = 121.0 + i * 0.1
                vd_group.attrs['position_lat'] = 25.0 + i * 0.1
                vd_group.attrs['road_id'] = f'R00{i+1}'
                vd_group.attrs['road_name'] = f'Test Road {i+1}'
                vd_group.attrs['lane_num'] = 2 + i
            
            # Create realistic traffic data
            np.random.seed(42)  # For reproducibility
            data = f.create_group('data')
            
            # Generate synthetic traffic data
            features = np.zeros((200, 3, 5), dtype=np.float32)
            for t in range(200):
                for v in range(3):
                    # Simulate traffic patterns
                    base_speed = 60 + 10 * np.sin(t * 0.1) + np.random.normal(0, 5)
                    base_volume = 100 + 50 * np.sin(t * 0.15) + np.random.normal(0, 10)
                    
                    features[t, v, 0] = max(0, base_speed)  # avg_speed
                    features[t, v, 1] = max(0, base_volume)  # total_volume
                    features[t, v, 2] = np.random.uniform(0.1, 0.8)  # avg_occupancy
                    features[t, v, 3] = np.random.uniform(0, 10)  # speed_std
                    features[t, v, 4] = 2 + v  # lane_count
            
            data.create_dataset('features', data=features)
            
            # Add file attributes
            f.attrs['description'] = 'Test traffic data'
            f.attrs['num_timesteps'] = 200
            f.attrs['num_locations'] = 3
            f.attrs['num_features'] = 5
    
    def test_traffic_timeseries_basic_functionality(self):
        """Test basic TrafficTimeSeries functionality."""
        with tempfile.NamedTemporaryFile(suffix='.h5', delete=False) as tmp:
            tmp_path = Path(tmp.name)
            
        try:
            self.create_test_hdf5(tmp_path)
            
            config = TrafficDatasetConfig(
                hdf5_path=tmp_path,
                sequence_length=30,
                prediction_length=10,
                train_ratio=0.7,
                val_ratio=0.2,
                test_ratio=0.1
            )
            
            # Test training dataset
            train_dataset = TrafficTimeSeries(config, split='train')
            
            assert len(train_dataset) > 0
            assert len(train_dataset.selected_vdids) == 3
            assert len(train_dataset.selected_features) == 5
            
            # Test sample structure
            sample = train_dataset[0]
            assert 'input_seq' in sample
            assert 'target_seq' in sample
            assert 'input_time_feat' in sample
            assert 'target_time_feat' in sample
            assert 'input_mask' in sample
            assert 'target_mask' in sample
            
            # Check shapes
            assert sample['input_seq'].shape == (30, 3, 5)  # seq_len, num_vds, num_features
            assert sample['target_seq'].shape == (10, 3, 5)  # pred_len, num_vds, num_features
            assert sample['input_time_feat'].shape == (30, 9)  # seq_len, time_feat_dim
            assert sample['target_time_feat'].shape == (10, 9)  # pred_len, time_feat_dim
            assert sample['input_mask'].shape == (30, 3)  # seq_len, num_vds
            assert sample['target_mask'].shape == (10, 3)  # pred_len, num_vds
            
        finally:
            tmp_path.unlink()
    
    def test_traffic_timeseries_normalization(self):
        """Test normalization functionality."""
        with tempfile.NamedTemporaryFile(suffix='.h5', delete=False) as tmp:
            tmp_path = Path(tmp.name)
            
        try:
            self.create_test_hdf5(tmp_path)
            
            config = TrafficDatasetConfig(
                hdf5_path=tmp_path,
                sequence_length=30,
                prediction_length=10,
                normalize=True,
                normalization_method='standard'
            )
            
            train_dataset = TrafficTimeSeries(config, split='train')
            
            # Check that scaler was created
            assert train_dataset.scaler is not None
            
            # Check that data is normalized (approximately)
            sample = train_dataset[0]
            input_data = sample['input_seq'].numpy()
            
            # For normalized data, values should be roughly in [-10, 10] range
            # (allowing for outliers in synthetic data)
            assert np.all(input_data >= -10)
            assert np.all(input_data <= 10)
            
        finally:
            tmp_path.unlink()
    
    def test_traffic_datamodule_functionality(self):
        """Test TrafficDataModule functionality."""
        with tempfile.NamedTemporaryFile(suffix='.h5', delete=False) as tmp:
            tmp_path = Path(tmp.name)
            
        try:
            self.create_test_hdf5(tmp_path)
            
            config = TrafficDatasetConfig(
                hdf5_path=tmp_path,
                sequence_length=30,
                prediction_length=10,
                batch_size=16
            )
            
            datamodule = TrafficDataModule(config)
            
            # Test setup
            datamodule.setup('fit')
            
            assert datamodule.train_dataset is not None
            assert datamodule.val_dataset is not None
            assert datamodule.shared_scaler is not None
            
            # Test that scaler is shared
            assert datamodule.val_dataset.scaler is datamodule.shared_scaler
            
            # Test dataloaders
            train_loader = datamodule.train_dataloader()
            val_loader = datamodule.val_dataloader()
            
            assert train_loader is not None
            assert val_loader is not None
            
            # Test batch
            train_batch = next(iter(train_loader))
            assert train_batch['input_seq'].shape[0] == 16  # batch_size
            assert train_batch['input_seq'].shape[1] == 30  # sequence_length
            assert train_batch['input_seq'].shape[2] == 3   # num_vds
            assert train_batch['input_seq'].shape[3] == 5   # num_features
            
            # Test data info
            data_info = datamodule.get_data_info()
            assert data_info['num_vds'] == 3
            assert data_info['num_features'] == 5
            assert data_info['sequence_length'] == 30
            assert data_info['prediction_length'] == 10
            
        finally:
            tmp_path.unlink()
    
    def test_scaler_sharing_between_splits(self):
        """Test that scaler is properly shared between train/val/test splits."""
        with tempfile.NamedTemporaryFile(suffix='.h5', delete=False) as tmp:
            tmp_path = Path(tmp.name)
            
        try:
            self.create_test_hdf5(tmp_path)
            
            config = TrafficDatasetConfig(
                hdf5_path=tmp_path,
                sequence_length=10,  # Smaller sequence length for testing
                prediction_length=5,   # Smaller prediction length
                normalize=True
            )
            
            # Create train dataset first
            train_dataset = TrafficTimeSeries(config, split='train')
            train_scaler = train_dataset.get_scaler()
            
            # Create val dataset with shared scaler
            val_dataset = TrafficTimeSeries(config, split='val', scaler=train_scaler)
            
            # Create test dataset with shared scaler
            test_dataset = TrafficTimeSeries(config, split='test', scaler=train_scaler)
            
            # Check that scalers are the same object
            assert val_dataset.scaler is train_scaler
            assert test_dataset.scaler is train_scaler
            
            # Check that normalization is consistent
            train_sample = train_dataset[0]
            val_sample = val_dataset[0]
            test_sample = test_dataset[0]
            
            # All samples should be in similar normalized range
            for sample in [train_sample, val_sample, test_sample]:
                data = sample['input_seq'].numpy()
                assert np.all(data >= -10)
                assert np.all(data <= 10)
                
        finally:
            tmp_path.unlink()
    
    def test_backward_compatibility(self):
        """Test that refactored code maintains backward compatibility."""
        with tempfile.NamedTemporaryFile(suffix='.h5', delete=False) as tmp:
            tmp_path = Path(tmp.name)
            
        try:
            self.create_test_hdf5(tmp_path)
            
            # Test that all imports work
            from social_xlstm.dataset import (
                TrafficDatasetConfig, TrafficTimeSeries, TrafficDataModule,
                TrafficHDF5Config, TrafficHDF5Reader, TrafficDataProcessor
            )
            
            # Test that basic usage still works
            config = TrafficDatasetConfig(hdf5_path=tmp_path)
            dataset = TrafficTimeSeries(config, split='train')
            datamodule = TrafficDataModule(config)
            
            assert len(dataset) > 0
            assert datamodule is not None
            
        finally:
            tmp_path.unlink()
    
    def test_error_handling(self):
        """Test error handling in refactored components."""
        with tempfile.NamedTemporaryFile(suffix='.h5', delete=False) as tmp:
            tmp_path = Path(tmp.name)
            
        try:
            self.create_test_hdf5(tmp_path)
            
            # Test invalid split
            config = TrafficDatasetConfig(hdf5_path=tmp_path)
            with pytest.raises(ValueError, match="Unknown split"):
                TrafficTimeSeries(config, split='invalid')
            
            # Test invalid VDIDs
            config = TrafficDatasetConfig(
                hdf5_path=tmp_path,
                selected_vdids=['VD999']  # Non-existent VDID
            )
            with pytest.raises(ValueError, match="No valid VDIDs found"):
                TrafficTimeSeries(config, split='train')
            
            # Test invalid features
            config = TrafficDatasetConfig(
                hdf5_path=tmp_path,
                selected_features=['invalid_feature']
            )
            with pytest.raises(ValueError, match="No valid features found"):
                TrafficTimeSeries(config, split='train')
                
        finally:
            tmp_path.unlink()