"""Tests for HDF5 operations."""

import pytest
import h5py
import numpy as np
import tempfile
import json
from pathlib import Path
from unittest.mock import MagicMock, patch

from social_xlstm.dataset.config import TrafficHDF5Config
from social_xlstm.dataset.storage.h5_converter import TrafficHDF5Converter, TrafficFeatureExtractor
from social_xlstm.dataset.storage.h5_reader import TrafficHDF5Reader, create_traffic_hdf5


class TestTrafficFeatureExtractor:
    """Test TrafficFeatureExtractor class."""
    
    def test_extract_lane_features(self):
        """Test lane feature extraction."""
        # Mock lane object
        lane = MagicMock()
        lane.Speed = 60.0
        lane.Occupancy = 0.15
        lane.Vehicles = [
            MagicMock(Volume=10),
            MagicMock(Volume=15),
            MagicMock(Volume=20)
        ]
        
        features = TrafficFeatureExtractor.extract_lane_features(lane)
        
        assert features['speed'] == 60.0
        assert features['occupancy'] == 0.15
        assert features['volume'] == 45.0  # 10 + 15 + 20
    
    def test_extract_lane_features_with_nan(self):
        """Test lane feature extraction with NaN values."""
        lane = MagicMock()
        lane.Speed = np.nan
        lane.Occupancy = np.nan
        lane.Vehicles = None
        
        features = TrafficFeatureExtractor.extract_lane_features(lane)
        
        assert np.isnan(features['speed'])
        assert np.isnan(features['occupancy'])
        assert features['volume'] == 0.0
    
    def test_aggregate_vd_features(self):
        """Test VD feature aggregation."""
        # Mock VD detail object
        vd_detail = MagicMock()
        
        # Create mock lanes
        lane1 = MagicMock()
        lane1.Speed = 60.0
        lane1.Occupancy = 0.10
        lane1.Vehicles = [MagicMock(Volume=10)]
        
        lane2 = MagicMock()
        lane2.Speed = 70.0
        lane2.Occupancy = 0.20
        lane2.Vehicles = [MagicMock(Volume=20)]
        
        vd_detail.Lanes = [lane1, lane2]
        
        feature_names = ['avg_speed', 'total_volume', 'avg_occupancy', 'speed_std', 'lane_count']
        
        features = TrafficFeatureExtractor.aggregate_vd_features(vd_detail, feature_names)
        
        assert len(features) == 5
        assert features[0] == 65.0  # avg_speed: (60 + 70) / 2
        assert features[1] == 30.0  # total_volume: 10 + 20
        assert abs(features[2] - 0.15) < 1e-10  # avg_occupancy: (0.10 + 0.20) / 2
        assert features[3] == 5.0   # speed_std: std([60, 70])
        assert features[4] == 2.0   # lane_count: 2 lanes
    
    def test_aggregate_vd_features_empty_lanes(self):
        """Test VD feature aggregation with empty lanes."""
        vd_detail = MagicMock()
        vd_detail.Lanes = []
        
        feature_names = ['avg_speed', 'total_volume', 'avg_occupancy', 'speed_std', 'lane_count']
        
        features = TrafficFeatureExtractor.aggregate_vd_features(vd_detail, feature_names)
        
        assert len(features) == 5
        assert all(np.isnan(f) for f in features)


class TestTrafficHDF5Reader:
    """Test TrafficHDF5Reader class."""
    
    def create_test_hdf5(self, filepath):
        """Create a test HDF5 file."""
        with h5py.File(filepath, 'w') as f:
            # Create metadata
            metadata = f.create_group('metadata')
            metadata.create_dataset('vdids', data=np.array(['VD001', 'VD002'], dtype=h5py.string_dtype()))
            metadata.create_dataset('feature_names', data=np.array(['speed', 'volume'], dtype=h5py.string_dtype()))
            metadata.create_dataset('timestamps', data=np.array(['2023-01-01T12:00:00', '2023-01-01T12:01:00'], dtype=h5py.string_dtype()))
            
            # Create VD info
            vd_info = metadata.create_group('vd_info')
            vd001 = vd_info.create_group('VD001')
            vd001.attrs['position_lon'] = 121.5
            vd001.attrs['position_lat'] = 25.0
            vd001.attrs['road_id'] = 'R001'
            vd001.attrs['road_name'] = 'Test Road'
            vd001.attrs['lane_num'] = 3
            
            # Create data
            data = f.create_group('data')
            features = np.array([
                [[60.0, 100], [70.0, 150]],  # Time 0
                [[65.0, 110], [75.0, 160]]   # Time 1
            ], dtype=np.float32)
            data.create_dataset('features', data=features)
            
            # Add file attributes
            f.attrs['description'] = 'Test traffic data'
            f.attrs['num_timesteps'] = 2
            f.attrs['num_locations'] = 2
            f.attrs['num_features'] = 2
    
    def test_init_with_valid_file(self):
        """Test initialization with valid HDF5 file."""
        with tempfile.NamedTemporaryFile(suffix='.h5', delete=False) as tmp:
            tmp_path = Path(tmp.name)
            
        try:
            self.create_test_hdf5(tmp_path)
            reader = TrafficHDF5Reader(tmp_path)
            assert reader.hdf5_path == tmp_path
        finally:
            tmp_path.unlink()
    
    def test_init_with_invalid_file(self):
        """Test initialization with invalid HDF5 file."""
        invalid_path = Path("/nonexistent/file.h5")
        
        with pytest.raises(FileNotFoundError, match="HDF5 file not found"):
            TrafficHDF5Reader(invalid_path)
    
    def test_get_metadata(self):
        """Test metadata retrieval."""
        with tempfile.NamedTemporaryFile(suffix='.h5', delete=False) as tmp:
            tmp_path = Path(tmp.name)
            
        try:
            self.create_test_hdf5(tmp_path)
            reader = TrafficHDF5Reader(tmp_path)
            
            metadata = reader.get_metadata()
            
            assert metadata['vdids'] == ['VD001', 'VD002']
            assert metadata['feature_names'] == ['speed', 'volume']
            assert metadata['data_shape'] == (2, 2, 2)
            assert metadata['description'] == 'Test traffic data'
            assert metadata['num_timesteps'] == 2
        finally:
            tmp_path.unlink()
    
    def test_get_timestamps(self):
        """Test timestamp retrieval."""
        with tempfile.NamedTemporaryFile(suffix='.h5', delete=False) as tmp:
            tmp_path = Path(tmp.name)
            
        try:
            self.create_test_hdf5(tmp_path)
            reader = TrafficHDF5Reader(tmp_path)
            
            timestamps = reader.get_timestamps()
            
            assert timestamps == ['2023-01-01T12:00:00', '2023-01-01T12:01:00']
        finally:
            tmp_path.unlink()
    
    def test_get_vd_info(self):
        """Test VD info retrieval."""
        with tempfile.NamedTemporaryFile(suffix='.h5', delete=False) as tmp:
            tmp_path = Path(tmp.name)
            
        try:
            self.create_test_hdf5(tmp_path)
            reader = TrafficHDF5Reader(tmp_path)
            
            vd_info = reader.get_vd_info('VD001')
            
            assert vd_info is not None
            assert vd_info['position_lon'] == 121.5
            assert vd_info['position_lat'] == 25.0
            assert vd_info['road_id'] == 'R001'
            assert vd_info['road_name'] == 'Test Road'
            assert vd_info['lane_num'] == 3
            
            # Test non-existent VD
            assert reader.get_vd_info('VD999') is None
        finally:
            tmp_path.unlink()
    
    def test_get_features(self):
        """Test feature data retrieval."""
        with tempfile.NamedTemporaryFile(suffix='.h5', delete=False) as tmp:
            tmp_path = Path(tmp.name)
            
        try:
            self.create_test_hdf5(tmp_path)
            reader = TrafficHDF5Reader(tmp_path)
            
            # Test full data retrieval
            features = reader.get_features()
            assert features.shape == (2, 2, 2)
            
            # Test time slicing
            features = reader.get_features(time_slice=slice(0, 1))
            assert features.shape == (1, 2, 2)
            
            # Test VD indexing
            features = reader.get_features(vd_indices=[0])
            assert features.shape == (2, 1, 2)
            
            # Test feature indexing
            features = reader.get_features(feature_indices=[0])
            assert features.shape == (2, 2, 1)
            
            # Test combined indexing
            features = reader.get_features(
                time_slice=slice(0, 1),
                vd_indices=[0],
                feature_indices=[0]
            )
            assert features.shape == (1, 1, 1)
            assert features[0, 0, 0] == 60.0
        finally:
            tmp_path.unlink()
    
    def test_get_vd_timeseries(self):
        """Test VD time series retrieval."""
        with tempfile.NamedTemporaryFile(suffix='.h5', delete=False) as tmp:
            tmp_path = Path(tmp.name)
            
        try:
            self.create_test_hdf5(tmp_path)
            reader = TrafficHDF5Reader(tmp_path)
            
            # Test valid VDID
            timeseries = reader.get_vd_timeseries('VD001')
            assert timeseries.shape == (2, 2)  # 2 timesteps, 2 features
            assert timeseries[0, 0] == 60.0  # First timestep, speed
            assert timeseries[0, 1] == 100.0  # First timestep, volume
            
            # Test invalid VDID
            with pytest.raises(ValueError, match="VDID VD999 not found"):
                reader.get_vd_timeseries('VD999')
        finally:
            tmp_path.unlink()


class TestCreateTrafficHDF5:
    """Test create_traffic_hdf5 function."""
    
    def test_create_traffic_hdf5_basic(self):
        """Test basic HDF5 creation."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            source_dir = Path(tmp_dir) / "source"
            source_dir.mkdir()
            output_path = Path(tmp_dir) / "output.h5"
            
            # Create mock data structure
            time_dir = source_dir / "2023-01-01_12-00-00"
            time_dir.mkdir()
            
            # Create VDList.json
            vd_list = {
                "VDList": [
                    {
                        "VDID": "VD001",
                        "PositionLon": 121.5,
                        "PositionLat": 25.0,
                        "RoadID": "R001",
                        "RoadName": "Test Road",
                        "LaneNum": 2
                    }
                ]
            }
            with open(time_dir / "VDList.json", 'w') as f:
                json.dump(vd_list, f)
            
            # Create VDLiveList.json
            vd_live_list = {
                "LiveTrafficData": [
                    {
                        "VDID": "VD001",
                        "DataCollectTime": "2023-01-01T12:00:00",
                        "Lanes": [
                            {
                                "Speed": 60.0,
                                "Occupancy": 0.15,
                                "Vehicles": [{"Volume": 10}]
                            }
                        ]
                    }
                ]
            }
            with open(time_dir / "VDLiveList.json", 'w') as f:
                json.dump(vd_live_list, f)
            
            # Mock the converter to avoid actual conversion
            with patch('social_xlstm.dataset.storage.h5_converter.TrafficHDF5Converter') as mock_converter:
                mock_instance = MagicMock()
                mock_converter.return_value = mock_instance
                
                config = create_traffic_hdf5(
                    source_dir=source_dir,
                    output_path=output_path,
                    selected_vdids=['VD001']
                )
                
                assert isinstance(config, TrafficHDF5Config)
                assert config.source_dir == source_dir
                assert config.output_path == output_path
                assert config.selected_vdids == ['VD001']
                
                # Verify converter was called
                mock_converter.assert_called_once()
                mock_instance.convert.assert_called_once()