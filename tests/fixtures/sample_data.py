"""
Sample data generators for testing.
"""

import numpy as np
import torch
import h5py
from pathlib import Path
import json
from datetime import datetime, timedelta


def generate_sample_traffic_data(num_samples=1000, num_features=3, seq_length=12):
    """Generate sample traffic time series data."""
    np.random.seed(42)
    
    # Generate realistic traffic patterns
    time_steps = np.arange(num_samples)
    
    # Base patterns with daily and weekly cycles
    daily_cycle = np.sin(2 * np.pi * time_steps / 24)  # Daily pattern
    weekly_cycle = np.sin(2 * np.pi * time_steps / (24 * 7))  # Weekly pattern
    
    # Generate features: volume, speed, occupancy
    volume = 50 + 30 * daily_cycle + 10 * weekly_cycle + np.random.normal(0, 5, num_samples)
    speed = 60 - 20 * daily_cycle - 5 * weekly_cycle + np.random.normal(0, 3, num_samples)
    occupancy = 0.3 + 0.2 * daily_cycle + 0.1 * weekly_cycle + np.random.normal(0, 0.05, num_samples)
    
    # Ensure realistic bounds
    volume = np.clip(volume, 0, 200)
    speed = np.clip(speed, 10, 100)
    occupancy = np.clip(occupancy, 0, 1)
    
    if num_features == 3:
        data = np.column_stack([volume, speed, occupancy])
    else:
        # Add more features if needed
        data = np.column_stack([volume, speed, occupancy])
        for i in range(num_features - 3):
            extra_feature = np.random.normal(0, 1, num_samples)
            data = np.column_stack([data, extra_feature])
    
    return data


def generate_sample_hdf5(file_path, num_vds=5, num_samples=1000):
    """Generate sample HDF5 file with traffic data."""
    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    
    with h5py.File(file_path, 'w') as f:
        for i in range(num_vds):
            vd_id = f"VD{i+1:03d}"
            vd_group = f.create_group(vd_id)
            
            # Generate traffic data
            traffic_data = generate_sample_traffic_data(num_samples)
            
            # Create datasets
            vd_group.create_dataset('volume', data=traffic_data[:, 0])
            vd_group.create_dataset('speed', data=traffic_data[:, 1])
            vd_group.create_dataset('occupancy', data=traffic_data[:, 2])
            
            # Add metadata
            vd_group.attrs['location'] = f'Test Location {i+1}'
            vd_group.attrs['coordinates'] = [120.0 + i * 0.01, 24.0 + i * 0.01]
            vd_group.attrs['road_type'] = 'highway'
    
    return file_path


def generate_sample_json_data(num_vds=5, num_samples=100):
    """Generate sample JSON data structure."""
    vd_data = {}
    
    for i in range(num_vds):
        vd_id = f"VD{i+1:03d}"
        
        # Generate time series data
        base_time = datetime.now() - timedelta(hours=num_samples)
        time_series = []
        
        for j in range(num_samples):
            timestamp = base_time + timedelta(hours=j)
            
            # Generate realistic traffic values
            hour = timestamp.hour
            volume = max(0, 50 + 30 * np.sin(2 * np.pi * hour / 24) + np.random.normal(0, 5))
            speed = max(10, 60 - 20 * np.sin(2 * np.pi * hour / 24) + np.random.normal(0, 3))
            occupancy = max(0, min(1, 0.3 + 0.2 * np.sin(2 * np.pi * hour / 24) + np.random.normal(0, 0.05)))
            
            time_series.append({
                'timestamp': timestamp.isoformat(),
                'volume': float(volume),
                'speed': float(speed),
                'occupancy': float(occupancy)
            })
        
        vd_data[vd_id] = {
            'metadata': {
                'location': f'Test Location {i+1}',
                'coordinates': [120.0 + i * 0.01, 24.0 + i * 0.01],
                'road_type': 'highway'
            },
            'data': time_series
        }
    
    return vd_data


def create_sample_torch_dataset(num_samples=100, seq_length=12, num_features=3):
    """Create sample PyTorch dataset."""
    # Generate continuous time series
    full_data = generate_sample_traffic_data(num_samples + seq_length, num_features)
    
    # Create sequences
    inputs = []
    targets = []
    
    for i in range(num_samples):
        # Input sequence
        input_seq = full_data[i:i+seq_length]
        # Target is next timestep
        target = full_data[i+seq_length]
        
        inputs.append(input_seq)
        targets.append(target)
    
    # Convert to tensors
    inputs = torch.FloatTensor(np.array(inputs))
    targets = torch.FloatTensor(np.array(targets))
    
    return inputs, targets


def create_sample_multi_vd_dataset(num_samples=100, seq_length=12, num_vds=5, num_features=3):
    """Create sample multi-VD dataset."""
    # Generate data for each VD
    all_vd_data = []
    
    for vd in range(num_vds):
        vd_data = generate_sample_traffic_data(num_samples + seq_length, num_features)
        all_vd_data.append(vd_data)
    
    # Stack VD data
    all_vd_data = np.array(all_vd_data)  # Shape: (num_vds, timesteps, features)
    
    # Create sequences
    inputs = []
    targets = []
    
    for i in range(num_samples):
        # Input sequence: (seq_length, num_vds, num_features)
        input_seq = all_vd_data[:, i:i+seq_length, :].transpose(1, 0, 2)
        # Target: (num_vds, num_features)
        target = all_vd_data[:, i+seq_length, :]
        
        inputs.append(input_seq)
        targets.append(target)
    
    # Convert to tensors
    inputs = torch.FloatTensor(np.array(inputs))
    targets = torch.FloatTensor(np.array(targets))
    
    return inputs, targets


class SampleDataGenerator:
    """Utility class for generating various types of sample data."""
    
    @staticmethod
    def traffic_time_series(num_samples=1000, num_features=3, add_noise=True):
        """Generate realistic traffic time series."""
        return generate_sample_traffic_data(num_samples, num_features)
    
    @staticmethod
    def hdf5_file(file_path, num_vds=5, num_samples=1000):
        """Generate sample HDF5 file."""
        return generate_sample_hdf5(file_path, num_vds, num_samples)
    
    @staticmethod
    def json_data(num_vds=5, num_samples=100):
        """Generate sample JSON data."""
        return generate_sample_json_data(num_vds, num_samples)
    
    @staticmethod
    def torch_dataset(num_samples=100, seq_length=12, num_features=3):
        """Generate sample PyTorch dataset."""
        return create_sample_torch_dataset(num_samples, seq_length, num_features)
    
    @staticmethod
    def multi_vd_dataset(num_samples=100, seq_length=12, num_vds=5, num_features=3):
        """Generate sample multi-VD dataset."""
        return create_sample_multi_vd_dataset(num_samples, seq_length, num_vds, num_features)