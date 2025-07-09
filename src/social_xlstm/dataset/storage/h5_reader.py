"""HDF5 reader for traffic data."""

import h5py
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Any, Union

from ..config import TrafficHDF5Config


class TrafficHDF5Reader:
    """Reader for traffic HDF5 files."""
    
    def __init__(self, hdf5_path: Path):
        self.hdf5_path = Path(hdf5_path)
        if not self.hdf5_path.exists():
            raise FileNotFoundError(f"HDF5 file not found: {hdf5_path}")
    
    def get_metadata(self) -> Dict[str, Any]:
        """Get metadata from HDF5 file."""
        with h5py.File(self.hdf5_path, 'r') as h5file:
            metadata = {}
            
            # File attributes
            for key in h5file.attrs:
                metadata[key] = h5file.attrs[key]
            
            # VDIDs and feature names
            metadata['vdids'] = [s.decode('utf-8') for s in h5file['metadata/vdids'][:]]
            metadata['feature_names'] = [s.decode('utf-8') for s in h5file['metadata/feature_names'][:]]
            
            # Shape information
            features_shape = h5file['data/features'].shape
            metadata['data_shape'] = features_shape
            
            return metadata
    
    def get_timestamps(self) -> List[str]:
        """Get all timestamps."""
        with h5py.File(self.hdf5_path, 'r') as h5file:
            return [s.decode('utf-8') for s in h5file['metadata/timestamps'][:]]
    
    def get_vd_info(self, vdid: str) -> Optional[Dict[str, Any]]:
        """Get information for a specific VD."""
        with h5py.File(self.hdf5_path, 'r') as h5file:
            if f'metadata/vd_info/{vdid}' in h5file:
                vd_group = h5file[f'metadata/vd_info/{vdid}']
                return dict(vd_group.attrs)
            return None
    
    def get_features(self, time_slice: Optional[slice] = None, 
                    vd_indices: Optional[List[int]] = None,
                    feature_indices: Optional[List[int]] = None) -> np.ndarray:
        """Get feature data with optional slicing."""
        with h5py.File(self.hdf5_path, 'r') as h5file:
            features = h5file['data/features']
            
            # Load data progressively to avoid HDF5 fancy indexing limitations
            # Step 1: Apply time slicing
            if time_slice is not None:
                data = features[time_slice, :, :]
            else:
                data = features[:, :, :]
            
            # Step 2: Apply VD indexing
            if vd_indices is not None:
                data = data[:, vd_indices, :]
            
            # Step 3: Apply feature indexing
            if feature_indices is not None:
                data = data[:, :, feature_indices]
            
            return data
    
    def get_vd_timeseries(self, vdid: str) -> np.ndarray:
        """Get complete time series for a specific VD."""
        metadata = self.get_metadata()
        try:
            vd_index = metadata['vdids'].index(vdid)
            return self.get_features(vd_indices=[vd_index])[:, 0, :]
        except ValueError:
            raise ValueError(f"VDID {vdid} not found in dataset")


def create_traffic_hdf5(source_dir: Union[str, Path], 
                       output_path: Union[str, Path],
                       selected_vdids: Optional[List[str]] = None,
                       overwrite: bool = False,
                       check_consistency: bool = True,
                       **kwargs) -> TrafficHDF5Config:
    """Convenience function to create traffic HDF5 file with smart checking."""
    from .h5_converter import TrafficHDF5Converter
    
    config = TrafficHDF5Config(
        source_dir=source_dir,
        output_path=output_path,
        selected_vdids=selected_vdids,
        overwrite=overwrite,
        check_consistency=check_consistency,
        **kwargs
    )
    
    converter = TrafficHDF5Converter(config)
    converter.convert()
    
    return config


def ensure_traffic_hdf5(source_dir: Union[str, Path], 
                       output_path: Union[str, Path],
                       **kwargs) -> TrafficHDF5Reader:
    """Ensure HDF5 exists and return reader. Create only if necessary."""
    from .h5_converter import TrafficHDF5Converter
    
    output_path = Path(output_path)
    
    # Default to no overwrite for this function
    kwargs.setdefault('overwrite', False)
    kwargs.setdefault('check_consistency', True)
    
    # Create HDF5 if it doesn't exist or needs updating
    if not output_path.exists():
        create_traffic_hdf5(source_dir, output_path, **kwargs)
    else:
        # Check if file needs updating
        config = TrafficHDF5Config(
            source_dir=source_dir,
            output_path=output_path,
            **kwargs
        )
        converter = TrafficHDF5Converter(config)
        if converter._is_hdf5_outdated():
            converter.convert()
    
    return TrafficHDF5Reader(output_path)