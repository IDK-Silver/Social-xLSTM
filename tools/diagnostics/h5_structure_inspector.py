#!/usr/bin/env python3
"""
Quick H5 file structure checker
"""
import h5py
import numpy as np

def check_h5_structure(h5_path):
    print(f"🔍 Checking H5 file structure: {h5_path}")
    
    with h5py.File(h5_path, 'r') as h5file:
        print(f"\n📁 Root level keys: {list(h5file.keys())}")
        
        def print_structure(name, obj):
            if isinstance(obj, h5py.Group):
                print(f"📁 Group: {name}")
            elif isinstance(obj, h5py.Dataset):
                print(f"📄 Dataset: {name}, shape: {obj.shape}, dtype: {obj.dtype}")
        
        print(f"\n🏗️ Full structure:")
        h5file.visititems(print_structure)
        
        # If there's a 'data' group, check its contents
        if 'data' in h5file:
            data_group = h5file['data']
            print(f"\n📊 Data group contents: {list(data_group.keys())}")
            
            # Check for VD data
            for key in list(data_group.keys())[:5]:  # Check first 5
                item = data_group[key]
                if isinstance(item, h5py.Group):
                    print(f"   📁 {key}: {list(item.keys())}")
                elif isinstance(item, h5py.Dataset):
                    print(f"   📄 {key}: shape={item.shape}")

if __name__ == "__main__":
    check_h5_structure("blob/dataset/pre-processed/h5/traffic_features_default.h5")