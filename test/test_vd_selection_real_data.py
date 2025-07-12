"""
Real data integration test for VD selection functionality.

This test uses actual data structures to verify VD selection correctness.
"""

import pytest
import torch
import numpy as np
from pathlib import Path

from social_xlstm.dataset.core.datamodule import TrafficDataModule
from social_xlstm.dataset.config.base import TrafficDatasetConfig
from social_xlstm.training.without_social_pooling.single_vd_trainer import (
    SingleVDTrainer, 
    SingleVDTrainingConfig
)


@pytest.mark.skipif(
    not Path("blob/dataset/pre-processed/h5/traffic_features.h5").exists(),
    reason="Real data file not available"
)
class TestVDSelectionRealData:
    """Test VD selection with real traffic data."""
    
    def test_vd_selection_with_real_data(self):
        """Test VD selection using actual traffic dataset."""
        
        # Create data configuration for testing
        data_config = TrafficDatasetConfig(
            hdf5_path="blob/dataset/pre-processed/h5/traffic_features.h5",
            sequence_length=5,
            batch_size=2,
            selected_vdids=["VD-11-0020-002-001", "VD-28-0740-000-001"],  # Select 2 VDs
            train_ratio=0.8,
            val_ratio=0.1,
            test_ratio=0.1,
            num_workers=0  # Avoid multiprocessing in tests
        )
        
        # Create data module
        data_module = TrafficDataModule(data_config)
        data_module.setup()
        
        # Get a real batch
        train_loader = data_module.train_dataloader()
        real_batch = next(iter(train_loader))
        
        # Verify batch structure
        assert 'input_seq' in real_batch
        assert 'target_seq' in real_batch
        assert 'vdids' in real_batch
        
        print(f"Real batch structure:")
        print(f"  input_seq.shape: {real_batch['input_seq'].shape}")
        print(f"  target_seq.shape: {real_batch['target_seq'].shape}")
        print(f"  vdids: {real_batch['vdids']}")
        
        # Test with both VDs
        for target_vd in ["VD-11-0020-002-001", "VD-28-0740-000-001"]:
            
            # Create trainer configuration
            config = SingleVDTrainingConfig(
                epochs=1,
                batch_size=2,
                select_vd_id=target_vd,
                experiment_name=f"test_real_data_{target_vd.replace('-', '_')}",
                save_dir="/tmp"
            )
            
            # Create mock model
            model = torch.nn.Linear(5, 5)  # Real model with parameters
            
            # Create trainer
            trainer = SingleVDTrainer(model, config, train_loader, None, None)
            
            # Test prepare_batch
            inputs, targets = trainer.prepare_batch(real_batch)
            
            # Verify shapes
            batch_size = real_batch['input_seq'].shape[0]
            assert inputs.shape == (batch_size, 5, 5)  # [batch, seq, features]
            assert targets.shape == (batch_size, 1, 5)  # [batch, pred_steps, features]
            
            # Verify we got actual data (no NaN, and it's a valid tensor)
            assert not torch.isnan(inputs).any()
            assert inputs.dtype == torch.float32
            
            print(f"✅ VD {target_vd} selection test passed")
            print(f"   Input range: [{inputs.min():.3f}, {inputs.max():.3f}]")
    
    def test_vd_selection_data_consistency(self):
        """Test that VD selection gives consistent results."""
        
        # Test with multiple VDs
        selected_vds = ["VD-11-0020-002-001", "VD-28-0740-000-001", "VD-13-0660-000-002"]
        
        data_config = TrafficDatasetConfig(
            hdf5_path="blob/dataset/pre-processed/h5/traffic_features.h5",
            sequence_length=5,
            batch_size=4,
            selected_vdids=selected_vds,
            train_ratio=0.8,
            val_ratio=0.1,
            test_ratio=0.1,
            num_workers=0
        )
        
        data_module = TrafficDataModule(data_config)
        data_module.setup()
        
        train_loader = data_module.train_dataloader()
        real_batch = next(iter(train_loader))
        
        # Test each VD selection
        selected_data = {}
        
        for i, target_vd in enumerate(selected_vds):
            config = SingleVDTrainingConfig(
                epochs=1,
                select_vd_id=target_vd,
                experiment_name=f"consistency_test_{i}",
                save_dir="/tmp"
            )
            
            model = torch.nn.Linear(5, 5)
            trainer = SingleVDTrainer(model, config, train_loader, None, None)
            
            inputs, targets = trainer.prepare_batch(real_batch)
            selected_data[target_vd] = inputs
            
            # Verify this matches manual selection
            expected_inputs = real_batch['input_seq'][:, :, i, :]
            torch.testing.assert_close(inputs, expected_inputs, rtol=1e-5, atol=1e-8)
            
            print(f"✅ {target_vd} data consistency verified")
        
        # Verify different VDs give different data
        vd_list = list(selected_data.keys())
        for i in range(len(vd_list)):
            for j in range(i + 1, len(vd_list)):
                vd1, vd2 = vd_list[i], vd_list[j]
                data1, data2 = selected_data[vd1], selected_data[vd2]
                
                # Data should be different (unless by extreme coincidence)
                assert not torch.allclose(data1, data2, rtol=1e-3)
                print(f"✅ {vd1} ≠ {vd2} (data is different)")
    
    def test_vd_selection_error_handling_real_data(self):
        """Test error handling with real data."""
        
        data_config = TrafficDatasetConfig(
            hdf5_path="blob/dataset/pre-processed/h5/traffic_features.h5",
            sequence_length=5,
            batch_size=2,
            selected_vdids=["VD-11-0020-002-001"],  # Only one VD
            train_ratio=0.8,
            val_ratio=0.1,
            test_ratio=0.1,
            num_workers=0
        )
        
        data_module = TrafficDataModule(data_config)
        data_module.setup()
        
        train_loader = data_module.train_dataloader()
        real_batch = next(iter(train_loader))
        
        # Test with non-existent VD
        config = SingleVDTrainingConfig(
            epochs=1,
            select_vd_id="VD-DOES-NOT-EXIST",
            experiment_name="error_handling_test",
            save_dir="/tmp"
        )
        
        model = torch.nn.Linear(5, 5)
        trainer = SingleVDTrainer(model, config, train_loader, None, None)
        
        # Should fall back to first VD without crashing
        inputs, targets = trainer.prepare_batch(real_batch)
        
        # Should get the same result as manual first VD selection
        expected_inputs = real_batch['input_seq'][:, :, 0, :]
        torch.testing.assert_close(inputs, expected_inputs)
        
        print("✅ Error handling test passed - graceful fallback to first VD")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])