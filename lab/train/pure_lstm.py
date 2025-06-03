import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import sys
import time
from dataclasses import dataclass

from social_xlstm.models.pure.traffic_lstm import TrafficLSTM
from social_xlstm.dataset.loader import TrafficDatasetConfig, TrafficTimeSeries
from social_xlstm.evaluation.evaluator import ModelEvaluator
from social_xlstm.visualization.model import TrafficResultsPlotter
from typing import Optional


@dataclass
class TrainingConfig:
    """Configuration for training the TrafficLSTM model."""
    batch_size: int = 4
    epochs: int = 500
    selected_features: list[str] = None
    selected_vdids: list[str] = None
    hidden_size: int = 64
    num_layers: int = 2
    prediction_length: int = 3
    dropout: float = 0.0
    learning_rate: float = 0.001
    
    def __post_init__(self):
        # Set default values if None
        if self.selected_features is None:
            self.selected_features = ['avg_speed', 'total_volume', 'avg_occupancy']
        if self.selected_vdids is None:
            self.selected_vdids = ['VD-11-0020-002-001', 'VD-11-0020-008-001']


def create_dataloader(dataset, config: TrainingConfig, shuffle: bool = True) -> DataLoader:
    """Helper function to create DataLoader with given dataset and parameters."""
    return DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=shuffle,
        num_workers=0,  # Set to 0 to avoid multiprocessing issues
        pin_memory=False,  # Not needed for small datasets
        drop_last=False  # Do not drop the last incomplete batch
    )


def create_model(config: TrainingConfig) -> TrafficLSTM:
    """Create and return the TrafficLSTM model."""
    model = TrafficLSTM(
        input_size=len(config.selected_features),
        hidden_size=config.hidden_size,
        num_layers=config.num_layers,
        prediction_length=config.prediction_length,
        dropout=config.dropout
    )
    return model


def train_model(model, train_loader, val_loader, config: TrainingConfig, device, vd_index=0):
    """Train the model for a specific VD."""
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
    
    model.to(device)
    train_losses = []
    val_losses = []
    
    print(f"=== Training for VD index {vd_index} ===")
    
    for epoch in range(config.epochs):
        # Training phase
        model.train()
        epoch_train_loss = 0.0
        num_batches = 0
        
        for batch_idx, batch in enumerate(train_loader):
            # Extract data for specific VD
            if isinstance(batch, dict):
                if 'input_seq' in batch and 'target_seq' in batch:
                    # Original shape: [batch_size, seq_len, num_vds, num_features]
                    # Select specific VD: [batch_size, seq_len, num_features]
                    data = batch['input_seq'][:, :, vd_index, :]  # Select VD at index
                    target = batch['target_seq'][:, :, vd_index, :]  # Select VD at index
                else:
                    raise ValueError(f"Cannot identify data and target keys in batch: {list(batch.keys())}")
            else:
                raise ValueError("Expected dictionary format from dataset")
            
            data, target = data.to(device), target.to(device)
            
            # Print shapes for debugging (only first few batches)
            if batch_idx < 3 and epoch == 0:
                print(f"Batch {batch_idx}: data shape = {data.shape}, target shape = {target.shape}")
            
            optimizer.zero_grad()
            output = model(data)  # Now model expects 3D input
            
            # Debug output shape
            if batch_idx < 3 and epoch == 0:
                print(f"Batch {batch_idx}: output shape = {output.shape}")
            
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            epoch_train_loss += loss.item()
            num_batches += 1
        
        avg_train_loss = epoch_train_loss / num_batches
        train_losses.append(avg_train_loss)
        
        # Validation phase (similar logic)
        if val_loader is not None and len(val_loader) > 0:
            model.eval()
            epoch_val_loss = 0.0
            val_batches = 0
            
            with torch.no_grad():
                for batch in val_loader:
                    if isinstance(batch, dict):
                        data = batch['input_seq'][:, :, vd_index, :]
                        target = batch['target_seq'][:, :, vd_index, :]
                    
                    data, target = data.to(device), target.to(device)
                    output = model(data)
                    loss = criterion(output, target)
                    epoch_val_loss += loss.item()
                    val_batches += 1
            
            avg_val_loss = epoch_val_loss / val_batches
            val_losses.append(avg_val_loss)
            
            if epoch % 10 == 0:
                print(f"Epoch {epoch}/{config.epochs}: Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}")
        else:
            val_losses.append(0.0)
            if epoch % 10 == 0:
                print(f"Epoch {epoch}/{config.epochs}: Train Loss: {avg_train_loss:.6f}")
    
    return train_losses, val_losses


def main(config: TrainingConfig = TrainingConfig()):
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create dataset configuration
    dataset_loader_config = TrafficDatasetConfig(
        hdf5_path=Path("blob/lab/pre-processed/h5/traffic_features.h5"),
        sequence_length=5,
        prediction_length=config.prediction_length,
        selected_vdids=config.selected_vdids,  # Still load all VDs
        selected_features=config.selected_features,
        train_ratio=0.8,
        val_ratio=0.2,
        test_ratio=0.0,
        batch_size=config.batch_size,
        num_workers=0,
        pin_memory=False,
        normalize=True,
        fill_missing='interpolate'
    )
    
    print("=== Creating datasets ===")
    try:
        train_dataset = TrafficTimeSeries(dataset_loader_config, split='train')
        validation_dataset = TrafficTimeSeries(dataset_loader_config, split='val')
        test_dataset = TrafficTimeSeries(dataset_loader_config, split='test')
        
        print(f"Train dataset created: {len(train_dataset)} samples")
        print(f"Validation dataset created: {len(validation_dataset)} samples")
        print(f"Test dataset created: {len(test_dataset)} samples")
        
        if len(train_dataset) == 0:
            raise ValueError("Training dataset is empty! Need to adjust parameters.")
        
    except Exception as e:
        print(f"Error creating datasets: {e}")
        raise
    
    # Create data loaders
    train_loader = create_dataloader(train_dataset, config, shuffle=True)
    val_loader = create_dataloader(validation_dataset, config, shuffle=False) if len(validation_dataset) > 0 else None
    
    # Train separate models for each VD
    trained_models = {}
    
    for vd_index, vdid in enumerate(config.selected_vdids):
        print(f"\n=== Training model for {vdid} (index {vd_index}) ===")
        
        # Create model for single VD (3D input/output)
        model = create_model(config)
        print(f"Model created with {sum(p.numel() for p in model.parameters())} parameters")
        
        # Train model for this specific VD
        train_losses, val_losses = train_model(
            model, train_loader, val_loader, config, device, vd_index=vd_index
        )
        
        # Store trained model
        trained_models[vdid] = {
            'model': model,
            'train_losses': train_losses,
            'val_losses': val_losses
        }
        
        print(f"Training completed for {vdid}")
        print(f"Final train loss: {train_losses[-1]:.6f}")
        if val_losses and val_losses[-1] > 0:
            print(f"Final val loss: {val_losses[-1]:.6f}")
    
    print("\nTraining completed!")
    

    
    # 創建繪圖器
    plotter = TrafficResultsPlotter()
    
    # 評估每個模型
    evaluators = {}
    for vd_index, vdid in enumerate(config.selected_vdids):
        print(f"\n=== Evaluating model for {vdid} ===")
        
        # 獲取訓練好的模型
        model_info = trained_models[vdid]
        
        # 創建評估器 - 傳入訓練和驗證的 data loader
        evaluator = ModelEvaluator(
            model=model_info['model'],
            train_losses=model_info['train_losses'],
            val_losses=model_info['val_losses'],
            config=config,
            train_loader=train_loader,
            val_loader=val_loader,  # 新增驗證集 loader
            device=device,
            vd_index=vd_index
        )
        
        # 獲取評估數據
        eval_data = evaluator.get_evaluation_data()
        
        # 顯示訓練指標
        train_metrics = eval_data['train_metrics']
        print(f"Training Metrics for {vdid}:")
        print(f"  MAE:  {train_metrics['mae']:.6f}")
        print(f"  RMSE: {train_metrics['rmse']:.6f}")
        print(f"  MAPE: {train_metrics['mape']:.2f}%")
        print(f"  R²:   {train_metrics['r2']:.6f}")
        
        # 如果有驗證集，顯示驗證指標
        if 'val_metrics' in eval_data:
            val_metrics = eval_data['val_metrics']
            print(f"Validation Metrics for {vdid}:")
            print(f"  MAE:  {val_metrics['mae']:.6f}")
            print(f"  RMSE: {val_metrics['rmse']:.6f}")
            print(f"  MAPE: {val_metrics['mape']:.2f}%")
            print(f"  R²:   {val_metrics['r2']:.6f}")
        
        # 生成圖表
        fig = plotter.create_evaluation_dashboard(
            eval_data, 
            save_path=f"evaluation_results_{vdid.replace('-', '_')}.png"
        )
        
        evaluators[vdid] = evaluator
    
    return trained_models, evaluators

if __name__ == '__main__':
    trained_models, evaluators = main()