"""
Distributed Social xLSTM Model for traffic prediction.

NOTE: This is the authoritative Social-xLSTM implementation.
Any new work must occur here; legacy files are frozen for reference.

This is the main model class that integrates VDXLSTMManager and Social Pooling
for distributed per-VD xLSTM processing with spatial interaction modeling.
"""

import torch
import torch.nn as nn
import pytorch_lightning as pl
from typing import Dict, List, Optional, Any
from collections import OrderedDict
import torchmetrics

from .vd_xlstm_manager import VDXLSTMManager
from .xlstm import TrafficXLSTMConfig
from ..pooling.xlstm_pooling import XLSTMSocialPoolingLayer, create_mock_positions


class SocialPoolingLayer(nn.Module):
    def __init__(self, hidden_dim: int, pool_type: str = "mean"):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.pool_type = pool_type
    
    def forward(self, neighbor_states: List[torch.Tensor]) -> torch.Tensor:
        if not neighbor_states:
            return torch.zeros(1, self.hidden_dim)
        
        if len(neighbor_states) == 1:
            return neighbor_states[0]
        
        stacked = torch.stack(neighbor_states, dim=0)
        return torch.mean(stacked, dim=0)


class DistributedSocialXLSTMModel(pl.LightningModule):
    def __init__(
        self,
        xlstm_config: TrafficXLSTMConfig,
        num_features: int,
        hidden_dim: int = 128,
        prediction_length: int = 3,
        social_pool_type: str = "mean",
        learning_rate: float = 1e-3,
        enable_gradient_checkpointing: bool = True,
        enable_spatial_pooling: bool = False,
        spatial_radius: float = 2.0
    ):
        super().__init__()
        
        self.save_hyperparameters()
        
        self.num_features = num_features
        self.hidden_dim = hidden_dim
        self.prediction_length = prediction_length
        self.learning_rate = learning_rate
        self.enable_spatial_pooling = enable_spatial_pooling
        self.spatial_radius = spatial_radius
        
        # Initialize TorchMetrics for proper metric calculation
        self.train_mae = torchmetrics.MeanAbsoluteError()
        self.val_mae = torchmetrics.MeanAbsoluteError()
        self.train_mse = torchmetrics.MeanSquaredError()
        self.val_mse = torchmetrics.MeanSquaredError()
        self.train_rmse = torchmetrics.MeanSquaredError(squared=False)  # RMSE
        self.val_rmse = torchmetrics.MeanSquaredError(squared=False)    # RMSE
        self.train_r2 = torchmetrics.R2Score()
        self.val_r2 = torchmetrics.R2Score()
        
        # VD Manager
        self.vd_manager = VDXLSTMManager(
            xlstm_config=xlstm_config,
            lazy_init=True,
            enable_gradient_checkpointing=enable_gradient_checkpointing
        )
        
        # Social pooling - use spatial or legacy version
        if enable_spatial_pooling:
            self.social_pooling = XLSTMSocialPoolingLayer(
                hidden_dim=hidden_dim,
                radius=spatial_radius,
                pool_type=social_pool_type,
                learnable_radius=False
            )
        else:
            self.social_pooling = SocialPoolingLayer(
                hidden_dim=hidden_dim,
                pool_type=social_pool_type
            )
        
        # Fusion layer
        self.fusion_layer = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        # Prediction head
        self.prediction_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, prediction_length * num_features)
        )
        
        self.criterion = nn.MSELoss()
    
    def _generate_neighbor_map(self, vd_ids: List[str]) -> Dict[str, List[str]]:
        neighbor_map = {}
        for vd_id in vd_ids:
            neighbors = [other_vd for other_vd in vd_ids if other_vd != vd_id]
            neighbor_map[vd_id] = neighbors
        return neighbor_map
    
    def forward(
        self,
        vd_inputs: Dict[str, torch.Tensor],
        neighbor_map: Optional[Dict[str, List[str]]] = None,
        positions: Optional[Dict[str, torch.Tensor]] = None
    ) -> Dict[str, torch.Tensor]:
        
        if not vd_inputs:
            raise ValueError("vd_inputs cannot be empty")
        
        if not isinstance(vd_inputs, OrderedDict):
            vd_inputs = OrderedDict(sorted(vd_inputs.items()))
        
        vd_ids = list(vd_inputs.keys())
        
        if neighbor_map is None:
            neighbor_map = self._generate_neighbor_map(vd_ids)
        
        # Process through VD manager  
        individual_hidden_states = self.vd_manager(vd_inputs)
        
        # Social pooling - spatial or legacy
        if self.enable_spatial_pooling and positions is not None:
            # Use spatial-aware pooling
            social_contexts = self.social_pooling(
                agent_hidden_states=individual_hidden_states,
                agent_positions=positions,
                target_agent_ids=vd_ids
            )
        else:
            # Use legacy neighbor-based pooling
            social_contexts = OrderedDict()
            
            for vd_id in vd_ids:
                neighbors = neighbor_map.get(vd_id, [])
                neighbor_states = []
                
                for neighbor_id in neighbors:
                    if neighbor_id in individual_hidden_states:
                        neighbor_hidden = individual_hidden_states[neighbor_id][:, -1, :]
                        neighbor_states.append(neighbor_hidden)
                
                if neighbor_states:
                    # In legacy mode, manually compute mean pooling 
                    # (spatial pooling layer expects different interface)
                    stacked = torch.stack(neighbor_states, dim=0)
                    social_context = torch.mean(stacked, dim=0)
                else:
                    batch_size = individual_hidden_states[vd_id].shape[0]
                    device = individual_hidden_states[vd_id].device
                    social_context = torch.zeros(batch_size, self.hidden_dim, device=device)
                
                social_contexts[vd_id] = social_context
        
        # Fusion and prediction
        predictions = OrderedDict()
        
        for vd_id in vd_ids:
            individual_hidden = individual_hidden_states[vd_id][:, -1, :]
            social_context = social_contexts[vd_id]
            
            fused_features = torch.cat([individual_hidden, social_context], dim=-1)
            fused_output = self.fusion_layer(fused_features)
            
            prediction = self.prediction_head(fused_output)
            predictions[vd_id] = prediction
        
        return predictions
    
    def training_step(self, batch: Dict[str, Any], batch_idx: int) -> torch.Tensor:
        vd_inputs = batch['features']
        vd_targets = batch['targets']
        
        predictions = self(vd_inputs)
        
        total_loss = 0.0
        all_preds = []
        all_targets = []
        num_vds = 0
        
        for vd_id, pred in predictions.items():
            if vd_id in vd_targets:
                target = vd_targets[vd_id]
                target_flat = target.reshape(target.shape[0], -1)
                
                vd_loss = self.criterion(pred, target_flat)
                total_loss += vd_loss
                num_vds += 1
                
                # Collect predictions and targets for metric calculation
                all_preds.append(pred)
                all_targets.append(target_flat)
        
        avg_loss = total_loss / num_vds if num_vds > 0 else total_loss
        
        # Calculate metrics using TorchMetrics if we have data
        if all_preds and all_targets:
            preds_tensor = torch.cat(all_preds, dim=0)
            targets_tensor = torch.cat(all_targets, dim=0)
            
            # Update and log metrics
            self.train_mae(preds_tensor, targets_tensor)
            self.train_mse(preds_tensor, targets_tensor)
            self.train_rmse(preds_tensor, targets_tensor)
            self.train_r2(preds_tensor, targets_tensor)
            
            self.log('train_loss', avg_loss, prog_bar=True)
            self.log('train_mae', self.train_mae, prog_bar=False)
            self.log('train_mse', self.train_mse, prog_bar=False)
            self.log('train_rmse', self.train_rmse, prog_bar=False)
            self.log('train_r2', self.train_r2, prog_bar=False)
            self.log('num_vds', float(num_vds), prog_bar=True)
        
        return avg_loss
    
    def validation_step(self, batch: Dict[str, Any], batch_idx: int) -> torch.Tensor:
        vd_inputs = batch['features']
        vd_targets = batch['targets']
        
        predictions = self(vd_inputs)
        
        total_loss = 0.0
        all_preds = []
        all_targets = []
        num_vds = 0
        
        for vd_id, pred in predictions.items():
            if vd_id in vd_targets:
                target = vd_targets[vd_id]
                target_flat = target.reshape(target.shape[0], -1)
                
                vd_loss = self.criterion(pred, target_flat)
                total_loss += vd_loss
                num_vds += 1
                
                # Collect predictions and targets for metric calculation
                all_preds.append(pred)
                all_targets.append(target_flat)
        
        avg_loss = total_loss / num_vds if num_vds > 0 else total_loss
        
        # Calculate metrics using TorchMetrics if we have data
        if all_preds and all_targets:
            preds_tensor = torch.cat(all_preds, dim=0)
            targets_tensor = torch.cat(all_targets, dim=0)
            
            # Update and log metrics
            self.val_mae(preds_tensor, targets_tensor)
            self.val_mse(preds_tensor, targets_tensor)
            self.val_rmse(preds_tensor, targets_tensor)
            self.val_r2(preds_tensor, targets_tensor)
            
            self.log('val_loss', avg_loss, prog_bar=True)
            self.log('val_mae', self.val_mae, prog_bar=False)
            self.log('val_mse', self.val_mse, prog_bar=False)
            self.log('val_rmse', self.val_rmse, prog_bar=False)
            self.log('val_r2', self.val_r2, prog_bar=False)
        
        return avg_loss
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=10
        )
        
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': 'val_loss'
            }
        }
    
    def get_model_info(self) -> Dict[str, Any]:
        vd_manager_info = self.vd_manager.get_memory_usage()
        
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'vd_manager_info': vd_manager_info,
            'hidden_dim': self.hidden_dim,
            'prediction_length': self.prediction_length
        }