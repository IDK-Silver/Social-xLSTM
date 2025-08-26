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
from .distributed_config import DistributedSocialXLSTMConfig
from ..pooling.xlstm_pooling import XLSTMSocialPoolingLayer, create_mock_positions




class DistributedSocialXLSTMModel(pl.LightningModule):
    def __init__(self, config: DistributedSocialXLSTMConfig):
        super().__init__()
        
        # Validate configuration
        if not isinstance(config, DistributedSocialXLSTMConfig):
            raise TypeError("config must be an instance of DistributedSocialXLSTMConfig")
        
        self.config = config
        self.save_hyperparameters()
        
        # Extract frequently used values
        self.num_features = config.num_features
        self.prediction_length = config.prediction_length
        self.learning_rate = config.learning_rate
        # VDXLSTMManager returns embedding_dim dimensional hidden states, not hidden_size
        self.hidden_dim = config.xlstm.embedding_dim
        
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
            xlstm_config=config.xlstm,
            lazy_init=True,
            enable_gradient_checkpointing=config.enable_gradient_checkpointing
        )
        
        # Social pooling - spatial-only mode
        if config.social_pooling.enabled:
            # XLSTMSocialPoolingLayer operates on the actual hidden states dimension
            # which comes from xlstm.embedding_dim (VDXLSTMManager returns embedding_dim)
            self.social_pooling = XLSTMSocialPoolingLayer(
                hidden_dim=config.xlstm.embedding_dim,  # Use xlstm embedding_dim
                radius=config.social_pooling.radius,
                pool_type=config.social_pooling.pool_type,
                learnable_radius=False
            )
        else:
            self.social_pooling = None
        
        # Simplified dimension handling
        # Both individual and social contexts use xlstm.embedding_dim (what VDXLSTMManager actually returns)
        hidden_dim = config.xlstm.embedding_dim
        self.social_projection = None  # No projection needed
        
        # Fusion layer combines individual + social contexts (both same dimension)
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
            nn.Linear(hidden_dim // 2, config.prediction_length * config.num_features)
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
        
        # Social pooling - spatial-only or disabled
        if self.social_pooling is not None and positions is not None:
            # Use spatial-aware pooling
            social_contexts = self.social_pooling(
                agent_hidden_states=individual_hidden_states,
                agent_positions=positions,
                target_agent_ids=vd_ids
            )
        else:
            # Social pooling disabled - use zero contexts
            social_contexts = OrderedDict()
            
            for vd_id in vd_ids:
                batch_size = individual_hidden_states[vd_id].shape[0]
                device = individual_hidden_states[vd_id].device
                # Use xlstm embedding_dim for consistency (what VDXLSTMManager actually returns)
                social_context = torch.zeros(batch_size, self.config.xlstm.embedding_dim, device=device)
                social_contexts[vd_id] = social_context
        
        # Fusion and prediction
        predictions = OrderedDict()
        
        for vd_id in vd_ids:
            individual_hidden = individual_hidden_states[vd_id][:, -1, :]
            social_context = social_contexts[vd_id]
            
            # No projection needed - both contexts use same dimension
            
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
            self.log('social_pooling_enabled', float(self.config.social_pooling.enabled), prog_bar=False)
        
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
            'prediction_length': self.prediction_length,
            'social_pooling_enabled': self.config.social_pooling.enabled,
            'social_pooling_config': {
                'radius': self.config.social_pooling.radius,
                'aggregation': self.config.social_pooling.pool_type,
                'hidden_dim': self.config.xlstm.embedding_dim  # Use xlstm embedding_dim
            } if self.config.social_pooling.enabled else None
        }