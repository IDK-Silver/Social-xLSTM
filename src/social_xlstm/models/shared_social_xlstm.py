"""
Shared-Encoder Social xLSTM Model for traffic prediction.

Processes all VD sensors with a single TrafficXLSTM by flattening the VD axis
into the batch dimension, then restores [B,N,T,E] to compute per-VD social pooling.

This maximizes GPU utilization (fewer small kernels), reduces parameter/optimizer
memory (one encoder instead of per-VD encoders), and preserves per-VD representations
for pooling and per-VD predictions.
"""

from collections import OrderedDict
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
import pytorch_lightning as pl
import torchmetrics

from .xlstm import TrafficXLSTM, TrafficXLSTMConfig
from .distributed_config import DistributedSocialXLSTMConfig
from ..pooling.xlstm_pooling import XLSTMSocialPoolingLayer


class SharedSocialXLSTMModel(pl.LightningModule):
    def __init__(self, config: DistributedSocialXLSTMConfig):
        super().__init__()

        if not isinstance(config, DistributedSocialXLSTMConfig):
            raise TypeError("config must be DistributedSocialXLSTMConfig")

        self.config = config
        self.save_hyperparameters()

        self.num_features = config.num_features
        self.prediction_length = config.prediction_length
        self.learning_rate = config.learning_rate

        # Shared encoder
        self.encoder = TrafficXLSTM(config.xlstm)
        self.hidden_dim = config.xlstm.embedding_dim

        # Metrics (track both normalized and real-scale)
        # Normalized-scale metrics (used for optimization stability)
        self.train_mae_norm = torchmetrics.MeanAbsoluteError()
        self.val_mae_norm = torchmetrics.MeanAbsoluteError()
        self.train_mse_norm = torchmetrics.MeanSquaredError()
        self.val_mse_norm = torchmetrics.MeanSquaredError()
        self.train_rmse_norm = torchmetrics.MeanSquaredError(squared=False)
        self.val_rmse_norm = torchmetrics.MeanSquaredError(squared=False)
        self.train_r2_norm = torchmetrics.R2Score()
        self.val_r2_norm = torchmetrics.R2Score()

        # Real-scale metrics (comparable to papers; require inverse-transform)
        self.train_mae_real = torchmetrics.MeanAbsoluteError()
        self.val_mae_real = torchmetrics.MeanAbsoluteError()
        self.train_mse_real = torchmetrics.MeanSquaredError()
        self.val_mse_real = torchmetrics.MeanSquaredError()
        self.train_rmse_real = torchmetrics.MeanSquaredError(squared=False)
        self.val_rmse_real = torchmetrics.MeanSquaredError(squared=False)
        self.train_r2_real = torchmetrics.R2Score()
        self.val_r2_real = torchmetrics.R2Score()

        # Social pooling layer (spatial)
        if config.social_pooling.enabled:
            self.social_pooling = XLSTMSocialPoolingLayer(
                hidden_dim=self.hidden_dim,
                radius=config.social_pooling.radius,
                pool_type=config.social_pooling.pool_type,
                learnable_radius=False
            )
        else:
            self.social_pooling = None

        # Fusion and prediction
        self.fusion_layer = nn.Sequential(
            nn.Linear(self.hidden_dim * 2, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
        )
        self.prediction_head = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(self.hidden_dim // 2, config.prediction_length * config.num_features),
        )

        self.criterion = nn.MSELoss()

    def forward(
        self,
        features: torch.Tensor,  # [B, T, N, F]
        vd_ids: Optional[List[str]] = None,
        positions_xy: Optional[torch.Tensor] = None,  # [N, 2] or None
    ) -> Dict[str, torch.Tensor]:
        if features.dim() != 4:
            raise ValueError(f"Expected features [B,T,N,F], got {features.shape}")

        B, T, N, F = features.shape
        if F != self.num_features:
            raise ValueError(f"num_features mismatch: cfg={self.num_features}, got {F}")

        # Flatten VD axis into batch, encode, then restore
        x_flat = features.permute(0, 2, 1, 3).reshape(B * N, T, F)  # [B*N,T,F]
        h_flat = self.encoder.get_hidden_states(x_flat)             # [B*N,T,E]
        H = h_flat.view(B, N, T, self.hidden_dim)                   # [B,N,T,E]

        # Prepare dict-of-VD hidden states for pooling/fusion
        # Keep VD order deterministic using vd_ids if provided; else use index
        if vd_ids is None:
            vd_ids = [str(i) for i in range(N)]

        if len(vd_ids) != N:
            raise ValueError("vd_ids length does not match N")

        hidden_states_dict: Dict[str, torch.Tensor] = OrderedDict()
        for i, vd in enumerate(vd_ids):
            hidden_states_dict[vd] = H[:, i, :, :]  # [B,T,E]

        # Social contexts (strict requirement if enabled)
        if self.social_pooling is not None:
            if positions_xy is None:
                raise RuntimeError(
                    "Social pooling is enabled but 'positions_xy' is missing in batch. "
                    "Ensure centralized collate attaches positions_xy [N,2] from HDF5 metadata."
                )
            if positions_xy.dim() != 2 or positions_xy.size(0) != N or positions_xy.size(1) != 2:
                raise RuntimeError(
                    f"positions_xy must be [N,2] matching N={N}, got {tuple(positions_xy.shape)}."
                )
            if torch.isnan(positions_xy).any():
                raise RuntimeError("positions_xy contains NaNs while social pooling is enabled.")

            # Build positions dict per VD as [B,T,2] by repeating static XY
            positions_dict: Dict[str, torch.Tensor] = OrderedDict()
            for i, vd in enumerate(vd_ids):
                xy = positions_xy[i]  # [2]
                base = xy.view(1, 1, 2).to(H.device)
                positions_dict[vd] = base.expand(B, T, 2)

            social_contexts = self.social_pooling(
                agent_hidden_states=hidden_states_dict,
                agent_positions=positions_dict,
                target_agent_ids=vd_ids,
            )
        else:
            social_contexts = OrderedDict()
            for vd in vd_ids:
                social_contexts[vd] = torch.zeros(B, self.hidden_dim, device=H.device)

        # Fusion + prediction per VD
        predictions: Dict[str, torch.Tensor] = OrderedDict()
        for i, vd in enumerate(vd_ids):
            individual_hidden = H[:, i, -1, :]           # [B,E]
            social_context = social_contexts[vd]         # [B,E]
            fused = torch.cat([individual_hidden, social_context], dim=-1)
            fused = self.fusion_layer(fused)
            pred = self.prediction_head(fused)           # [B, P*F]
            predictions[vd] = pred

        return predictions

    def training_step(self, batch: Dict[str, Any], batch_idx: int) -> torch.Tensor:
        features = batch['features']              # [B,T,N,F]
        targets = batch['targets']                # [B,P,N,F]
        vd_ids = batch.get('vd_ids')
        positions_xy = batch.get('positions_xy')  # [N,2]

        preds = self(features, vd_ids=vd_ids, positions_xy=positions_xy)

        total_loss = 0.0
        all_preds = []
        all_targets = []
        num_vds = 0

        B = features.size(0)
        P = targets.size(1)

        for idx, vd in enumerate(vd_ids):
            # targets[:, :, idx, :] -> [B,P,F] -> flatten to [B, P*F]
            tgt_flat = targets[:, :, idx, :].reshape(B, -1)
            if vd in preds:
                vd_loss = self.criterion(preds[vd], tgt_flat)
                total_loss += vd_loss
                num_vds += 1
                all_preds.append(preds[vd])
                all_targets.append(tgt_flat)

        avg_loss = total_loss / num_vds if num_vds > 0 else total_loss

        if all_preds and all_targets:
            preds_tensor = torch.cat(all_preds, dim=0)    # [B*N, P*F]
            targets_tensor = torch.cat(all_targets, dim=0) # [B*N, P*F]

            # Update normalized-scale metrics
            self.train_mae_norm(preds_tensor, targets_tensor)
            self.train_mse_norm(preds_tensor, targets_tensor)
            self.train_rmse_norm(preds_tensor, targets_tensor)
            self.train_r2_norm(preds_tensor, targets_tensor)

            # Try to compute real-scale metrics via inverse transform (if scaler is available)
            preds_real = None
            targets_real = None
            scaler = None
            try:
                if hasattr(self, 'trainer') and getattr(self.trainer, 'datamodule', None) is not None:
                    scaler = getattr(self.trainer.datamodule, 'shared_scaler', None)
            except Exception:
                scaler = None

            if scaler is not None:
                with torch.no_grad():
                    BxN = preds_tensor.size(0)
                    P = self.prediction_length
                    F = self.num_features
                    # Shape back to [BxN*P, F] for inverse_transform, then restore to [BxN, P*F]
                    preds_np = preds_tensor.detach().cpu().view(BxN, P, F).reshape(-1, F).numpy()
                    targs_np = targets_tensor.detach().cpu().view(BxN, P, F).reshape(-1, F).numpy()
                    try:
                        preds_inv = scaler.inverse_transform(preds_np)
                        targs_inv = scaler.inverse_transform(targs_np)
                        preds_real = torch.from_numpy(preds_inv).view(BxN, P * F)
                        targets_real = torch.from_numpy(targs_inv).view(BxN, P * F)
                        # Move to same device/dtype as predictions for metrics
                        device = preds_tensor.device
                        dtype = preds_tensor.dtype
                        preds_real = preds_real.to(device=device, dtype=dtype)
                        targets_real = targets_real.to(device=device, dtype=dtype)
                    except Exception:
                        preds_real = None
                        targets_real = None

            if preds_real is None or targets_real is None:
                # Fallback to normalized values if inverse-transform not possible
                preds_real = preds_tensor.detach()
                targets_real = targets_tensor.detach()

            # Update real-scale metrics
            self.train_mae_real(preds_real, targets_real)
            self.train_mse_real(preds_real, targets_real)
            self.train_rmse_real(preds_real, targets_real)
            self.train_r2_real(preds_real, targets_real)

            # Log primary metrics on real scale for comparability
            self.log('train_loss', avg_loss, prog_bar=True)
            self.log('train_mae', self.train_mae_real, prog_bar=False)
            self.log('train_mse', self.train_mse_real, prog_bar=False)
            self.log('train_rmse', self.train_rmse_real, prog_bar=False)
            self.log('train_r2', self.train_r2_real, prog_bar=False)

            # Also log normalized metrics for debugging/reference
            self.log('train_mae_norm', self.train_mae_norm, prog_bar=False)
            self.log('train_mse_norm', self.train_mse_norm, prog_bar=False)
            self.log('train_rmse_norm', self.train_rmse_norm, prog_bar=False)
            self.log('train_r2_norm', self.train_r2_norm, prog_bar=False)

            self.log('num_vds', float(num_vds), prog_bar=True)
            self.log('social_pooling_enabled', float(self.config.social_pooling.enabled), prog_bar=False)

        return avg_loss

    def validation_step(self, batch: Dict[str, Any], batch_idx: int) -> torch.Tensor:
        features = batch['features']
        targets = batch['targets']
        vd_ids = batch.get('vd_ids')
        positions_xy = batch.get('positions_xy')

        preds = self(features, vd_ids=vd_ids, positions_xy=positions_xy)

        total_loss = 0.0
        all_preds = []
        all_targets = []
        num_vds = 0

        B = features.size(0)
        for idx, vd in enumerate(vd_ids):
            tgt_flat = targets[:, :, idx, :].reshape(B, -1)
            if vd in preds:
                vd_loss = self.criterion(preds[vd], tgt_flat)
                total_loss += vd_loss
                num_vds += 1
                all_preds.append(preds[vd])
                all_targets.append(tgt_flat)

        avg_loss = total_loss / num_vds if num_vds > 0 else total_loss

        if all_preds and all_targets:
            preds_tensor = torch.cat(all_preds, dim=0)    # [B*N, P*F]
            targets_tensor = torch.cat(all_targets, dim=0) # [B*N, P*F]

            # Update normalized-scale metrics
            self.val_mae_norm(preds_tensor, targets_tensor)
            self.val_mse_norm(preds_tensor, targets_tensor)
            self.val_rmse_norm(preds_tensor, targets_tensor)
            self.val_r2_norm(preds_tensor, targets_tensor)

            # Inverse-transform to real scale if scaler available
            preds_real = None
            targets_real = None
            scaler = None
            try:
                if hasattr(self, 'trainer') and getattr(self.trainer, 'datamodule', None) is not None:
                    scaler = getattr(self.trainer.datamodule, 'shared_scaler', None)
            except Exception:
                scaler = None

            if scaler is not None:
                with torch.no_grad():
                    BxN = preds_tensor.size(0)
                    P = self.prediction_length
                    F = self.num_features
                    preds_np = preds_tensor.detach().cpu().view(BxN, P, F).reshape(-1, F).numpy()
                    targs_np = targets_tensor.detach().cpu().view(BxN, P, F).reshape(-1, F).numpy()
                    try:
                        preds_inv = scaler.inverse_transform(preds_np)
                        targs_inv = scaler.inverse_transform(targs_np)
                        preds_real = torch.from_numpy(preds_inv).view(BxN, P * F)
                        targets_real = torch.from_numpy(targs_inv).view(BxN, P * F)
                        # Move to same device/dtype as predictions for metrics
                        device = preds_tensor.device
                        dtype = preds_tensor.dtype
                        preds_real = preds_real.to(device=device, dtype=dtype)
                        targets_real = targets_real.to(device=device, dtype=dtype)
                    except Exception:
                        preds_real = None
                        targets_real = None

            if preds_real is None or targets_real is None:
                preds_real = preds_tensor.detach()
                targets_real = targets_tensor.detach()

            # Update real-scale metrics
            self.val_mae_real(preds_real, targets_real)
            self.val_mse_real(preds_real, targets_real)
            self.val_rmse_real(preds_real, targets_real)
            self.val_r2_real(preds_real, targets_real)

            # Log primary metrics on real scale for comparability
            self.log('val_loss', avg_loss, prog_bar=True)
            self.log('val_mae', self.val_mae_real, prog_bar=False)
            self.log('val_mse', self.val_mse_real, prog_bar=False)
            self.log('val_rmse', self.val_rmse_real, prog_bar=False)
            self.log('val_r2', self.val_r2_real, prog_bar=False)

            # Also log normalized metrics for debugging/reference
            self.log('val_mae_norm', self.val_mae_norm, prog_bar=False)
            self.log('val_mse_norm', self.val_mse_norm, prog_bar=False)
            self.log('val_rmse_norm', self.val_rmse_norm, prog_bar=False)
            self.log('val_r2_norm', self.val_r2_norm, prog_bar=False)

        return avg_loss

    def configure_optimizers(self):
        if self.config.optimizer is not None:
            opt_config = self.config.optimizer
            lr = opt_config.lr

            if opt_config.name.lower() == "adamw":
                optimizer = torch.optim.AdamW(
                    self.parameters(), lr=lr,
                    weight_decay=opt_config.weight_decay,
                    betas=opt_config.betas,
                    eps=opt_config.eps,
                )
            elif opt_config.name.lower() == "sgd":
                optimizer = torch.optim.SGD(
                    self.parameters(), lr=lr,
                    weight_decay=opt_config.weight_decay,
                    momentum=opt_config.momentum,
                )
            else:
                optimizer = torch.optim.Adam(
                    self.parameters(), lr=lr,
                    weight_decay=opt_config.weight_decay,
                    betas=opt_config.betas,
                    eps=opt_config.eps,
                )
        else:
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
