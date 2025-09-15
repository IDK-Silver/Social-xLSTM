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
from ..pooling.st_attention_pooling import STAttentionPooling


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

        # Social pooling layer selection
        self.social_pooling = None
        if config.social_pooling.enabled:
            sp_cfg = config.social_pooling
            if getattr(sp_cfg, 'type', 'legacy') == 'st_attention':
                self.social_pooling = STAttentionPooling(
                    hidden_dim=self.hidden_dim,
                    knn_k=sp_cfg.knn_k,
                    time_window=sp_cfg.time_window,
                    heads=sp_cfg.heads,
                    learnable_tau=sp_cfg.learnable_tau,
                    tau_init=sp_cfg.tau_init,
                    dropout=sp_cfg.dropout,
                    use_radius_mask=sp_cfg.use_radius_mask,
                    radius=sp_cfg.radius,
                    bias_scale=getattr(sp_cfg, 'bias_scale', 1.0),
                )
            else:
                self.social_pooling = XLSTMSocialPoolingLayer(
                    hidden_dim=self.hidden_dim,
                    radius=sp_cfg.radius,
                    pool_type=sp_cfg.pool_type,
                    learnable_radius=False
                )

        # Fusion and prediction
        self.fusion_layer = nn.Sequential(
            nn.Linear(self.hidden_dim * 2, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
        )
        # Gate for social context (learned per-VD, per-feature)
        # Takes [h_t, context] and outputs sigmoid gate in [0,1] for context
        self.gate_layer = nn.Sequential(
            nn.Linear(self.hidden_dim * 2, self.hidden_dim),
            nn.Sigmoid(),
        )
        self.prediction_head = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(self.hidden_dim // 2, config.prediction_length * config.num_features),
        )

        self.criterion = nn.MSELoss()

        # Normalization buffers (set in on_fit_start)
        self.norm_kind: Optional[str] = None  # 'standard' | 'minmax' | None
        # Registered at runtime if scaler available
        # self.register_buffer('feature_mean', ...)
        # self.register_buffer('feature_scale', ...)
        # self.register_buffer('data_min', ...)
        # self.register_buffer('data_range', ...)

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

        # Prepare VD ids
        if vd_ids is None:
            vd_ids = [str(i) for i in range(N)]
        if len(vd_ids) != N:
            raise ValueError("vd_ids length does not match N")

        # Social contexts
        social_context_tensor: Optional[torch.Tensor] = None  # [B,N,E]
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

            # Branch by pooling implementation
            if isinstance(self.social_pooling, STAttentionPooling):
                # Vectorized ST-Attention pooling
                social_context_tensor = self.social_pooling(H, positions_xy.to(H.device))  # [B,N,E]
            else:
                # Legacy pooling expects dict-of-VD with [B,T,E] and [B,T,2]
                hidden_states_dict: Dict[str, torch.Tensor] = OrderedDict()
                positions_dict: Dict[str, torch.Tensor] = OrderedDict()
                for i, vd in enumerate(vd_ids):
                    hidden_states_dict[vd] = H[:, i, :, :]  # [B,T,E]
                    base = positions_xy[i].view(1, 1, 2).to(H.device)
                    positions_dict[vd] = base.expand(B, T, 2)

                social_contexts = self.social_pooling(
                    agent_hidden_states=hidden_states_dict,
                    agent_positions=positions_dict,
                    target_agent_ids=vd_ids,
                )
                # Stack to tensor [B,N,E]
                social_context_tensor = torch.stack([social_contexts[vd] for vd in vd_ids], dim=1)
        else:
            social_context_tensor = torch.zeros(B, N, self.hidden_dim, device=H.device)

        # Fusion + prediction per VD (vectorized over N)
        individual_hidden = H[:, :, -1, :]                # [B,N,E]
        # Apply learned gate on social context when pooling is enabled
        if self.social_pooling is not None:
            gate_in = torch.cat([individual_hidden, social_context_tensor], dim=-1)  # [B,N,2E]
            gate = self.gate_layer(gate_in)  # [B,N,E]
            social_context_tensor = gate * social_context_tensor
        fused = torch.cat([individual_hidden, social_context_tensor], dim=-1)  # [B,N,2E]
        fused = self.fusion_layer(fused.view(B * N, -1))  # [B*N,E]
        pred = self.prediction_head(fused).view(B, N, -1) # [B,N,P*F]

        predictions: Dict[str, torch.Tensor] = OrderedDict()
        for i, vd in enumerate(vd_ids):
            predictions[vd] = pred[:, i, :]

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

            # Compute real-scale tensors using registered buffers (pure torch)
            preds_real, targets_real = self._prepare_real_scale(
                preds_tensor, targets_tensor, self.prediction_length, self.num_features
            )

            # Update real-scale metrics
            self.train_mae_real(preds_real, targets_real)
            self.train_mse_real(preds_real, targets_real)
            self.train_rmse_real(preds_real, targets_real)
            self.train_r2_real(preds_real, targets_real)

            # Log primary metrics on real scale for comparability
            self.log('train_loss', avg_loss, prog_bar=True, on_step=False, on_epoch=True, batch_size=B)
            self.log('train_mae', self.train_mae_real, prog_bar=False, on_step=False, on_epoch=True, batch_size=B)
            self.log('train_mse', self.train_mse_real, prog_bar=False, on_step=False, on_epoch=True, batch_size=B)
            self.log('train_rmse', self.train_rmse_real, prog_bar=False, on_step=False, on_epoch=True, batch_size=B)
            self.log('train_r2', self.train_r2_real, prog_bar=False, on_step=False, on_epoch=True, batch_size=B)

            # Also log normalized metrics for debugging/reference
            self.log('train_mae_norm', self.train_mae_norm, prog_bar=False, on_step=False, on_epoch=True, batch_size=B)
            self.log('train_mse_norm', self.train_mse_norm, prog_bar=False, on_step=False, on_epoch=True, batch_size=B)
            self.log('train_rmse_norm', self.train_rmse_norm, prog_bar=False, on_step=False, on_epoch=True, batch_size=B)
            self.log('train_r2_norm', self.train_r2_norm, prog_bar=False, on_step=False, on_epoch=True, batch_size=B)

            self.log('num_vds', float(num_vds), prog_bar=True, on_step=False, on_epoch=True, batch_size=B)
            self.log('social_pooling_enabled', float(self.config.social_pooling.enabled), prog_bar=False, on_step=False, on_epoch=True, batch_size=B)

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

            # Compute real-scale tensors using registered buffers (pure torch)
            preds_real, targets_real = self._prepare_real_scale(
                preds_tensor, targets_tensor, self.prediction_length, self.num_features
            )

            # Update real-scale metrics
            self.val_mae_real(preds_real, targets_real)
            self.val_mse_real(preds_real, targets_real)
            self.val_rmse_real(preds_real, targets_real)
            self.val_r2_real(preds_real, targets_real)

            # Log primary metrics on real scale for comparability
            self.log('val_loss', avg_loss, prog_bar=True, on_step=False, on_epoch=True, batch_size=B)
            self.log('val_mae', self.val_mae_real, prog_bar=False, on_step=False, on_epoch=True, batch_size=B)
            self.log('val_mse', self.val_mse_real, prog_bar=False, on_step=False, on_epoch=True, batch_size=B)
            self.log('val_rmse', self.val_rmse_real, prog_bar=False, on_step=False, on_epoch=True, batch_size=B)
            self.log('val_r2', self.val_r2_real, prog_bar=False, on_step=False, on_epoch=True, batch_size=B)

            # Also log normalized metrics for debugging/reference
            self.log('val_mae_norm', self.val_mae_norm, prog_bar=False, on_step=False, on_epoch=True, batch_size=B)
            self.log('val_mse_norm', self.val_mse_norm, prog_bar=False, on_step=False, on_epoch=True, batch_size=B)
            self.log('val_rmse_norm', self.val_rmse_norm, prog_bar=False, on_step=False, on_epoch=True, batch_size=B)
            self.log('val_r2_norm', self.val_r2_norm, prog_bar=False, on_step=False, on_epoch=True, batch_size=B)

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

    # ----------------------
    # Normalization utilities
    # ----------------------
    def on_fit_start(self) -> None:
        """Extract scaler stats from datamodule and register as buffers for torch-only inverse-transform."""
        try:
            dm = getattr(self.trainer, 'datamodule', None)
            scaler = getattr(dm, 'shared_scaler', None) if dm is not None else None
        except Exception:
            scaler = None

        if scaler is None:
            self.norm_kind = None
            return

        # Detect scaler type by attributes; register buffers on CPU, Lightning moves them to device
        if hasattr(scaler, 'mean_') and hasattr(scaler, 'scale_'):
            # StandardScaler
            mean = torch.tensor(scaler.mean_, dtype=torch.float32)
            scale = torch.tensor(scaler.scale_, dtype=torch.float32)
            # Avoid degenerate scale
            eps = torch.finfo(torch.float32).eps
            scale = torch.clamp(scale, min=eps)
            self.register_buffer('feature_mean', mean)
            self.register_buffer('feature_scale', scale)
            self.norm_kind = 'standard'
        elif hasattr(scaler, 'data_min_') and hasattr(scaler, 'data_range_'):
            # MinMaxScaler
            data_min = torch.tensor(scaler.data_min_, dtype=torch.float32)
            data_range = torch.tensor(scaler.data_range_, dtype=torch.float32)
            eps = torch.finfo(torch.float32).eps
            data_range = torch.clamp(data_range, min=eps)
            self.register_buffer('data_min', data_min)
            self.register_buffer('data_range', data_range)
            self.norm_kind = 'minmax'
        else:
            # Unknown scaler type
            self.norm_kind = None

    def _inverse_transform(self, x: torch.Tensor, P: int, F: int) -> torch.Tensor:
        """Inverse-transform last-dimension features using registered buffers; x shape [*, P*F]."""
        if self.norm_kind is None:
            return x
        orig_shape = x.shape
        x_view = x.view(-1, P, F)
        if self.norm_kind == 'standard' and hasattr(self, 'feature_mean') and hasattr(self, 'feature_scale'):
            mean = self.feature_mean.to(device=x_view.device, dtype=x_view.dtype)
            scale = self.feature_scale.to(device=x_view.device, dtype=x_view.dtype)
            x_real = x_view * scale + mean
        elif self.norm_kind == 'minmax' and hasattr(self, 'data_min') and hasattr(self, 'data_range'):
            data_min = self.data_min.to(device=x_view.device, dtype=x_view.dtype)
            data_range = self.data_range.to(device=x_view.device, dtype=x_view.dtype)
            x_real = x_view * data_range + data_min
        else:
            return x
        return x_real.view(orig_shape)

    def _prepare_real_scale(self, preds: torch.Tensor, targets: torch.Tensor, P: int, F: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Return real-scale tensors using pure-torch inverse-transform when possible; otherwise return inputs."""
        try:
            preds_real = self._inverse_transform(preds, P, F)
            targets_real = self._inverse_transform(targets, P, F)
            return preds_real, targets_real
        except Exception:
            return preds.detach(), targets.detach()
