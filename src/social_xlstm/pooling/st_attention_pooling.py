"""
ST-Attention Social Pooling with kNN and learnable temperature (tau).

Vectorized implementation that aggregates per-VD temporal representations using
masked multi-head attention over k nearest spatial neighbors.

Inputs:
- Hidden states H: [B, N, T, E]
- Positions XY:   [N, 2] (static per VD, in projected meters)

Outputs:
- Social context per VD: [B, N, E]

Key features:
- kNN sparsification for scalability (O(B·N·k·E))
- Distance kernel w(d) = exp(-d / tau) with learnable tau
- Optional radius mask (hard zeroing beyond radius)
- Multi-head attention with position-aware bias from distances
"""

from __future__ import annotations

from typing import Optional
import torch
import torch.nn as nn
import math


class STAttentionPooling(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        knn_k: int = 16,
        time_window: int = 4,
        heads: int = 4,
        learnable_tau: bool = True,
        tau_init: float = 1.0,
        dropout: float = 0.1,
        use_radius_mask: bool = False,
        radius: float = 0.0,
        bias_scale: float = 1.0,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.knn_k = int(knn_k)
        self.time_window = int(time_window)
        self.heads = int(heads)
        self.dropout = nn.Dropout(dropout)
        self.use_radius_mask = bool(use_radius_mask)
        self.radius = float(radius)
        self.bias_scale = float(bias_scale)

        # Head dimensions
        self.head_dim = max(1, hidden_dim // heads)
        self.proj_dim = self.head_dim * heads

        # Projections
        self.q_proj = nn.Linear(hidden_dim, self.proj_dim)
        self.k_proj = nn.Linear(hidden_dim, self.proj_dim)
        self.v_proj = nn.Linear(hidden_dim, self.proj_dim)
        self.out_proj = nn.Linear(self.proj_dim, hidden_dim)

        # Learnable temperature for distance kernel
        if learnable_tau:
            self.tau = nn.Parameter(torch.tensor(float(tau_init), dtype=torch.float32))
        else:
            self.register_buffer('tau', torch.tensor(float(tau_init), dtype=torch.float32))

    @staticmethod
    def _pairwise_distances(xy: torch.Tensor) -> torch.Tensor:
        """Compute pairwise Euclidean distance matrix for [N,2] coordinates -> [N,N]."""
        # Use (x - y)^2 = x^2 + y^2 - 2xy trick
        x2 = (xy**2).sum(dim=1, keepdim=True)  # [N,1]
        y2 = x2.t()                             # [1,N]
        xy_inner = xy @ xy.t()                  # [N,N]
        d2 = torch.clamp(x2 + y2 - 2.0 * xy_inner, min=0.0)
        return torch.sqrt(d2 + 1e-9)

    def forward(self, hidden: torch.Tensor, positions_xy: torch.Tensor) -> torch.Tensor:
        """
        Args:
            hidden: [B, N, T, E] encoder hidden states per VD
            positions_xy: [N, 2] static XY positions for kNN

        Returns:
            social_context: [B, N, E]
        """
        if hidden.dim() != 4:
            raise ValueError(f"hidden must be [B,N,T,E], got {tuple(hidden.shape)}")
        if positions_xy.dim() != 2 or positions_xy.size(1) != 2:
            raise ValueError(f"positions_xy must be [N,2], got {tuple(positions_xy.shape)}")

        B, N, T, E = hidden.shape
        device = hidden.device

        # Query: target VD last timestep
        h_last = hidden[:, :, -1, :]  # [B, N, E]

        # kNN on positions (no batch dimension)
        with torch.no_grad():
            D = self._pairwise_distances(positions_xy.to(device=device, dtype=hidden.dtype))  # [N,N]
            # Exclude self
            D.fill_diagonal_(float('inf'))
            # topk smallest distances
            k = min(self.knn_k, max(1, N - 1))
            knn_dists, knn_idx = torch.topk(D, k=k, dim=1, largest=False)  # [N,k]

        # Gather neighbor sequence representations: use last L timesteps mean
        L = min(self.time_window, T)
        if L <= 0:
            raise ValueError("time_window must be >= 1")
        neigh_seq = hidden[:, :, -L:, :]  # [B,N,L,E]

        # Gather neighbors for each target node i -> [B, N, k, L, E]
        # Expand indices for batch gather
        knn_idx_exp = knn_idx.unsqueeze(0).unsqueeze(-1).unsqueeze(-1).expand(B, N, k, L, E)
        # Prepare source for gather along N dim (dim=1)
        # To gather neighbors in one go, we reshape to merge (N,L) dims then unmerge after
        neigh_seq_expand = neigh_seq.unsqueeze(2).expand(B, N, N, L, E)
        neigh_repr = torch.gather(neigh_seq_expand, dim=2, index=knn_idx_exp).contiguous()  # [B,N,k,L,E]
        # Temporal compression: mean over last L steps -> [B,N,k,E]
        neigh_repr = neigh_repr.mean(dim=3)

        # Distance bias with local normalization to stabilize scale:
        # d_hat = d / mean_k(d), then log_w = - d_hat / tau
        eps = torch.finfo(hidden.dtype).eps
        local_scale = knn_dists.mean(dim=1, keepdim=True).clamp_min(eps)  # [N,1]
        d_hat = (knn_dists / local_scale).to(device=device, dtype=hidden.dtype)  # [N,k]
        tau_safe = torch.clamp(self.tau.to(device=device, dtype=hidden.dtype), min=1e-4)
        log_w = -d_hat / tau_safe  # [N,k]
        if self.use_radius_mask and self.radius > 0.0:
            in_radius = (knn_dists.to(device) <= self.radius)
            # Large negative bias for out-of-radius neighbors (approximate -inf)
            log_w = torch.where(in_radius, log_w, torch.full_like(log_w, -10.0))
        # Optional global scaling and clamp for stability
        log_w = (self.bias_scale * log_w).clamp(min=-10.0, max=0.0)

        # Projections
        Q = self.q_proj(h_last)                      # [B,N,proj]
        K = self.k_proj(neigh_repr)                  # [B,N,k,proj]
        V = self.v_proj(neigh_repr)                  # [B,N,k,proj]

        # Reshape to heads
        Hh = self.heads
        Dh = self.head_dim
        Q = Q.view(B, N, Hh, Dh).transpose(2, 3)     # [B,N,Dh,Hh]
        K = K.view(B, N, k, Hh, Dh).permute(0, 1, 3, 2, 4)  # [B,N,Hh,k,Dh]
        V = V.view(B, N, k, Hh, Dh).permute(0, 1, 3, 2, 4)  # [B,N,Hh,k,Dh]

        # Attention logits: Q·K^T / sqrt(d) + log_w bias
        # Q: [B,N,Dh,Hh], K: [B,N,Hh,k,Dh] -> logits: [B,N,Hh,k]
        # Compute (Q^T * K^T) along Dh
        Qt = Q.transpose(2, 3).unsqueeze(-2)         # [B,N,Hh,1,Dh]
        logits = torch.matmul(Qt, K.transpose(-1, -2)).squeeze(-2)  # [B,N,Hh,k]
        logits = logits / math.sqrt(Dh)

        # Add distance bias
        bias = log_w.view(1, N, 1, k)
        logits = logits + bias

        attn = torch.softmax(logits, dim=-1)         # [B,N,Hh,k]
        attn = self.dropout(attn)

        # Weighted sum over V: [B,N,Hh,k,Dh]
        context_heads = torch.matmul(attn.unsqueeze(-2), V).squeeze(-2)  # [B,N,Hh,Dh]
        context = context_heads.transpose(2, 3).contiguous().view(B, N, Hh * Dh)  # [B,N,proj]
        context = self.out_proj(context)             # [B,N,E]
        return context
