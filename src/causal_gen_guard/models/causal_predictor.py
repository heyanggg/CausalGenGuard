'''Behavior causal predictor for discrete event tensors.

The predictor consumes windows with shape [B, tau, C] and predicts the next
multi-label event vector [B, C]. It is intentionally lightweight and CPU-friendly
for this stage; causal graph discovery is handled by gradients in causal_graph.py.
'''

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class BehaviorCausalPredictor(nn.Module):
    '''GRU predictor for next-event behavior channels.'''

    def __init__(
        self,
        input_channels: int,
        hidden_dim: int = 64,
        time_feature_dim: int = 0,
        num_layers: int = 1,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if input_channels <= 0:
            raise ValueError('input_channels must be positive')
        if hidden_dim <= 0:
            raise ValueError('hidden_dim must be positive')
        if num_layers <= 0:
            raise ValueError('num_layers must be positive')
        if time_feature_dim < 0:
            raise ValueError('time_feature_dim must be non-negative')
        self.input_channels = input_channels
        self.hidden_dim = hidden_dim
        self.time_feature_dim = time_feature_dim
        self.num_layers = num_layers
        self.input_projection = nn.Linear(input_channels + time_feature_dim, hidden_dim)
        gru_dropout = dropout if num_layers > 1 else 0.0
        self.gru = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=gru_dropout,
            batch_first=True,
        )
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, input_channels),
        )

    def forward(self, windows: torch.Tensor, time_features: Optional[torch.Tensor] = None) -> torch.Tensor:
        '''Predict next-event logits from behavior windows.'''
        if windows.dim() != 3:
            raise ValueError('windows must have shape [B, tau, C]')
        if windows.size(-1) != self.input_channels:
            raise ValueError(f'expected {self.input_channels} input channels, got {windows.size(-1)}')
        windows = windows.to(dtype=torch.float32)
        if self.time_feature_dim > 0:
            if time_features is None:
                time_features = torch.zeros(
                    windows.size(0),
                    windows.size(1),
                    self.time_feature_dim,
                    dtype=windows.dtype,
                    device=windows.device,
                )
            if time_features.shape[:2] != windows.shape[:2] or time_features.size(-1) != self.time_feature_dim:
                raise ValueError(
                    'time_features must have shape [B, tau, time_feature_dim] matching windows when enabled'
                )
            model_input = torch.cat([windows, time_features.to(dtype=torch.float32, device=windows.device)], dim=-1)
        else:
            model_input = windows

        projected = F.relu(self.input_projection(model_input))
        _, hidden = self.gru(projected)
        return self.head(hidden[-1])

    def compute_loss(self, pred_logits: torch.Tensor, targets: torch.Tensor, reduction: str = 'mean') -> torch.Tensor:
        '''Compute BCE-with-logits loss for one-hot or multi-hot next events.'''
        return F.binary_cross_entropy_with_logits(pred_logits, targets.to(dtype=torch.float32), reduction=reduction)
