'''SmartGuard-style reconstruction backbone for CausalGenGuard.

This module is adapted from the local SmartGuard project into a standalone,
importable PyTorch module. It keeps the core SmartGuard idea: reconstruct device
control tokens from behavior sequences using time-aware positional embeddings
(TTPE) and a Transformer encoder-decoder. It does not import from or modify the
source SmartGuard project.
'''

from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class TimeAwarePositionalEmbedding(nn.Module):
    '''Time-aware positional embedding for day, hour, duration, and order.'''

    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.day_weight = nn.Parameter(torch.tensor(0.4, dtype=torch.float32))
        self.hour_weight = nn.Parameter(torch.tensor(0.4, dtype=torch.float32))
        self.duration_weight = nn.Parameter(torch.tensor(0.7, dtype=torch.float32))
        self.order_weight = nn.Parameter(torch.tensor(0.1, dtype=torch.float32))

    def _sinusoidal(self, values: torch.Tensor) -> torch.Tensor:
        values = values.to(dtype=torch.float32)
        device = values.device
        even_count = (self.hidden_dim + 1) // 2
        div_term = torch.exp(
            torch.arange(0, even_count, dtype=torch.float32, device=device)
            * (-(math.log(10000.0) / max(self.hidden_dim, 1)))
            * 2.0
        )
        scaled = values.unsqueeze(-1) * div_term
        embedding = torch.zeros(*values.shape, self.hidden_dim, dtype=torch.float32, device=device)
        embedding[..., 0::2] = torch.sin(scaled)
        if self.hidden_dim > 1:
            embedding[..., 1::2] = torch.cos(scaled[..., : embedding[..., 1::2].shape[-1]])
        return embedding

    def forward(
        self,
        behavior_fields: torch.Tensor,
        duration_input: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        '''Return TTPE embeddings with shape [B, T, hidden_dim].'''
        if behavior_fields.dim() != 3 or behavior_fields.size(-1) != 4:
            raise ValueError('behavior_fields must have shape [B, T, 4]')

        day = behavior_fields[..., 0].to(dtype=torch.float32)
        hour = behavior_fields[..., 1].to(dtype=torch.float32)
        duration = behavior_fields[..., 2].to(dtype=torch.float32)
        if duration_input is not None:
            if duration_input.dim() == 3 and duration_input.size(-1) == self.hidden_dim:
                duration_embedding = duration_input.to(dtype=torch.float32, device=behavior_fields.device)
            else:
                duration = duration_input.squeeze(-1).to(dtype=torch.float32, device=behavior_fields.device)
                duration_embedding = self._sinusoidal(duration)
        else:
            duration_embedding = self._sinusoidal(duration)

        batch_size, seq_len, _ = behavior_fields.shape
        order = torch.arange(seq_len, dtype=torch.float32, device=behavior_fields.device).unsqueeze(0).expand(batch_size, -1)

        return (
            self.day_weight * self._sinusoidal(day)
            + self.hour_weight * self._sinusoidal(hour)
            + self.duration_weight * duration_embedding
            + self.order_weight * self._sinusoidal(order)
        )


class SmartGuardBackbone(nn.Module):
    '''Transformer encoder-decoder backbone for control reconstruction.'''

    def __init__(
        self,
        vocab_size: int,
        hidden_dim: int = 128,
        nhead: int = 4,
        num_layers: int = 2,
        dropout: float = 0.1,
        use_ttpe: bool = True,
    ) -> None:
        super().__init__()
        if vocab_size <= 0:
            raise ValueError('vocab_size must be positive')
        if hidden_dim % nhead != 0:
            raise ValueError('hidden_dim must be divisible by nhead')
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.use_ttpe = use_ttpe
        self.control_embedding = nn.Embedding(vocab_size, hidden_dim)
        self.time_embedding = TimeAwarePositionalEmbedding(hidden_dim)
        self.transformer = nn.Transformer(
            d_model=hidden_dim,
            nhead=nhead,
            num_encoder_layers=num_layers,
            num_decoder_layers=num_layers,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True,
        )
        self.output_projection = nn.Linear(hidden_dim, vocab_size)

    def _normalize_input(self, input_ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if input_ids.dim() == 2:
            if input_ids.size(1) % 4 != 0:
                raise ValueError('flat SmartGuard input must have a feature dimension divisible by 4')
            behavior_fields = input_ids.reshape(input_ids.size(0), input_ids.size(1) // 4, 4)
        elif input_ids.dim() == 3 and input_ids.size(-1) == 4:
            behavior_fields = input_ids
        else:
            raise ValueError('input_ids must have shape [B, 40] or [B, T, 4]')

        behavior_fields = behavior_fields.to(dtype=torch.float32)
        targets = behavior_fields[..., 3].to(dtype=torch.long)
        if targets.numel() > 0:
            min_target = int(targets.min().item())
            max_target = int(targets.max().item())
            if min_target < 0 or max_target >= self.vocab_size:
                raise ValueError(
                    f'control ids must be in [0, {self.vocab_size - 1}], got min={min_target}, max={max_target}'
                )
        return behavior_fields, targets

    def forward(
        self,
        input_ids: torch.Tensor,
        duration_input: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        '''Run reconstruction and return logits, hidden_states, and token_losses.'''
        behavior_fields, targets = self._normalize_input(input_ids)
        controls = targets
        hidden_states = self.control_embedding(controls)
        if self.use_ttpe:
            hidden_states = hidden_states + self.time_embedding(behavior_fields, duration_input=duration_input)

        key_padding_mask = None
        if attention_mask is not None:
            key_padding_mask = attention_mask.to(device=input_ids.device) <= 0

        decoded = self.transformer(
            hidden_states,
            hidden_states,
            src_key_padding_mask=key_padding_mask,
            tgt_key_padding_mask=key_padding_mask,
            memory_key_padding_mask=key_padding_mask,
        )
        logits = self.output_projection(decoded)
        token_losses = compute_reconstruction_loss(logits, targets, mask=attention_mask, reduction='none')
        return {
            'logits': logits,
            'hidden_states': decoded,
            'token_losses': token_losses,
        }


def compute_reconstruction_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    reduction: str = 'none',
) -> torch.Tensor:
    '''Compute per-token control reconstruction cross entropy.'''
    if logits.dim() != 3:
        raise ValueError('logits must have shape [B, T, vocab_size]')
    if targets.shape != logits.shape[:2]:
        raise ValueError('targets must have shape [B, T] matching logits')
    losses = F.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        targets.to(dtype=torch.long).reshape(-1),
        reduction='none',
    ).reshape(targets.shape)

    weights = None
    if mask is not None:
        weights = mask.to(dtype=losses.dtype, device=losses.device)
        losses = losses * weights

    if reduction == 'none':
        return losses
    if reduction == 'sum':
        return losses.sum()
    if reduction == 'mean':
        if weights is None:
            return losses.mean()
        return losses.sum() / weights.sum().clamp_min(1.0)
    raise ValueError('reduction must be none, sum, or mean')
