'''Loss and weighting utilities for CausalGenGuard.

These helpers are small CPU-safe tensor transforms. They do not train models by
themselves; training code may use them to down-weight noisy reconstruction
signals and recover high-causal-centrality behavior events.
'''

from __future__ import annotations

from typing import Any

import torch


def _float_tensor(values: Any) -> torch.Tensor:
    if isinstance(values, torch.Tensor):
        return values.to(dtype=torch.float32)
    return torch.as_tensor(values, dtype=torch.float32)


def _minmax_normalize(values: torch.Tensor) -> torch.Tensor:
    values = values.to(dtype=torch.float32)
    span = values.max() - values.min()
    if span.abs() < 1e-12:
        return torch.zeros_like(values)
    return (values - values.min()) / span


def noise_aware_weight_from_loss_vector(loss_vector: Any, mu: float) -> torch.Tensor:
    '''Convert reconstruction losses into noise-aware weights.

    Higher reconstruction loss is treated as less reliable evidence, so its base
    weight decreases. The output is in (0, 1], with mu controlling sharpness.
    '''
    losses = _float_tensor(loss_vector)
    normalized_loss = _minmax_normalize(losses)
    return torch.exp(-float(mu) * normalized_loss)


def causal_aware_noise_weight(
    loss_vector: Any,
    causal_centrality: Any,
    mu: float,
    gamma: float,
    min_w: float,
    max_w: float,
) -> torch.Tensor:
    '''Blend noise-aware reconstruction weights with causal centrality.

    High-loss, low-centrality events stay suppressed. High-loss events that are
    also causally central are boosted by the centrality term and clipped to the
    requested weight range.
    '''
    losses = _float_tensor(loss_vector)
    centrality = _float_tensor(causal_centrality).to(device=losses.device)
    if losses.shape != centrality.shape:
        raise ValueError('loss_vector and causal_centrality must have the same shape')
    if min_w > max_w:
        raise ValueError('min_w must be <= max_w')

    base_weight = noise_aware_weight_from_loss_vector(losses, mu).to(device=losses.device)
    centrality_norm = _minmax_normalize(centrality)
    boosted = base_weight * (1.0 + float(gamma) * centrality_norm)
    return boosted.clamp(min=float(min_w), max=float(max_w))
