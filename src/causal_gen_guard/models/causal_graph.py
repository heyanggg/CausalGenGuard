'''Causal graph utilities for discrete behavior event tensors.

This module adapts the GCAD gradient-causality idea to [B, tau, C] event
windows. Each target channel is backpropagated independently and absolute input
gradients are accumulated over lag positions to estimate source -> target
influence.
'''

from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn.functional as F


def _to_tensor(values: Any) -> torch.Tensor:
    if isinstance(values, torch.Tensor):
        return values.to(dtype=torch.float32)
    return torch.as_tensor(values, dtype=torch.float32)


def _restore_type(reference: Any, tensor: torch.Tensor) -> Any:
    if isinstance(reference, np.ndarray):
        return tensor.detach().cpu().numpy()
    return tensor


def compute_channel_losses(pred_logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    '''Return BCE-with-logits losses with shape [B, C].'''
    if pred_logits.shape != targets.shape:
        raise ValueError('pred_logits and targets must have the same shape [B, C]')
    return F.binary_cross_entropy_with_logits(pred_logits, targets.to(dtype=torch.float32), reduction='none')


def compute_gradient_causality(model: torch.nn.Module, windows: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    '''Compute batched gradient causality matrices with shape [B, C, C].

    Matrix entry A[b, i, j] is the lag-aggregated absolute gradient from source
    channel i in the input window to target channel j in the predicted event.
    '''
    if windows.dim() != 3:
        raise ValueError('windows must have shape [B, tau, C]')
    if targets.dim() != 2 or targets.shape[0] != windows.shape[0] or targets.shape[1] != windows.shape[2]:
        raise ValueError('targets must have shape [B, C] matching windows')

    was_training = model.training
    model.eval()
    model.zero_grad(set_to_none=True)
    causal_windows = windows.detach().clone().to(dtype=torch.float32)
    causal_windows.requires_grad_(True)
    target_tensor = targets.detach().clone().to(dtype=torch.float32, device=causal_windows.device)
    pred_logits = model(causal_windows)
    channel_losses = compute_channel_losses(pred_logits, target_tensor)

    batch_size, _, channel_count = causal_windows.shape
    matrices = torch.zeros(batch_size, channel_count, channel_count, dtype=torch.float32, device=causal_windows.device)
    for target_channel in range(channel_count):
        model.zero_grad(set_to_none=True)
        if causal_windows.grad is not None:
            causal_windows.grad.zero_()
        loss_j = channel_losses[:, target_channel].sum()
        grad = torch.autograd.grad(
            loss_j,
            causal_windows,
            retain_graph=target_channel < channel_count - 1,
            create_graph=False,
            allow_unused=False,
        )[0]
        matrices[:, :, target_channel] = grad.detach().abs().sum(dim=1)

    if was_training:
        model.train()
    return matrices


def sparsify_causality(A: Any, threshold: float) -> Any:
    '''Keep positive asymmetric causality above threshold.'''
    tensor = _to_tensor(A)
    if tensor.dim() not in (2, 3):
        raise ValueError('A must have shape [C, C] or [B, C, C]')
    asymmetric = torch.clamp(tensor - tensor.transpose(-1, -2), min=0.0)
    sparse = torch.where(asymmetric >= float(threshold), asymmetric, torch.zeros_like(asymmetric))
    return _restore_type(A, sparse)


def normal_causal_pattern(graphs: Any, method: str = 'mean') -> Any:
    '''Compute a normal causal pattern from graph samples.'''
    tensor = _to_tensor(graphs)
    if tensor.dim() == 2:
        pattern = tensor
    elif tensor.dim() == 3:
        if method == 'mean':
            pattern = tensor.mean(dim=0)
        elif method == 'median':
            pattern = tensor.median(dim=0).values
        else:
            raise ValueError('method must be mean or median')
    else:
        raise ValueError('graphs must have shape [C, C] or [B, C, C]')
    return _restore_type(graphs, pattern)


def causal_deviation_score(A_test: Any, A_norm: Any, beta: float = 0.0) -> Any:
    '''Score deviation from the normal causal pattern.'''
    test = _to_tensor(A_test)
    norm = _to_tensor(A_norm).to(device=test.device)
    if test.dim() == 2:
        diff = test - norm
        score = torch.linalg.vector_norm(diff)
        if beta:
            score = score + float(beta) * torch.abs(test.sum() - norm.sum()) / max(test.numel(), 1)
    elif test.dim() == 3:
        diff = test - norm.unsqueeze(0)
        score = torch.linalg.vector_norm(diff, dim=(1, 2))
        if beta:
            density_gap = torch.abs(test.sum(dim=(1, 2)) - norm.sum()) / max(norm.numel(), 1)
            score = score + float(beta) * density_gap
    else:
        raise ValueError('A_test must have shape [C, C] or [B, C, C]')
    return _restore_type(A_test, score)


def graph_centrality(A_norm: Any) -> Any:
    '''Return in+out weighted centrality for each behavior channel.'''
    tensor = _to_tensor(A_norm)
    if tensor.dim() != 2:
        raise ValueError('A_norm must have shape [C, C]')
    centrality = tensor.sum(dim=0) + tensor.sum(dim=1)
    return _restore_type(A_norm, centrality)
