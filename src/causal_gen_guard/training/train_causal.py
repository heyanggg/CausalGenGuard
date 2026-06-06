'''Minimal training entry point for the behavior causal predictor.

This stage reads normalized BehaviorSequence JSONL records, builds event-tensor
sliding windows, trains a next-event multi-label predictor, and saves a normal
causal pattern estimated from gradient causality. It does not run a full GCAD
experiment and does not use source GCAD dataloaders.
'''

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from causal_gen_guard.data.behavior_event_tensor import build_vocab, sequence_to_event_tensor, sliding_windows_from_tensor
from causal_gen_guard.data.schemas import BehaviorSequence
from causal_gen_guard.models.causal_graph import (
    compute_gradient_causality,
    graph_centrality,
    normal_causal_pattern,
    sparsify_causality,
)
from causal_gen_guard.models.causal_predictor import BehaviorCausalPredictor


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_jsonl_sequences(path: str, max_sequences: Optional[int] = None) -> List[BehaviorSequence]:
    sequences: List[BehaviorSequence] = []
    with Path(path).open('r', encoding='utf-8') as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                sequences.append(BehaviorSequence.from_dict(json.loads(line)))
            except Exception as exc:
                raise ValueError(f'Failed to parse {path}:{line_number}: {exc}') from exc
            if max_sequences is not None and len(sequences) >= max_sequences:
                break
    if not sequences:
        raise ValueError(f'No sequences loaded from {path}')
    return sequences


def _windows_for_sequence(sequence: BehaviorSequence, vocab: Dict[object, int], window_size: int, pred_horizon: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    X, time_features, mask = sequence_to_event_tensor(sequence, vocab, max_len=None, include_time_features=True)
    valid_len = int(mask.sum())
    X = X[:valid_len]
    time_features = time_features[:valid_len]
    windows, targets = sliding_windows_from_tensor(X, window_size=window_size, pred_horizon=pred_horizon)
    count = windows.shape[0]
    time_windows = np.zeros((count, window_size, time_features.shape[-1]), dtype=np.float32)
    for index in range(count):
        time_windows[index] = time_features[index : index + window_size]
    return windows.astype(np.float32), time_windows.astype(np.float32), targets.astype(np.float32)


def build_window_arrays(
    sequences: List[BehaviorSequence],
    vocab: Dict[object, int],
    window_size: int,
    pred_horizon: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    all_windows: List[np.ndarray] = []
    all_time_windows: List[np.ndarray] = []
    all_targets: List[np.ndarray] = []
    for sequence in sequences:
        windows, time_windows, targets = _windows_for_sequence(sequence, vocab, window_size, pred_horizon)
        if windows.shape[0] == 0:
            continue
        all_windows.append(windows)
        all_time_windows.append(time_windows)
        all_targets.append(targets)
    if not all_windows:
        raise ValueError('No sliding windows were created; reduce window_size or pred_horizon')
    return np.concatenate(all_windows, axis=0), np.concatenate(all_time_windows, axis=0), np.concatenate(all_targets, axis=0)


class WindowDataset(Dataset):
    def __init__(self, windows: np.ndarray, time_windows: np.ndarray, targets: np.ndarray) -> None:
        self.windows = torch.from_numpy(windows).float()
        self.time_windows = torch.from_numpy(time_windows).float()
        self.targets = torch.from_numpy(targets).float()

    def __len__(self) -> int:
        return self.windows.shape[0]

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.windows[index], self.time_windows[index], self.targets[index]


def train(args: argparse.Namespace) -> Dict[str, str]:
    set_seed(args.seed)
    sequences = load_jsonl_sequences(args.input_jsonl, max_sequences=args.max_sequences)
    vocab = build_vocab(sequences)
    windows, time_windows, targets = build_window_arrays(sequences, vocab, args.window_size, args.pred_horizon)
    if not args.include_time_features:
        time_windows = np.zeros((windows.shape[0], windows.shape[1], 0), dtype=np.float32)

    dataset = WindowDataset(windows, time_windows, targets)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    model = BehaviorCausalPredictor(
        input_channels=len(vocab),
        hidden_dim=args.hidden_dim,
        time_feature_dim=time_windows.shape[-1],
        num_layers=args.num_layers,
        dropout=args.dropout,
    )
    device = torch.device('cpu')
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        total_count = 0
        for batch_windows, batch_time, batch_targets in loader:
            batch_windows = batch_windows.to(device)
            batch_time = batch_time.to(device)
            batch_targets = batch_targets.to(device)
            optimizer.zero_grad()
            pred_logits = model(batch_windows, batch_time if batch_time.shape[-1] else None)
            loss = model.compute_loss(pred_logits, batch_targets)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.detach().item()) * batch_windows.size(0)
            total_count += int(batch_windows.size(0))
        print(f'Epoch {epoch + 1}/{args.epochs} causal_loss={total_loss / max(total_count, 1):.6f}')

    sample_count = min(args.causality_samples, len(dataset))
    sample_windows = dataset.windows[:sample_count].to(device)
    sample_targets = dataset.targets[:sample_count].to(device)
    graphs = compute_gradient_causality(model, sample_windows, sample_targets)
    sparse_graphs = sparsify_causality(graphs, threshold=args.sparse_threshold)
    A_norm = normal_causal_pattern(sparse_graphs, method=args.pattern_method)
    centrality = graph_centrality(A_norm)

    checkpoint_path = Path(args.checkpoint_path)
    graph_path = Path(args.graph_output)
    centrality_path = Path(args.centrality_output)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    graph_path.parent.mkdir(parents=True, exist_ok=True)
    centrality_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            'model_state_dict': model.state_dict(),
            'vocab': dict(vocab),
            'inverse_vocab': list(vocab.inverse_vocab),
            'model_config': {
                'input_channels': len(vocab),
                'hidden_dim': args.hidden_dim,
                'time_feature_dim': time_windows.shape[-1],
                'num_layers': args.num_layers,
                'dropout': args.dropout,
            },
            'window_size': args.window_size,
            'pred_horizon': args.pred_horizon,
        },
        checkpoint_path,
    )
    np.save(graph_path, A_norm.detach().cpu().numpy())
    np.save(centrality_path, centrality.detach().cpu().numpy())
    print(f'Saved causal predictor checkpoint to {checkpoint_path}')
    print(f'Saved normal causal graph to {graph_path}')
    print(f'Saved causal centrality to {centrality_path}')
    return {'checkpoint_path': str(checkpoint_path), 'graph_output': str(graph_path), 'centrality_output': str(centrality_path)}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Train behavior causal predictor and estimate normal causal graph.')
    parser.add_argument('--input-jsonl', required=True)
    parser.add_argument('--checkpoint-path', default='outputs/checkpoints/causal_predictor.pt')
    parser.add_argument('--graph-output', default='outputs/results/A_norm.npy')
    parser.add_argument('--centrality-output', default='outputs/results/causal_centrality.npy')
    parser.add_argument('--window-size', type=int, default=4)
    parser.add_argument('--pred-horizon', type=int, default=1)
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--hidden-dim', type=int, default=64)
    parser.add_argument('--num-layers', type=int, default=1)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--learning-rate', type=float, default=1e-3)
    parser.add_argument('--sparse-threshold', type=float, default=0.0)
    parser.add_argument('--causality-samples', type=int, default=64)
    parser.add_argument('--pattern-method', choices=['mean', 'median'], default='mean')
    parser.add_argument('--max-sequences', type=int, default=None)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--include-time-features', action='store_true')
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if args.window_size <= 0:
        raise ValueError('window-size must be positive')
    if args.pred_horizon <= 0:
        raise ValueError('pred-horizon must be positive')
    if args.epochs <= 0:
        raise ValueError('epochs must be positive')
    if args.causality_samples <= 0:
        raise ValueError('causality-samples must be positive')
    train(args)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
