'''Minimal SmartGuard backbone training entry point.

This trains only the reconstruction backbone from JSONL BehaviorSequence files.
It is intentionally small, CPU-only, and suitable for toy runs. It does not call
source SmartGuard code and does not launch any anomaly detector training.
'''

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from causal_gen_guard.data.behavior_event_tensor import build_vocab
from causal_gen_guard.data.schemas import BehaviorSequence
from causal_gen_guard.models.smartguard_backbone import SmartGuardBackbone, compute_reconstruction_loss


def _control_values(control_id: Any) -> List[Any]:
    if isinstance(control_id, (str, bytes)):
        return [control_id]
    if isinstance(control_id, dict):
        return [repr(control_id)]
    if isinstance(control_id, Iterable):
        values = list(control_id)
        return values if values else [None]
    return [control_id]


def _as_float(value: Any, default: float = 0.0) -> float:
    if value is None:
        return default
    if hasattr(value, 'item'):
        try:
            value = value.item()
        except Exception:
            pass
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def load_jsonl_sequences(path: str | Path, max_sequences: Optional[int] = None) -> List[BehaviorSequence]:
    '''Load BehaviorSequence records from JSONL.'''
    path = Path(path)
    sequences: List[BehaviorSequence] = []
    with path.open('r', encoding='utf-8') as handle:
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


def sequence_to_smartguard_input(sequence: BehaviorSequence, vocab: Dict[Any, int], max_len: int) -> Tuple[torch.Tensor, torch.Tensor]:
    '''Convert one sequence to [T, 4] day/hour/duration/control_index plus mask.'''
    fields = torch.zeros(max_len, 4, dtype=torch.float32)
    mask = torch.zeros(max_len, dtype=torch.float32)
    for index, event in enumerate(sequence.events[:max_len]):
        control_key = _control_values(event.control_id)[0]
        if control_key not in vocab:
            raise KeyError(f'control_id {control_key!r} is missing from vocab')
        fields[index, 0] = _as_float(event.day)
        fields[index, 1] = _as_float(event.hour)
        fields[index, 2] = _as_float(event.duration)
        fields[index, 3] = float(vocab[control_key])
        mask[index] = 1.0
    return fields, mask


class SequenceTensorDataset(Dataset):
    '''Small in-memory dataset for BehaviorSequence reconstruction training.'''

    def __init__(self, sequences: List[BehaviorSequence], vocab: Dict[Any, int], max_len: int) -> None:
        self.inputs: List[torch.Tensor] = []
        self.masks: List[torch.Tensor] = []
        for sequence in sequences:
            fields, mask = sequence_to_smartguard_input(sequence, vocab, max_len)
            self.inputs.append(fields)
            self.masks.append(mask)

    def __len__(self) -> int:
        return len(self.inputs)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.inputs[index], self.masks[index]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def train(args: argparse.Namespace) -> Dict[str, Any]:
    '''Train the reconstruction backbone and save checkpoint artifacts.'''
    set_seed(args.seed)
    sequences = load_jsonl_sequences(args.input_jsonl, max_sequences=args.max_sequences)
    vocab = build_vocab(sequences)
    if args.max_len is None:
        max_len = max(len(sequence.events) for sequence in sequences)
    else:
        max_len = args.max_len
    if max_len <= 0:
        raise ValueError('max_len must be positive')

    dataset = SequenceTensorDataset(sequences, vocab, max_len)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    model = SmartGuardBackbone(
        vocab_size=len(vocab),
        hidden_dim=args.hidden_dim,
        nhead=args.nhead,
        num_layers=args.num_layers,
        dropout=args.dropout,
        use_ttpe=not args.disable_ttpe,
    )
    device = torch.device('cpu')
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    loss_sums = torch.zeros(len(vocab), dtype=torch.float32)
    loss_counts = torch.zeros(len(vocab), dtype=torch.float32)
    history: List[Dict[str, float]] = []

    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0
        running_tokens = 0.0
        loss_sums.zero_()
        loss_counts.zero_()
        for input_batch, mask_batch in loader:
            input_batch = input_batch.to(device)
            mask_batch = mask_batch.to(device)
            optimizer.zero_grad()
            outputs = model(input_batch, attention_mask=mask_batch)
            targets = input_batch[..., 3].to(dtype=torch.long)
            loss = compute_reconstruction_loss(outputs['logits'], targets, mask=mask_batch, reduction='mean')
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                token_losses = outputs['token_losses'].detach().cpu()
                targets_cpu = targets.detach().cpu()
                mask_cpu = mask_batch.detach().cpu() > 0
                valid_losses = token_losses[mask_cpu]
                valid_targets = targets_cpu[mask_cpu]
                ones = torch.ones_like(valid_losses, dtype=torch.float32)
                if valid_losses.numel() > 0:
                    loss_sums.scatter_add_(0, valid_targets, valid_losses)
                    loss_counts.scatter_add_(0, valid_targets, ones)
                    running_loss += float(valid_losses.sum().item())
                    running_tokens += float(valid_losses.numel())
        mean_loss = running_loss / max(running_tokens, 1.0)
        history.append({'epoch': float(epoch + 1), 'train_loss': float(mean_loss)})
        print(f'Epoch {epoch + 1}/{args.epochs} train_loss={mean_loss:.6f}')

    loss_vector = loss_sums / loss_counts.clamp_min(1.0)

    checkpoint_path = Path(args.checkpoint_path)
    loss_vector_path = Path(args.loss_vector_output)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    loss_vector_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            'model_state_dict': model.state_dict(),
            'vocab': dict(vocab),
            'inverse_vocab': list(vocab.inverse_vocab),
            'max_len': max_len,
            'model_config': {
                'vocab_size': len(vocab),
                'hidden_dim': args.hidden_dim,
                'nhead': args.nhead,
                'num_layers': args.num_layers,
                'dropout': args.dropout,
                'use_ttpe': not args.disable_ttpe,
            },
            'history': history,
        },
        checkpoint_path,
    )
    np.save(loss_vector_path, loss_vector.numpy())
    print(f'Saved checkpoint to {checkpoint_path}')
    print(f'Saved loss vector to {loss_vector_path}')
    return {'checkpoint_path': str(checkpoint_path), 'loss_vector_path': str(loss_vector_path), 'history': history}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Train the CausalGenGuard SmartGuard reconstruction backbone.')
    parser.add_argument('--input-jsonl', required=True, help='JSONL file produced by prepare_smartguard_data.py or toy data.')
    parser.add_argument('--checkpoint-path', default='outputs/checkpoints/smartguard_backbone.pt')
    parser.add_argument('--loss-vector-output', default='outputs/results/loss_vector.npy')
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--hidden-dim', type=int, default=64)
    parser.add_argument('--nhead', type=int, default=4)
    parser.add_argument('--num-layers', type=int, default=1)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--learning-rate', type=float, default=1e-3)
    parser.add_argument('--max-len', type=int, default=None)
    parser.add_argument('--max-sequences', type=int, default=None)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--disable-ttpe', action='store_true')
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if args.epochs <= 0:
        raise ValueError('epochs must be positive')
    if args.batch_size <= 0:
        raise ValueError('batch-size must be positive')
    train(args)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
