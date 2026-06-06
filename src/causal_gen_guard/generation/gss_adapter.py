'''Graph and transition hints for offline SmartGen-style prompting.

GSS here means lightweight graph-structure support: transition statistics from
observed behavior and causal edges from CausalGenGuard's A_norm matrix.
'''

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any, Dict, List

import numpy as np

from causal_gen_guard.data.behavior_event_tensor import build_vocab
from causal_gen_guard.data.schemas import BehaviorSequence


def _control_key(control_id: Any) -> Any:
    if hasattr(control_id, 'item'):
        try:
            control_id = control_id.item()
        except Exception:
            pass
    try:
        hash(control_id)
    except TypeError:
        return repr(control_id)
    return control_id


def build_transition_matrix(sequences: List[BehaviorSequence], top_k: int = 5) -> Dict[str, Any]:
    '''Build channel transition counts and top transitions from sequences.'''
    if top_k <= 0:
        raise ValueError('top_k must be positive')
    vocab = build_vocab(sequences)
    channel_count = len(vocab)
    matrix = np.zeros((channel_count, channel_count), dtype=np.float32)
    counts: Dict[int, Counter] = defaultdict(Counter)
    for sequence in sequences:
        controls = [_control_key(event.control_id) for event in sequence.events]
        for src, dst in zip(controls, controls[1:]):
            if src in vocab and dst in vocab:
                src_idx = vocab[src]
                dst_idx = vocab[dst]
                matrix[src_idx, dst_idx] += 1.0
                counts[src_idx][dst_idx] += 1
    top_transitions: Dict[int, List[tuple[int, int]]] = {}
    for src_idx, counter in counts.items():
        top_transitions[src_idx] = counter.most_common(top_k)
    return {
        'matrix': matrix,
        'vocab': dict(vocab),
        'inverse_vocab': list(vocab.inverse_vocab),
        'top_k': top_k,
        'top_transitions': top_transitions,
    }


def _label(inverse_vocab: List[Any], index: int) -> str:
    if 0 <= index < len(inverse_vocab):
        return str(inverse_vocab[index])
    return str(index)


def export_json_hints(matrix: Any, inverse_vocab: List[Any] | None = None) -> List[Dict[str, Any]]:
    '''Export high-frequency transition hints as JSON-serializable records.'''
    if isinstance(matrix, dict):
        inverse = matrix.get('inverse_vocab', inverse_vocab or [])
        top_transitions = matrix.get('top_transitions', {})
        counts = matrix.get('matrix')
        hints: List[Dict[str, Any]] = []
        for src_idx, transitions in top_transitions.items():
            total = float(counts[src_idx].sum()) if counts is not None else float(sum(count for _, count in transitions))
            for dst_idx, count in transitions:
                hints.append(
                    {
                        'from': _label(inverse, int(src_idx)),
                        'to': _label(inverse, int(dst_idx)),
                        'count': int(count),
                        'probability': float(count) / total if total else 0.0,
                    }
                )
        return hints

    array = np.asarray(matrix)
    inverse = inverse_vocab or list(range(array.shape[0]))
    hints = []
    for src_idx in range(array.shape[0]):
        total = float(array[src_idx].sum())
        ranked = np.argsort(array[src_idx])[::-1]
        for dst_idx in ranked:
            count = float(array[src_idx, dst_idx])
            if count <= 0:
                continue
            hints.append(
                {
                    'from': _label(inverse, int(src_idx)),
                    'to': _label(inverse, int(dst_idx)),
                    'count': int(count),
                    'probability': count / total if total else 0.0,
                }
            )
    return hints


def build_causal_json_hints(A_norm: Any, inverse_vocab: List[Any], top_k_edges: int = 10) -> List[Dict[str, Any]]:
    '''Export strongest source -> target causal edges from A_norm.'''
    if top_k_edges <= 0:
        raise ValueError('top_k_edges must be positive')
    array = np.asarray(A_norm, dtype=np.float32)
    if array.ndim != 2:
        raise ValueError('A_norm must have shape [C, C]')
    edges: List[Dict[str, Any]] = []
    for src_idx in range(array.shape[0]):
        for dst_idx in range(array.shape[1]):
            weight = float(array[src_idx, dst_idx])
            if weight > 0:
                edges.append(
                    {
                        'from': _label(inverse_vocab, src_idx),
                        'to': _label(inverse_vocab, dst_idx),
                        'weight': weight,
                    }
                )
    edges.sort(key=lambda item: item['weight'], reverse=True)
    return edges[:top_k_edges]
