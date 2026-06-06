'''Explanation and visualization utilities for CausalGenGuard evaluation.'''

from __future__ import annotations

import json
import math
import warnings
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from causal_gen_guard.data.schemas import BehaviorSequence


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    if hasattr(value, 'detach'):
        try:
            return _json_safe(value.detach().cpu().numpy())
        except Exception:
            pass
    if hasattr(value, 'tolist'):
        try:
            return _json_safe(value.tolist())
        except Exception:
            pass
    if hasattr(value, 'item'):
        try:
            return value.item()
        except Exception:
            pass
    return value


def _to_numpy(value: Any) -> np.ndarray:
    if value is None:
        return np.asarray([], dtype=np.float32)
    if isinstance(value, np.ndarray):
        return value.astype(np.float32)
    if hasattr(value, 'detach'):
        try:
            return value.detach().cpu().numpy().astype(np.float32)
        except Exception:
            pass
    return np.asarray(value, dtype=np.float32)


def _event_payload(sequence: BehaviorSequence, position: int) -> Dict[str, Any]:
    if position < 0 or position >= len(sequence.events):
        return {}
    event = sequence.events[position]
    return event.to_dict()


def _detector_metadata(detector: Any) -> Dict[str, Any]:
    return {
        'weights': {
            'alpha': getattr(detector, 'alpha', None),
            'beta': getattr(detector, 'beta', None),
            'delta': getattr(detector, 'delta', None),
        },
        'threshold': getattr(detector, 'threshold', None),
        'normalization': {
            'rec_mean': getattr(detector, 'rec_mean', None),
            'rec_std': getattr(detector, 'rec_std', None),
            'causal_mean': getattr(detector, 'causal_mean', None),
            'causal_std': getattr(detector, 'causal_std', None),
            'context_mean': getattr(detector, 'context_mean', None),
            'context_std': getattr(detector, 'context_std', None),
        },
        'vocab_size': len(getattr(detector, 'vocab', {}) or {}),
    }


def _summarize_tokens(sequence: BehaviorSequence, tokens: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    summarized: List[Dict[str, Any]] = []
    for token in tokens:
        position = int(token.get('position', -1))
        event_payload = _event_payload(sequence, position)
        summarized.append(
            {
                'rank': len(summarized) + 1,
                'position': position,
                'control_id': token.get('control_id'),
                'device_id': event_payload.get('device_id'),
                'timestamp': event_payload.get('timestamp'),
                'day': event_payload.get('day', token.get('day')),
                'hour': event_payload.get('hour', token.get('hour')),
                'duration': event_payload.get('duration'),
                'reconstruction_loss': token.get('rec_loss'),
                'raw_fields': event_payload.get('raw_fields', {}),
            }
        )
    return summarized


def _summarize_edges(edges: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    summarized: List[Dict[str, Any]] = []
    for edge in edges:
        summarized.append(
            {
                'rank': len(summarized) + 1,
                'from': edge.get('from'),
                'to': edge.get('to'),
                'weight': edge.get('weight'),
            }
        )
    return summarized


def explain_one_sequence(
    detector: Any,
    sequence: BehaviorSequence,
    top_k_tokens: int = 5,
    top_k_edges: int = 5,
) -> Dict[str, Any]:
    '''Build a case-study-ready explanation for one BehaviorSequence.'''
    detail = detector.score_sequence(sequence)
    tokens = sorted(detail.get('token_details', []), key=lambda item: item.get('rec_loss', 0.0), reverse=True)[:top_k_tokens]
    edges = detail.get('edge_details', [])[:top_k_edges]
    pred = detail.get('pred')
    threshold = getattr(detector, 'threshold', None)
    if pred is None and threshold is not None:
        pred = int(detail.get('score', 0.0) > threshold)

    graph = _to_numpy(detail.get('graph'))
    graph_summary = {
        'shape': list(graph.shape) if graph.size else [],
        'nonzero_edges': int(np.count_nonzero(graph)) if graph.size else 0,
        'max_edge_weight': float(np.max(graph)) if graph.size else 0.0,
        'mean_edge_weight': float(np.mean(graph)) if graph.size else 0.0,
    }

    return {
        'sequence_id': sequence.sequence_id,
        'label': sequence.label,
        'anomaly_type': sequence.anomaly_type,
        'prediction': pred,
        'scores': {
            'score': detail.get('score'),
            'reconstruction_score': detail.get('rec_score'),
            'causal_score': detail.get('causal_score'),
            'context_score': detail.get('context_score'),
            'z_reconstruction': detail.get('z_rec'),
            'z_causal': detail.get('z_causal'),
            'z_context': detail.get('z_context'),
            'threshold': threshold,
        },
        'context': dict(sequence.context),
        'context_graph_id': detail.get('context_graph'),
        'sequence_summary': {
            'event_count': len(sequence.events),
            'first_timestamp': sequence.events[0].timestamp if sequence.events else None,
            'last_timestamp': sequence.events[-1].timestamp if sequence.events else None,
            'unique_controls': sorted({str(event.control_id) for event in sequence.events}),
            'unique_devices': sorted({str(event.device_id) for event in sequence.events if event.device_id is not None}),
        },
        'top_abnormal_tokens': _summarize_tokens(sequence, tokens),
        'top_abnormal_causal_edges': _summarize_edges(edges),
        'causal_graph_summary': graph_summary,
        'detector': _detector_metadata(detector),
    }


def save_explanation_json(explanation: Dict[str, Any], path: str | Path) -> None:
    '''Save one explanation dictionary as JSON.'''
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('w', encoding='utf-8') as handle:
        json.dump(_json_safe(explanation), handle, ensure_ascii=False, indent=2)


def _edge_label(value: Any, max_len: int = 18) -> str:
    text = str(value)
    if len(text) > max_len:
        return text[: max_len - 3] + '...'
    return text


def _top_edges(diff: np.ndarray, top_k_edges: int) -> List[Tuple[int, int, float]]:
    edges: List[Tuple[int, int, float]] = []
    if diff.ndim != 2:
        return edges
    for src in range(diff.shape[0]):
        for dst in range(diff.shape[1]):
            weight = float(diff[src, dst])
            if weight > 0.0:
                edges.append((src, dst, weight))
    edges.sort(key=lambda item: item[2], reverse=True)
    return edges[: max(1, int(top_k_edges))]


def plot_causal_graph_diff(
    A_test: Any,
    A_norm: Any,
    inverse_vocab: Sequence[Any],
    output_path: str | Path,
    top_k_edges: int = 20,
    max_nodes: int = 20,
) -> None:
    '''Plot positive causal graph deviations A_test - A_norm.

    The function uses matplotlib only. For large graphs it visualizes only nodes
    touched by the strongest top_k_edges deviations.
    '''
    import matplotlib
    matplotlib.use('Agg', force=True)
    import matplotlib.pyplot as plt

    test = _to_numpy(A_test)
    norm = _to_numpy(A_norm)
    if test.ndim != 2 or norm.ndim != 2:
        raise ValueError('A_test and A_norm must be 2-D matrices')
    rows = min(test.shape[0], norm.shape[0])
    cols = min(test.shape[1], norm.shape[1])
    if rows == 0 or cols == 0:
        raise ValueError('A_test and A_norm must not be empty')
    diff = np.maximum(test[:rows, :cols] - norm[:rows, :cols], 0.0)
    edges = _top_edges(diff, top_k_edges=top_k_edges)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 6))
    if not edges:
        ax.text(0.5, 0.5, 'No positive causal graph deviations', ha='center', va='center')
        ax.set_axis_off()
        fig.tight_layout()
        fig.savefig(output, dpi=160)
        plt.close(fig)
        return

    nodes = sorted({src for src, _, _ in edges} | {dst for _, dst, _ in edges})
    if len(nodes) > max_nodes:
        node_set = set()
        limited_edges: List[Tuple[int, int, float]] = []
        for src, dst, weight in edges:
            if len(node_set | {src, dst}) <= max_nodes:
                limited_edges.append((src, dst, weight))
                node_set.update([src, dst])
        edges = limited_edges or edges[:1]
        nodes = sorted({src for src, _, _ in edges} | {dst for _, dst, _ in edges})

    angle_by_node = {node: 2.0 * math.pi * index / max(len(nodes), 1) for index, node in enumerate(nodes)}
    positions = {node: (math.cos(angle), math.sin(angle)) for node, angle in angle_by_node.items()}

    for node in nodes:
        x, y = positions[node]
        ax.scatter([x], [y], s=500)
        label = inverse_vocab[node] if node < len(inverse_vocab) else node
        ax.text(x, y, _edge_label(label), ha='center', va='center', fontsize=8)

    max_weight = max(weight for _, _, weight in edges)
    for src, dst, weight in edges:
        start = positions[src]
        end = positions[dst]
        width = 0.5 + 3.0 * weight / max(max_weight, 1e-8)
        ax.annotate(
            '',
            xy=end,
            xytext=start,
            arrowprops={'arrowstyle': '->', 'lw': width, 'shrinkA': 18, 'shrinkB': 18},
        )
        mid_x = (start[0] + end[0]) / 2.0
        mid_y = (start[1] + end[1]) / 2.0
        ax.text(mid_x, mid_y, '{:.3g}'.format(weight), fontsize=7)

    ax.set_title('Causal graph deviation: top {} edges'.format(len(edges)))
    ax.set_aspect('equal')
    ax.set_axis_off()
    fig.tight_layout()
    fig.savefig(output, dpi=160)
    plt.close(fig)


def plot_score_distribution(normal_scores: Iterable[float], anomaly_scores: Iterable[float], output_path: str | Path) -> None:
    '''Plot normal/anomaly score distributions with matplotlib.'''
    import matplotlib
    matplotlib.use('Agg', force=True)
    import matplotlib.pyplot as plt

    normal = np.asarray(list(normal_scores), dtype=np.float32)
    anomaly = np.asarray(list(anomaly_scores), dtype=np.float32)
    if normal.size == 0 and anomaly.size == 0:
        raise ValueError('normal_scores and anomaly_scores are both empty')
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 4))
    bins = min(30, max(5, int(math.sqrt(max(normal.size + anomaly.size, 1)))))
    if normal.size:
        ax.hist(normal, bins=bins, alpha=0.6, label='normal')
    if anomaly.size:
        ax.hist(anomaly, bins=bins, alpha=0.6, label='anomaly')
    ax.set_xlabel('Fusion anomaly score')
    ax.set_ylabel('Count')
    ax.set_title('Score distribution')
    ax.legend()
    fig.tight_layout()
    fig.savefig(output, dpi=160)
    plt.close(fig)


def safe_plot(callable_obj: Any, warning_list: List[str], description: str, *args: Any, **kwargs: Any) -> Optional[str]:
    '''Run plotting without allowing visualization failure to fail evaluation.'''
    try:
        callable_obj(*args, **kwargs)
        return str(args[-1]) if args else None
    except Exception as exc:  # pragma: no cover - intentionally defensive.
        message = '{} failed: {}'.format(description, exc)
        warning_list.append(message)
        warnings.warn(message)
        return None
