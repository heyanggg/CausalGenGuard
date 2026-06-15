#!/usr/bin/env python3
'''Run bounded Feasibility v2 over canonical SmartGuard FR data.

This script intentionally keeps training artifacts in memory. It only writes
the requested v2 result JSON files and markdown report.
'''

from __future__ import annotations

import argparse
import copy
import json
import random
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

sys.dont_write_bytecode = True

import numpy as np
import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / 'src'
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from causal_gen_guard.data.behavior_event_tensor import build_vocab
from causal_gen_guard.data.schemas import BehaviorEvent, BehaviorSequence
from causal_gen_guard.evaluation.metrics import compute_binary_metrics
from causal_gen_guard.models.causal_graph import compute_gradient_causality, normal_causal_pattern, sparsify_causality
from causal_gen_guard.models.causal_predictor import BehaviorCausalPredictor
from causal_gen_guard.models.fusion_detector import FusionDetector
from causal_gen_guard.models.smartguard_backbone import SmartGuardBackbone, compute_reconstruction_loss
from causal_gen_guard.training.train_backbone import SequenceTensorDataset
from causal_gen_guard.training.train_causal import WindowDataset, build_window_arrays


METHODS = ('reconstruction_only', 'causal_only', 'fusion')
METRIC_FIELDS = ('precision', 'recall', 'f1', 'fpr', 'auroc', 'auprc')
CAUSAL_TYPES = ('edge_break', 'edge_injection', 'lag_delay')


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, 'detach'):
        try:
            return json_safe(value.detach().cpu().numpy())
        except Exception:
            pass
    if hasattr(value, 'tolist'):
        try:
            return json_safe(value.tolist())
        except Exception:
            pass
    if hasattr(value, 'item'):
        try:
            return value.item()
        except Exception:
            pass
    if isinstance(value, float) and np.isnan(value):
        return None
    return value


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_safe(payload), ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_jsonl(path: Path, max_sequences: Optional[int] = None) -> List[BehaviorSequence]:
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


def label_to_int(label: Any) -> int:
    if label is None:
        return 0
    if isinstance(label, bool):
        return int(label)
    if isinstance(label, (int, float)):
        return int(label != 0)
    text = str(label).strip().lower()
    if text in ('0', 'normal', 'benign', 'clean', 'false', 'none'):
        return 0
    return 1


def normal_copy(sequence: BehaviorSequence, suffix: str = '') -> BehaviorSequence:
    copied = copy.deepcopy(sequence)
    copied.sequence_id = copied.sequence_id + suffix
    copied.label = 0
    copied.anomaly_type = 'normal'
    copied.context = dict(copied.context)
    return copied


def split_normals(sequences: List[BehaviorSequence]) -> Tuple[List[BehaviorSequence], List[BehaviorSequence], List[BehaviorSequence]]:
    if len(sequences) < 10:
        raise ValueError('Need at least 10 normal sequences for Feasibility v2')
    train_end = max(1, int(len(sequences) * 0.6))
    val_end = max(train_end + 1, int(len(sequences) * 0.8))
    return sequences[:train_end], sequences[train_end:val_end], sequences[val_end:]


def max_event_len(sequences: List[BehaviorSequence]) -> int:
    return max((len(sequence.events) for sequence in sequences), default=0)


def safe_nhead(hidden_dim: int) -> int:
    for candidate in (8, 4, 2, 1):
        if hidden_dim % candidate == 0:
            return candidate
    return 1


def train_backbone(
    sequences: List[BehaviorSequence],
    vocab: Dict[Any, int],
    epochs: int,
    hidden_dim: int,
    batch_size: int,
    seed: int,
) -> Tuple[SmartGuardBackbone, Dict[str, Any]]:
    set_seed(seed)
    max_len = max_event_len(sequences)
    if max_len <= 0:
        raise ValueError('Cannot train backbone on empty sequences')
    dataset = SequenceTensorDataset(sequences, vocab, max_len)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    model = SmartGuardBackbone(
        vocab_size=len(vocab),
        hidden_dim=hidden_dim,
        nhead=safe_nhead(hidden_dim),
        num_layers=1,
        dropout=0.1,
        use_ttpe=True,
    )
    model.control_vocab = dict(vocab)
    model.max_len = max_len
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    history: List[Dict[str, float]] = []
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        total_tokens = 0.0
        for fields, mask in loader:
            optimizer.zero_grad()
            outputs = model(fields, attention_mask=mask)
            targets = fields[..., 3].long()
            loss = compute_reconstruction_loss(outputs['logits'], targets, mask=mask, reduction='mean')
            loss.backward()
            optimizer.step()
            token_count = float(mask.sum().item())
            total_loss += float(loss.detach().item()) * token_count
            total_tokens += token_count
        mean_loss = total_loss / max(total_tokens, 1.0)
        history.append({'epoch': epoch + 1, 'train_loss': mean_loss})
        print(f'v2 backbone epoch {epoch + 1}/{epochs} loss={mean_loss:.6f}')
    model.eval()
    return model, {'history': history, 'max_len': max_len}


def train_causal(
    sequences: List[BehaviorSequence],
    vocab: Dict[Any, int],
    epochs: int,
    hidden_dim: int,
    batch_size: int,
    window_size: int,
    causality_samples: int,
    seed: int,
) -> Tuple[BehaviorCausalPredictor, np.ndarray, Dict[str, Any]]:
    set_seed(seed)
    window_size = max(1, min(window_size, max_event_len(sequences) - 1))
    windows, time_windows, targets = build_window_arrays(sequences, vocab, window_size=window_size, pred_horizon=1)
    dataset = WindowDataset(windows, time_windows, targets)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    model = BehaviorCausalPredictor(
        input_channels=len(vocab),
        hidden_dim=hidden_dim,
        time_feature_dim=time_windows.shape[-1],
        num_layers=1,
        dropout=0.1,
    )
    model.window_size = window_size
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    history: List[Dict[str, float]] = []
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        total_count = 0
        for batch_windows, batch_time, batch_targets in loader:
            optimizer.zero_grad()
            pred = model(batch_windows, batch_time)
            loss = model.compute_loss(pred, batch_targets)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.detach().item()) * int(batch_windows.size(0))
            total_count += int(batch_windows.size(0))
        mean_loss = total_loss / max(total_count, 1)
        history.append({'epoch': epoch + 1, 'train_loss': mean_loss})
        print(f'v2 causal epoch {epoch + 1}/{epochs} loss={mean_loss:.6f}')
    model.eval()
    sample_count = min(causality_samples, len(dataset))
    graph_batch = compute_gradient_causality(model, dataset.windows[:sample_count], dataset.targets[:sample_count])
    A_norm = normal_causal_pattern(sparsify_causality(graph_batch, threshold=0.0)).detach().cpu().numpy().astype(np.float32)
    info = {
        'history': history,
        'window_size': window_size,
        'window_count': int(len(dataset)),
        'causality_samples': int(sample_count),
        'A_norm_shape': list(A_norm.shape),
    }
    return model, A_norm, info


def build_detector(mode: str, backbone: Any, causal_model: Any, vocab: Dict[Any, int], A_norm: np.ndarray) -> FusionDetector:
    if mode == 'reconstruction_only':
        return FusionDetector(backbone=backbone, causal_model=None, vocab=vocab, A_norm_bank=A_norm, alpha=1.0, beta=0.0)
    if mode == 'causal_only':
        return FusionDetector(backbone=None, causal_model=causal_model, vocab=vocab, A_norm_bank=A_norm, alpha=0.0, beta=1.0)
    if mode == 'fusion':
        return FusionDetector(backbone=backbone, causal_model=causal_model, vocab=vocab, A_norm_bank=A_norm, alpha=0.6, beta=0.4)
    raise ValueError(f'Unknown mode: {mode}')


def evaluate_method(
    mode: str,
    backbone: Any,
    causal_model: Any,
    vocab: Dict[Any, int],
    A_norm: np.ndarray,
    val_normals: List[BehaviorSequence],
    test_sequences: List[BehaviorSequence],
) -> Dict[str, Any]:
    detector = build_detector(mode, backbone, causal_model, vocab, A_norm)
    threshold = detector.calibrate_threshold(val_normals, quantile=0.95)
    scores = detector.score_batch(test_sequences)
    y_true = [label_to_int(sequence.label) for sequence in test_sequences]
    y_score = [float(item['score']) for item in scores]
    metrics = compute_binary_metrics(y_true, y_score, threshold)
    y_pred = [int(score > threshold) for score in y_score]
    return {
        'method': mode,
        'threshold': threshold,
        'metrics': metrics,
        'score_summary': {
            'normal_count': int(sum(1 for item in y_true if item == 0)),
            'anomaly_count': int(sum(1 for item in y_true if item == 1)),
            'predicted_anomaly_count': int(sum(y_pred)),
            'normal_mean_score': float(np.mean([s for s, y in zip(y_score, y_true) if y == 0])) if any(y == 0 for y in y_true) else None,
            'anomaly_mean_score': float(np.mean([s for s, y in zip(y_score, y_true) if y == 1])) if any(y == 1 for y in y_true) else None,
        },
    }


def evaluate_all_methods(
    backbone: Any,
    causal_model: Any,
    vocab: Dict[Any, int],
    A_norm: np.ndarray,
    val_normals: List[BehaviorSequence],
    test_sequences: List[BehaviorSequence],
) -> Dict[str, Any]:
    return {
        method: evaluate_method(method, backbone, causal_model, vocab, A_norm, val_normals, test_sequences)
        for method in METHODS
    }


def inverse_vocab(vocab: Dict[Any, int]) -> List[Any]:
    inverse: List[Any] = [None] * len(vocab)
    for control, index in dict(vocab).items():
        if 0 <= int(index) < len(inverse):
            inverse[int(index)] = control
    return inverse


def top_edges(A_norm: np.ndarray, vocab: Dict[Any, int], limit: int) -> List[Dict[str, Any]]:
    inverse = inverse_vocab(vocab)
    edges: List[Dict[str, Any]] = []
    matrix = np.asarray(A_norm, dtype=np.float32)
    for src in range(matrix.shape[0]):
        for dst in range(matrix.shape[1]):
            if src == dst:
                continue
            edges.append({'src': inverse[src], 'dst': inverse[dst], 'weight': float(matrix[src, dst])})
    edges.sort(key=lambda item: item['weight'], reverse=True)
    return edges[:limit]


def event_controls(event: BehaviorEvent) -> List[Any]:
    controls = []
    raw = event.raw_fields or {}
    canonical = raw.get('canonical_control')
    if canonical and str(canonical).lower() != 'unknown':
        controls.append(canonical)
    controls.append(event.control_id)
    seen = set()
    unique = []
    for control in controls:
        key = repr(control)
        if key not in seen:
            seen.add(key)
            unique.append(control)
    return unique


def event_matches(event: BehaviorEvent, control: Any) -> bool:
    return any(candidate == control for candidate in event_controls(event))


def find_index(sequence: BehaviorSequence, control: Any, start: int = 0) -> Optional[int]:
    for index, event in enumerate(sequence.events[start:], start=start):
        if event_matches(event, control):
            return index
    return None


def find_pair(sequence: BehaviorSequence, src: Any, dst: Any) -> Tuple[Optional[int], Optional[int]]:
    src_index = find_index(sequence, src)
    if src_index is None:
        return None, None
    dst_index = find_index(sequence, dst, start=src_index + 1)
    return src_index, dst_index


def clone_event(event: BehaviorEvent, control_id: Any = None, **updates: Any) -> BehaviorEvent:
    cloned = copy.deepcopy(event)
    raw_fields = dict(cloned.raw_fields)
    if control_id is not None:
        cloned.control_id = control_id
        raw_fields['canonical_control'] = control_id
        if isinstance(control_id, str) and ':' in control_id:
            device, action = control_id.split(':', 1)
            cloned.device_id = device
            raw_fields['device'] = device
            raw_fields['action'] = action
    for key, value in updates.items():
        setattr(cloned, key, value)
        if key in ('day', 'hour', 'duration'):
            raw_fields[key] = value
    raw_fields['causal_attack'] = True
    cloned.raw_fields = raw_fields
    return cloned


def make_causal_sequence(
    source: BehaviorSequence,
    events: List[BehaviorEvent],
    anomaly_type: str,
    edge: Dict[str, Any],
    operation: str,
) -> BehaviorSequence:
    context = dict(source.context)
    context.update({'attack_injected': True, 'top_edge': edge, 'attack_operation': operation})
    return BehaviorSequence(
        sequence_id=f'{source.sequence_id}::{anomaly_type}',
        events=events,
        context=context,
        label=1,
        anomaly_type=anomaly_type,
    )


def inject_top_edge_causal(source: BehaviorSequence, anomaly_type: str, edges: List[Dict[str, Any]]) -> Tuple[Optional[BehaviorSequence], Dict[str, Any]]:
    for edge in edges:
        src = edge['src']
        dst = edge['dst']
        if anomaly_type == 'edge_break':
            src_index, dst_index = find_pair(source, src, dst)
            if dst_index is None:
                continue
            events = copy.deepcopy(source.events)
            removed = events.pop(dst_index)
            if not events:
                continue
            return make_causal_sequence(source, events, anomaly_type, edge, 'remove_top_edge_target'), {
                'status': 'injected',
                'anomaly_type': anomaly_type,
                'sequence_id': source.sequence_id,
                'edge': edge,
                'removed_control': removed.control_id,
            }
        if anomaly_type == 'edge_injection':
            dst_index = find_index(source, dst)
            if dst_index is None:
                continue
            events = copy.deepcopy(source.events)
            injected_event = clone_event(source.events[dst_index], control_id=src)
            events = events[: dst_index + 1] + [injected_event] + events[dst_index + 1 :]
            return make_causal_sequence(source, events, anomaly_type, edge, 'inject_reversed_top_edge'), {
                'status': 'injected',
                'anomaly_type': anomaly_type,
                'sequence_id': source.sequence_id,
                'edge': edge,
                'injected_control': src,
            }
        if anomaly_type == 'lag_delay':
            src_index, dst_index = find_pair(source, src, dst)
            if dst_index is None:
                continue
            events = copy.deepcopy(source.events)
            target = events[dst_index]
            try:
                delayed_hour = float(target.hour) + 4.0
            except Exception:
                delayed_hour = 4.0
            try:
                delayed_duration = float(target.duration) + 12.0
            except Exception:
                delayed_duration = 12.0
            events[dst_index] = clone_event(target, hour=delayed_hour, duration=delayed_duration)
            return make_causal_sequence(source, events, anomaly_type, edge, 'delay_top_edge_target'), {
                'status': 'injected',
                'anomaly_type': anomaly_type,
                'sequence_id': source.sequence_id,
                'edge': edge,
                'delayed_control': target.control_id,
            }
    return None, {'status': 'skipped', 'anomaly_type': anomaly_type, 'skipped_reason': 'top_edge_controls_not_found'}


def build_top_edge_anomalies(
    normals: List[BehaviorSequence],
    edges: List[Dict[str, Any]],
    per_type: int,
    seed: int,
) -> Tuple[List[BehaviorSequence], Dict[str, Any]]:
    rng = random.Random(seed)
    anomalies: List[BehaviorSequence] = []
    injected_reports: List[Dict[str, Any]] = []
    skipped_reports: List[Dict[str, Any]] = []
    per_type_success = {name: 0 for name in CAUSAL_TYPES}
    for anomaly_type in CAUSAL_TYPES:
        attempts = 0
        while per_type_success[anomaly_type] < per_type and attempts < max(len(normals) * 6, per_type * 4):
            attempts += 1
            source = rng.choice(normals)
            injected, report = inject_top_edge_causal(source, anomaly_type, edges)
            if injected is None:
                skipped_reports.append(report)
                continue
            injected.sequence_id = f'{injected.sequence_id}::v2_{per_type_success[anomaly_type]:04d}'
            anomalies.append(injected)
            injected_reports.append(report)
            per_type_success[anomaly_type] += 1
    return anomalies, {
        'per_type_target': per_type,
        'injected_count': len(anomalies),
        'per_type_success': per_type_success,
        'skipped_count': len(skipped_reports),
        'injected': injected_reports[:30],
        'skipped': skipped_reports[:30],
    }


def repeat_normals(normals: List[BehaviorSequence], count: int) -> List[BehaviorSequence]:
    if not normals or count <= 0:
        return []
    return [normal_copy(normals[index % len(normals)], suffix=f'::v2_normal_{index:04d}') for index in range(count)]


def best_method(methods: Dict[str, Any]) -> Tuple[str, Dict[str, float]]:
    name = max(methods, key=lambda item: float(methods[item]['metrics'].get('f1', 0.0)))
    return name, methods[name]['metrics']


def metric_value(methods: Dict[str, Any], method: str, metric: str) -> float:
    value = methods[method]['metrics'].get(metric)
    return float(value) if value is not None else float('nan')


def positive_signal(methods: Dict[str, Any], method: str) -> bool:
    rec_f1 = metric_value(methods, 'reconstruction_only', 'f1')
    rec_auroc = metric_value(methods, 'reconstruction_only', 'auroc')
    method_f1 = metric_value(methods, method, 'f1')
    method_auroc = metric_value(methods, method, 'auroc')
    return bool(method_f1 > rec_f1 or method_auroc > rec_auroc)


def metric_line(methods: Dict[str, Any], method: str) -> str:
    metrics = methods[method]['metrics']
    return '| {} | {:.4f} | {:.4f} | {:.4f} | {:.4f} | {:.4f} | {:.4f} |'.format(
        method,
        float(metrics.get('precision', float('nan'))),
        float(metrics.get('recall', float('nan'))),
        float(metrics.get('f1', float('nan'))),
        float(metrics.get('fpr', float('nan'))),
        float(metrics.get('auroc', float('nan'))),
        float(metrics.get('auprc', float('nan'))),
    )


def write_report(path: Path, summary: Dict[str, Any]) -> None:
    exp1 = summary['experiment_1']
    exp2 = summary['experiment_2']
    label_report = summary['labeled_report']
    lines = [
        '# Feasibility Report V2',
        '',
        'Project path: `/home/heyang/projects/CausalGenGuard`',
        'Environment: `/home/heyang/miniconda3/envs/smartguard_env/bin/python`',
        '',
        '## Run Bounds',
        '',
        '- Max normal sequences: {}'.format(summary['run_bounds']['max_normal_sequences']),
        '- Epochs: {}'.format(summary['run_bounds']['epochs']),
        '- Train/val/test normal split: train={train}, val={val}, test={test}'.format(**summary['normal_split']),
        '- Vocab size: {}'.format(summary['vocab_size']),
        '- Window size: {}'.format(summary['causal_training']['window_size']),
        '',
        '## Input Files',
        '',
    ]
    lines.extend('- `{}`'.format(path_value) for path_value in summary['input_files'])
    lines.extend([
        '',
        '## Labeled Injection Report',
        '',
        '- named_injection_success_count: {}'.format(label_report.get('named_injection_success_count')),
        '- fallback_numeric_injection_count: {}'.format(label_report.get('fallback_numeric_injection_count')),
        '- skipped_count: {}'.format(label_report.get('skipped_count')),
        '',
        '## Experiment 1: SmartGuard-Style Named Semantic Anomaly Sanity',
        '',
        '| method | precision | recall | f1 | fpr | auroc | auprc |',
        '| --- | --- | --- | --- | --- | --- | --- |',
    ])
    lines.extend(metric_line(exp1['methods'], method) for method in METHODS)
    lines.extend([
        '',
        'Output: `outputs/results/sanity_fr_v2.json`',
        '',
        '## Experiment 2: A_norm Top-Edge Causal Anomaly Smoke',
        '',
        '| method | precision | recall | f1 | fpr | auroc | auprc |',
        '| --- | --- | --- | --- | --- | --- | --- |',
    ])
    lines.extend(metric_line(exp2['methods'], method) for method in METHODS)
    lines.extend([
        '',
        'Output: `outputs/results/causal_anomaly_smoke_v2.json`',
        '',
        '## Signals',
        '',
        '- Experiment 2 causal branch better than reconstruction_only: `{}`'.format(summary['signals']['causal_branch_positive']),
        '- Experiment 2 fusion better than reconstruction_only: `{}`'.format(summary['signals']['fusion_positive']),
        '- Recommend SmartGen alignment / Causal-TOF v2: `{}`'.format(summary['recommendations']['smartgen_alignment_causal_tof_v2']),
        '- Recommend formal main experiments: `{}`'.format(summary['recommendations']['formal_main_experiment']),
        '',
        '## Notes',
        '',
    ])
    lines.extend('- {}'.format(note) for note in summary['notes'])
    lines.append('')
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text('\n'.join(lines), encoding='utf-8')


def run(args: argparse.Namespace) -> Dict[str, Any]:
    if args.max_normal_sequences > 1000:
        raise ValueError('--max-normal-sequences must be <= 1000')
    if args.epochs > 3:
        raise ValueError('--epochs must be <= 3')
    set_seed(args.seed)
    normal_path = PROJECT_ROOT / args.normal_jsonl
    labeled_path = PROJECT_ROOT / args.labeled_jsonl
    labeled_report_path = PROJECT_ROOT / args.labeled_report
    id_to_control_path = PROJECT_ROOT / args.id_to_control
    sanity_output = PROJECT_ROOT / 'outputs/results/sanity_fr_v2.json'
    causal_output = PROJECT_ROOT / 'outputs/results/causal_anomaly_smoke_v2.json'
    report_output = PROJECT_ROOT / 'outputs/logs/FEASIBILITY_REPORT_V2.md'

    normal_sequences = [normal_copy(sequence) for sequence in load_jsonl(normal_path, max_sequences=args.max_normal_sequences)]
    labeled_sequences = load_jsonl(labeled_path)
    labeled_report = json.loads(labeled_report_path.read_text(encoding='utf-8'))
    id_to_control = json.loads(id_to_control_path.read_text(encoding='utf-8'))
    train_normals, val_normals, test_normals = split_normals(normal_sequences)
    vocab = build_vocab(normal_sequences + labeled_sequences)

    backbone, backbone_info = train_backbone(
        train_normals,
        vocab,
        epochs=args.epochs,
        hidden_dim=args.hidden_dim,
        batch_size=args.batch_size,
        seed=args.seed,
    )
    causal_model, A_norm, causal_info = train_causal(
        train_normals,
        vocab,
        epochs=args.epochs,
        hidden_dim=args.hidden_dim,
        batch_size=args.batch_size,
        window_size=args.window_size,
        causality_samples=args.causality_samples,
        seed=args.seed,
    )

    rng = random.Random(args.seed)
    exp1_sequences = [copy.deepcopy(sequence) for sequence in labeled_sequences]
    rng.shuffle(exp1_sequences)
    exp1_methods = evaluate_all_methods(backbone, causal_model, vocab, A_norm, val_normals, exp1_sequences)
    exp1_best, exp1_best_metrics = best_method(exp1_methods)
    exp1_results = {
        'experiment': 'SmartGuard-style named semantic anomaly sanity',
        'input': str(labeled_path),
        'counts': {
            'normal': int(sum(1 for sequence in exp1_sequences if label_to_int(sequence.label) == 0)),
            'anomaly': int(sum(1 for sequence in exp1_sequences if label_to_int(sequence.label) == 1)),
        },
        'methods': exp1_methods,
        'best_method': exp1_best,
        'best_metrics': exp1_best_metrics,
        'labeled_report': labeled_report,
    }
    write_json(sanity_output, exp1_results)

    edge_candidates = top_edges(A_norm, vocab, args.top_edges)
    causal_anomalies, causal_injection_report = build_top_edge_anomalies(
        test_normals,
        edge_candidates,
        per_type=args.causal_anomalies_per_type,
        seed=args.seed + 1,
    )
    causal_normals = repeat_normals(test_normals, len(causal_anomalies))
    exp2_sequences = causal_normals + causal_anomalies
    rng.shuffle(exp2_sequences)
    exp2_methods = evaluate_all_methods(backbone, causal_model, vocab, A_norm, val_normals, exp2_sequences)
    exp2_best, exp2_best_metrics = best_method(exp2_methods)
    causal_positive = positive_signal(exp2_methods, 'causal_only')
    fusion_positive = positive_signal(exp2_methods, 'fusion')
    exp2_results = {
        'experiment': 'A_norm top-edge causal anomaly smoke',
        'normal_source': str(normal_path),
        'counts': {
            'normal': len(causal_normals),
            'anomaly': len(causal_anomalies),
        },
        'top_edges': edge_candidates[:20],
        'injection_report': causal_injection_report,
        'methods': exp2_methods,
        'best_method': exp2_best,
        'best_metrics': exp2_best_metrics,
        'signals': {
            'causal_branch_positive': causal_positive,
            'fusion_positive': fusion_positive,
        },
    }
    write_json(causal_output, exp2_results)

    recommend_tof_v2 = bool(labeled_report.get('named_injection_success_count', 0) > 0 and labeled_report.get('fallback_numeric_injection_count') == 0)
    recommend_formal = bool(causal_positive or fusion_positive)
    summary = {
        'run_bounds': {
            'max_normal_sequences': args.max_normal_sequences,
            'epochs': args.epochs,
            'seed': args.seed,
            'batch_size': args.batch_size,
            'hidden_dim': args.hidden_dim,
        },
        'input_files': [str(normal_path), str(labeled_path), str(labeled_report_path), str(id_to_control_path)],
        'normal_split': {'train': len(train_normals), 'val': len(val_normals), 'test': len(test_normals)},
        'vocab_size': len(vocab),
        'id_to_control_count': len(id_to_control),
        'backbone_training': backbone_info,
        'causal_training': causal_info,
        'labeled_report': labeled_report,
        'experiment_1': exp1_results,
        'experiment_2': exp2_results,
        'signals': {
            'causal_branch_positive': causal_positive,
            'fusion_positive': fusion_positive,
        },
        'recommendations': {
            'smartgen_alignment_causal_tof_v2': recommend_tof_v2,
            'formal_main_experiment': recommend_formal,
        },
        'notes': [
            'Bounded v2 run used canonical FR data only; no source projects were modified.',
            'Normal training was limited to {} sequences and {} epochs.'.format(len(train_normals), args.epochs),
            'Formal main experiments should wait for broader semantic coverage if many named attack types remain skipped.',
        ],
    }
    write_report(report_output, summary)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Run bounded Feasibility v2 over canonical SmartGuard FR data.')
    parser.add_argument('--normal-jsonl', default='outputs/processed/fr_sequences_canonical.jsonl')
    parser.add_argument('--labeled-jsonl', default='outputs/labels/fr_smartguard_style_labeled.jsonl')
    parser.add_argument('--labeled-report', default='outputs/labels/fr_smartguard_style_labeled_report.json')
    parser.add_argument('--id-to-control', default='outputs/mappings/smartguard/fr/id_to_control.json')
    parser.add_argument('--max-normal-sequences', type=int, default=100)
    parser.add_argument('--epochs', type=int, default=2)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--hidden-dim', type=int, default=64)
    parser.add_argument('--window-size', type=int, default=4)
    parser.add_argument('--causality-samples', type=int, default=64)
    parser.add_argument('--top-edges', type=int, default=25)
    parser.add_argument('--causal-anomalies-per-type', type=int, default=20)
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    summary = run(args)
    exp1_best = summary['experiment_1']['best_method']
    exp2_best = summary['experiment_2']['best_method']
    print('Wrote outputs/results/sanity_fr_v2.json')
    print('Wrote outputs/results/causal_anomaly_smoke_v2.json')
    print('Wrote outputs/logs/FEASIBILITY_REPORT_V2.md')
    print('experiment_1_best_method={}'.format(exp1_best))
    print('experiment_2_best_method={}'.format(exp2_best))
    print('causal_branch_positive={}'.format(summary['signals']['causal_branch_positive']))
    print('fusion_positive={}'.format(summary['signals']['fusion_positive']))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
