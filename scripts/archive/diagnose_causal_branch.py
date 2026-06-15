#!/usr/bin/env python3
'''Diagnose why the causal branch lacks positive signal in Feasibility v2.'''

from __future__ import annotations

import argparse
import copy
import csv
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
from causal_gen_guard.models.causal_graph import (
    compute_gradient_causality,
    normal_causal_pattern,
    sparsify_causality,
)
from causal_gen_guard.models.causal_predictor import BehaviorCausalPredictor
from causal_gen_guard.models.fusion_detector import FusionDetector
from causal_gen_guard.training.train_causal import WindowDataset, build_window_arrays


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
    return 0 if text in ('0', 'normal', 'benign', 'clean', 'false', 'none') else 1


def normal_copy(sequence: BehaviorSequence, suffix: str = '') -> BehaviorSequence:
    copied = copy.deepcopy(sequence)
    copied.sequence_id = copied.sequence_id + suffix
    copied.label = 0
    copied.anomaly_type = 'normal'
    return copied


def event(control: str, index: int, sequence_id: str, hour_offset: int = 0) -> BehaviorEvent:
    return BehaviorEvent(
        day=0,
        hour=(index + hour_offset) % 8,
        duration=1.0,
        device_id=control,
        control_id=control,
        raw_fields={'canonical_control': control, 'device': control, 'action': control, 'source_sequence_id': sequence_id},
    )


def make_toy_normal(index: int, rng: random.Random) -> BehaviorSequence:
    sequence_id = f'toy_normal_{index:04d}'
    events: List[BehaviorEvent] = []
    position = 0
    for _ in range(4):
        events.append(event('A', position, sequence_id))
        position += 1
        events.append(event('B', position, sequence_id))
        position += 1
        if rng.random() < 0.35:
            events.append(event('C', position, sequence_id))
            position += 1
    return BehaviorSequence(sequence_id=sequence_id, events=events, context={'toy_pattern': 'A_to_B'}, label=0, anomaly_type='normal')


def make_toy_anomaly(source: BehaviorSequence, anomaly_type: str, index: int) -> BehaviorSequence:
    sequence_id = f'{source.sequence_id}::{anomaly_type}_{index:04d}'
    events = copy.deepcopy(source.events)
    if anomaly_type == 'edge_break':
        events = [item for item in events if item.control_id != 'B']
    elif anomaly_type == 'edge_injection':
        injected: List[BehaviorEvent] = []
        for item in events:
            injected.append(item)
            if item.control_id == 'A':
                injected.append(event('C', len(injected), sequence_id))
        events = injected
    elif anomaly_type == 'lag_delay':
        delayed: List[BehaviorEvent] = []
        for item in events:
            delayed.append(item)
            if item.control_id == 'A':
                delayed.append(event('C', len(delayed), sequence_id))
                delayed.append(event('C', len(delayed), sequence_id))
        events = delayed
    else:
        raise ValueError(f'Unsupported toy anomaly {anomaly_type}')
    return BehaviorSequence(sequence_id=sequence_id, events=events, context={'toy_pattern': 'abnormal', 'attack': anomaly_type}, label=1, anomaly_type=anomaly_type)


def split_three(sequences: List[BehaviorSequence]) -> Tuple[List[BehaviorSequence], List[BehaviorSequence], List[BehaviorSequence]]:
    train_end = max(1, int(len(sequences) * 0.6))
    val_end = max(train_end + 1, int(len(sequences) * 0.8))
    return sequences[:train_end], sequences[train_end:val_end], sequences[val_end:]


def max_event_len(sequences: List[BehaviorSequence]) -> int:
    return max((len(sequence.events) for sequence in sequences), default=0)


def train_causal_model(
    sequences: List[BehaviorSequence],
    vocab: Dict[Any, int],
    epochs: int,
    hidden_dim: int,
    batch_size: int,
    window_size: int,
    causality_samples: int,
    seed: int,
) -> Tuple[BehaviorCausalPredictor, np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
    set_seed(seed)
    window_size = max(1, min(window_size, max_event_len(sequences) - 1))
    windows, time_windows, targets = build_window_arrays(sequences, vocab, window_size=window_size, pred_horizon=1)
    time_windows = np.zeros((windows.shape[0], windows.shape[1], 0), dtype=np.float32)
    dataset = WindowDataset(windows, time_windows, targets)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    model = BehaviorCausalPredictor(input_channels=len(vocab), hidden_dim=hidden_dim, time_feature_dim=0, num_layers=1, dropout=0.1)
    model.window_size = window_size
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    history: List[Dict[str, float]] = []
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        total_count = 0
        for batch_windows, batch_time, batch_targets in loader:
            optimizer.zero_grad()
            pred = model(batch_windows, None)
            loss = model.compute_loss(pred, batch_targets)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.detach().item()) * int(batch_windows.size(0))
            total_count += int(batch_windows.size(0))
        mean_loss = total_loss / max(total_count, 1)
        history.append({'epoch': epoch + 1, 'train_loss': mean_loss})
        print(f'diagnostic causal epoch {epoch + 1}/{epochs} loss={mean_loss:.6f}')
    model.eval()
    sample_count = min(causality_samples, len(dataset))
    graph_batch = compute_gradient_causality(model, dataset.windows[:sample_count], dataset.targets[:sample_count])
    A_norm = normal_causal_pattern(sparsify_causality(graph_batch, threshold=0.0)).detach().cpu().numpy().astype(np.float32)
    info = {'history': history, 'window_size': window_size, 'window_count': len(dataset), 'causality_samples': sample_count}
    return model, A_norm, dataset.windows.numpy(), dataset.targets.numpy(), info


def causal_detector(model: BehaviorCausalPredictor, vocab: Dict[Any, int], A_norm: np.ndarray) -> FusionDetector:
    return FusionDetector(backbone=None, causal_model=model, vocab=vocab, A_norm_bank=A_norm, alpha=0.0, beta=1.0)


def causal_scores(
    model: BehaviorCausalPredictor,
    vocab: Dict[Any, int],
    A_norm: np.ndarray,
    val_normals: List[BehaviorSequence],
    test_sequences: List[BehaviorSequence],
) -> Tuple[Dict[str, float], float, List[Dict[str, Any]]]:
    detector = causal_detector(model, vocab, A_norm)
    threshold = detector.calibrate_threshold(val_normals, quantile=0.95)
    scored = detector.score_batch(test_sequences)
    y_true = [label_to_int(sequence.label) for sequence in test_sequences]
    y_score = [float(item['score']) for item in scored]
    metrics = compute_binary_metrics(y_true, y_score, threshold)
    return metrics, threshold, scored


def threshold_sweep(y_true: Sequence[int], y_score: Sequence[float], quantile_threshold: float) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    scores = np.asarray(y_score, dtype=np.float64)
    thresholds = sorted(set(float(value) for value in scores))
    if thresholds:
        thresholds = [min(thresholds) - 1e-9] + thresholds + [max(thresholds) + 1e-9]
    rows: List[Dict[str, Any]] = []
    best: Optional[Dict[str, Any]] = None
    for threshold in thresholds:
        metrics = compute_binary_metrics(y_true, y_score, threshold)
        row = {'threshold': threshold, **metrics, 'kind': 'oracle_candidate'}
        rows.append(row)
        if best is None or float(metrics['f1']) > float(best['f1']):
            best = dict(row)
    quantile_metrics = compute_binary_metrics(y_true, y_score, quantile_threshold)
    quantile_row = {'threshold': quantile_threshold, **quantile_metrics, 'kind': 'quantile_0.95'}
    rows.append(quantile_row)
    return rows, {'oracle_best': best or {}, 'quantile': quantile_row}


def inverse_vocab(vocab: Dict[Any, int]) -> List[Any]:
    inverse: List[Any] = [None] * len(vocab)
    for control, index in dict(vocab).items():
        if 0 <= int(index) < len(inverse):
            inverse[int(index)] = control
    return inverse


def top_edges(A_norm: np.ndarray, vocab: Dict[Any, int], limit: int) -> List[Dict[str, Any]]:
    inverse = inverse_vocab(vocab)
    matrix = np.asarray(A_norm, dtype=np.float32)
    rows: List[Dict[str, Any]] = []
    for src in range(matrix.shape[0]):
        for dst in range(matrix.shape[1]):
            if src == dst:
                continue
            rows.append({'src': inverse[src], 'dst': inverse[dst], 'weight': float(matrix[src, dst])})
    rows.sort(key=lambda item: item['weight'], reverse=True)
    return rows[:limit]


def edge_density(A_norm: np.ndarray) -> Dict[str, Any]:
    matrix = np.asarray(A_norm, dtype=np.float32)
    mask = ~np.eye(matrix.shape[0], dtype=bool)
    values = matrix[mask]
    nonzero = values > 1e-8
    return {
        'node_count': int(matrix.shape[0]),
        'possible_directed_edges': int(values.size),
        'nonzero_edges': int(nonzero.sum()),
        'density': float(nonzero.sum() / max(values.size, 1)),
        'mean_weight': float(values.mean()) if values.size else 0.0,
        'max_weight': float(values.max()) if values.size else 0.0,
    }


def top_edge_stability(model: BehaviorCausalPredictor, windows: np.ndarray, targets: np.ndarray, vocab: Dict[Any, int], top_k: int) -> Dict[str, Any]:
    if windows.shape[0] < 4:
        return {'stable': False, 'jaccard_top_k': 0.0, 'reason': 'too_few_windows'}
    midpoint = windows.shape[0] // 2
    graphs = []
    for start, end in ((0, midpoint), (midpoint, windows.shape[0])):
        graph_batch = compute_gradient_causality(model, torch.from_numpy(windows[start:end]).float(), torch.from_numpy(targets[start:end]).float())
        graph = normal_causal_pattern(sparsify_causality(graph_batch, threshold=0.0)).detach().cpu().numpy().astype(np.float32)
        graphs.append(graph)
    edge_sets = []
    for graph in graphs:
        edge_sets.append({(item['src'], item['dst']) for item in top_edges(graph, vocab, top_k)})
    union = edge_sets[0] | edge_sets[1]
    intersection = edge_sets[0] & edge_sets[1]
    jaccard = len(intersection) / max(len(union), 1)
    return {'stable': bool(jaccard >= 0.25), 'jaccard_top_k': float(jaccard), 'top_k': top_k}


def event_controls(event: BehaviorEvent) -> List[Any]:
    controls = []
    canonical = (event.raw_fields or {}).get('canonical_control')
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
    for index, item in enumerate(sequence.events[start:], start=start):
        if event_matches(item, control):
            return index
    return None


def find_pair(sequence: BehaviorSequence, src: Any, dst: Any) -> Tuple[Optional[int], Optional[int]]:
    src_index = find_index(sequence, src)
    if src_index is None:
        return None, None
    return src_index, find_index(sequence, dst, start=src_index + 1)


def clone_event(source: BehaviorEvent, control_id: Any = None, **updates: Any) -> BehaviorEvent:
    cloned = copy.deepcopy(source)
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
    raw_fields['causal_diagnostic_attack'] = True
    cloned.raw_fields = raw_fields
    return cloned


def make_real_causal_anomaly(source: BehaviorSequence, anomaly_type: str, edges: List[Dict[str, Any]], index: int) -> Optional[BehaviorSequence]:
    for edge in edges:
        src, dst = edge['src'], edge['dst']
        if anomaly_type == 'edge_break':
            _, dst_index = find_pair(source, src, dst)
            if dst_index is None:
                continue
            events = copy.deepcopy(source.events)
            events.pop(dst_index)
        elif anomaly_type == 'edge_injection':
            dst_index = find_index(source, dst)
            if dst_index is None:
                continue
            events = copy.deepcopy(source.events)
            injected = clone_event(source.events[dst_index], control_id=src)
            events = events[: dst_index + 1] + [injected] + events[dst_index + 1 :]
        elif anomaly_type == 'lag_delay':
            _, dst_index = find_pair(source, src, dst)
            if dst_index is None:
                continue
            events = copy.deepcopy(source.events)
            target = events[dst_index]
            events[dst_index] = clone_event(
                target,
                hour=float(target.hour or 0) + 4.0,
                duration=float(target.duration or 0) + 12.0,
            )
        else:
            raise ValueError(f'Unsupported real anomaly {anomaly_type}')
        context = dict(source.context)
        context.update({'diagnostic_attack': anomaly_type, 'top_edge': edge})
        return BehaviorSequence(
            sequence_id=f'{source.sequence_id}::{anomaly_type}_diag_{index:04d}',
            events=events,
            context=context,
            label=1,
            anomaly_type=anomaly_type,
        )
    return None


def build_real_anomalies(normals: List[BehaviorSequence], edges: List[Dict[str, Any]], per_type: int, seed: int) -> Tuple[List[BehaviorSequence], Dict[str, Any]]:
    rng = random.Random(seed)
    anomalies: List[BehaviorSequence] = []
    success = {name: 0 for name in CAUSAL_TYPES}
    skipped = {name: 0 for name in CAUSAL_TYPES}
    for anomaly_type in CAUSAL_TYPES:
        attempts = 0
        while success[anomaly_type] < per_type and attempts < max(per_type * 5, len(normals) * 5):
            attempts += 1
            source = rng.choice(normals)
            anomaly = make_real_causal_anomaly(source, anomaly_type, edges, success[anomaly_type])
            if anomaly is None:
                skipped[anomaly_type] += 1
                continue
            anomalies.append(anomaly)
            success[anomaly_type] += 1
    return anomalies, {'per_type_target': per_type, 'success': success, 'skipped': skipped, 'injected_count': len(anomalies)}


def repeat_normals(normals: List[BehaviorSequence], count: int) -> List[BehaviorSequence]:
    return [normal_copy(normals[index % len(normals)], suffix=f'::diag_normal_{index:04d}') for index in range(count)]


def score_summary(scored: List[Dict[str, Any]], sequences: List[BehaviorSequence]) -> Dict[str, Any]:
    normal_scores = [float(item['score']) for item, sequence in zip(scored, sequences) if label_to_int(sequence.label) == 0]
    anomaly_scores = [float(item['score']) for item, sequence in zip(scored, sequences) if label_to_int(sequence.label) == 1]
    return {
        'normal_count': len(normal_scores),
        'anomaly_count': len(anomaly_scores),
        'normal_mean': float(np.mean(normal_scores)) if normal_scores else None,
        'normal_std': float(np.std(normal_scores)) if normal_scores else None,
        'anomaly_mean': float(np.mean(anomaly_scores)) if anomaly_scores else None,
        'anomaly_std': float(np.std(anomaly_scores)) if anomaly_scores else None,
        'normal_scores': normal_scores,
        'anomaly_scores': anomaly_scores,
    }


def write_csv(path: Path, rows: Iterable[Dict[str, Any]], fieldnames: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json_safe(row.get(key)) for key in fieldnames})


def run_toy(args: argparse.Namespace, output_dir: Path) -> Dict[str, Any]:
    rng = random.Random(args.seed)
    toy_normals = [make_toy_normal(index, rng) for index in range(args.toy_normal_count)]
    train, val, test_normals = split_three(toy_normals)
    toy_anomalies: List[BehaviorSequence] = []
    for index, source in enumerate(test_normals):
        for anomaly_type in CAUSAL_TYPES:
            toy_anomalies.append(make_toy_anomaly(source, anomaly_type, index))
    test_sequences = [normal_copy(item, suffix='::toy_eval') for item in test_normals] + toy_anomalies
    vocab = build_vocab(train + val + test_sequences)
    model, A_norm, _, _, train_info = train_causal_model(
        train,
        vocab,
        epochs=args.epochs,
        hidden_dim=args.hidden_dim,
        batch_size=args.batch_size,
        window_size=args.window_size,
        causality_samples=args.causality_samples,
        seed=args.seed,
    )
    metrics, threshold, scored = causal_scores(model, vocab, A_norm, val, test_sequences)
    result = {
        'passed': bool(metrics['auroc'] >= 0.75),
        'metrics': metrics,
        'threshold': threshold,
        'train_info': train_info,
        'counts': {'train': len(train), 'val': len(val), 'test_normals': len(test_normals), 'anomalies': len(toy_anomalies)},
        'score_summary': score_summary(scored, test_sequences),
        'top_edges': top_edges(A_norm, vocab, 10),
    }
    write_json(output_dir / 'toy_causal_sanity.json', result)
    return result


def run_real(args: argparse.Namespace, output_dir: Path) -> Dict[str, Any]:
    normal_path = PROJECT_ROOT / args.normal_jsonl
    id_to_control_path = PROJECT_ROOT / args.id_to_control
    previous_result_path = PROJECT_ROOT / args.previous_causal_result
    previous_report_path = PROJECT_ROOT / args.previous_feasibility_report
    normals = [normal_copy(item) for item in load_jsonl(normal_path, max_sequences=args.max_normal_sequences)]
    train, val, test_normals = split_three(normals)
    id_to_control = json.loads(id_to_control_path.read_text(encoding='utf-8'))
    previous_result = json.loads(previous_result_path.read_text(encoding='utf-8')) if previous_result_path.exists() else {}
    previous_report_excerpt = previous_report_path.read_text(encoding='utf-8')[:2000] if previous_report_path.exists() else ''
    vocab = build_vocab(normals)
    model, A_norm, windows, targets, train_info = train_causal_model(
        train,
        vocab,
        epochs=args.epochs,
        hidden_dim=args.hidden_dim,
        batch_size=args.batch_size,
        window_size=args.window_size,
        causality_samples=args.causality_samples,
        seed=args.seed + 10,
    )
    edges = top_edges(A_norm, vocab, 50)
    density = edge_density(A_norm)
    stability = top_edge_stability(model, windows, targets, vocab, top_k=min(20, len(vocab) * max(len(vocab) - 1, 1)))
    anomalies, injection_report = build_real_anomalies(test_normals, edges, per_type=args.real_anomalies_per_type, seed=args.seed + 11)
    eval_normals = repeat_normals(test_normals, len(anomalies))
    eval_sequences = eval_normals + anomalies
    rng = random.Random(args.seed + 12)
    rng.shuffle(eval_sequences)
    metrics, quantile_threshold, scored = causal_scores(model, vocab, A_norm, val, eval_sequences)
    y_true = [label_to_int(sequence.label) for sequence in eval_sequences]
    y_score = [float(item['score']) for item in scored]
    sweep_rows, sweep_summary = threshold_sweep(y_true, y_score, quantile_threshold)
    score_dist = score_summary(scored, eval_sequences)

    write_csv(output_dir / 'top_50_edges.csv', edges, ['src', 'dst', 'weight'])
    write_json(output_dir / 'edge_density.json', density)
    write_json(output_dir / 'score_distribution.json', score_dist)
    write_csv(
        output_dir / 'threshold_sweep.csv',
        sweep_rows,
        ['threshold', 'precision', 'recall', 'f1', 'fpr', 'fnr', 'auroc', 'auprc', 'kind'],
    )
    result = {
        'metrics_quantile_threshold': metrics,
        'quantile_threshold': quantile_threshold,
        'oracle_best': sweep_summary['oracle_best'],
        'quantile_sweep_row': sweep_summary['quantile'],
        'score_distribution': score_dist,
        'edge_density': density,
        'top_edge_stability': stability,
        'top_edges': edges,
        'injection_report': injection_report,
        'train_info': train_info,
        'counts': {'train': len(train), 'val': len(val), 'test_normals': len(test_normals), 'anomalies': len(anomalies)},
        'id_to_control_count': len(id_to_control),
        'previous_causal_result': previous_result,
        'previous_feasibility_report_excerpt': previous_report_excerpt,
    }
    write_json(output_dir / 'real_fr_causal_diagnostic.json', result)
    return result


def diagnose_problem(toy: Dict[str, Any], real: Dict[str, Any]) -> str:
    if not toy.get('passed'):
        return 'implementation issue in causal graph or sliding-window target alignment'
    real_auroc = float(real['metrics_quantile_threshold'].get('auroc', 0.0))
    oracle_f1 = float(real['oracle_best'].get('f1', 0.0))
    quantile_f1 = float(real['metrics_quantile_threshold'].get('f1', 0.0))
    if real_auroc < 0.6:
        return 'real FR causal scores have weak separability; likely data/anomaly construction and limited semantic coverage'
    if oracle_f1 > quantile_f1 + 0.1:
        return 'threshold calibration issue; oracle threshold improves causal-only F1 materially'
    return 'data too small / causal signal weaker than reconstruction; implementation sanity passed'


def write_report(path: Path, toy: Dict[str, Any], real: Dict[str, Any], args: argparse.Namespace) -> None:
    problem = diagnose_problem(toy, real)
    keep_causal = bool(toy.get('passed') and float(real['metrics_quantile_threshold'].get('auroc', 0.0)) >= 0.65)
    lines = [
        '# Causal Branch Diagnostic',
        '',
        'Project path: `/home/heyang/projects/CausalGenGuard`',
        'Environment: `/home/heyang/miniconda3/envs/smartguard_env/bin/python`',
        '',
        '## Run Bounds',
        '',
        '- Max normal sequences: {}'.format(args.max_normal_sequences),
        '- Epochs: {}'.format(args.epochs),
        '- Toy normal count: {}'.format(args.toy_normal_count),
        '- Real FR split: train={train}, val={val}, test={test}'.format(
            train=real['counts']['train'], val=real['counts']['val'], test=real['counts']['test_normals']
        ),
        '',
        '## Toy Causal Sanity Check',
        '',
        '- Passed: `{}`'.format(toy['passed']),
        '- causal_only AUROC: {:.4f}'.format(float(toy['metrics'].get('auroc', 0.0))),
        '- causal_only F1: {:.4f}'.format(float(toy['metrics'].get('f1', 0.0))),
        '- Top edges: `{}`'.format(toy['top_edges'][:5]),
        '',
        '## Real FR Diagnostic',
        '',
        '- causal_only AUROC: {:.4f}'.format(float(real['metrics_quantile_threshold'].get('auroc', 0.0))),
        '- quantile threshold F1: {:.4f}'.format(float(real['metrics_quantile_threshold'].get('f1', 0.0))),
        '- oracle best F1: {:.4f}'.format(float(real['oracle_best'].get('f1', 0.0))),
        '- edge_density: {:.4f}'.format(float(real['edge_density']['density'])),
        '- nonzero_edges: {}'.format(real['edge_density']['nonzero_edges']),
        '- top_edge_stability: `{}`'.format(real['top_edge_stability']),
        '- score_distribution normal_mean: {}'.format(real['score_distribution']['normal_mean']),
        '- score_distribution anomaly_mean: {}'.format(real['score_distribution']['anomaly_mean']),
        '- injection_report: `{}`'.format(real['injection_report']),
        '',
        '## Outputs',
        '',
        '- `outputs/diagnostics/causal_branch/toy_causal_sanity.json`',
        '- `outputs/diagnostics/causal_branch/real_fr_causal_diagnostic.json`',
        '- `outputs/diagnostics/causal_branch/top_50_edges.csv`',
        '- `outputs/diagnostics/causal_branch/edge_density.json`',
        '- `outputs/diagnostics/causal_branch/score_distribution.json`',
        '- `outputs/diagnostics/causal_branch/threshold_sweep.csv`',
        '',
        '## Diagnosis',
        '',
        '- Toy causal passed: `{}`'.format(toy['passed']),
        '- Real FR top edges stable: `{}`'.format(real['top_edge_stability'].get('stable')),
        '- Causal score separability: `{}`'.format('weak' if float(real['metrics_quantile_threshold'].get('auroc', 0.0)) < 0.65 else 'moderate'),
        '- Main problem judgement: `{}`'.format(problem),
        '- Recommend keeping causal branch as main innovation: `{}`'.format(keep_causal),
        '',
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text('\n'.join(lines), encoding='utf-8')


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Diagnose CausalGenGuard causal branch behavior.')
    parser.add_argument('--normal-jsonl', default='outputs/processed/fr_sequences_canonical.jsonl')
    parser.add_argument('--id-to-control', default='outputs/mappings/smartguard/fr/id_to_control.json')
    parser.add_argument('--previous-causal-result', default='outputs/results/causal_anomaly_smoke_v2.json')
    parser.add_argument('--previous-feasibility-report', default='outputs/logs/FEASIBILITY_REPORT_V2.md')
    parser.add_argument('--output-dir', default='outputs/diagnostics/causal_branch')
    parser.add_argument('--report', default='outputs/logs/CAUSAL_BRANCH_DIAGNOSTIC.md')
    parser.add_argument('--max-normal-sequences', type=int, default=100)
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--hidden-dim', type=int, default=64)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--window-size', type=int, default=4)
    parser.add_argument('--causality-samples', type=int, default=64)
    parser.add_argument('--toy-normal-count', type=int, default=120)
    parser.add_argument('--real-anomalies-per-type', type=int, default=20)
    parser.add_argument('--seed', type=int, default=42)
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if args.max_normal_sequences > 500:
        raise ValueError('--max-normal-sequences must be <= 500')
    if args.epochs > 5:
        raise ValueError('--epochs must be <= 5')
    output_dir = PROJECT_ROOT / args.output_dir
    report_path = PROJECT_ROOT / args.report
    output_dir.mkdir(parents=True, exist_ok=True)
    toy = run_toy(args, output_dir)
    real = run_real(args, output_dir)
    write_report(report_path, toy, real, args)
    print('toy_causal_auroc={:.6f}'.format(float(toy['metrics'].get('auroc', 0.0))))
    print('true_fr_causal_auroc={:.6f}'.format(float(real['metrics_quantile_threshold'].get('auroc', 0.0))))
    print('oracle_best_f1={:.6f}'.format(float(real['oracle_best'].get('f1', 0.0))))
    print('quantile_threshold_f1={:.6f}'.format(float(real['metrics_quantile_threshold'].get('f1', 0.0))))
    print('problem={}'.format(diagnose_problem(toy, real)))
    print('report={}'.format(report_path))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
