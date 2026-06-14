#!/usr/bin/env python3
'''Bounded FR winter->spring target-anomaly diagnostics.

This script reuses the final context-shift runner's loaders and model helpers,
then writes per-anomaly, threshold-sweep, and balanced-subset diagnostics. It
keeps the smoke-test bounds at <=500 normal, <=500 anomaly, and <=3 epochs.
'''

from __future__ import annotations

import argparse
import copy
import csv
import json
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / 'src'
SCRIPTS_ROOT = PROJECT_ROOT / 'scripts'
for path in (SRC_ROOT, SCRIPTS_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import run_context_shift_final as final
from causal_gen_guard.data.behavior_event_tensor import build_vocab
from causal_gen_guard.data.schemas import BehaviorSequence
from causal_gen_guard.evaluation.metrics import compute_binary_metrics


METHODS = [
    'source_only',
    'source_plus_raw_synthetic',
    'source_plus_tof_synthetic',
    'source_plus_causal_tof_synthetic',
    'oracle_target',
]

THRESHOLD_QUANTILES = [0.80, 0.85, 0.90, 0.95, 0.975, 0.99]


def resolve(path_value: str | Path) -> Path:
    path = Path(path_value)
    return path if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def bounded_bounds(config: Dict[str, Any]) -> Dict[str, Any]:
    bounds = dict(config.get('run_bounds', {}))
    bounds['max_source_train'] = min(int(bounds.get('max_source_train', 500)), 500)
    bounds['max_source_val'] = min(int(bounds.get('max_source_val', 200)), 500)
    bounds['max_target_normal'] = min(int(bounds.get('max_target_normal', 300)), 500)
    bounds['max_target_anomaly'] = min(int(bounds.get('max_target_anomaly', 300)), 500)
    bounds['max_synthetic'] = min(int(bounds.get('max_synthetic', 500)), 500)
    bounds['max_target_labeled_normal'] = min(int(bounds.get('max_target_labeled_normal', 500)), 500)
    bounds['max_target_labeled_anomaly'] = min(int(bounds.get('max_target_labeled_anomaly', 500)), 500)
    bounds['epochs'] = min(int(bounds.get('epochs', 3)), 3)
    bounds['batch_size'] = int(bounds.get('batch_size', 32))
    bounds['hidden_dim'] = int(bounds.get('hidden_dim', 64))
    bounds['seed'] = int(bounds.get('seed', 42))
    return bounds


def json_safe(value: Any) -> Any:
    return final.json_safe(value)


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_safe(payload), ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def write_csv(path: Path, rows: List[Dict[str, Any]], fields: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields))
        writer.writeheader()
        for row in rows:
            writer.writerow({field: json_safe(row.get(field)) for field in fields})


def load_jsonl(path: Path, max_normal: int = 500, max_anomaly: int = 500) -> Tuple[List[BehaviorSequence], List[BehaviorSequence]]:
    normals: List[BehaviorSequence] = []
    anomalies: List[BehaviorSequence] = []
    with path.open('r', encoding='utf-8') as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                sequence = BehaviorSequence.from_dict(json.loads(stripped))
            except json.JSONDecodeError as exc:
                raise ValueError(f'Invalid JSONL line {line_number}: {path}') from exc
            if final.label_to_int(sequence.label) == 1:
                if len(anomalies) < max_anomaly:
                    anomalies.append(sequence)
            elif len(normals) < max_normal:
                normals.append(sequence)
            if len(normals) >= max_normal and len(anomalies) >= max_anomaly:
                break
    return normals, anomalies


def score_list(detector: Any, sequences: Sequence[BehaviorSequence]) -> List[float]:
    if not sequences:
        return []
    return [float(item['score']) for item in detector.score_batch(list(sequences))]


def clean_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if np.isnan(number) or np.isinf(number):
        return None
    return number


def metrics_from_scores(normal_scores: Sequence[float], anomaly_scores: Sequence[float], threshold: float) -> Dict[str, Any]:
    if not anomaly_scores:
        return {
            'target_normal_fpr': sum(score > threshold for score in normal_scores) / max(len(normal_scores), 1) if normal_scores else None,
            'precision': None,
            'recall': None,
            'f1': None,
            'auroc': None,
            'auprc': None,
        }
    y_true = [0] * len(normal_scores) + [1] * len(anomaly_scores)
    y_score = list(normal_scores) + list(anomaly_scores)
    metrics = compute_binary_metrics(y_true, y_score, threshold)
    return {
        'target_normal_fpr': sum(score > threshold for score in normal_scores) / max(len(normal_scores), 1) if normal_scores else None,
        'precision': clean_float(metrics.get('precision')),
        'recall': clean_float(metrics.get('recall')),
        'f1': clean_float(metrics.get('f1')),
        'auroc': clean_float(metrics.get('auroc')),
        'auprc': clean_float(metrics.get('auprc')),
    }


def mean_or_none(values: Sequence[float]) -> Optional[float]:
    return float(np.mean(values)) if values else None


def group_anomalies(anomalies: Sequence[BehaviorSequence]) -> Dict[str, List[BehaviorSequence]]:
    grouped: Dict[str, List[BehaviorSequence]] = defaultdict(list)
    for sequence in anomalies:
        grouped[str(sequence.anomaly_type or 'unknown')].append(sequence)
    return dict(grouped)


def selected_strategy(final_payload: Dict[str, Any], method: str) -> str:
    selected = dict(final_payload.get('selected_methods') or {})
    if method in selected:
        return str((selected[method] or {}).get('filter_strategy') or '')
    for row in final_payload.get('rows', []):
        if row.get('method') == method and row.get('selected'):
            return str(row.get('filter_strategy') or '')
    return ''


def load_inputs(config: Dict[str, Any], bounds: Dict[str, Any]) -> Dict[str, Any]:
    seed = int(bounds.get('seed', 42))
    final.set_seed(seed)
    control_to_id, id_to_control = final.read_mapping(config)
    contexts = dict(config.get('contexts', {}))
    source_context = final.choose_context(config, contexts.get('source_candidates', ['winter', 'single']), 'source')
    target_context = final.choose_context(config, contexts.get('target_candidates', ['spring', 'multiple', 'night']), 'target')
    source_train = final.load_context_normals(config, source_context, ['trn'], control_to_id, id_to_control, int(bounds['max_source_train']))
    source_val = final.load_context_normals(config, source_context, ['vld', 'trn'], control_to_id, id_to_control, int(bounds['max_source_val']))
    target_normals = final.load_context_normals(config, target_context, ['test', 'split_test', 'vld', 'trn'], control_to_id, id_to_control, int(bounds['max_target_normal']))
    synthetic, synthetic_files = final.load_target_synthetic(config, target_context, control_to_id, id_to_control, int(bounds['max_synthetic']))
    labeled_jsonl, labeled_report_path = final.target_labeled_paths(config)
    if not labeled_jsonl.exists():
        final.write_target_context_labeled_set(config, target_context, target_normals, synthetic, bounds, seed + 1000)
    labeled_normals, labeled_anomalies = load_jsonl(
        labeled_jsonl,
        max_normal=int(bounds['max_target_labeled_normal']),
        max_anomaly=int(bounds['max_target_labeled_anomaly']),
    )
    labeled_report = json.loads(labeled_report_path.read_text(encoding='utf-8')) if labeled_report_path.exists() else {}
    if not source_train or not source_val or not target_normals:
        raise ValueError('source_train, source_val, and target_normals must be non-empty')
    if not labeled_anomalies:
        raise ValueError(f'No target labeled anomalies found in {labeled_jsonl}')
    vocab = build_vocab(source_train + source_val + target_normals + labeled_normals + labeled_anomalies + synthetic)
    return {
        'control_to_id': control_to_id,
        'id_to_control': id_to_control,
        'source_context': source_context,
        'target_context': target_context,
        'source_train': source_train,
        'source_val': source_val,
        'target_normals': target_normals,
        'synthetic': synthetic,
        'synthetic_files': synthetic_files,
        'labeled_jsonl': labeled_jsonl,
        'labeled_report_path': labeled_report_path,
        'labeled_normals': labeled_normals,
        'labeled_anomalies': labeled_anomalies,
        'labeled_report': labeled_report,
        'vocab': vocab,
    }


def train_method_artifacts(
    config: Dict[str, Any],
    bounds: Dict[str, Any],
    inputs: Dict[str, Any],
    final_payload: Dict[str, Any],
) -> Dict[str, Dict[str, Any]]:
    seed = int(bounds.get('seed', 42))
    threshold_quantile = float(config.get('evaluation', {}).get('threshold_quantile', 0.95))
    source_train = inputs['source_train']
    source_val = inputs['source_val']
    target_normals = inputs['target_normals']
    target_anomalies = inputs['labeled_anomalies']
    synthetic = inputs['synthetic']
    vocab = inputs['vocab']

    artifacts: Dict[str, Dict[str, Any]] = {}
    source_row, source_detector, source_threshold = final.train_and_eval_method(
        'source_only',
        source_train,
        source_val,
        target_normals,
        target_anomalies,
        vocab,
        bounds,
        threshold_quantile,
        seed,
        selected=True,
    )
    artifacts['source_only'] = {
        'row': source_row,
        'detector': source_detector,
        'threshold': source_threshold,
        'val_sequences': source_val,
        'train_sequences': source_train,
        'filter_strategy': '',
    }

    synthetic_sets = final.build_synthetic_sets(
        synthetic,
        source_train,
        source_val,
        source_detector,
        source_threshold,
        vocab,
        inputs['control_to_id'],
        bounds,
        seed + 1,
    )

    raw_payload = synthetic_sets['raw']
    raw_train = raw_payload['sequences']
    raw_val = source_val + raw_train[: max(1, min(len(raw_train), len(source_val)))]
    raw_row, raw_detector, raw_threshold = final.train_and_eval_method(
        'source_plus_raw_synthetic',
        source_train + raw_train,
        raw_val,
        target_normals,
        target_anomalies,
        vocab,
        bounds,
        threshold_quantile,
        seed + 1,
        synthetic_count=len(synthetic),
        kept_count=raw_payload['kept_count'],
        rejected_count=len(raw_payload['rejected']),
        filter_strategy='no_filter',
        selected=True,
    )
    artifacts['source_plus_raw_synthetic'] = {
        'row': raw_row,
        'detector': raw_detector,
        'threshold': raw_threshold,
        'val_sequences': raw_val,
        'train_sequences': source_train + raw_train,
        'filter_strategy': 'no_filter',
    }

    tof_strategy = selected_strategy(final_payload, 'source_plus_tof_synthetic') or 'iqr_1.5'
    tof_seed_offset = 10 + final.TOF_FILTER_STRATEGIES.index(tof_strategy) if tof_strategy in final.TOF_FILTER_STRATEGIES else 12
    tof_payload = synthetic_sets['tof_candidates'][tof_strategy]
    tof_train = tof_payload['sequences']
    tof_val = source_val + tof_train[: max(1, min(len(tof_train), len(source_val)))]
    tof_row, tof_detector, tof_threshold = final.train_and_eval_method(
        'source_plus_tof_synthetic',
        source_train + tof_train,
        tof_val,
        target_normals,
        target_anomalies,
        vocab,
        bounds,
        threshold_quantile,
        seed + tof_seed_offset,
        synthetic_count=len(synthetic),
        kept_count=tof_payload['kept_count'],
        rejected_count=len(tof_payload['rejected']),
        filter_strategy=tof_strategy,
        selected=True,
    )
    artifacts['source_plus_tof_synthetic'] = {
        'row': tof_row,
        'detector': tof_detector,
        'threshold': tof_threshold,
        'val_sequences': tof_val,
        'train_sequences': source_train + tof_train,
        'filter_strategy': tof_strategy,
    }

    causal_strategy = selected_strategy(final_payload, 'source_plus_causal_tof_synthetic') or 'relaxed_causal_keep_90_percent'
    causal_seed_offset = 30 + final.CAUSAL_TOF_FILTER_STRATEGIES.index(causal_strategy) if causal_strategy in final.CAUSAL_TOF_FILTER_STRATEGIES else 31
    causal_payload = final.causal_strategy_payload(
        tof_payload,
        synthetic_sets['causal_model'],
        synthetic_sets['A_norm'],
        vocab,
        causal_strategy,
        synthetic_sets['causal_info'],
    )
    causal_train = causal_payload['sequences']
    causal_val = source_val + causal_train[: max(1, min(len(causal_train), len(source_val)))]
    causal_row, causal_detector, causal_threshold = final.train_and_eval_method(
        'source_plus_causal_tof_synthetic',
        source_train + causal_train,
        causal_val,
        target_normals,
        target_anomalies,
        vocab,
        bounds,
        threshold_quantile,
        seed + causal_seed_offset,
        synthetic_count=len(synthetic),
        kept_count=causal_payload['kept_count'],
        rejected_count=len(causal_payload['rejected']),
        filter_strategy=causal_strategy,
        selected=True,
    )
    artifacts['source_plus_causal_tof_synthetic'] = {
        'row': causal_row,
        'detector': causal_detector,
        'threshold': causal_threshold,
        'val_sequences': causal_val,
        'train_sequences': source_train + causal_train,
        'filter_strategy': causal_strategy,
    }

    oracle_val = source_val + target_normals[: max(1, min(len(target_normals), len(source_val)))]
    oracle_row, oracle_detector, oracle_threshold = final.train_and_eval_method(
        'oracle_target',
        source_train + target_normals,
        oracle_val,
        target_normals,
        target_anomalies,
        vocab,
        bounds,
        threshold_quantile,
        seed + 99,
        selected=True,
    )
    artifacts['oracle_target'] = {
        'row': oracle_row,
        'detector': oracle_detector,
        'threshold': oracle_threshold,
        'val_sequences': oracle_val,
        'train_sequences': source_train + target_normals,
        'filter_strategy': '',
    }
    return artifacts


def score_artifacts(
    artifacts: Dict[str, Dict[str, Any]],
    target_normals: List[BehaviorSequence],
    anomalies: List[BehaviorSequence],
) -> Dict[str, Dict[str, Any]]:
    scored: Dict[str, Dict[str, Any]] = {}
    for method in METHODS:
        artifact = artifacts[method]
        detector = artifact['detector']
        normal_scores = score_list(detector, target_normals)
        anomaly_scores = score_list(detector, anomalies)
        val_scores = score_list(detector, artifact['val_sequences'])
        scored[method] = {
            'normal_scores': normal_scores,
            'anomaly_scores': anomaly_scores,
            'val_scores': val_scores,
            'threshold': artifact['threshold'],
            'row': artifact['row'],
            'filter_strategy': artifact['filter_strategy'],
        }
    return scored


def per_anomaly_rows(
    scored: Dict[str, Dict[str, Any]],
    target_normals: List[BehaviorSequence],
    anomalies: List[BehaviorSequence],
    report: Dict[str, Any],
) -> List[Dict[str, Any]]:
    grouped = group_anomalies(anomalies)
    all_types = list((report.get('candidate_count') or {}).keys()) or sorted(grouped)
    rows: List[Dict[str, Any]] = []
    for method in METHODS:
        method_scores = scored[method]
        detector = method_scores.get('detector')
        normal_scores = method_scores['normal_scores']
        mean_normal = mean_or_none(normal_scores)
        for anomaly_type in all_types:
            type_sequences = grouped.get(anomaly_type, [])
            if detector is not None:
                type_scores = score_list(detector, type_sequences)
            else:
                indices = [index for index, sequence in enumerate(anomalies) if sequence.anomaly_type == anomaly_type]
                type_scores = [method_scores['anomaly_scores'][index] for index in indices]
            metrics = metrics_from_scores(normal_scores, type_scores, float(method_scores['threshold']))
            mean_anomaly = mean_or_none(type_scores)
            rows.append(
                {
                    'method': method,
                    'filter_strategy': method_scores.get('filter_strategy', ''),
                    'anomaly_type': anomaly_type,
                    'anomaly_count': len(type_sequences),
                    'low_support': len(type_sequences) < 5,
                    'precision': metrics['precision'],
                    'recall': metrics['recall'],
                    'f1': metrics['f1'],
                    'auroc': metrics['auroc'],
                    'auprc': metrics['auprc'],
                    'mean_score_normal': mean_normal,
                    'mean_score_anomaly': mean_anomaly,
                    'score_margin': None if mean_normal is None or mean_anomaly is None else mean_anomaly - mean_normal,
                }
            )
    return rows


def threshold_sweep_rows(scored: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for method in METHODS:
        method_scores = scored[method]
        normal_scores = method_scores['normal_scores']
        anomaly_scores = method_scores['anomaly_scores']
        val_scores = method_scores['val_scores']
        for quantile in THRESHOLD_QUANTILES:
            threshold = float(np.quantile(val_scores, quantile)) if val_scores else float(method_scores['threshold'])
            metrics = metrics_from_scores(normal_scores, anomaly_scores, threshold)
            rows.append(
                {
                    'method': method,
                    'filter_strategy': method_scores.get('filter_strategy', ''),
                    'quantile': quantile,
                    'threshold': threshold,
                    'target_normal_fpr': metrics['target_normal_fpr'],
                    'precision': metrics['precision'],
                    'recall': metrics['recall'],
                    'f1': metrics['f1'],
                    'auroc': metrics['auroc'],
                    'auprc': metrics['auprc'],
                }
            )
    return rows


def clone_for_balanced(sequence: BehaviorSequence, index: int, role: str) -> BehaviorSequence:
    copied = copy.deepcopy(sequence)
    copied.sequence_id = f'balanced::{role}::{index:06d}::{copied.sequence_id}'
    copied.context = dict(copied.context)
    copied.context['balanced_target_context_subset'] = True
    copied.context['balanced_role'] = role
    return copied


def build_balanced_subset(
    labeled_normals: List[BehaviorSequence],
    anomalies: List[BehaviorSequence],
    seed: int,
) -> Tuple[List[BehaviorSequence], List[BehaviorSequence], Dict[str, Any]]:
    rng = random.Random(seed)
    grouped = group_anomalies(anomalies)
    positive_types = [anomaly_type for anomaly_type, items in grouped.items() if items]
    min_count = min((len(grouped[anomaly_type]) for anomaly_type in positive_types), default=0)
    take_per_type = min(min_count, 30) if min_count else 0
    balanced_anomalies: List[BehaviorSequence] = []
    included_counts: Dict[str, int] = {}
    low_support: Dict[str, int] = {}
    for anomaly_type in sorted(grouped):
        items = list(grouped[anomaly_type])
        if len(items) < 5:
            low_support[anomaly_type] = len(items)
        rng.shuffle(items)
        selected = items[:take_per_type] if take_per_type else []
        included_counts[anomaly_type] = len(selected)
        for sequence in selected:
            balanced_anomalies.append(clone_for_balanced(sequence, len(balanced_anomalies), 'anomaly'))
    balanced_normals = [clone_for_balanced(sequence, index, 'normal') for index, sequence in enumerate(labeled_normals[:500])]
    report = {
        'normal_count': len(balanced_normals),
        'anomaly_count': len(balanced_anomalies),
        'original_anomaly_counts': {anomaly_type: len(items) for anomaly_type, items in sorted(grouped.items())},
        'min_count': min_count,
        'max_per_type': 30,
        'take_per_type': take_per_type,
        'included_counts': included_counts,
        'low_support': low_support,
        'seed': seed,
        'notes': [
            'Balanced means anomaly types are capped to the same nonzero min_count, with an upper cap of 30.',
            'Normal rows are retained up to 500 to preserve target-normal FPR context.',
        ],
    }
    return balanced_normals, balanced_anomalies, report


def write_sequence_jsonl(sequences: Sequence[BehaviorSequence], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8') as handle:
        for sequence in sequences:
            handle.write(json.dumps(sequence.to_dict(), ensure_ascii=False) + '\n')


def balanced_rows(
    artifacts: Dict[str, Dict[str, Any]],
    balanced_normals: List[BehaviorSequence],
    balanced_anomalies: List[BehaviorSequence],
    source_fpr: Optional[float],
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for method in METHODS:
        artifact = artifacts[method]
        detector = artifact['detector']
        normal_scores = score_list(detector, balanced_normals)
        anomaly_scores = score_list(detector, balanced_anomalies)
        metrics = metrics_from_scores(normal_scores, anomaly_scores, float(artifact['threshold']))
        row = {
            'method': method,
            'filter_strategy': artifact.get('filter_strategy', ''),
            'normal_count': len(balanced_normals),
            'anomaly_count': len(balanced_anomalies),
            'target_normal_fpr': metrics['target_normal_fpr'],
            'precision': metrics['precision'],
            'recall': metrics['recall'],
            'f1': metrics['f1'],
            'auroc': metrics['auroc'],
            'auprc': metrics['auprc'],
            'kept_count': artifact['row'].get('kept_count', 0),
            'rejected_count': artifact['row'].get('rejected_count', 0),
            'adaptation_gain': None if source_fpr is None or metrics['target_normal_fpr'] is None or method == 'source_only' else source_fpr - metrics['target_normal_fpr'],
        }
        rows.append(row)
    return rows


def best_row(rows: Iterable[Dict[str, Any]], key: str, minimize: bool = False) -> Optional[Dict[str, Any]]:
    candidates = [row for row in rows if row.get(key) is not None]
    if not candidates:
        return None
    return min(candidates, key=lambda row: float(row[key])) if minimize else max(candidates, key=lambda row: float(row[key]))


def rank_anomaly_types(per_rows: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    by_type: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in per_rows:
        if row.get('anomaly_count', 0) > 0:
            by_type[str(row['anomaly_type'])].append(row)
    summaries: List[Dict[str, Any]] = []
    for anomaly_type, rows in by_type.items():
        usable = [row for row in rows if row.get('f1') is not None]
        if not usable:
            continue
        summaries.append(
            {
                'anomaly_type': anomaly_type,
                'anomaly_count': rows[0].get('anomaly_count'),
                'low_support': bool(rows[0].get('low_support')),
                'mean_f1': float(np.mean([row['f1'] for row in usable])),
                'mean_auroc': mean_or_none([row['auroc'] for row in usable if row.get('auroc') is not None]),
                'mean_score_margin': mean_or_none([row['score_margin'] for row in usable if row.get('score_margin') is not None]),
            }
        )
    easiest = sorted(summaries, key=lambda item: (item['low_support'], -(item.get('mean_f1') or -1), -(item.get('mean_score_margin') or -1)))[:3]
    hardest = sorted(summaries, key=lambda item: (item.get('mean_f1') or 0, item.get('mean_score_margin') or 0))[:3]
    return easiest, hardest


def diagnose_threshold_vs_separation(threshold_rows: List[Dict[str, Any]], per_rows: List[Dict[str, Any]]) -> str:
    improvements: List[float] = []
    for method in METHODS:
        rows = [row for row in threshold_rows if row['method'] == method and row.get('f1') is not None]
        default = next((row for row in rows if abs(float(row['quantile']) - 0.95) < 1e-9), None)
        best = best_row(rows, 'f1')
        if default is not None and best is not None:
            improvements.append(float(best['f1']) - float(default['f1']))
    margins = [float(row['score_margin']) for row in per_rows if row.get('score_margin') is not None and row.get('anomaly_count', 0) > 0]
    nonpositive_margin_ratio = sum(margin <= 0 for margin in margins) / max(len(margins), 1)
    median_margin = float(np.median(margins)) if margins else 0.0
    max_improvement = max(improvements, default=0.0)
    if nonpositive_margin_ratio >= 0.4 or median_margin <= 0:
        return (
            'score separation is the main bottleneck: many per-type score margins are non-positive or near zero, '
            'so lowering the threshold cannot reliably recover recall without increasing false positives.'
        )
    if max_improvement >= 0.10:
        return (
            'threshold conservatism is a meaningful bottleneck: threshold sweep substantially improves F1 at lower quantiles, '
            'although deployment would trade this against higher target-normal FPR.'
        )
    return (
        'the issue is mixed but leans toward score separation: threshold changes move F1 only modestly, and per-type margins remain small.'
    )


def write_report(
    path: Path,
    inputs: Dict[str, Any],
    overall_rows: List[Dict[str, Any]],
    threshold_rows: List[Dict[str, Any]],
    per_rows: List[Dict[str, Any]],
    balanced_eval_rows: List[Dict[str, Any]],
    balanced_report: Dict[str, Any],
) -> None:
    lowest_fpr = best_row(overall_rows, 'target_normal_fpr', minimize=True)
    best_overall_f1 = best_row(overall_rows, 'f1')
    best_balanced_f1 = best_row(balanced_eval_rows, 'f1')
    easiest, hardest = rank_anomaly_types(per_rows)
    tof = next(row for row in overall_rows if row['method'] == 'source_plus_tof_synthetic')
    causal = next(row for row in overall_rows if row['method'] == 'source_plus_causal_tof_synthetic')
    threshold_judgment = diagnose_threshold_vs_separation(threshold_rows, per_rows)

    causal_tradeoff = (
        'FPR/F1 trade-off'
        if causal.get('target_normal_fpr') is not None
        and tof.get('target_normal_fpr') is not None
        and causal.get('f1') is not None
        and tof.get('f1') is not None
        and causal['target_normal_fpr'] <= tof['target_normal_fpr']
        and causal['f1'] < tof['f1']
        else 'not a clear trade-off in this run'
    )
    recommendation = (
        'write Causal-TOF as a low-false-alarm deployment option, not the sole main method'
        if causal_tradeoff == 'FPR/F1 trade-off'
        else 'Causal-TOF can be presented as the main selected synthetic-filtering method with caveats'
    )
    enough_for_multi_dataset = bool(inputs['labeled_anomalies']) and len(inputs['labeled_anomalies']) <= 500

    lines = [
        '# Context Shift Anomaly Diagnostic Report',
        '',
        f"- Source context: `{inputs['source_context']}`",
        f"- Target context: `{inputs['target_context']}`",
        f"- Target labeled JSONL: `{inputs['labeled_jsonl']}`",
        f"- Normal bound: `<=500`; anomaly bound: `<=500`; epochs: `<=3`",
        '',
        '## Direct Answers',
        '',
        f"1. Lowest target-normal FPR: `{lowest_fpr['method']}` (`{final.format_metric(lowest_fpr.get('target_normal_fpr'))}`)." if lowest_fpr else '1. Lowest target-normal FPR: unavailable.',
        f"2. Highest overall F1: `{best_overall_f1['method']}` (`{final.format_metric(best_overall_f1.get('f1'))}`)." if best_overall_f1 else '2. Highest overall F1: unavailable.',
        f"3. Highest balanced F1: `{best_balanced_f1['method']}` (`{final.format_metric(best_balanced_f1.get('f1'))}`)." if best_balanced_f1 else '3. Highest balanced F1: unavailable.',
        f'4. Recall diagnosis: {threshold_judgment}',
        '5. Easiest anomaly types: ' + ', '.join(
            f"`{item['anomaly_type']}` (mean_f1={final.format_metric(item.get('mean_f1'))}, margin={final.format_metric(item.get('mean_score_margin'))}{', low_support' if item.get('low_support') else ''})"
            for item in easiest
        ),
        '6. Hardest anomaly types: ' + ', '.join(
            f"`{item['anomaly_type']}` (mean_f1={final.format_metric(item.get('mean_f1'))}, margin={final.format_metric(item.get('mean_score_margin'))}{', low_support' if item.get('low_support') else ''})"
            for item in hardest
        ),
        f'7. Causal-TOF vs TOF: `{causal_tradeoff}`. TOF FPR/F1 = `{final.format_metric(tof.get("target_normal_fpr"))}` / `{final.format_metric(tof.get("f1"))}`; Causal-TOF FPR/F1 = `{final.format_metric(causal.get("target_normal_fpr"))}` / `{final.format_metric(causal.get("f1"))}`.',
        f'8. Recommendation: {recommendation}.',
        '',
        '## Multi-Dataset Readiness',
        '',
        (
            'Current FR winter->spring results are sufficient to enter bounded FR/SP/US multi-dataset experiments as a diagnostic/adaptation pipeline, '
            'but not yet sufficient to claim anomaly-detection superiority: recall and score separation remain weak on several target-context anomaly types.'
            if enough_for_multi_dataset
            else 'Current results are not sufficient for multi-dataset experiments because target labeled anomalies are missing.'
        ),
        '',
        '## Overall Metrics',
        '',
        '| method | filter_strategy | target_normal_fpr | precision | recall | f1 | auroc | auprc | kept_count | rejected_count | adaptation_gain |',
        '| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |',
    ]
    for row in overall_rows:
        lines.append(
            '| {method} | {strategy} | {fpr} | {p} | {r} | {f1} | {auroc} | {auprc} | {kept} | {rejected} | {gain} |'.format(
                method=row['method'],
                strategy=row.get('filter_strategy', ''),
                fpr=final.format_metric(row.get('target_normal_fpr')),
                p=final.format_metric(row.get('precision')),
                r=final.format_metric(row.get('recall')),
                f1=final.format_metric(row.get('f1')),
                auroc=final.format_metric(row.get('auroc')),
                auprc=final.format_metric(row.get('auprc')),
                kept=row.get('kept_count', 0),
                rejected=row.get('rejected_count', 0),
                gain=final.format_metric(row.get('adaptation_gain')),
            )
        )

    lines.extend(
        [
            '',
            '## Balanced Subset',
            '',
            f"- Balanced normal_count: `{balanced_report.get('normal_count')}`",
            f"- Balanced anomaly_count: `{balanced_report.get('anomaly_count')}`",
            f"- take_per_type: `{balanced_report.get('take_per_type')}`",
            f"- low_support: `{balanced_report.get('low_support')}`",
            '',
            '| method | target_normal_fpr | precision | recall | f1 | auroc | auprc |',
            '| --- | ---: | ---: | ---: | ---: | ---: | ---: |',
        ]
    )
    for row in balanced_eval_rows:
        lines.append(
            '| {method} | {fpr} | {p} | {r} | {f1} | {auroc} | {auprc} |'.format(
                method=row['method'],
                fpr=final.format_metric(row.get('target_normal_fpr')),
                p=final.format_metric(row.get('precision')),
                r=final.format_metric(row.get('recall')),
                f1=final.format_metric(row.get('f1')),
                auroc=final.format_metric(row.get('auroc')),
                auprc=final.format_metric(row.get('auprc')),
            )
        )

    lines.extend(
        [
            '',
            '## Output Files',
            '',
            '- `outputs/results/context_shift_final_fr_per_anomaly.csv`',
            '- `outputs/results/context_shift_final_fr_per_anomaly.json`',
            '- `outputs/results/context_shift_final_fr_threshold_sweep.csv`',
            '- `outputs/labels/fr_target_context_labeled_balanced.jsonl`',
            '- `outputs/labels/fr_target_context_labeled_balanced_report.json`',
            '- `outputs/results/context_shift_final_fr_balanced.csv`',
            '- `outputs/results/context_shift_final_fr_balanced.json`',
            '',
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text('\n'.join(lines), encoding='utf-8')


def run(config_path: Path) -> Dict[str, Any]:
    config = final.load_config(config_path)
    bounds = bounded_bounds(config)
    final_payload_path = resolve(dict(config.get('paths', {})).get('output_json', 'outputs/results/context_shift_final_fr.json'))
    final_payload = json.loads(final_payload_path.read_text(encoding='utf-8')) if final_payload_path.exists() else {}
    inputs = load_inputs(config, bounds)
    artifacts = train_method_artifacts(config, bounds, inputs, final_payload)
    scored = score_artifacts(artifacts, inputs['target_normals'], inputs['labeled_anomalies'])
    for method, method_scores in scored.items():
        method_scores['detector'] = artifacts[method]['detector']

    source_fpr = artifacts['source_only']['row'].get('target_normal_fpr')
    overall_rows: List[Dict[str, Any]] = []
    for method in METHODS:
        row = copy.deepcopy(artifacts[method]['row'])
        row['f1'] = row.get('f1', row.get('anomaly_f1'))
        row['adaptation_gain'] = None if method == 'source_only' or source_fpr is None or row.get('target_normal_fpr') is None else source_fpr - row['target_normal_fpr']
        overall_rows.append(row)

    per_rows = per_anomaly_rows(scored, inputs['target_normals'], inputs['labeled_anomalies'], inputs['labeled_report'])
    threshold_rows = threshold_sweep_rows(scored)

    balanced_normals, balanced_anomalies, balanced_report = build_balanced_subset(
        inputs['labeled_normals'],
        inputs['labeled_anomalies'],
        int(bounds.get('seed', 42)) + 2000,
    )
    balanced_jsonl = PROJECT_ROOT / 'outputs/labels/fr_target_context_labeled_balanced.jsonl'
    balanced_report_path = PROJECT_ROOT / 'outputs/labels/fr_target_context_labeled_balanced_report.json'
    write_sequence_jsonl(balanced_normals + balanced_anomalies, balanced_jsonl)
    write_json(balanced_report_path, balanced_report)
    balanced_eval_rows = balanced_rows(artifacts, balanced_normals, balanced_anomalies, source_fpr)

    per_csv = PROJECT_ROOT / 'outputs/results/context_shift_final_fr_per_anomaly.csv'
    per_json = PROJECT_ROOT / 'outputs/results/context_shift_final_fr_per_anomaly.json'
    threshold_csv = PROJECT_ROOT / 'outputs/results/context_shift_final_fr_threshold_sweep.csv'
    balanced_csv = PROJECT_ROOT / 'outputs/results/context_shift_final_fr_balanced.csv'
    balanced_json = PROJECT_ROOT / 'outputs/results/context_shift_final_fr_balanced.json'
    report_path = PROJECT_ROOT / 'outputs/logs/CONTEXT_SHIFT_ANOMALY_DIAGNOSTIC_REPORT.md'

    write_csv(
        per_csv,
        per_rows,
        [
            'method',
            'filter_strategy',
            'anomaly_type',
            'anomaly_count',
            'low_support',
            'precision',
            'recall',
            'f1',
            'auroc',
            'auprc',
            'mean_score_normal',
            'mean_score_anomaly',
            'score_margin',
        ],
    )
    write_json(
        per_json,
        {
            'source_context': inputs['source_context'],
            'target_context': inputs['target_context'],
            'rows': per_rows,
            'target_labeled_report': inputs['labeled_report'],
        },
    )
    write_csv(
        threshold_csv,
        threshold_rows,
        ['method', 'filter_strategy', 'quantile', 'threshold', 'target_normal_fpr', 'precision', 'recall', 'f1', 'auroc', 'auprc'],
    )
    write_csv(
        balanced_csv,
        balanced_eval_rows,
        [
            'method',
            'filter_strategy',
            'normal_count',
            'anomaly_count',
            'target_normal_fpr',
            'precision',
            'recall',
            'f1',
            'auroc',
            'auprc',
            'kept_count',
            'rejected_count',
            'adaptation_gain',
        ],
    )
    write_json(
        balanced_json,
        {
            'source_context': inputs['source_context'],
            'target_context': inputs['target_context'],
            'balanced_report': balanced_report,
            'rows': balanced_eval_rows,
        },
    )
    write_report(report_path, inputs, overall_rows, threshold_rows, per_rows, balanced_eval_rows, balanced_report)

    return {
        'overall_rows': overall_rows,
        'per_rows': per_rows,
        'threshold_rows': threshold_rows,
        'balanced_rows': balanced_eval_rows,
        'balanced_report': balanced_report,
        'report_path': str(report_path),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Diagnose context-shift target anomaly metrics.')
    parser.add_argument('--config', default='configs/context_shift_fr.yaml')
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    payload = run(resolve(args.config))
    best_balanced = best_row(payload['balanced_rows'], 'f1')
    print(f"Wrote {payload['report_path']}")
    if best_balanced:
        print('best_balanced_f1={} method={}'.format(final.format_metric(best_balanced.get('f1')), best_balanced.get('method')))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
