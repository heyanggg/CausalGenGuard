#!/usr/bin/env python3
'''Run small CausalGenGuard feasibility experiments.

This runner is intentionally bounded: it uses at most 500 SmartGuard sequences
by default and trains each model for at most 2 epochs unless explicitly lowered.
It never modifies source projects and writes all artifacts under outputs/.
'''

from __future__ import annotations

import argparse
import copy
import json
import random
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / 'src'
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from causal_gen_guard.data.attack_injector import inject_causal_anomaly, inject_smartguard_style
from causal_gen_guard.data.behavior_event_tensor import build_vocab
from causal_gen_guard.data.schemas import BehaviorEvent, BehaviorSequence
from causal_gen_guard.data.smartguard_adapter import load_smartguard_dataset
from causal_gen_guard.evaluation.metrics import compute_binary_metrics
from causal_gen_guard.generation.causal_tof import run_causal_tof
from causal_gen_guard.generation.offline_synth_loader import load_smartgen_offline_sequences
from causal_gen_guard.models.fusion_detector import FusionDetector
from causal_gen_guard.training.train_fusion import train_backbone_stage, train_causal_stage


SMARTGUARD_STYLE_SPECS = [
    'SD_light_flickering',
    'SD_camera_flickering',
    'SD_tv_flickering',
    'DM_window_open_midnight',
    'DD_microwave_long_time',
]
CAUSAL_SPECS = ['causal_edge_break', 'causal_edge_injection', 'lag_delay']
METRIC_FIELDS = ['precision', 'recall', 'f1', 'fpr', 'fnr', 'auroc', 'auprc']


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
    try:
        if isinstance(value, float) and np.isnan(value):
            return None
    except Exception:
        pass
    return value


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8') as handle:
        json.dump(json_safe(payload), handle, ensure_ascii=False, indent=2)


def write_jsonl(path: Path, sequences: Iterable[BehaviorSequence]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open('w', encoding='utf-8') as handle:
        for sequence in sequences:
            handle.write(json.dumps(json_safe(sequence.to_dict()), ensure_ascii=False) + '\n')
            count += 1
    return count


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def normal_copy(sequence: BehaviorSequence, suffix: str = '') -> BehaviorSequence:
    copied = copy.deepcopy(sequence)
    copied.sequence_id = copied.sequence_id + suffix
    copied.label = 0
    copied.anomaly_type = None
    copied.context = dict(copied.context)
    return copied


def load_normal_sequences(root: Path, dataset: str, limit: int, seed: int) -> List[BehaviorSequence]:
    sequences = [normal_copy(sequence) for sequence in load_smartguard_dataset(root, dataset)]
    rng = random.Random(seed)
    rng.shuffle(sequences)
    return sequences[:limit]


def split_sequences(sequences: List[BehaviorSequence]) -> Tuple[List[BehaviorSequence], List[BehaviorSequence], List[BehaviorSequence]]:
    if len(sequences) < 30:
        raise ValueError('Need at least 30 sequences for feasibility split')
    train_end = max(1, int(len(sequences) * 0.6))
    val_end = max(train_end + 1, int(len(sequences) * 0.8))
    return sequences[:train_end], sequences[train_end:val_end], sequences[val_end:]


def experiment_config(epochs: int) -> Dict[str, Any]:
    epochs = max(1, min(int(epochs), 3))
    return {
        'model': {
            'backbone': {
                'embedding_dim': 64,
                'layers': 1,
                'epochs': epochs,
                'batch_size': 64,
                'learning_rate': 1e-3,
            },
            'causal': {
                'tau': 4,
                'hidden_dim': 64,
                'epochs': epochs,
                'batch_size': 64,
                'learning_rate': 1e-3,
                'threshold_h': 0.0,
                'causality_samples': 64,
            },
        },
        'fusion': {'alpha': 0.6, 'beta': 0.4, 'threshold_quantile': 0.95},
        'experiments': {'use_causal': True},
    }


def clone_event(event: BehaviorEvent, **updates: Any) -> BehaviorEvent:
    cloned = copy.deepcopy(event)
    for key, value in updates.items():
        setattr(cloned, key, value)
    cloned.raw_fields = dict(cloned.raw_fields)
    cloned.raw_fields['feasibility_fallback_attack'] = True
    return cloned


def fallback_smartguard_style(sequence: BehaviorSequence, variant: int) -> BehaviorSequence:
    events = copy.deepcopy(sequence.events)
    if not events:
        raise ValueError('cannot inject fallback anomaly into empty sequence')
    if variant % 3 == 0:
        anchor = clone_event(events[0])
        events = [events[0], clone_event(anchor), clone_event(anchor), clone_event(anchor)] + events[1:]
        anomaly_type = 'fallback_SD_numeric_flickering'
    elif variant % 3 == 1:
        event = clone_event(events[min(1, len(events) - 1)], hour=0)
        events = [event] + events
        anomaly_type = 'fallback_DM_numeric_midnight'
    else:
        mid = len(events) // 2
        duration = events[mid].duration
        try:
            duration = float(duration) * 8.0
        except Exception:
            duration = 120.0
        events[mid] = clone_event(events[mid], duration=max(duration, 120.0))
        anomaly_type = 'fallback_DD_numeric_long_duration'
    context = dict(sequence.context)
    context['fallback_reason'] = 'SmartGuard source controls are numeric, semantic name matching was unavailable'
    return BehaviorSequence(
        sequence_id='{}::{}'.format(sequence.sequence_id, anomaly_type),
        events=events,
        context=context,
        label=1,
        anomaly_type=anomaly_type,
    )


def inject_smartguard_style_set(normals: List[BehaviorSequence], target_count: int, seed: int) -> Tuple[List[BehaviorSequence], Dict[str, Any]]:
    rng = random.Random(seed)
    anomalies: List[BehaviorSequence] = []
    reports: List[Dict[str, Any]] = []
    attempts = 0
    max_attempts = max(target_count * 4, len(normals))
    while len(anomalies) < target_count and attempts < max_attempts:
        attempts += 1
        source = normals[attempts % len(normals)]
        spec = rng.choice(SMARTGUARD_STYLE_SPECS)
        injected, report = inject_smartguard_style(source, spec)
        reports.append(report)
        if injected is not None:
            anomalies.append(injected)
    fallback_count = 0
    while len(anomalies) < target_count:
        source = normals[len(anomalies) % len(normals)]
        anomalies.append(fallback_smartguard_style(source, fallback_count))
        fallback_count += 1
    return anomalies, {
        'target_count': target_count,
        'injected_by_named_smartguard_specs': target_count - fallback_count,
        'fallback_count': fallback_count,
        'attempts': attempts,
        'skipped_named_reports': [item for item in reports if item.get('status') == 'skipped'][:20],
    }


def inject_causal_set(normals: List[BehaviorSequence], per_type: int) -> Tuple[List[BehaviorSequence], Dict[str, Any]]:
    anomalies: List[BehaviorSequence] = []
    reports: List[Dict[str, Any]] = []
    skipped: List[Dict[str, Any]] = []
    index = 0
    for spec in CAUSAL_SPECS:
        count = 0
        attempts = 0
        while count < per_type and attempts < len(normals) * 3:
            source = normals[index % len(normals)]
            index += 1
            attempts += 1
            injected, report = inject_causal_anomaly(source, spec)
            if injected is None:
                skipped.append(report)
                continue
            anomalies.append(injected)
            reports.append(report)
            count += 1
    return anomalies, {'per_type': per_type, 'injected_count': len(anomalies), 'injected': reports[:20], 'skipped': skipped[:20]}


def labels_for(sequences: Sequence[BehaviorSequence]) -> List[int]:
    return [0 if sequence.label in (None, 0, '0', 'normal') else 1 for sequence in sequences]


def build_detector(backbone: Any, causal_model: Any, vocab: Dict[Any, int], A_norm: Any, mode: str) -> FusionDetector:
    if mode == 'reconstruction_only':
        return FusionDetector(backbone=backbone, causal_model=None, vocab=vocab, A_norm_bank=A_norm, alpha=1.0, beta=0.0)
    if mode == 'causal_only':
        return FusionDetector(backbone=None, causal_model=causal_model, vocab=vocab, A_norm_bank=A_norm, alpha=0.0, beta=1.0)
    if mode == 'fusion':
        return FusionDetector(backbone=backbone, causal_model=causal_model, vocab=vocab, A_norm_bank=A_norm, alpha=0.6, beta=0.4)
    raise ValueError('unknown detector mode {}'.format(mode))


def evaluate_detector(
    mode: str,
    backbone: Any,
    causal_model: Any,
    vocab: Dict[Any, int],
    A_norm: Any,
    val_normals: List[BehaviorSequence],
    test_sequences: List[BehaviorSequence],
) -> Dict[str, Any]:
    detector = build_detector(backbone, causal_model, vocab, A_norm, mode)
    threshold = detector.calibrate_threshold(val_normals, quantile=0.95)
    scored = detector.score_batch(test_sequences)
    y_true = labels_for(test_sequences)
    y_score = [float(item['score']) for item in scored]
    metrics = compute_binary_metrics(y_true, y_score, threshold)
    y_pred = [int(score > threshold) for score in y_score]
    normal_scores = [score for score, label in zip(y_score, y_true) if label == 0]
    anomaly_scores = [score for score, label in zip(y_score, y_true) if label == 1]
    return {
        'method': mode,
        'threshold': float(threshold),
        'metrics': metrics,
        'score_summary': {
            'normal_mean': float(np.mean(normal_scores)) if normal_scores else None,
            'anomaly_mean': float(np.mean(anomaly_scores)) if anomaly_scores else None,
            'normal_count': len(normal_scores),
            'anomaly_count': len(anomaly_scores),
            'predicted_anomaly_count': int(sum(y_pred)),
        },
    }


def raw_component_thresholds(detector: FusionDetector, val_normals: List[BehaviorSequence]) -> Dict[str, float]:
    raw = [detector._raw_score_sequence(sequence) for sequence in val_normals]
    rec_values = np.asarray([item['rec_score'] for item in raw], dtype=np.float32)
    causal_values = np.asarray([item['causal_score'] for item in raw], dtype=np.float32)
    return {
        'reconstruction_threshold': float(np.quantile(rec_values, 0.95)),
        'causal_threshold': float(np.quantile(causal_values, 0.95)),
    }


def synthetic_toy_sequences(normals: List[BehaviorSequence], limit: int) -> List[BehaviorSequence]:
    candidates: List[BehaviorSequence] = []
    normal_like_count = max(1, int(limit * 0.75))
    for index, sequence in enumerate(normals[:normal_like_count]):
        copied = normal_copy(sequence, suffix='::toy_synthetic')
        copied.sequence_id = 'toy_synthetic_{:04d}'.format(index)
        copied.context.update({'source_project': 'toy_synthetic', 'synthetic_fallback': True})
        candidates.append(copied)
    reject_count = max(1, limit - normal_like_count)
    for index, sequence in enumerate(normals[:reject_count]):
        event = copy.deepcopy(sequence.events[0])
        events = [copy.deepcopy(event) for _ in range(6)]
        candidate = BehaviorSequence(
            sequence_id='toy_synthetic_repetitive_{:04d}'.format(index),
            events=events,
            context={'source_project': 'toy_synthetic', 'synthetic_fallback': True, 'designed_to_test_legality_filter': True},
            label=0,
        )
        candidates.append(candidate)
    return candidates[:limit]


def safe_detector_fpr(detector: FusionDetector, sequences: List[BehaviorSequence]) -> Tuple[Optional[float], Optional[str]]:
    if not sequences:
        return None, 'no sequences'
    try:
        scored = detector.score_batch(sequences)
        threshold = detector.threshold if detector.threshold is not None else 0.0
        fpr = sum(1 for item in scored if float(item['score']) > threshold) / max(len(scored), 1)
        return float(fpr), None
    except Exception as exc:
        return None, str(exc)


def run_experiments(args: argparse.Namespace) -> Dict[str, Any]:
    set_seed(args.seed)
    output_root = PROJECT_ROOT / 'outputs'
    results_dir = output_root / 'results'
    logs_dir = output_root / 'logs'
    processed_dir = output_root / 'processed'
    feasibility_dir = output_root / 'feasibility'
    synthetic_dir = output_root / 'synthetic' / 'feasibility_tof'
    for directory in (results_dir, logs_dir, processed_dir, feasibility_dir, synthetic_dir):
        directory.mkdir(parents=True, exist_ok=True)

    smartguard_root = (PROJECT_ROOT / args.smartguard_root).resolve() if not Path(args.smartguard_root).is_absolute() else Path(args.smartguard_root)
    normal_sequences = load_normal_sequences(smartguard_root, args.dataset, args.max_sequences, args.seed)
    train_normals, val_normals, test_normals = split_sequences(normal_sequences)
    write_jsonl(processed_dir / 'feasibility_{}_normal_train.jsonl'.format(args.dataset), train_normals)
    write_jsonl(processed_dir / 'feasibility_{}_normal_val.jsonl'.format(args.dataset), val_normals)
    write_jsonl(processed_dir / 'feasibility_{}_normal_test.jsonl'.format(args.dataset), test_normals)

    config = experiment_config(args.epochs)
    # Use the bounded experiment pool for vocabulary discovery so val/test controls do not become OOV during smoke scoring.
    vocab = build_vocab(normal_sequences)
    backbone, backbone_info = train_backbone_stage(train_normals, vocab, config, feasibility_dir, 'feasibility_{}'.format(args.dataset), smoke_test=False)
    causal_model, A_norm, centrality, causal_info = train_causal_stage(train_normals, vocab, config, feasibility_dir, 'feasibility_{}'.format(args.dataset), smoke_test=False)

    # Experiment 1: SmartGuard standard sanity.
    standard_target = min(len(test_normals), args.anomalies_per_experiment)
    smartguard_anomalies, smartguard_injection_report = inject_smartguard_style_set(test_normals, standard_target, args.seed)
    sanity_mixed = [normal_copy(sequence) for sequence in test_normals[:standard_target]] + smartguard_anomalies
    random.Random(args.seed).shuffle(sanity_mixed)
    sanity_results = {
        'experiment': 'SmartGuard standard sanity check',
        'dataset': args.dataset,
        'counts': {'train_normals': len(train_normals), 'val_normals': len(val_normals), 'test_normals': standard_target, 'anomalies': len(smartguard_anomalies)},
        'injection_report': smartguard_injection_report,
        'methods': {},
    }
    for mode in ('reconstruction_only', 'fusion'):
        sanity_results['methods'][mode] = evaluate_detector(mode, backbone, causal_model, vocab, A_norm, val_normals, sanity_mixed)
    write_json(results_dir / 'sanity_fr.json', sanity_results)

    # Experiment 2: causal anomaly smoke.
    per_type = max(1, args.anomalies_per_experiment // len(CAUSAL_SPECS))
    causal_anomalies, causal_injection_report = inject_causal_set(test_normals, per_type)
    causal_normal_count = min(len(test_normals), len(causal_anomalies))
    causal_mixed = [normal_copy(sequence) for sequence in test_normals[:causal_normal_count]] + causal_anomalies
    random.Random(args.seed + 1).shuffle(causal_mixed)
    causal_results = {
        'experiment': 'causal anomaly smoke test',
        'dataset': args.dataset,
        'counts': {'normal_test': causal_normal_count, 'causal_anomalies': len(causal_anomalies)},
        'injection_report': causal_injection_report,
        'methods': {},
    }
    for mode in ('reconstruction_only', 'causal_only', 'fusion'):
        causal_results['methods'][mode] = evaluate_detector(mode, backbone, causal_model, vocab, A_norm, val_normals, causal_mixed)
    rec_f1 = causal_results['methods']['reconstruction_only']['metrics']['f1']
    causal_f1 = causal_results['methods']['causal_only']['metrics']['f1']
    fusion_f1 = causal_results['methods']['fusion']['metrics']['f1']
    causal_results['trend'] = {
        'causal_branch_improves_over_reconstruction': bool(max(causal_f1, fusion_f1) > rec_f1),
        'best_method': max(causal_results['methods'], key=lambda key: causal_results['methods'][key]['metrics']['f1']),
    }
    write_json(results_dir / 'causal_anomaly_smoke.json', causal_results)

    # Experiment 3: SmartGen offline synthetic + Causal-TOF.
    synthetic_source = 'SmartGen offline'
    synthetic_notes: List[str] = []
    try:
        synthetic_candidates = load_smartgen_offline_sequences(PROJECT_ROOT / args.smartgen_root, dataset=args.dataset, transition='ST', limit=args.synthetic_limit)
        if not synthetic_candidates:
            raise ValueError('SmartGen loader returned no sequences')
    except Exception as exc:
        synthetic_source = 'toy synthetic fallback'
        synthetic_notes.append('SmartGen offline load failed or unavailable: {}'.format(exc))
        synthetic_candidates = synthetic_toy_sequences(test_normals, args.synthetic_limit)
    write_jsonl(synthetic_dir / 'candidates.jsonl', synthetic_candidates)

    fusion_detector = build_detector(backbone, causal_model, vocab, A_norm, 'fusion')
    fusion_detector.calibrate_threshold(val_normals, quantile=0.95)
    before_fpr, before_fpr_error = safe_detector_fpr(fusion_detector, synthetic_candidates)
    component_thresholds = raw_component_thresholds(fusion_detector, val_normals)
    tof_report = run_causal_tof(
        synthetic_candidates,
        output_dir=synthetic_dir,
        backbone=backbone,
        reconstruction_threshold=component_thresholds['reconstruction_threshold'],
        causal_model=causal_model,
        A_norm=A_norm,
        vocab=vocab,
        causal_threshold=component_thresholds['causal_threshold'],
        train_sequences=train_normals,
        val_sequences=val_normals,
        utility_config={'enabled': False},
    )
    kept_path = Path(tof_report['outputs']['kept_jsonl'])
    kept_sequences: List[BehaviorSequence] = []
    if kept_path.exists():
        with kept_path.open('r', encoding='utf-8') as handle:
            for line in handle:
                line = line.strip()
                if line:
                    kept_sequences.append(BehaviorSequence.from_dict(json.loads(line)))
    after_fpr, after_fpr_error = safe_detector_fpr(fusion_detector, kept_sequences)
    synthetic_results = {
        'experiment': 'synthetic data pipeline smoke test',
        'synthetic_source': synthetic_source,
        'notes': synthetic_notes,
        'candidate_count': len(synthetic_candidates),
        'kept_count': len(kept_sequences),
        'rejected_count': int(tof_report.get('final_rejected_count', 0)),
        'component_thresholds': component_thresholds,
        'proxy_target_normal_fpr_before_tof': before_fpr,
        'proxy_target_normal_fpr_after_tof': after_fpr,
        'proxy_fpr_errors': {'before': before_fpr_error, 'after': after_fpr_error},
        'tof_outputs': tof_report.get('outputs', {}),
        'tof_report_path': str(synthetic_dir / 'report.json'),
        'observed_target_normal_fpr_reduction': None if before_fpr is None or after_fpr is None else bool(after_fpr < before_fpr),
    }
    write_json(results_dir / 'synthetic_pipeline_smoke.json', synthetic_results)

    report = {
        'config': {'dataset': args.dataset, 'max_sequences': args.max_sequences, 'epochs': args.epochs, 'seed': args.seed},
        'normal_split': {'train': len(train_normals), 'val': len(val_normals), 'test': len(test_normals)},
        'vocab_size': len(vocab),
        'backbone': backbone_info,
        'causal': causal_info,
        'experiments': {
            'sanity_fr': sanity_results,
            'causal_anomaly_smoke': causal_results,
            'synthetic_pipeline_smoke': synthetic_results,
        },
    }
    write_feasibility_markdown(logs_dir / 'FEASIBILITY_REPORT.md', report)
    write_json(results_dir / 'feasibility_summary.json', report)
    return report


def metric_line(result: Dict[str, Any], method: str) -> str:
    payload = result['methods'][method]
    metrics = payload['metrics']
    return '| {} | {:.4f} | {:.4f} | {:.4f} | {:.4f} | {:.4f} | {:.4f} |'.format(
        method,
        metrics.get('precision', float('nan')),
        metrics.get('recall', float('nan')),
        metrics.get('f1', float('nan')),
        metrics.get('fpr', float('nan')),
        metrics.get('auroc', float('nan')),
        metrics.get('auprc', float('nan')),
    )


def write_feasibility_markdown(path: Path, report: Dict[str, Any]) -> None:
    sanity = report['experiments']['sanity_fr']
    causal = report['experiments']['causal_anomaly_smoke']
    synth = report['experiments']['synthetic_pipeline_smoke']
    lines = [
        '# Feasibility Report',
        '',
        'Project path: `/home/heyang/projects/CausalGenGuard`',
        'Environment: `/home/heyang/miniconda3/envs/smartguard_env/bin/python`',
        '',
        '## Run Bounds',
        '',
        '- Dataset: `{}`'.format(report['config']['dataset']),
        '- Max sequences loaded: {}'.format(report['config']['max_sequences']),
        '- Epochs per model: {}'.format(report['config']['epochs']),
        '- Normal split: train={train}, val={val}, test={test}'.format(**report['normal_split']),
        '- Vocab size: {}'.format(report['vocab_size']),
        '',
        '## Experiment 1: SmartGuard Standard Sanity Check',
        '',
        'Status: ran through.',
        '',
        '| method | precision | recall | f1 | fpr | auroc | auprc |',
        '| --- | --- | --- | --- | --- | --- | --- |',
        metric_line(sanity, 'reconstruction_only'),
        metric_line(sanity, 'fusion'),
        '',
        'Injection summary:',
        '',
        '- Named SmartGuard injections: {}'.format(sanity['injection_report']['injected_by_named_smartguard_specs']),
        '- Fallback numeric-control injections: {}'.format(sanity['injection_report']['fallback_count']),
        '',
        '## Experiment 2: Causal Anomaly Smoke Test',
        '',
        'Status: ran through.',
        '',
        '| method | precision | recall | f1 | fpr | auroc | auprc |',
        '| --- | --- | --- | --- | --- | --- | --- |',
        metric_line(causal, 'reconstruction_only'),
        metric_line(causal, 'causal_only'),
        metric_line(causal, 'fusion'),
        '',
        '- Causal branch improves over reconstruction: `{}`'.format(causal['trend']['causal_branch_improves_over_reconstruction']),
        '- Best method by F1: `{}`'.format(causal['trend']['best_method']),
        '',
        '## Experiment 3: Synthetic Data Pipeline Smoke Test',
        '',
        'Status: ran through.',
        '',
        '- Synthetic source: `{}`'.format(synth['synthetic_source']),
        '- Candidate count: {}'.format(synth['candidate_count']),
        '- Kept count: {}'.format(synth['kept_count']),
        '- Rejected count: {}'.format(synth['rejected_count']),
        '- Proxy target-normal FPR before Causal-TOF: {}'.format(synth['proxy_target_normal_fpr_before_tof']),
        '- Proxy target-normal FPR after Causal-TOF: {}'.format(synth['proxy_target_normal_fpr_after_tof']),
        '- Observed proxy FPR reduction: `{}`'.format(synth['observed_target_normal_fpr_reduction']),
        '- Kept JSONL: `{}`'.format(synth['tof_outputs'].get('kept_jsonl')),
        '- Rejected JSONL: `{}`'.format(synth['tof_outputs'].get('rejected_jsonl')),
        '- TOF report JSON: `{}`'.format(synth['tof_outputs'].get('report_json')),
        '',
        'Notes:',
        '',
    ]
    if synth['notes']:
        lines.extend('- {}'.format(note) for note in synth['notes'])
    else:
        lines.append('- SmartGen offline synthetic data was found and parsed.')
    lines.extend([
        '',
        '## Current Biggest Issues',
        '',
        '- SmartGuard FR controls are numeric in the prepared 10x4 data, so semantic SmartGuard-style attacks may require fallback numeric-control perturbations unless a device/action dictionary is mapped in.',
        '- These are feasibility metrics from very small runs, not final paper numbers.',
        '- Target-normal FPR reduction is only a proxy unless real target-context normal data and labels are fixed.',
        '- More stable conclusions require repeated seeds and frozen train/validation/test splits.',
        '',
    ])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text('\n'.join(lines), encoding='utf-8')


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Run bounded CausalGenGuard feasibility experiments.')
    parser.add_argument('--smartguard-root', default='../SmartGuard')
    parser.add_argument('--smartgen-root', default='../SmartGen')
    parser.add_argument('--dataset', default='fr')
    parser.add_argument('--max-sequences', type=int, default=500)
    parser.add_argument('--epochs', type=int, default=2)
    parser.add_argument('--anomalies-per-experiment', type=int, default=100)
    parser.add_argument('--synthetic-limit', type=int, default=80)
    parser.add_argument('--seed', type=int, default=42)
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if args.max_sequences > 1000:
        raise ValueError('--max-sequences must be <= 1000 for feasibility experiments')
    if args.epochs > 3:
        raise ValueError('--epochs must be <= 3 for feasibility experiments')
    run_experiments(args)
    print('Wrote outputs/results/sanity_fr.json')
    print('Wrote outputs/results/causal_anomaly_smoke.json')
    print('Wrote outputs/results/synthetic_pipeline_smoke.json')
    print('Wrote outputs/synthetic/feasibility_tof/report.json')
    print('Wrote outputs/logs/FEASIBILITY_REPORT.md')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
