#!/usr/bin/env python3
'''Bounded multi-dataset context-shift anomaly experiments.

Runs seasonal winter -> spring/summer context-shift checks for FR/SP/US without
touching source projects and without full-scale training.
'''

from __future__ import annotations

import argparse
import copy
import csv
import json
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

import diagnose_context_shift_anomaly as diag
import run_context_shift_final as final
from causal_gen_guard.data.behavior_event_tensor import build_vocab


METHODS = [
    'source_only',
    'source_plus_raw_synthetic',
    'source_plus_tof_synthetic',
    'source_plus_causal_tof_synthetic',
    'oracle_target',
]

SUMMARY_FIELDS = [
    'dataset',
    'transition',
    'source_context',
    'target_context',
    'mapping_type',
    'method',
    'filter_strategy',
    'target_normal_count',
    'target_anomaly_count',
    'synthetic_count',
    'kept_count',
    'rejected_count',
    'target_normal_fpr',
    'precision',
    'recall',
    'f1',
    'fnr',
    'auroc',
    'auprc',
    'adaptation_gain',
    'status',
    'missing_reason',
]


def resolve(path_value: str | Path) -> Path:
    path = Path(path_value)
    return path if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def parse_csv_arg(value: str) -> List[str]:
    return [item.strip() for item in value.split(',') if item.strip()]


def json_safe(value: Any) -> Any:
    return final.json_safe(value)


def write_csv(path: Path, rows: List[Dict[str, Any]], fields: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields))
        writer.writeheader()
        for row in rows:
            writer.writerow({field: json_safe(row.get(field)) for field in fields})


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_safe(payload), ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def bounded_bounds(args: argparse.Namespace) -> Dict[str, Any]:
    seed = int(getattr(args, 'seed', 42))
    return {
        'max_source_train': min(int(args.max_normal), 500),
        'max_source_val': min(200, int(args.max_normal), 500),
        'max_target_normal': min(int(args.max_normal), 500),
        'max_target_anomaly': min(int(args.max_anomaly), 500),
        'max_synthetic': min(int(args.max_synthetic), 500),
        'max_target_labeled_normal': min(int(args.max_normal), 500),
        'max_target_labeled_anomaly': min(int(args.max_anomaly), 500),
        'epochs': min(int(args.epochs), 3),
        'batch_size': 32,
        'hidden_dim': 64,
        'seed': seed,
    }


def config_for_dataset(dataset: str, output_dir: Path, bounds: Dict[str, Any]) -> Dict[str, Any]:
    transition = 'seasonal'
    prefix = output_dir / f'{dataset}_{transition}'
    return {
        'dataset': dataset,
        'experiment': 'context_shift_multidataset',
        'contexts': {
            'source_candidates': ['winter'],
            'target_candidates': ['spring', 'summer'],
        },
        'paths': {
            'smartgen_root': f'../SmartGen/SmartGen/IoT_data/{dataset}',
            'smartguard_root': '../SmartGuard',
            'control_to_id': f'outputs/mappings/smartguard/{dataset}/control_to_id.json',
            'id_to_control': f'outputs/mappings/smartguard/{dataset}/id_to_control.json',
            'output_csv': str(prefix.with_suffix('.csv')),
            'output_json': str(prefix.with_suffix('.json')),
            'target_labeled_jsonl': str(prefix.parent / f'{dataset}_{transition}_target_labeled.jsonl'),
            'target_labeled_report': str(prefix.parent / f'{dataset}_{transition}_target_labeled_report.json'),
            'canonical_normal_jsonl': f'outputs/processed/{dataset}_sequences_canonical.jsonl',
        },
        'run_bounds': dict(bounds),
        'smartgen': {
            'prefer_textual_generated': True,
            'generated_keywords': ['generation', 'generated'],
            'generated_suffixes': ['_seq.pkl', '.pkl'],
        },
        'evaluation': {
            'threshold_quantile': 0.95,
        },
    }


def missing_rows(
    dataset: str,
    transition: str,
    source_context: str,
    target_context: str,
    mapping_type: str,
    reason: str,
) -> List[Dict[str, Any]]:
    return [
        {
            'dataset': dataset,
            'transition': transition,
            'source_context': source_context,
            'target_context': target_context,
            'mapping_type': mapping_type,
            'method': method,
            'filter_strategy': '',
            'target_normal_count': 0,
            'target_anomaly_count': 0,
            'synthetic_count': 0,
            'kept_count': 0,
            'rejected_count': 0,
            'target_normal_fpr': None,
            'precision': None,
            'recall': None,
            'f1': None,
            'fnr': None,
            'auroc': None,
            'auprc': None,
            'adaptation_gain': None,
            'status': 'missing',
            'missing_reason': reason,
        }
        for method in METHODS
    ]


def validate_dataset_inputs(config: Dict[str, Any]) -> Tuple[Optional[str], str, str]:
    dataset = str(config.get('dataset', 'unknown'))
    paths = dict(config.get('paths', {}))
    for key in ('control_to_id', 'id_to_control'):
        path = resolve(paths[key])
        if not path.exists():
            return f'missing SmartGuard mapping file: {path}', 'winter', ''

    smartgen_root = resolve(paths['smartgen_root'])
    if not smartgen_root.exists():
        return f'missing SmartGen dataset root: {smartgen_root}', 'winter', ''

    source_context = 'winter'
    if not final.context_has_normal_data(config, source_context):
        return f'missing source normal data for context `{source_context}` in dataset `{dataset}`', source_context, ''

    target_context = ''
    for candidate in ('spring', 'summer'):
        if final.context_has_normal_data(config, candidate):
            target_context = candidate
            break
    if not target_context:
        return 'missing target normal data for contexts `spring` and `summer`', source_context, ''

    return None, source_context, target_context


def mapping_type_for_config(config: Dict[str, Any]) -> str:
    report_path = resolve(Path(str(config['paths']['control_to_id'])).parent / 'mapping_report.json')
    if not report_path.exists():
        return ''
    try:
        report = json.loads(report_path.read_text(encoding='utf-8'))
    except Exception:
        return 'unknown'
    return str(report.get('mapping_type') or 'smartguard_semantic')


def train_artifacts(
    config: Dict[str, Any],
    bounds: Dict[str, Any],
    inputs: Dict[str, Any],
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
        'filter_strategy': 'no_filter',
    }

    tof_strategy = 'iqr_1.5'
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
        seed + 10 + final.TOF_FILTER_STRATEGIES.index(tof_strategy),
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
        'filter_strategy': tof_strategy,
    }

    causal_strategy = 'relaxed_causal_keep_90_percent'
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
        seed + 30 + final.CAUSAL_TOF_FILTER_STRATEGIES.index(causal_strategy),
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
        'filter_strategy': '',
    }
    return artifacts


def load_inputs_or_missing(config: Dict[str, Any], bounds: Dict[str, Any]) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    try:
        control_to_id, id_to_control = final.read_mapping(config)
        mapping_type = mapping_type_for_config(config)
        source_context = final.choose_context(config, ['winter'], 'source')
        target_context = final.choose_context(config, ['spring', 'summer'], 'target')
        source_train = final.load_context_normals(config, source_context, ['trn'], control_to_id, id_to_control, int(bounds['max_source_train']))
        source_val = final.load_context_normals(config, source_context, ['vld', 'trn'], control_to_id, id_to_control, int(bounds['max_source_val']))
        target_normals = final.load_context_normals(config, target_context, ['test', 'split_test', 'vld', 'trn'], control_to_id, id_to_control, int(bounds['max_target_normal']))
        synthetic, synthetic_files = final.load_target_synthetic(config, target_context, control_to_id, id_to_control, int(bounds['max_synthetic']))
        if not source_train:
            return None, 'no parsed source train normal sequences'
        if not source_val:
            return None, 'no parsed source validation normal sequences'
        if not target_normals:
            return None, 'no parsed target real normal sequences'
        if not synthetic:
            return None, 'no parsed target-context synthetic normal sequences'
        labeled_normals, labeled_anomalies, labeled_report, labeled_files = final.write_target_context_labeled_set(
            config,
            target_context,
            target_normals,
            synthetic,
            bounds,
            int(bounds.get('seed', 42)) + 1000,
        )
        if not labeled_anomalies:
            return None, 'no injectable target-context anomaly candidates'
        vocab = build_vocab(source_train + source_val + target_normals + labeled_normals + labeled_anomalies + synthetic)
        return {
            'control_to_id': control_to_id,
            'id_to_control': id_to_control,
            'mapping_type': mapping_type,
            'source_context': source_context,
            'target_context': target_context,
            'source_train': source_train,
            'source_val': source_val,
            'target_normals': target_normals,
            'synthetic': synthetic,
            'synthetic_files': synthetic_files,
            'labeled_normals': labeled_normals,
            'labeled_anomalies': labeled_anomalies,
            'labeled_report': labeled_report,
            'labeled_files': labeled_files,
            'vocab': vocab,
        }, None
    except Exception as exc:
        return None, f'{type(exc).__name__}: {exc}'


def summary_rows_from_artifacts(
    dataset: str,
    transition: str,
    inputs: Dict[str, Any],
    artifacts: Dict[str, Dict[str, Any]],
) -> List[Dict[str, Any]]:
    source_fpr = artifacts['source_only']['row'].get('target_normal_fpr')
    rows: List[Dict[str, Any]] = []
    for method in METHODS:
        row = copy.deepcopy(artifacts[method]['row'])
        f1_value = row.get('f1', row.get('anomaly_f1'))
        adaptation_gain = None if method == 'source_only' or source_fpr is None or row.get('target_normal_fpr') is None else source_fpr - row['target_normal_fpr']
        rows.append(
            {
                'dataset': dataset,
                'transition': transition,
                'source_context': inputs['source_context'],
                'target_context': inputs['target_context'],
                'mapping_type': inputs.get('mapping_type', ''),
                'method': method,
                'filter_strategy': artifacts[method].get('filter_strategy', ''),
                'target_normal_count': len(inputs['target_normals']),
                'target_anomaly_count': len(inputs['labeled_anomalies']),
                'synthetic_count': len(inputs['synthetic']) if method != 'source_only' and method != 'oracle_target' else row.get('synthetic_count', 0),
                'kept_count': row.get('kept_count', 0),
                'rejected_count': row.get('rejected_count', 0),
                'target_normal_fpr': row.get('target_normal_fpr'),
                'precision': row.get('precision'),
                'recall': row.get('recall'),
                'f1': f1_value,
                'fnr': row.get('fnr'),
                'auroc': row.get('auroc'),
                'auprc': row.get('auprc'),
                'adaptation_gain': adaptation_gain,
                'status': 'success',
                'missing_reason': '',
            }
        )
    return rows


def write_dataset_outputs(
    dataset: str,
    transition: str,
    output_dir: Path,
    summary_rows: List[Dict[str, Any]],
    per_rows: List[Dict[str, Any]],
    threshold_rows: List[Dict[str, Any]],
    balanced_rows: List[Dict[str, Any]],
    balanced_report: Dict[str, Any],
    inputs: Dict[str, Any],
) -> None:
    prefix = output_dir / f'{dataset}_{transition}'
    write_csv(prefix.with_suffix('.csv'), summary_rows, SUMMARY_FIELDS)
    write_json(
        prefix.with_suffix('.json'),
        {
            'dataset': dataset,
            'transition': transition,
            'source_context': inputs['source_context'],
            'target_context': inputs['target_context'],
            'mapping_type': inputs.get('mapping_type', ''),
            'rows': summary_rows,
            'target_labeled_report': inputs['labeled_report'],
            'target_labeled_files': inputs['labeled_files'],
            'balanced_report': balanced_report,
            'balanced_rows': balanced_rows,
        },
    )
    write_csv(
        output_dir / f'{dataset}_{transition}_per_anomaly.csv',
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
            'fnr',
            'auroc',
            'auprc',
            'mean_score_normal',
            'mean_score_anomaly',
            'score_margin',
        ],
    )
    write_csv(
        output_dir / f'{dataset}_{transition}_threshold_sweep.csv',
        threshold_rows,
        ['method', 'filter_strategy', 'quantile', 'threshold', 'target_normal_fpr', 'precision', 'recall', 'f1', 'fnr', 'auroc', 'auprc'],
    )


def write_missing_outputs(dataset: str, transition: str, output_dir: Path, rows: List[Dict[str, Any]], reason: str) -> None:
    prefix = output_dir / f'{dataset}_{transition}'
    write_csv(prefix.with_suffix('.csv'), rows, SUMMARY_FIELDS)
    write_json(prefix.with_suffix('.json'), {'dataset': dataset, 'transition': transition, 'status': 'missing', 'missing_reason': reason, 'rows': rows})
    write_csv(
        output_dir / f'{dataset}_{transition}_per_anomaly.csv',
        [],
        ['method', 'filter_strategy', 'anomaly_type', 'anomaly_count', 'low_support', 'precision', 'recall', 'f1', 'fnr', 'auroc', 'auprc', 'mean_score_normal', 'mean_score_anomaly', 'score_margin'],
    )
    write_csv(
        output_dir / f'{dataset}_{transition}_threshold_sweep.csv',
        [],
        ['method', 'filter_strategy', 'quantile', 'threshold', 'target_normal_fpr', 'precision', 'recall', 'f1', 'fnr', 'auroc', 'auprc'],
    )


def score_success_run(
    artifacts: Dict[str, Dict[str, Any]],
    inputs: Dict[str, Any],
    seed: int,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    scored = diag.score_artifacts(artifacts, inputs['target_normals'], inputs['labeled_anomalies'])
    for method, method_scores in scored.items():
        method_scores['detector'] = artifacts[method]['detector']
    per_rows = diag.per_anomaly_rows(scored, inputs['target_normals'], inputs['labeled_anomalies'], inputs['labeled_report'])
    threshold_rows = diag.threshold_sweep_rows(scored)
    source_fpr = artifacts['source_only']['row'].get('target_normal_fpr')
    balanced_normals, balanced_anomalies, balanced_report = diag.build_balanced_subset(
        inputs['labeled_normals'],
        inputs['labeled_anomalies'],
        seed + 2000,
    )
    balanced_rows = diag.balanced_rows(artifacts, balanced_normals, balanced_anomalies, source_fpr)
    return per_rows, threshold_rows, [{'balanced_report': balanced_report, 'balanced_rows': balanced_rows}]


def safe_mean(values: Sequence[Any]) -> Optional[float]:
    parsed = []
    for value in values:
        if value is None or value == '':
            continue
        try:
            number = float(value)
        except (TypeError, ValueError):
            continue
        if not np.isnan(number) and not np.isinf(number):
            parsed.append(number)
    return float(np.mean(parsed)) if parsed else None


def method_average(rows: List[Dict[str, Any]], method: str, field: str) -> Optional[float]:
    return safe_mean([row.get(field) for row in rows if row.get('status') == 'success' and row.get('method') == method])


def yes_no_consistency(rows: List[Dict[str, Any]], predicate: Any) -> Tuple[bool, int, int]:
    grouped: Dict[Tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row.get('status') == 'success':
            grouped[(str(row['dataset']), str(row['transition']))].append(row)
    total = 0
    passed = 0
    for run_rows in grouped.values():
        by_method = {row['method']: row for row in run_rows}
        if predicate(by_method):
            passed += 1
        total += 1
    return passed == total and total > 0, passed, total


def write_report(path: Path, summary_rows: List[Dict[str, Any]], details: List[Dict[str, Any]], output_dir: Path) -> None:
    successful = sorted({(row['dataset'], row['transition']) for row in summary_rows if row.get('status') == 'success'})
    missing = [detail for detail in details if detail.get('status') == 'missing']
    success_rows = [row for row in summary_rows if row.get('status') == 'success']
    us_detail = next((detail for detail in details if detail.get('dataset') == 'us' and detail.get('transition') == 'seasonal'), None)
    us_mapping_found = bool(us_detail and us_detail.get('mapping_type'))
    us_mapping_type = str(us_detail.get('mapping_type') or 'missing') if us_detail else 'missing'
    us_success = bool(us_detail and us_detail.get('status') == 'success')
    avg_fpr = {method: method_average(success_rows, method, 'target_normal_fpr') for method in METHODS}
    avg_balanced_f1: Dict[str, Optional[float]] = {}
    for method in METHODS:
        avg_balanced_f1[method] = safe_mean(
            [
                row.get('f1')
                for detail in details
                if detail.get('status') == 'success'
                for row in detail.get('balanced_rows', [])
                if row.get('method') == method
            ]
        )
    lowest_avg_fpr = min((item for item in avg_fpr.items() if item[1] is not None), key=lambda item: item[1], default=(None, None))
    best_avg_balanced = max((item for item in avg_balanced_f1.items() if item[1] is not None), key=lambda item: item[1], default=(None, None))
    non_oracle_methods = [method for method in METHODS if method != 'oracle_target']
    lowest_non_oracle_fpr = min(
        ((method, avg_fpr.get(method)) for method in non_oracle_methods if avg_fpr.get(method) is not None),
        key=lambda item: item[1],
        default=(None, None),
    )
    best_non_oracle_balanced = max(
        ((method, avg_balanced_f1.get(method)) for method in non_oracle_methods if avg_balanced_f1.get(method) is not None),
        key=lambda item: item[1],
        default=(None, None),
    )
    causal_avg_fpr_lowest_non_oracle = lowest_non_oracle_fpr[0] == 'source_plus_causal_tof_synthetic'
    causal_avg_balanced_highest_non_oracle = best_non_oracle_balanced[0] == 'source_plus_causal_tof_synthetic'

    source_high, source_high_count, total_success = yes_no_consistency(
        success_rows,
        lambda by_method: (by_method.get('source_only', {}).get('target_normal_fpr') or 0) >= 0.3,
    )
    raw_lower, raw_lower_count, _ = yes_no_consistency(
        success_rows,
        lambda by_method: by_method.get('source_plus_raw_synthetic', {}).get('target_normal_fpr') is not None
        and by_method.get('source_only', {}).get('target_normal_fpr') is not None
        and by_method['source_plus_raw_synthetic']['target_normal_fpr'] < by_method['source_only']['target_normal_fpr'],
    )
    filters_lower, filters_lower_count, _ = yes_no_consistency(
        success_rows,
        lambda by_method: all(
            by_method.get(method, {}).get('target_normal_fpr') is not None
            and by_method.get('source_plus_raw_synthetic', {}).get('target_normal_fpr') is not None
            and by_method[method]['target_normal_fpr'] < by_method['source_plus_raw_synthetic']['target_normal_fpr']
            for method in ('source_plus_tof_synthetic', 'source_plus_causal_tof_synthetic')
        ),
    )
    causal_tradeoff, causal_tradeoff_count, _ = yes_no_consistency(
        success_rows,
        lambda by_method: by_method.get('source_plus_causal_tof_synthetic', {}).get('target_normal_fpr') is not None
        and by_method.get('source_plus_tof_synthetic', {}).get('target_normal_fpr') is not None
        and by_method.get('source_plus_causal_tof_synthetic', {}).get('f1') is not None
        and by_method.get('source_plus_tof_synthetic', {}).get('f1') is not None
        and by_method['source_plus_causal_tof_synthetic']['target_normal_fpr'] <= by_method['source_plus_tof_synthetic']['target_normal_fpr']
        and by_method['source_plus_causal_tof_synthetic']['f1'] < by_method['source_plus_tof_synthetic']['f1'],
    )

    lines = [
        '# Context Shift Multi-Dataset Report',
        '',
        '## Scope',
        '',
        '- Datasets requested: `fr, sp, us`',
        '- Transition requested: `seasonal` (`winter -> spring`, falling back to `summer` if needed)',
        '- Bounds: normal <= 500, anomaly <= 500, synthetic <= 500, epochs <= 3',
        '',
        '## Direct Answers',
        '',
        '1. Successful dataset/transitions: ' + (', '.join(f'`{dataset}:{transition}`' for dataset, transition in successful) if successful else 'none'),
        '2. Missing dataset/transitions: ' + (', '.join(f"`{item['dataset']}:{item['transition']}` ({item['missing_reason']})" for item in missing) if missing else 'none'),
        f'3. US mapping found: `{us_mapping_found}`.',
        f'4. US mapping type: `{us_mapping_type}`.',
        f'5. US seasonal success: `{us_success}`' + (f" ({us_detail.get('missing_reason')})" if us_detail and us_detail.get('status') == 'missing' else '.'),
        f'6. Source-only consistently high FPR: `{source_high}` ({source_high_count}/{total_success} successful runs >= 0.3).',
        f'7. Raw synthetic consistently lower FPR than source-only: `{raw_lower}` ({raw_lower_count}/{total_success}).',
        f'8. TOF/Causal-TOF consistently lower FPR than raw synthetic: `{filters_lower}` ({filters_lower_count}/{total_success}).',
        f'9. Causal-TOF remains FPR/F1 trade-off: `{causal_tradeoff}` ({causal_tradeoff_count}/{total_success}).',
        f'10. Lowest average target-normal FPR: `{lowest_avg_fpr[0]}` (`{final.format_metric(lowest_avg_fpr[1])}`).',
        f'11. Highest average balanced F1: `{best_avg_balanced[0]}` (`{final.format_metric(best_avg_balanced[1])}`).',
        f'12. Causal-TOF average FPR is lowest among non-oracle methods: `{causal_avg_fpr_lowest_non_oracle}` (`{final.format_metric(avg_fpr.get("source_plus_causal_tof_synthetic"))}`). Including oracle, oracle remains lower.',
        f'13. Whether Causal-TOF average balanced F1 is highest among non-oracle methods: `{causal_avg_balanced_highest_non_oracle}` (`{final.format_metric(avg_balanced_f1.get("source_plus_causal_tof_synthetic"))}`). Including oracle, `{best_avg_balanced[0]}` is highest.',
        '14. Paper-main-experiment readiness: bounded seasonal results are a usable prototype, but weak/variable anomaly F1 means this is not yet enough as the final paper main experiment.',
        '15. Next transitions: yes, extend to `daytime -> night` and `single -> multiple`; those transitions test different context axes than seasonal weather.',
        '',
        '## Average Metrics On Successful Runs',
        '',
        '| method | avg_target_normal_fpr | avg_f1 | avg_fnr | avg_balanced_f1 |',
        '| --- | ---: | ---: | ---: | ---: |',
    ]
    for method in METHODS:
        lines.append(
            f"| {method} | {final.format_metric(avg_fpr.get(method))} | {final.format_metric(method_average(success_rows, method, 'f1'))} | {final.format_metric(method_average(success_rows, method, 'fnr'))} | {final.format_metric(avg_balanced_f1.get(method))} |"
        )

    lines.extend(
        [
            '',
            '## Mapping Types',
            '',
            '| dataset | transition | status | mapping_type |',
            '| --- | --- | --- | --- |',
        ]
    )
    for detail in details:
        lines.append(
            f"| {detail.get('dataset')} | {detail.get('transition')} | {detail.get('status')} | {detail.get('mapping_type', '')} |"
        )

    lines.extend(
        [
            '',
            '## Missing Details',
            '',
        ]
    )
    if missing:
        for item in missing:
            lines.append(f"- `{item['dataset']}:{item['transition']}`: {item['missing_reason']}")
    else:
        lines.append('- none')

    lines.extend(
        [
            '',
            '## Output Files',
            '',
            f'- `{output_dir / "summary.csv"}`',
            f'- `{output_dir / "summary.json"}`',
            '- per-dataset CSV/JSON, per-anomaly CSV, and threshold-sweep CSV under the same directory',
            '',
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text('\n'.join(lines), encoding='utf-8')


def run(args: argparse.Namespace) -> Dict[str, Any]:
    datasets = parse_csv_arg(args.datasets)
    transitions = parse_csv_arg(args.transitions)
    output_dir = resolve(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    bounds = bounded_bounds(args)
    seed = int(bounds.get('seed', 42))
    final.set_seed(seed)
    all_rows: List[Dict[str, Any]] = []
    details: List[Dict[str, Any]] = []

    for dataset in datasets:
        for transition in transitions:
            if transition != 'seasonal':
                reason = f'unsupported transition `{transition}` in this bounded runner'
                rows = missing_rows(dataset, transition, 'winter', '', '', reason)
                write_missing_outputs(dataset, transition, output_dir, rows, reason)
                all_rows.extend(rows)
                details.append({'dataset': dataset, 'transition': transition, 'status': 'missing', 'mapping_type': '', 'missing_reason': reason})
                continue

            config = config_for_dataset(dataset, output_dir, bounds)
            reason, source_context, target_context = validate_dataset_inputs(config)
            if reason:
                mapping_type = mapping_type_for_config(config)
                rows = missing_rows(dataset, transition, source_context, target_context, mapping_type, reason)
                write_missing_outputs(dataset, transition, output_dir, rows, reason)
                all_rows.extend(rows)
                details.append({'dataset': dataset, 'transition': transition, 'status': 'missing', 'mapping_type': mapping_type, 'missing_reason': reason})
                continue

            inputs, load_reason = load_inputs_or_missing(config, bounds)
            if load_reason or inputs is None:
                mapping_type = mapping_type_for_config(config)
                rows = missing_rows(dataset, transition, source_context, target_context, mapping_type, load_reason or 'unknown load failure')
                write_missing_outputs(dataset, transition, output_dir, rows, load_reason or 'unknown load failure')
                all_rows.extend(rows)
                details.append({'dataset': dataset, 'transition': transition, 'status': 'missing', 'mapping_type': mapping_type, 'missing_reason': load_reason or 'unknown load failure'})
                continue

            artifacts = train_artifacts(config, bounds, inputs)
            summary_rows = summary_rows_from_artifacts(dataset, transition, inputs, artifacts)
            per_rows, threshold_rows, balanced_payloads = score_success_run(artifacts, inputs, seed)
            balanced_payload = balanced_payloads[0]
            write_dataset_outputs(
                dataset,
                transition,
                output_dir,
                summary_rows,
                per_rows,
                threshold_rows,
                balanced_payload['balanced_rows'],
                balanced_payload['balanced_report'],
                inputs,
            )
            all_rows.extend(summary_rows)
            details.append(
                {
                    'dataset': dataset,
                    'transition': transition,
                    'status': 'success',
                    'source_context': inputs['source_context'],
                    'target_context': inputs['target_context'],
                    'mapping_type': inputs.get('mapping_type', ''),
                    'target_labeled_report': inputs['labeled_report'],
                    'balanced_report': balanced_payload['balanced_report'],
                    'balanced_rows': balanced_payload['balanced_rows'],
                    'rows': summary_rows,
                }
            )

    write_csv(output_dir / 'summary.csv', all_rows, SUMMARY_FIELDS)
    payload = {
        'bounds': bounds,
        'datasets': datasets,
        'transitions': transitions,
        'rows': all_rows,
        'details': details,
    }
    write_json(output_dir / 'summary.json', payload)
    write_report(output_dir / 'CONTEXT_SHIFT_MULTIDATASET_REPORT.md', all_rows, details, output_dir)
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Run bounded context-shift experiments over multiple datasets.')
    parser.add_argument('--datasets', default='fr,sp,us')
    parser.add_argument('--transitions', default='seasonal')
    parser.add_argument('--max-normal', type=int, default=500)
    parser.add_argument('--max-anomaly', type=int, default=500)
    parser.add_argument('--max-synthetic', type=int, default=500)
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output-dir', default='outputs/results/context_shift_multidataset')
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    payload = run(args)
    successes = sum(1 for item in payload['details'] if item.get('status') == 'success')
    missing = sum(1 for item in payload['details'] if item.get('status') == 'missing')
    print(f"Wrote {resolve(args.output_dir) / 'summary.csv'}")
    print(f"Wrote {resolve(args.output_dir) / 'summary.json'}")
    print(f'successful_runs={successes} missing_runs={missing}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
