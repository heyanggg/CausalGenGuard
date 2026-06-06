'''Ablation runner for context-shift CausalGenGuard experiments.

Each ablation variant calls the same train_fusion pipeline with small config
overrides. Missing datasets or optional artifacts are recorded as failed rows in
the output CSV rather than hiding partial progress.
'''

from __future__ import annotations

import argparse
import csv
import math
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from causal_gen_guard.training.train_fusion import load_config, project_root, resolve_path, run_pipeline


CSV_FIELDS = [
    'method',
    'dataset',
    'transition',
    'precision',
    'recall',
    'f1',
    'fpr',
    'fnr',
    'auroc',
    'auprc',
    'threshold',
    'notes',
]


def slugify(value: str) -> str:
    value = value.lower().replace('+', 'plus')
    value = re.sub(r'[^a-z0-9]+', '_', value).strip('_')
    return value or 'method'


def transition_from_config(config: Dict[str, Any]) -> str:
    return '{}->{}'.format(config.get('source_context', 'source'), config.get('target_context', 'target'))


def nan_metrics() -> Dict[str, float]:
    return {
        'precision': float('nan'),
        'recall': float('nan'),
        'f1': float('nan'),
        'fpr': float('nan'),
        'fnr': float('nan'),
        'auroc': float('nan'),
        'auprc': float('nan'),
    }


def format_value(value: Any) -> Any:
    if isinstance(value, float) and math.isnan(value):
        return 'nan'
    return value


def ablation_variants() -> List[Dict[str, Any]]:
    return [
        {
            'method': 'SmartGuard only',
            'overrides': {
                'experiments': {'use_smartgen': False, 'use_causal': False, 'use_causal_tof': False},
                'fusion': {'alpha': 1.0, 'beta': 0.0},
            },
            'notes': 'reconstruction score only',
        },
        {
            'method': 'SmartGuard + SmartGen',
            'overrides': {
                'experiments': {'use_smartgen': True, 'use_causal': False, 'use_causal_tof': False},
                'fusion': {'alpha': 1.0, 'beta': 0.0},
            },
            'notes': 'synthetic target normal augmentation without causal branch',
        },
        {
            'method': 'SmartGuard + Causal',
            'overrides': {
                'experiments': {'use_smartgen': False, 'use_causal': True, 'use_causal_tof': False},
                'fusion': {'alpha': 0.6, 'beta': 0.4},
            },
            'notes': 'source normal plus behavior causal graph deviation',
        },
        {
            'method': 'SmartGuard + SmartGen + Causal',
            'overrides': {
                'experiments': {'use_smartgen': True, 'use_causal': True, 'use_causal_tof': False},
                'fusion': {'alpha': 0.6, 'beta': 0.4},
            },
            'notes': 'SmartGen augmentation and causal branch, no Causal-TOF',
        },
        {
            'method': 'Full CausalGenGuard + CausalTOF + causal-aware NWRL',
            'overrides': {
                'experiments': {'use_smartgen': True, 'use_causal': True, 'use_causal_tof': True, 'use_causal_aware_nwrl': True},
                'fusion': {'alpha': 0.6, 'beta': 0.4},
            },
            'notes': 'Causal-TOF enabled; causal-aware NWRL fine-tune enabled after causal centrality estimation',
        },
    ]


def row_from_summary(method: str, config: Dict[str, Any], summary: Dict[str, Any], base_notes: str) -> Dict[str, Any]:
    metrics = dict(summary.get('metrics') or nan_metrics())
    notes = list(summary.get('notes') or [])
    if base_notes:
        notes.insert(0, base_notes)
    return {
        'method': method,
        'dataset': summary.get('dataset', config.get('dataset', 'unknown')),
        'transition': summary.get('transition', transition_from_config(config)),
        'precision': metrics.get('precision', float('nan')),
        'recall': metrics.get('recall', float('nan')),
        'f1': metrics.get('f1', float('nan')),
        'fpr': metrics.get('fpr', float('nan')),
        'fnr': metrics.get('fnr', float('nan')),
        'auroc': metrics.get('auroc', float('nan')),
        'auprc': metrics.get('auprc', float('nan')),
        'threshold': summary.get('threshold', float('nan')),
        'notes': '; '.join(str(item) for item in notes if item),
    }


def failed_row(method: str, config: Dict[str, Any], error: Exception, base_notes: str) -> Dict[str, Any]:
    metrics = nan_metrics()
    return {
        'method': method,
        'dataset': config.get('dataset', 'unknown'),
        'transition': transition_from_config(config),
        'precision': metrics['precision'],
        'recall': metrics['recall'],
        'f1': metrics['f1'],
        'fpr': metrics['fpr'],
        'fnr': metrics['fnr'],
        'auroc': metrics['auroc'],
        'auprc': metrics['auprc'],
        'threshold': float('nan'),
        'notes': '{}; failed: {}'.format(base_notes, error),
    }


def default_output_path(config: Dict[str, Any]) -> Path:
    dataset = str(config.get('dataset', 'unknown'))
    return project_root() / 'outputs' / 'results' / 'context_shift_{}.csv'.format(dataset)


def run_ablation(config: Dict[str, Any], smoke_test: bool = False, output: Optional[Path] = None) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for variant in ablation_variants():
        method = variant['method']
        overrides = dict(variant['overrides'])
        overrides['run_name'] = '{}{}'.format(slugify(method), '_smoke' if smoke_test else '')
        try:
            summary = run_pipeline(config, smoke_test=smoke_test, overrides=overrides)
            rows.append(row_from_summary(method, config, summary, variant.get('notes', '')))
        except Exception as exc:
            rows.append(failed_row(method, config, exc, variant.get('notes', '')))
            print('Ablation method failed: {}: {}'.format(method, exc))

    output_path = output or default_output_path(config)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open('w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: format_value(row.get(key)) for key in CSV_FIELDS})
    print('Saved ablation CSV to {}'.format(output_path))
    return rows


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Run context-shift ablations for CausalGenGuard.')
    parser.add_argument('--config', required=True, help='YAML config such as configs/context_shift_fr.yaml')
    parser.add_argument('--smoke-test', action='store_true', help='Run each ablation on a tiny subset')
    parser.add_argument('--output', default=None, help='Optional CSV output path')
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    config = load_config(args.config)
    output = resolve_path(args.output, project_root()) if args.output else None
    run_ablation(config, smoke_test=args.smoke_test, output=output)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
