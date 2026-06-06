'''Summarize CausalGenGuard result files for paper tables.'''

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

METRIC_FIELDS = ['precision', 'recall', 'f1', 'fpr', 'fnr', 'auroc', 'auprc']
SUMMARY_FIELDS = [
    'category', 'dataset', 'transition', 'method',
    'precision', 'recall', 'f1', 'fpr', 'fnr', 'auroc', 'auprc',
    'threshold', 'delta_f1_vs_smartguard', 'relative_f1_improvement_pct',
    'adaptation_gain_f1', 'runtime_seconds', 'source_file', 'notes',
]
SUMMARY_OUTPUT_NAMES = {'summary_main.csv', 'summary_ablation.csv', 'summary_low_data.csv', 'tables_for_paper.md'}
SMARTGUARD_BASELINE_NAMES = {'smartguard only', 'smartguard', 'smartguard baseline'}
ORACLE_HINTS = ('oracle-target', 'oracle target', 'target oracle', 'oracle_target')
LOW_DATA_HINTS = ('low_data', 'low-data', 'few-shot', 'few_shot', 'target_adaptation', 'adaptation')
CAUSAL_ANOMALY_HINTS = ('causal_anomaly', 'causal anomaly', 'dependency anomaly')
RUNTIME_KEYS = ('runtime_seconds', 'elapsed_seconds', 'duration_seconds', 'wall_time_seconds', 'train_seconds', 'eval_seconds')


def safe_float(value: Any) -> Optional[float]:
    if value in (None, '', 'nan', 'NaN', 'None'):
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(parsed) or math.isinf(parsed):
        return None
    return parsed


def format_cell(value: Any) -> str:
    number = safe_float(value)
    if number is not None:
        return '{:.4f}'.format(number)
    return '' if value is None else str(value)


def infer_dataset_from_path(path: Path) -> str:
    text = path.as_posix().lower()
    for token in ('fr', 'sp', 'us', 'kr', 'argus', 'smartsense', 'toy'):
        if re.search(r'(^|[^a-z0-9]){}([^a-z0-9]|$)'.format(token), text):
            return token
    return 'unknown'


def infer_transition(path: Path, payload: Dict[str, Any]) -> str:
    if payload.get('transition'):
        return str(payload['transition'])
    source = payload.get('source_context')
    target = payload.get('target_context')
    if source or target:
        return '{}->{}'.format(source or 'source', target or 'target')
    match = re.search(r'([a-z0-9]+)->([a-z0-9]+)', path.as_posix().lower())
    return match.group(0) if match else 'unknown'


def infer_method_from_json(path: Path, payload: Dict[str, Any]) -> str:
    for key in ('method', 'model', 'detector', 'experiment'):
        value = payload.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    stem = path.parent.name.lower() if path.name == 'summary.json' else path.stem.lower()
    known = {
        'smartguard_only': 'SmartGuard only',
        'smartguard_plus_smartgen': 'SmartGuard + SmartGen',
        'smartguard_plus_causal': 'SmartGuard + Causal',
        'smartguard_plus_smartgen_plus_causal': 'SmartGuard + SmartGen + Causal',
        'full_causalgenguard': 'Full CausalGenGuard + CausalTOF + causal-aware NWRL',
    }
    for hint, method in known.items():
        if hint in stem:
            return method
    weights = payload.get('weights')
    beta = safe_float(weights.get('beta')) if isinstance(weights, dict) else None
    return 'SmartGuard only' if beta == 0.0 else 'FusionDetector'


def extract_runtime(payload: Dict[str, Any]) -> Optional[float]:
    for key in RUNTIME_KEYS:
        value = safe_float(payload.get(key))
        if value is not None:
            return value
    for section_key in ('runtime', 'timing', 'time', 'profile'):
        section = payload.get(section_key)
        if isinstance(section, dict):
            for key in RUNTIME_KEYS:
                value = safe_float(section.get(key))
                if value is not None:
                    return value
    return None


def empty_row(path: Path) -> Dict[str, Any]:
    row = dict.fromkeys(SUMMARY_FIELDS, '')
    row['source_file'] = str(path)
    return row


def row_has_metric(row: Dict[str, Any]) -> bool:
    return any(safe_float(row.get(field)) is not None for field in METRIC_FIELDS)


def category_for(row: Dict[str, Any], path: Path) -> str:
    text = ' '.join([
        str(path).lower(), str(row.get('method', '')).lower(),
        str(row.get('transition', '')).lower(), str(row.get('notes', '')).lower(),
        str(row.get('anomaly_type', '')).lower(),
    ])
    if any(hint in text for hint in CAUSAL_ANOMALY_HINTS):
        return 'causal_anomaly'
    if any(hint in text for hint in LOW_DATA_HINTS):
        return 'low_data'
    method = str(row.get('method', '')).lower()
    ablation_methods = (
        'smartguard only',
        'smartguard + smartgen',
        'smartguard + causal',
        'smartguard + smartgen + causal',
        'full causalgenguard',
    )
    if any(item in method for item in ablation_methods) or 'causaltof' in method or 'ablation' in text:
        return 'ablation'
    return 'main'


def row_from_metrics(path: Path, payload: Dict[str, Any], metrics: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    if not isinstance(metrics, dict) or not any(field in metrics for field in METRIC_FIELDS):
        return None
    row = empty_row(path)
    row['dataset'] = str(payload.get('dataset') or infer_dataset_from_path(path))
    row['transition'] = infer_transition(path, payload)
    row['method'] = infer_method_from_json(path, payload)
    row['threshold'] = payload.get('threshold', '')
    row['runtime_seconds'] = extract_runtime(payload) or ''
    warnings_value = payload.get('warnings')
    if isinstance(warnings_value, list):
        row['notes'] = '; '.join(str(item) for item in warnings_value if item)
    else:
        notes_value = payload.get('notes')
        row['notes'] = '; '.join(str(item) for item in notes_value) if isinstance(notes_value, list) else str(notes_value or '')
    for field in METRIC_FIELDS:
        row[field] = metrics.get(field, '')
    row['category'] = category_for(row, path)
    return row if row_has_metric(row) else None


def rows_from_json(path: Path, missing: List[str]) -> List[Dict[str, Any]]:
    try:
        payload = json.loads(path.read_text(encoding='utf-8'))
    except Exception as exc:
        missing.append('{}: could not parse JSON ({})'.format(path, exc))
        return []
    if not isinstance(payload, dict):
        missing.append('{}: JSON root is not an object'.format(path))
        return []
    rows: List[Dict[str, Any]] = []
    direct = row_from_metrics(path, payload, payload.get('metrics') or {})
    if direct is not None:
        rows.append(direct)
    for key in ('results', 'rows', 'experiments', 'methods'):
        value = payload.get(key)
        if isinstance(value, list):
            for item in value:
                if isinstance(item, dict):
                    nested = dict(payload)
                    nested.update(item)
                    metrics = item.get('metrics') if isinstance(item.get('metrics'), dict) else item
                    nested_row = row_from_metrics(path, nested, metrics)
                    if nested_row is not None:
                        rows.append(nested_row)
    if not rows:
        missing.append('{}: no metric dictionary found; skipped'.format(path))
    return rows


def rows_from_csv(path: Path, missing: List[str]) -> List[Dict[str, Any]]:
    try:
        with path.open('r', encoding='utf-8', newline='') as handle:
            raw_rows = [dict(row) for row in csv.DictReader(handle)]
    except Exception as exc:
        missing.append('{}: could not parse CSV ({})'.format(path, exc))
        return []
    if not raw_rows:
        missing.append('{}: empty CSV; skipped'.format(path))
        return []
    rows: List[Dict[str, Any]] = []
    for raw in raw_rows:
        row = empty_row(path)
        row['dataset'] = raw.get('dataset') or infer_dataset_from_path(path)
        row['transition'] = raw.get('transition') or raw.get('context_shift') or 'unknown'
        row['method'] = raw.get('method') or raw.get('model') or 'unknown'
        row['threshold'] = raw.get('threshold', '')
        row['runtime_seconds'] = raw.get('runtime_seconds') or raw.get('elapsed_seconds') or ''
        row['notes'] = raw.get('notes', '')
        for field in METRIC_FIELDS:
            row[field] = raw.get(field, '')
        row['category'] = category_for(row, path)
        if row_has_metric(row):
            rows.append(row)
    if not rows:
        missing.append('{}: no metric columns found; skipped'.format(path))
    return rows


def discover_result_files(results_dir: Path, recursive: bool = False) -> List[Path]:
    candidates = results_dir.rglob('*') if recursive else results_dir.glob('*')
    files: List[Path] = []
    for path in candidates:
        if not path.is_file():
            continue
        if path.name in SUMMARY_OUTPUT_NAMES or path.name.startswith('summary_'):
            continue
        if path.suffix.lower() in ('.json', '.csv'):
            files.append(path)
    return sorted(files)


def load_result_rows(results_dir: Path, recursive: bool = False) -> Tuple[List[Dict[str, Any]], List[str]]:
    files = discover_result_files(results_dir, recursive=recursive)
    rows: List[Dict[str, Any]] = []
    missing: List[str] = []
    if not files:
        missing.append('{}: no JSON or CSV result files found'.format(results_dir))
        return rows, missing
    for path in files:
        if path.suffix.lower() == '.json':
            rows.extend(rows_from_json(path, missing))
        elif path.suffix.lower() == '.csv':
            rows.extend(rows_from_csv(path, missing))
    return rows, missing


def group_key(row: Dict[str, Any]) -> Tuple[str, str]:
    return (str(row.get('dataset') or 'unknown'), str(row.get('transition') or 'unknown'))


def is_smartguard_baseline(row: Dict[str, Any]) -> bool:
    method = str(row.get('method') or '').strip().lower()
    return method in SMARTGUARD_BASELINE_NAMES or method.startswith('smartguard only')


def is_oracle_target(row: Dict[str, Any]) -> bool:
    text = '{} {}'.format(row.get('method', ''), row.get('notes', '')).lower()
    return any(hint in text for hint in ORACLE_HINTS)


def add_relative_metrics(rows: List[Dict[str, Any]]) -> bool:
    baselines: Dict[Tuple[str, str], float] = {}
    oracles: Dict[Tuple[str, str], float] = {}
    for row in rows:
        f1 = safe_float(row.get('f1'))
        if f1 is None:
            continue
        key = group_key(row)
        if is_smartguard_baseline(row) and key not in baselines:
            baselines[key] = f1
        if is_oracle_target(row):
            oracles[key] = f1
    for row in rows:
        f1 = safe_float(row.get('f1'))
        if f1 is None:
            continue
        key = group_key(row)
        baseline = baselines.get(key)
        if baseline is not None:
            delta = f1 - baseline
            row['delta_f1_vs_smartguard'] = delta
            row['relative_f1_improvement_pct'] = '' if baseline == 0 else 100.0 * delta / baseline
        oracle = oracles.get(key)
        if baseline is not None and oracle is not None and oracle != baseline:
            row['adaptation_gain_f1'] = (f1 - baseline) / (oracle - baseline)
    return bool(oracles)


def sort_rows(rows: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return sorted(rows, key=lambda row: (str(row.get('dataset')), str(row.get('transition')), str(row.get('method')), str(row.get('source_file'))))


def split_summaries(rows: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    summaries: Dict[str, List[Dict[str, Any]]] = {'main': [], 'ablation': [], 'low_data': [], 'causal_anomaly': [], 'runtime': []}
    for row in rows:
        category = row.get('category') or 'main'
        if category == 'ablation':
            summaries['ablation'].append(row)
            method = str(row.get('method', '')).lower()
            if 'causalgenguard' in method or 'full' in method:
                summaries['main'].append(row)
        elif category == 'low_data':
            summaries['low_data'].append(row)
        elif category == 'causal_anomaly':
            summaries['causal_anomaly'].append(row)
        else:
            summaries['main'].append(row)
        if safe_float(row.get('runtime_seconds')) is not None:
            summaries['runtime'].append(row)
    return {key: sort_rows(value) for key, value in summaries.items()}


def write_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: format_cell(row.get(field, '')) for field in SUMMARY_FIELDS})


def markdown_table(title: str, rows: Sequence[Dict[str, Any]], fields: Sequence[str]) -> str:
    lines = ['## {}'.format(title), '']
    if not rows:
        lines.extend(['No results found for this table.', ''])
        return '\n'.join(lines)
    lines.append('| ' + ' | '.join(fields) + ' |')
    lines.append('| ' + ' | '.join(['---'] * len(fields)) + ' |')
    for row in rows:
        values = [format_cell(row.get(field, '')).replace('|', '/') for field in fields]
        lines.append('| ' + ' | '.join(values) + ' |')
    lines.append('')
    return '\n'.join(lines)


def write_markdown(path: Path, summaries: Dict[str, List[Dict[str, Any]]], missing: List[str], has_oracle: bool) -> None:
    metric_columns = ['dataset', 'transition', 'method'] + METRIC_FIELDS + ['delta_f1_vs_smartguard', 'adaptation_gain_f1', 'notes']
    runtime_columns = ['dataset', 'transition', 'method', 'runtime_seconds', 'source_file', 'notes']
    sections = [
        '# Tables For Paper', '',
        'Generated from local files under `outputs/results`. Missing result categories are reported instead of filled with synthetic values.', '',
        markdown_table('Main context shift results', summaries['main'], metric_columns),
        markdown_table('Low-data adaptation results', summaries['low_data'], metric_columns),
        markdown_table('Causal anomaly results', summaries['causal_anomaly'], metric_columns),
        markdown_table('Ablation results', summaries['ablation'], metric_columns),
        markdown_table('Runtime results if available', summaries['runtime'], runtime_columns),
        '## Missing Or Skipped', '',
    ]
    missing_lines: List[str] = []
    if not summaries['main']:
        missing_lines.append('Main context shift results: no rows found.')
    if not summaries['low_data']:
        missing_lines.append('Low-data adaptation results: no rows found.')
    if not summaries['causal_anomaly']:
        missing_lines.append('Causal anomaly results: no rows found.')
    if not summaries['ablation']:
        missing_lines.append('Ablation results: no rows found.')
    if not summaries['runtime']:
        missing_lines.append('Runtime results: no runtime fields found.')
    if not has_oracle:
        missing_lines.append('Oracle-target results: no rows found; adaptation_gain_f1 left blank.')
    missing_lines.extend(missing)
    if not missing_lines:
        missing_lines.append('No missing or skipped result files were detected.')
    sections.extend('- {}'.format(item) for item in missing_lines)
    sections.append('')
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text('\n'.join(sections), encoding='utf-8')


def summarize(results_dir: Path, recursive: bool = False) -> Dict[str, Any]:
    rows, missing = load_result_rows(results_dir, recursive=recursive)
    has_oracle = add_relative_metrics(rows)
    summaries = split_summaries(rows)
    write_csv(results_dir / 'summary_main.csv', summaries['main'])
    write_csv(results_dir / 'summary_ablation.csv', summaries['ablation'])
    write_csv(results_dir / 'summary_low_data.csv', summaries['low_data'])
    write_markdown(results_dir / 'tables_for_paper.md', summaries, missing, has_oracle)
    return {'rows': rows, 'summaries': summaries, 'missing': missing, 'has_oracle': has_oracle}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Summarize CausalGenGuard result JSON/CSV files for paper tables.')
    parser.add_argument('--results-dir', default='outputs/results')
    parser.add_argument('--recursive', action='store_true', help='Scan subdirectories under results-dir.')
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    results_dir = Path(args.results_dir)
    if not results_dir.exists():
        raise SystemExit('Results directory does not exist: {}'.format(results_dir))
    report = summarize(results_dir, recursive=args.recursive)
    print('Scanned {} metric rows from {}'.format(len(report['rows']), results_dir))
    print('Wrote {}'.format(results_dir / 'summary_main.csv'))
    print('Wrote {}'.format(results_dir / 'summary_ablation.csv'))
    print('Wrote {}'.format(results_dir / 'summary_low_data.csv'))
    print('Wrote {}'.format(results_dir / 'tables_for_paper.md'))
    if report['missing']:
        print('Skipped or missing entries: {}'.format(len(report['missing'])))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
