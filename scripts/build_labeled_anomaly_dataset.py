#!/usr/bin/env python3
'''Build a labeled SmartGuard anomaly JSONL from canonical normal sequences.'''

from __future__ import annotations

import argparse
import copy
import json
import random
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / 'src'
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from causal_gen_guard.data.attack_injector import inject_smartguard_style
from causal_gen_guard.data.schemas import BehaviorSequence


DEFAULT_NAMED_ATTACKS = [
    'SD_light_flickering',
    'SD_camera_flickering',
    'SD_tv_flickering',
    'MD_camera_off_while_lock',
    'MD_window_open_while_lock',
    'DM_window_open_midnight',
    'DM_watervalve_open_midnight',
    'DD_microwave_long_time',
]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Build labeled SmartGuard-style anomaly JSONL.')
    parser.add_argument(
        '--normal-jsonl',
        '--input-jsonl',
        '--input',
        dest='input_jsonl',
        required=True,
        help='Canonical normal SmartGuard JSONL input.',
    )
    parser.add_argument('--output-jsonl', '--output', required=True, help='Labeled JSONL output path.')
    parser.add_argument('--report', required=True, help='Generation report JSON path.')
    parser.add_argument(
        '--anomaly-types',
        nargs='+',
        default=DEFAULT_NAMED_ATTACKS,
        help='Named SmartGuard anomaly types to inject.',
    )
    parser.add_argument('--per-anomaly-type', type=int, default=1, help='Target injected examples per anomaly type.')
    parser.add_argument('--ratio', type=float, default=None, help='Target anomaly-to-normal ratio. Overrides --per-anomaly-type.')
    parser.add_argument('--seed', type=int, default=42, help='Random seed used to rotate source sequence attempts.')
    parser.add_argument('--max-sequences', type=int, default=None, help='Optional limit on normal sequences read from input.')
    parser.add_argument(
        '--max-attempts-per-type',
        type=int,
        default=None,
        help='Optional maximum attempts per anomaly type. Defaults to one pass over normal sequences.',
    )
    return parser


def _resolve_project_path(path_value: str) -> Path:
    path = Path(path_value)
    return path if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def load_jsonl(path: str | Path) -> List[BehaviorSequence]:
    sequences: List[BehaviorSequence] = []
    with Path(path).open('r', encoding='utf-8') as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                payload = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(f'Invalid JSON on line {line_number} of {path}') from exc
            sequences.append(BehaviorSequence.from_dict(payload))
    return sequences


def _normal_copy(sequence: BehaviorSequence) -> BehaviorSequence:
    copied = copy.deepcopy(sequence)
    copied.label = 0
    copied.anomaly_type = 'normal'
    return copied


def _clone_injected(sequence: BehaviorSequence, injection_index: int) -> BehaviorSequence:
    copied = copy.deepcopy(sequence)
    copied.sequence_id = f'{copied.sequence_id}::labeled_{injection_index:06d}'
    copied.context = dict(copied.context)
    copied.context['labeled_anomaly_index'] = injection_index
    return copied


def _scan_named_injection_pairs(
    normal_sequences: List[BehaviorSequence],
    anomaly_types: List[str],
) -> Tuple[Dict[str, List[int]], List[Dict[str, Any]]]:
    success_by_type = {anomaly_type: [] for anomaly_type in anomaly_types}
    skipped_reports: List[Dict[str, Any]] = []
    for anomaly_type in anomaly_types:
        for index, source in enumerate(normal_sequences):
            injected, report = inject_smartguard_style(source, anomaly_type)
            if injected is None:
                skipped_reports.append(report)
                continue
            success_by_type[anomaly_type].append(index)
    return success_by_type, skipped_reports


def _make_injected(
    normal_sequences: List[BehaviorSequence],
    anomaly_type: str,
    source_index: int,
    injection_index: int,
) -> Tuple[BehaviorSequence, Dict[str, Any]]:
    injected, report = inject_smartguard_style(normal_sequences[source_index], anomaly_type)
    if injected is None:
        raise RuntimeError(f'Previously successful injection failed for {anomaly_type}')
    return _clone_injected(injected, injection_index), report


def build_labeled_dataset(
    normal_sequences: List[BehaviorSequence],
    anomaly_types: List[str],
    per_anomaly_type: int,
    seed: int = 42,
    max_attempts_per_type: Optional[int] = None,
    ratio: Optional[float] = None,
) -> Tuple[List[BehaviorSequence], Dict[str, Any]]:
    if per_anomaly_type < 0:
        raise ValueError('--per-anomaly-type must be non-negative')
    if ratio is not None and ratio < 0:
        raise ValueError('--ratio must be non-negative')
    if not anomaly_types:
        raise ValueError('--anomaly-types must not be empty')

    labeled = [_normal_copy(sequence) for sequence in normal_sequences]
    rng = random.Random(seed)
    per_success = {anomaly_type: 0 for anomaly_type in anomaly_types}
    injected_reports: List[Dict[str, Any]] = []
    fallback_numeric_injection_count = 0
    success_by_type, skipped_reports = _scan_named_injection_pairs(normal_sequences, anomaly_types)
    available_pairs = [
        (anomaly_type, source_index)
        for anomaly_type, source_indices in success_by_type.items()
        for source_index in source_indices
    ]
    attempts_limit = max_attempts_per_type

    if ratio is not None:
        target_anomaly_count = int(round(len(normal_sequences) * ratio))
        if available_pairs:
            for injection_index in range(target_anomaly_count):
                anomaly_type, source_index = rng.choice(available_pairs)
                injected, report = _make_injected(normal_sequences, anomaly_type, source_index, injection_index)
                labeled.append(injected)
                injected_reports.append(report)
                per_success[anomaly_type] += 1
    else:
        target_anomaly_count = per_anomaly_type * len(anomaly_types)
        injection_index = 0
        for anomaly_type in anomaly_types:
            source_indices = list(success_by_type[anomaly_type])
            if not source_indices:
                continue
            rng.shuffle(source_indices)
            limit = per_anomaly_type if attempts_limit is None else min(per_anomaly_type, attempts_limit)
            for offset in range(limit):
                source_index = source_indices[offset % len(source_indices)]
                injected, report = _make_injected(normal_sequences, anomaly_type, source_index, injection_index)
                labeled.append(injected)
                injected_reports.append(report)
                per_success[anomaly_type] += 1
                injection_index += 1

    skipped_reason_counts = Counter(
        report.get('skipped_reason', 'unknown') for report in skipped_reports if report.get('status') == 'skipped'
    )
    named_injection_success_count = sum(per_success.values())
    report = {
        'normal_count': len(normal_sequences),
        'anomaly_count': named_injection_success_count + fallback_numeric_injection_count,
        'output_count': len(labeled),
        'target_anomaly_count': target_anomaly_count,
        'named_injection_success_count': named_injection_success_count,
        'fallback_numeric_injection_count': fallback_numeric_injection_count,
        'skipped_count': len(skipped_reports),
        'per_anomaly_type_success': per_success,
        'skipped_reasons': dict(sorted(skipped_reason_counts.items())),
        'seed': seed,
        'ratio': ratio,
        'per_anomaly_type_target': per_anomaly_type,
        'anomaly_types': anomaly_types,
        'injected': injected_reports,
        'skipped': skipped_reports,
    }
    return labeled, report


def write_jsonl(sequences: List[BehaviorSequence], path: str | Path) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open('w', encoding='utf-8') as handle:
        for sequence in sequences:
            handle.write(json.dumps(sequence.to_dict(), ensure_ascii=False) + '\n')


def write_report(report: Dict[str, Any], path: str | Path) -> None:
    report_path = Path(path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    input_jsonl = _resolve_project_path(args.input_jsonl)
    output_jsonl = _resolve_project_path(args.output_jsonl)
    report_path = _resolve_project_path(args.report)

    normal_sequences = load_jsonl(input_jsonl)
    if args.max_sequences is not None:
        if args.max_sequences <= 0:
            raise ValueError('--max-sequences must be positive when provided')
        normal_sequences = normal_sequences[: args.max_sequences]
    labeled, report = build_labeled_dataset(
        normal_sequences,
        list(args.anomaly_types),
        args.per_anomaly_type,
        seed=args.seed,
        max_attempts_per_type=args.max_attempts_per_type,
        ratio=args.ratio,
    )
    write_jsonl(labeled, output_jsonl)
    write_report(report, report_path)

    print(f'Wrote {len(labeled)} labeled sequences to {output_jsonl}')
    print(
        'named_injection_success_count={} fallback_numeric_injection_count={} skipped_count={}'.format(
            report['named_injection_success_count'],
            report['fallback_numeric_injection_count'],
            report['skipped_count'],
        )
    )
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
