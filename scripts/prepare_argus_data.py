#!/usr/bin/env python3
'''Prepare local ARGUS Home1-Home5 data for CausalGenGuard.

This script never downloads data. It expects --argus-root to point at a local
folder containing Home* directories or readable csv/json event files.
'''

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from causal_gen_guard.data.argus_adapter import load_argus_dataset, split_argus_sequences
from causal_gen_guard.data.schemas import BehaviorSequence


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if hasattr(value, 'item'):
        try:
            return value.item()
        except Exception:
            pass
    if hasattr(value, 'tolist'):
        try:
            return _json_safe(value.tolist())
        except Exception:
            pass
    return value


def write_jsonl(path: Path, sequences: Iterable[BehaviorSequence]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open('w', encoding='utf-8') as handle:
        for sequence in sequences:
            handle.write(json.dumps(_json_safe(sequence.to_dict()), ensure_ascii=False) + '\n')
            count += 1
    return count


def split_path(output: Path, name: str) -> Path:
    return output.with_name('{}_{}.jsonl'.format(output.stem, name))


def build_column_mapping(args: argparse.Namespace) -> Dict[str, str]:
    mapping: Dict[str, str] = {}
    if args.timestamp_column:
        mapping['timestamp'] = args.timestamp_column
    if args.device_column:
        mapping['device'] = args.device_column
    if args.control_column:
        mapping['control'] = args.control_column
    if args.action_column:
        mapping['action'] = args.action_column
        mapping['state'] = args.action_column
    return mapping


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Prepare local ARGUS Home1-Home5 data as BehaviorSequence JSONL.')
    parser.add_argument('--argus-root', required=True, help='Local ARGUS root containing Home* directories.')
    parser.add_argument('--output', default='outputs/processed/argus_sequences.jsonl')
    parser.add_argument('--split', choices=['temporal', 'leave_one_home'], default='temporal')
    parser.add_argument('--window-size', type=int, default=50)
    parser.add_argument('--leave-home-id', default=None, help='Home id held out for leave_one_home split, default is last sorted home.')
    parser.add_argument('--timestamp-column', default=None, help='Override timestamp column name if auto-detection fails.')
    parser.add_argument('--device-column', default=None, help='Override device column name if auto-detection fails.')
    parser.add_argument('--control-column', default=None, help='Override control column name if auto-detection fails.')
    parser.add_argument('--action-column', default=None, help='Override action/state column name if auto-detection fails.')
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if args.window_size <= 0:
        raise ValueError('--window-size must be positive')

    output = Path(args.output)
    columns = build_column_mapping(args) or None
    try:
        sequences = load_argus_dataset(args.argus_root, columns=columns, window_size=args.window_size)
    except FileNotFoundError as exc:
        raise SystemExit(str(exc)) from exc
    except Exception as exc:
        raise SystemExit('Failed to prepare ARGUS data: {}'.format(exc)) from exc

    total_count = write_jsonl(output, sequences)
    splits = split_argus_sequences(sequences, split=args.split, leave_home_id=args.leave_home_id)
    split_counts = {name: write_jsonl(split_path(output, name), rows) for name, rows in splits.items()}
    print('Wrote {} ARGUS sequences to {}'.format(total_count, output))
    for name in ('train', 'val', 'test'):
        print('Wrote {} {} sequences to {}'.format(split_counts.get(name, 0), name, split_path(output, name)))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
