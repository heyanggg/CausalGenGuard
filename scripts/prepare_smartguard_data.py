#!/usr/bin/env python3
'''Prepare SmartGuard FR/SP data for CausalGenGuard.

This script only converts SmartGuard samples into the unified JSONL schema. It
never trains a model and never modifies the SmartGuard source project.
'''

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / 'src'
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from causal_gen_guard.data.smartguard_adapter import load_smartguard_dataset
from causal_gen_guard.data.smartguard_dictionary import (
    build_mapping_report,
    load_smartguard_dictionary,
    parse_smartguard_dictionary,
    write_mapping_files,
)


def build_parser() -> argparse.ArgumentParser:
    '''Create the command-line parser.'''
    parser = argparse.ArgumentParser(description='Convert SmartGuard data into CausalGenGuard JSONL.')
    parser.add_argument('--smartguard-root', default='../SmartGuard', help='Path to the SmartGuard project root.')
    parser.add_argument('--dataset', default='fr', choices=['fr', 'sp', 'an'], help='SmartGuard dataset name.')
    parser.add_argument(
        '--output',
        default='outputs/processed/fr_sequences.jsonl',
        help='Output JSONL path relative to CausalGenGuard or absolute.',
    )
    parser.add_argument('--limit', type=int, default=None, help='Optional maximum number of sequences to write for smoke tests.')
    parser.add_argument(
        '--smartguard-dictionary',
        default=None,
        help='Optional SmartGuard dictionary.py path for canonical device/control mapping.',
    )
    parser.add_argument(
        '--emit-canonical-control',
        action='store_true',
        help='Emit canonical Device:action controls and preserve raw numeric ids in raw_fields.',
    )
    parser.add_argument(
        '--mapping-output-dir',
        default=None,
        help='Optional directory for device/control mapping JSON files.',
    )
    return parser


def _resolve_project_path(path_value: str | None) -> Path | None:
    if path_value is None:
        return None
    path = Path(path_value)
    return path if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def main(argv: list[str] | None = None) -> int:
    '''Load SmartGuard data and write one BehaviorSequence per JSONL line.'''
    args = build_parser().parse_args(argv)
    root_arg = Path(args.smartguard_root)
    smartguard_root = root_arg if root_arg.is_absolute() else (PROJECT_ROOT / root_arg).resolve()
    output = _resolve_project_path(args.output)
    dictionary_path = _resolve_project_path(args.smartguard_dictionary)
    mapping_output_dir = _resolve_project_path(args.mapping_output_dir)

    if args.limit is not None and args.limit <= 0:
        raise ValueError('--limit must be positive when provided')

    smartguard_mapping = None
    if dictionary_path is not None:
        smartguard_mapping = parse_smartguard_dictionary(dictionary_path, dataset=args.dataset)
    elif args.emit_canonical_control or mapping_output_dir is not None:
        smartguard_mapping = load_smartguard_dictionary(smartguard_root, args.dataset)

    if mapping_output_dir is not None and smartguard_mapping is not None:
        write_mapping_files(
            smartguard_mapping,
            mapping_output_dir,
        )

    sequences = load_smartguard_dataset(
        smartguard_root,
        args.dataset,
        smartguard_mapping=smartguard_mapping,
        emit_canonical_control=args.emit_canonical_control,
    )
    if args.limit is not None:
        sequences = sequences[: args.limit]
    assert output is not None
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('w', encoding='utf-8') as handle:
        for sequence in sequences:
            handle.write(json.dumps(sequence.to_dict(), ensure_ascii=False) + '\n')

    print('Wrote {} SmartGuard sequences to {}'.format(len(sequences), output))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
