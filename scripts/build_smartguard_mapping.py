#!/usr/bin/env python3
'''Build SmartGuard numeric-id mapping JSON files.'''

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / 'src'
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from causal_gen_guard.data.smartguard_dictionary import (
    build_mapping_report,
    load_smartguard_dictionary,
    write_mapping_files,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Build SmartGuard device/control id mapping JSON files.')
    parser.add_argument('--smartguard-root', default='../SmartGuard', help='Path to the SmartGuard project root.')
    parser.add_argument('--dataset', default='fr', choices=['fr', 'sp', 'an'], help='SmartGuard dataset name.')
    parser.add_argument(
        '--output-dir',
        default=None,
        help='Directory for mapping JSON outputs. Defaults to outputs/mappings/smartguard/<dataset>.',
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    root_arg = Path(args.smartguard_root)
    smartguard_root = root_arg if root_arg.is_absolute() else (PROJECT_ROOT / root_arg).resolve()

    if args.output_dir is None:
        output_dir = PROJECT_ROOT / 'outputs' / 'mappings' / 'smartguard' / args.dataset
    else:
        output_arg = Path(args.output_dir)
        output_dir = output_arg if output_arg.is_absolute() else (PROJECT_ROOT / output_arg).resolve()

    mapping = load_smartguard_dictionary(smartguard_root, args.dataset)
    report = build_mapping_report(mapping)
    written = write_mapping_files(mapping, output_dir, report=report)

    print(f'Wrote SmartGuard {args.dataset} mapping files to {output_dir}')
    print(f'devices={report["device_count"]} controls={report["control_count"]}')
    print(f'all_key_controls_present={report["all_key_controls_present"]}')
    for path in written.values():
        print(path)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
