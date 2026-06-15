#!/usr/bin/env python3
'''Build CausalGenGuard US mapping files from SmartGen textual dictionaries.'''

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / 'src'
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from causal_gen_guard.data.smartgen_us_mapping_adapter import (
    build_smartgen_mapping_report,
    load_smartgen_textual_mapping,
    write_smartgen_mapping_files,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Build US mapping JSON files from SmartGen textual dictionaries.')
    parser.add_argument('--smartgen-root', default='../SmartGen/SmartGen', help='Path to the SmartGen project root.')
    parser.add_argument('--dataset', default='us', choices=['us'], help='Dataset name. Only US is supported here.')
    parser.add_argument('--output-dir', default='outputs/mappings/smartguard/us', help='Mapping output directory.')
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    smartgen_root = Path(args.smartgen_root)
    if not smartgen_root.is_absolute():
        smartgen_root = (PROJECT_ROOT / smartgen_root).resolve()
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = (PROJECT_ROOT / output_dir).resolve()

    mapping = load_smartgen_textual_mapping(smartgen_root, args.dataset)
    report = build_smartgen_mapping_report(mapping)
    written = write_smartgen_mapping_files(mapping, output_dir, report=report)
    print(f'Wrote SmartGen textual {args.dataset} mapping files to {output_dir}')
    print(f'mapping_type={report["mapping_type"]}')
    print(f'devices={report["device_count"]} controls={report["control_count"]}')
    print(f'all_key_controls_present={report["all_key_controls_present"]}')
    for path in written.values():
        print(path)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
