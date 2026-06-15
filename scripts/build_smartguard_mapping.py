#!/usr/bin/env python
"""Build SmartGuard semantic mapping JSON files."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from causal_gen_guard.data.smartguard_dictionary import build_mapping_report, load_smartguard_dictionary


def _write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smartguard-root", required=True)
    parser.add_argument("--dataset", required=True, choices=["fr", "sp", "an"])
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--include-time-mappings", action="store_true")
    return parser


def main(argv=None) -> int:
    args = build_arg_parser().parse_args(argv)
    mapping = load_smartguard_dictionary(args.smartguard_root, args.dataset)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    payloads = mapping.json_payloads()
    file_names = ["device_to_id", "id_to_device", "control_to_id", "id_to_control"]
    if args.include_time_mappings:
        file_names.extend(["day_to_id", "id_to_day", "hour_to_id", "id_to_hour"])
    for name in file_names:
        _write_json(output_dir / f"{name}.json", payloads[name])

    report = build_mapping_report(mapping)
    _write_json(output_dir / "mapping_report.json", report)

    print(json.dumps({
        "status": "ok",
        "dataset": args.dataset,
        "dictionary_path": str(mapping.dictionary_path),
        "output_dir": str(output_dir),
        "device_count": report["device_count"],
        "control_count": report["control_count"],
        "day_count": report["day_count"],
        "hour_count": report["hour_count"],
        "missing_key_controls": report["missing_key_controls"],
        "all_key_controls_present": report["all_key_controls_present"],
        "can_run_named_smartguard_attacks": report["can_run_named_smartguard_attacks"],
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
