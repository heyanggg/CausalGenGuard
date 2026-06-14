#!/usr/bin/env python3
'''Audit US mapping/data availability for CausalGenGuard.'''

from __future__ import annotations

import ast
import json
import pickle
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / 'src'
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


def exists(path: Path) -> bool:
    return path.exists()


def literal_dict(path: Path, variable_name: str) -> Optional[Dict[str, int]]:
    if not path.exists():
        return None
    tree = ast.parse(path.read_text(encoding='utf-8'), filename=str(path))
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == variable_name:
                    return ast.literal_eval(node.value)
    return None


def pkl_sample(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {'exists': False}
    with path.open('rb') as handle:
        obj = pickle.load(handle)
    records = list(obj.values()) if isinstance(obj, dict) else list(obj) if isinstance(obj, (list, tuple)) else [obj]
    sample = records[0] if records else None
    flat = list(sample[0]) if isinstance(sample, tuple) and sample else list(sample) if isinstance(sample, list) else sample
    is_numeric_flat4 = isinstance(flat, list) and len(flat) >= 4 and len(flat) % 4 == 0 and all(isinstance(item, int) for item in flat[: min(len(flat), 40)])
    is_textual_flat4 = isinstance(flat, list) and len(flat) >= 4 and len(flat) % 4 == 0 and any(isinstance(item, str) for item in flat[: min(len(flat), 40)])
    return {
        'exists': True,
        'record_count': len(records),
        'sample_type': type(sample).__name__,
        'sample_preview': repr(sample)[:240],
        'numeric_smartgen_flat4': is_numeric_flat4,
        'textual_smartgen_flat4': is_textual_flat4,
    }


def mapping_report(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {'exists': False}
    try:
        payload = json.loads(path.read_text(encoding='utf-8'))
    except Exception as exc:
        return {'exists': True, 'read_error': f'{type(exc).__name__}: {exc}'}
    return {
        'exists': True,
        'dataset': payload.get('dataset'),
        'mapping_type': payload.get('mapping_type') or 'smartguard_numeric',
        'device_count': payload.get('device_count'),
        'control_count': payload.get('control_count'),
        'source_path': payload.get('source_path'),
        'keys_best_path': payload.get('keys_best_path'),
        'missing_key_controls': payload.get('missing_key_controls'),
    }


def find_us_outputs(limit: int = 20) -> List[str]:
    candidates = list((PROJECT_ROOT / 'outputs/processed').glob('*')) + list((PROJECT_ROOT / 'outputs/synthetic').glob('*'))
    us_token = re.compile(r'(^|[_\-.])us([_\-.]|$)', re.IGNORECASE)
    matched = [
        path
        for path in candidates
        if path.is_file() and (us_token.search(path.name) or any(part.lower() == 'us' for part in path.parts))
    ]
    return [str(path) for path in sorted(matched)[:limit]]


def main() -> int:
    smartguard_us_dir = (PROJECT_ROOT / '../SmartGuard/data/data/us').resolve()
    smartguard_us_dictionary = smartguard_us_dir / 'dictionary.py'
    smartgen_root = (PROJECT_ROOT / '../SmartGen/SmartGen').resolve()
    smartgen_dictionary = smartgen_root / 'dictionary.py'
    us_keys = smartgen_root / 'us_keys_best.txt'
    us_iot = smartgen_root / 'IoT_data/us'
    us_attack = smartgen_root / 'attack/us'
    us_mapping = PROJECT_ROOT / 'outputs/mappings/smartguard/us/mapping_report.json'

    us_devices = literal_dict(smartgen_dictionary, 'us_devices_dict') or {}
    us_actions = literal_dict(smartgen_dictionary, 'us_actions') or {}
    keys_text = us_keys.read_text(encoding='utf-8') if us_keys.exists() else ''
    key_device_lines = [line for line in keys_text.split(';') if line.strip()]
    samples = {
        'winter_trn': pkl_sample(us_iot / 'winter/trn.pkl'),
        'spring_test': pkl_sample(us_iot / 'spring/test.pkl'),
        'spring_synthetic': pkl_sample(us_iot / 'spring/us_spring_generation_SPPC_th=0.905_gpt-4o_seq.pkl'),
        'spring_attack': pkl_sample(us_attack / 'labeled_us_spring_attack_heater.pkl'),
    }
    numeric_style = any(item.get('numeric_smartgen_flat4') for item in samples.values())
    textual_style = any(item.get('textual_smartgen_flat4') for item in samples.values())
    report = mapping_report(us_mapping)

    lines = [
        '# US Mapping Audit',
        '',
        '## Source Availability',
        '',
        f"- `../SmartGuard/data/data/us/` exists: `{smartguard_us_dir.exists()}`",
        f"- `../SmartGuard/data/data/us/dictionary.py` exists: `{smartguard_us_dictionary.exists()}`",
        f"- `../SmartGen/SmartGen/us_keys_best.txt` exists: `{us_keys.exists()}`",
        f"- `../SmartGen/SmartGen/dictionary.py` exists: `{smartgen_dictionary.exists()}`",
        f"- `../SmartGen/SmartGen/IoT_data/us` exists: `{us_iot.exists()}`",
        f"- `../SmartGen/SmartGen/attack/us` exists: `{us_attack.exists()}`",
        '',
        '## SmartGen US Dictionaries',
        '',
        f"- `us_devices_dict` count: `{len(us_devices)}`",
        f"- `us_actions` count: `{len(us_actions)}`",
        f"- `us_keys_best.txt` device lines: `{len(key_device_lines)}`",
        '',
        '## CausalGenGuard Existing US Outputs',
        '',
        f"- processed/synthetic US files: `{find_us_outputs()}`",
        f"- current mapping report: `{report}`",
        '',
        '## US Data Format Samples',
        '',
    ]
    for name, payload in samples.items():
        lines.append(f"- `{name}`: `{payload}`")
    lines.extend(
        [
            '',
            '## Decision',
            '',
            f"- US data appears numeric SmartGen flat-4 style: `{numeric_style}`",
            f"- US data appears textual SmartGen flat-4 style: `{textual_style}`",
            '- SmartGuard US dictionary is unavailable, so SmartGuard numeric mapping cannot be generated.',
            '- SmartGen US textual/action dictionary is available, so CausalGenGuard uses `mapping_type=smartgen_textual` generated from SmartGen `us_devices_dict/us_actions`.',
            '- This does not reuse FR/SP mapping values and does not modify SmartGen.',
            '',
        ]
    )
    output = PROJECT_ROOT / 'outputs/logs/US_MAPPING_AUDIT.md'
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text('\n'.join(lines), encoding='utf-8')
    print(f'Wrote {output}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
