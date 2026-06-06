'''Utilities for loading SmartGuard device/control dictionaries.

The SmartGuard source dictionaries are Python files. This module parses them
with ``ast`` so the source project can stay read-only and no source code needs
to be executed just to recover numeric id mappings.
'''

from __future__ import annotations

import ast
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional


KEY_CONTROLS = [
    'Light:switch on',
    'Light:switch off',
    'Camera:switch on',
    'Camera:switch off',
    'Television:switch on',
    'Television:switch off',
    'SmartLock:lock lock',
    'SmartLock:lock unlock',
    'WaterValve:valve open',
    'WaterValve:valve close',
]


@dataclass(frozen=True)
class SmartGuardDictionary:
    '''Bidirectional SmartGuard id mappings for one dataset.'''

    dataset: str
    source_path: Path
    device_to_id: Dict[str, int]
    id_to_device: Dict[int, str]
    control_to_id: Dict[str, int]
    id_to_control: Dict[int, str]

    def json_payloads(self) -> Dict[str, Dict[str, Any]]:
        '''Return JSON-ready mapping payloads with deterministic ordering.'''
        return {
            'device_to_id': _sort_name_to_id(self.device_to_id),
            'id_to_device': _sort_id_to_name(self.id_to_device),
            'control_to_id': _sort_name_to_id(self.control_to_id),
            'id_to_control': _sort_id_to_name(self.id_to_control),
        }


def dictionary_path(smartguard_root: str | Path, dataset: str) -> Path:
    '''Resolve the SmartGuard dictionary.py path for a dataset.'''
    root = Path(smartguard_root).expanduser().resolve()
    dataset = dataset.lower()
    candidates = [
        root / 'data' / 'data' / dataset / 'dictionary.py',
        root / 'data' / dataset / 'dictionary.py',
        root / dataset / 'dictionary.py',
        root / 'dictionary.py',
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    joined = ', '.join(str(candidate) for candidate in candidates)
    raise FileNotFoundError(f'No SmartGuard dictionary.py found for dataset {dataset!r}; checked {joined}')


def load_smartguard_dictionary(smartguard_root: str | Path, dataset: str) -> SmartGuardDictionary:
    '''Load one SmartGuard dataset dictionary from a SmartGuard project root.'''
    path = dictionary_path(smartguard_root, dataset)
    return parse_smartguard_dictionary(path, dataset=dataset.lower())


def parse_smartguard_dictionary(path: str | Path, dataset: Optional[str] = None) -> SmartGuardDictionary:
    '''Parse device_dict and device_control_dict from a SmartGuard dictionary.py.'''
    source_path = Path(path).expanduser().resolve()
    text = source_path.read_text(encoding='utf-8')
    tree = ast.parse(text, filename=str(source_path))

    device_to_id = _extract_name_to_id(tree, 'device_dict')
    control_to_id = _extract_name_to_id(tree, 'device_control_dict')
    id_to_device = _invert_unique(device_to_id, 'device_dict')
    id_to_control = _invert_unique(control_to_id, 'device_control_dict')

    return SmartGuardDictionary(
        dataset=dataset or source_path.parent.name,
        source_path=source_path,
        device_to_id=device_to_id,
        id_to_device=id_to_device,
        control_to_id=control_to_id,
        id_to_control=id_to_control,
    )


def build_mapping_report(
    mapping: SmartGuardDictionary,
    key_controls: Optional[Iterable[str]] = None,
) -> Dict[str, Any]:
    '''Build a report describing mapping coverage for key semantic controls.'''
    controls = list(key_controls or KEY_CONTROLS)
    key_control_report: Dict[str, Dict[str, Any]] = {}
    missing: List[str] = []
    for control in controls:
        control_id = mapping.control_to_id.get(control)
        exists = control_id is not None
        entry: Dict[str, Any] = {'exists': exists}
        if exists:
            entry['id'] = control_id
        else:
            missing.append(control)
        key_control_report[control] = entry

    return {
        'dataset': mapping.dataset,
        'source_path': str(mapping.source_path),
        'device_count': len(mapping.device_to_id),
        'control_count': len(mapping.control_to_id),
        'id_to_device_count': len(mapping.id_to_device),
        'id_to_control_count': len(mapping.id_to_control),
        'all_key_controls_present': not missing,
        'missing_key_controls': missing,
        'key_controls': key_control_report,
    }


def write_mapping_files(
    mapping: SmartGuardDictionary,
    output_dir: str | Path,
    report: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Path]:
    '''Write mapping JSON files and return their paths.'''
    output_path = Path(output_dir).expanduser().resolve()
    output_path.mkdir(parents=True, exist_ok=True)

    payloads: Dict[str, Mapping[str, Any]] = mapping.json_payloads()
    payloads['mapping_report'] = dict(report or build_mapping_report(mapping))

    written: Dict[str, Path] = {}
    for stem, payload in payloads.items():
        path = output_path / f'{stem}.json'
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
        written[stem] = path
    return written


def _extract_name_to_id(tree: ast.Module, variable_name: str) -> Dict[str, int]:
    value_node: Optional[ast.AST] = None
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == variable_name:
                    value_node = node.value
        elif isinstance(node, ast.AnnAssign):
            target = node.target
            if isinstance(target, ast.Name) and target.id == variable_name:
                value_node = node.value

    if value_node is None:
        raise KeyError(f'{variable_name} was not found in SmartGuard dictionary.py')

    try:
        raw = ast.literal_eval(value_node)
    except (SyntaxError, ValueError) as exc:
        raise ValueError(f'{variable_name} must be a literal dictionary') from exc

    return _validate_name_to_id(raw, variable_name)


def _validate_name_to_id(raw: Any, variable_name: str) -> Dict[str, int]:
    if not isinstance(raw, dict):
        raise TypeError(f'{variable_name} must be a dictionary, got {type(raw).__name__}')

    mapping: Dict[str, int] = {}
    for name, numeric_id in raw.items():
        if not isinstance(name, str):
            raise TypeError(f'{variable_name} contains non-string key {name!r}')
        if isinstance(numeric_id, bool) or not isinstance(numeric_id, int):
            raise TypeError(f'{variable_name}[{name!r}] must be an int id, got {numeric_id!r}')
        mapping[name] = numeric_id
    return mapping


def _invert_unique(name_to_id: Mapping[str, int], variable_name: str) -> Dict[int, str]:
    id_to_name: Dict[int, str] = {}
    duplicates: Dict[int, List[str]] = {}
    for name, numeric_id in name_to_id.items():
        if numeric_id in id_to_name:
            duplicates.setdefault(numeric_id, [id_to_name[numeric_id]]).append(name)
        else:
            id_to_name[numeric_id] = name
    if duplicates:
        details = ', '.join(f'{numeric_id}: {names}' for numeric_id, names in sorted(duplicates.items()))
        raise ValueError(f'{variable_name} contains duplicate numeric ids: {details}')
    return id_to_name


def _sort_name_to_id(mapping: Mapping[str, int]) -> Dict[str, int]:
    return {name: numeric_id for name, numeric_id in sorted(mapping.items(), key=lambda item: (item[1], item[0]))}


def _sort_id_to_name(mapping: Mapping[int, str]) -> Dict[str, str]:
    return {str(numeric_id): name for numeric_id, name in sorted(mapping.items())}
