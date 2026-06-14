'''Build US mappings from SmartGen textual/action dictionaries.

SmartGuard does not provide a US dictionary in this workspace. SmartGen does
provide US textual controls and the numeric ids used by its generated pkl files
in ``SmartGen/dictionary.py``. This adapter reads those dictionaries without
executing SmartGen code and writes CausalGenGuard-compatible mapping JSON files.
'''

from __future__ import annotations

import ast
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional

from causal_gen_guard.data.smartguard_dictionary import KEY_CONTROLS


@dataclass(frozen=True)
class SmartGenTextualMapping:
    '''Bidirectional textual mapping extracted from SmartGen dictionaries.'''

    dataset: str
    mapping_type: str
    source_path: Path
    keys_best_path: Optional[Path]
    device_to_id: Dict[str, int]
    id_to_device: Dict[int, str]
    control_to_id: Dict[str, int]
    id_to_control: Dict[int, str]

    def json_payloads(self) -> Dict[str, Dict[str, Any]]:
        return {
            'device_to_id': _sort_name_to_id(self.device_to_id),
            'id_to_device': _sort_id_to_name(self.id_to_device),
            'control_to_id': _sort_name_to_id(self.control_to_id),
            'id_to_control': _sort_id_to_name(self.id_to_control),
        }


def smartgen_dictionary_path(smartgen_root: str | Path) -> Path:
    root = Path(smartgen_root).expanduser().resolve()
    candidates = [root / 'dictionary.py', root / 'SmartGen' / 'dictionary.py']
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError('No SmartGen dictionary.py found; checked {}'.format(', '.join(str(item) for item in candidates)))


def smartgen_keys_best_path(smartgen_root: str | Path, dataset: str) -> Optional[Path]:
    root = Path(smartgen_root).expanduser().resolve()
    candidates = [root / f'{dataset}_keys_best.txt', root / 'SmartGen' / f'{dataset}_keys_best.txt']
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def load_smartgen_textual_mapping(smartgen_root: str | Path, dataset: str = 'us') -> SmartGenTextualMapping:
    dataset = dataset.lower()
    dictionary = smartgen_dictionary_path(smartgen_root)
    tree = ast.parse(dictionary.read_text(encoding='utf-8'), filename=str(dictionary))
    device_to_id = _extract_literal_dict(tree, f'{dataset}_devices_dict')
    control_to_id = _extract_literal_dict(tree, f'{dataset}_actions')
    id_to_device = _invert_unique(device_to_id, f'{dataset}_devices_dict')
    id_to_control = _invert_unique(control_to_id, f'{dataset}_actions')
    return SmartGenTextualMapping(
        dataset=dataset,
        mapping_type='smartgen_textual',
        source_path=dictionary,
        keys_best_path=smartgen_keys_best_path(smartgen_root, dataset),
        device_to_id=device_to_id,
        id_to_device=id_to_device,
        control_to_id=control_to_id,
        id_to_control=id_to_control,
    )


def build_smartgen_mapping_report(
    mapping: SmartGenTextualMapping,
    key_controls: Optional[Iterable[str]] = None,
) -> Dict[str, Any]:
    controls = list(key_controls or KEY_CONTROLS)
    missing = []
    key_control_report: Dict[str, Dict[str, Any]] = {}
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
        'mapping_type': mapping.mapping_type,
        'source_project': 'SmartGen',
        'source_path': str(mapping.source_path),
        'keys_best_path': str(mapping.keys_best_path) if mapping.keys_best_path else None,
        'device_count': len(mapping.device_to_id),
        'control_count': len(mapping.control_to_id),
        'id_to_device_count': len(mapping.id_to_device),
        'id_to_control_count': len(mapping.id_to_control),
        'all_key_controls_present': not missing,
        'missing_key_controls': missing,
        'key_controls': key_control_report,
        'notes': [
            'SmartGuard US dictionary.py was not available in this workspace.',
            'This mapping is generated from SmartGen textual/action dictionaries and is not a SmartGuard numeric dictionary.',
            'No FR/SP mapping values are reused for US.',
        ],
    }


def write_smartgen_mapping_files(
    mapping: SmartGenTextualMapping,
    output_dir: str | Path,
    report: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Path]:
    output_path = Path(output_dir).expanduser().resolve()
    output_path.mkdir(parents=True, exist_ok=True)
    payloads: Dict[str, Mapping[str, Any]] = mapping.json_payloads()
    payloads['mapping_report'] = dict(report or build_smartgen_mapping_report(mapping))
    written: Dict[str, Path] = {}
    for stem, payload in payloads.items():
        path = output_path / f'{stem}.json'
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
        written[stem] = path
    return written


def _extract_literal_dict(tree: ast.Module, variable_name: str) -> Dict[str, int]:
    value_node = None
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == variable_name:
                    value_node = node.value
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name) and node.target.id == variable_name:
            value_node = node.value
    if value_node is None:
        raise KeyError(f'{variable_name} was not found in SmartGen dictionary.py')
    raw = ast.literal_eval(value_node)
    return _validate_name_to_id(raw, variable_name)


def _validate_name_to_id(raw: Any, variable_name: str) -> Dict[str, int]:
    if not isinstance(raw, dict):
        raise TypeError(f'{variable_name} must be a dictionary, got {type(raw).__name__}')
    parsed: Dict[str, int] = {}
    for name, numeric_id in raw.items():
        if not isinstance(name, str):
            raise TypeError(f'{variable_name} contains non-string key {name!r}')
        if isinstance(numeric_id, bool) or not isinstance(numeric_id, int):
            raise TypeError(f'{variable_name}[{name!r}] must be an int id, got {numeric_id!r}')
        parsed[name] = numeric_id
    return parsed


def _invert_unique(name_to_id: Mapping[str, int], variable_name: str) -> Dict[int, str]:
    id_to_name: Dict[int, str] = {}
    duplicates: Dict[int, list[str]] = {}
    for name, numeric_id in name_to_id.items():
        if numeric_id in id_to_name:
            duplicates.setdefault(numeric_id, [id_to_name[numeric_id]]).append(name)
        else:
            id_to_name[numeric_id] = name
    if duplicates:
        raise ValueError(f'{variable_name} contains duplicate numeric ids: {duplicates}')
    return id_to_name


def _sort_name_to_id(mapping: Mapping[str, int]) -> Dict[str, int]:
    return {name: numeric_id for name, numeric_id in sorted(mapping.items(), key=lambda item: (item[1], item[0]))}


def _sort_id_to_name(mapping: Mapping[int, str]) -> Dict[str, str]:
    return {str(numeric_id): name for numeric_id, name in sorted(mapping.items())}
