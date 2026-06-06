'''Tests for SmartGuard dictionary parsing and mapping output.'''

from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / 'src'
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from causal_gen_guard.data.smartguard_dictionary import (
    KEY_CONTROLS,
    build_mapping_report,
    load_smartguard_dictionary,
    parse_smartguard_dictionary,
)


def _smartguard_root() -> Path:
    root = PROJECT_ROOT.parent / 'SmartGuard'
    if not root.exists():
        pytest.skip('SmartGuard source project is not available')
    return root


def _load_build_script():
    path = PROJECT_ROOT / 'scripts' / 'build_smartguard_mapping.py'
    spec = importlib.util.spec_from_file_location('build_smartguard_mapping', path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_parse_dictionary_file_handles_single_line_literals() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / 'dictionary.py'
        path.write_text(
            "device_dict={'Light':13,'Camera':3};"
            "device_control_dict={'Light:switch on':78,'Camera:switch off':36}",
            encoding='utf-8',
        )

        mapping = parse_smartguard_dictionary(path, dataset='unit')
        report = build_mapping_report(mapping, key_controls=['Light:switch on', 'Camera:switch off'])

    assert mapping.device_to_id == {'Light': 13, 'Camera': 3}
    assert mapping.id_to_device[13] == 'Light'
    assert mapping.control_to_id['Light:switch on'] == 78
    assert mapping.id_to_control[36] == 'Camera:switch off'
    assert report['all_key_controls_present'] is True


def test_load_real_smartguard_fr_dictionary_has_required_key_controls() -> None:
    mapping = load_smartguard_dictionary(_smartguard_root(), 'fr')
    report = build_mapping_report(mapping)

    assert len(mapping.device_to_id) == len(mapping.id_to_device)
    assert len(mapping.control_to_id) == len(mapping.id_to_control)
    assert report['all_key_controls_present'] is True
    assert report['missing_key_controls'] == []
    for control in KEY_CONTROLS:
        control_id = mapping.control_to_id[control]
        assert mapping.id_to_control[control_id] == control


def test_load_real_smartguard_sp_and_an_dictionaries() -> None:
    root = _smartguard_root()
    sp_mapping = load_smartguard_dictionary(root, 'sp')
    an_mapping = load_smartguard_dictionary(root, 'an')
    sp_report = build_mapping_report(sp_mapping)
    an_report = build_mapping_report(an_mapping)

    assert sp_report['all_key_controls_present'] is True
    assert sp_report['missing_key_controls'] == []
    assert an_mapping.device_to_id
    assert an_mapping.control_to_id
    assert set(an_report['missing_key_controls']) == set(KEY_CONTROLS)


def test_build_smartguard_mapping_script_writes_expected_json_files() -> None:
    module = _load_build_script()
    with tempfile.TemporaryDirectory() as tmp:
        output_dir = Path(tmp) / 'mappings'
        result = module.main(
            [
                '--smartguard-root',
                str(_smartguard_root()),
                '--dataset',
                'fr',
                '--output-dir',
                str(output_dir),
            ]
        )

        assert result == 0
        expected = {
            'device_to_id.json',
            'id_to_device.json',
            'control_to_id.json',
            'id_to_control.json',
            'mapping_report.json',
        }
        assert {path.name for path in output_dir.iterdir()} == expected

        control_to_id = json.loads((output_dir / 'control_to_id.json').read_text(encoding='utf-8'))
        id_to_control = json.loads((output_dir / 'id_to_control.json').read_text(encoding='utf-8'))
        report = json.loads((output_dir / 'mapping_report.json').read_text(encoding='utf-8'))

    control_id = control_to_id['Light:switch on']
    assert id_to_control[str(control_id)] == 'Light:switch on'
    assert report['all_key_controls_present'] is True
    assert report['key_controls']['SmartLock:lock unlock']['exists'] is True
