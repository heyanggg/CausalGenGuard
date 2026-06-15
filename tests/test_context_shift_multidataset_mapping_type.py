'''Tests for context-shift mapping type output labels.'''

from __future__ import annotations

import importlib.util
import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _load_script():
    path = PROJECT_ROOT / 'scripts' / 'run_context_shift_multidataset.py'
    spec = importlib.util.spec_from_file_location('run_context_shift_multidataset', path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _config_for_mapping_dir(mapping_dir: Path) -> dict:
    return {
        'paths': {
            'control_to_id': str(mapping_dir / 'control_to_id.json'),
        }
    }


def test_old_smartguard_mapping_report_outputs_semantic_label(tmp_path: Path) -> None:
    module = _load_script()
    mapping_dir = tmp_path / 'mapping'
    mapping_dir.mkdir()
    (mapping_dir / 'mapping_report.json').write_text(
        json.dumps({'dataset': 'fr', 'control_count': 222}, ensure_ascii=False),
        encoding='utf-8',
    )

    assert module.mapping_type_for_config(_config_for_mapping_dir(mapping_dir)) == 'smartguard_semantic'


def test_explicit_mapping_type_is_preserved(tmp_path: Path) -> None:
    module = _load_script()
    mapping_dir = tmp_path / 'mapping'
    mapping_dir.mkdir()
    (mapping_dir / 'mapping_report.json').write_text(
        json.dumps({'dataset': 'us', 'mapping_type': 'smartgen_textual'}, ensure_ascii=False),
        encoding='utf-8',
    )

    assert module.mapping_type_for_config(_config_for_mapping_dir(mapping_dir)) == 'smartgen_textual'
