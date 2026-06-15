'''Tests for SmartGuard-style named attacks over canonical controls.'''

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / 'src'
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from causal_gen_guard.data.attack_injector import DICTIONARY_CONTROL_POOL, SOURCE_SEQUENCE_TEMPLATE, inject_smartguard_style
from causal_gen_guard.data.schemas import BehaviorEvent, BehaviorSequence


REQUIRED_NAMED_ATTACKS = [
    'SD_light_flickering',
    'SD_camera_flickering',
    'SD_tv_flickering',
    'MD_camera_off_while_lock',
    'MD_window_open_while_lock',
    'DM_window_open_midnight',
    'DM_watervalve_open_midnight',
    'DD_microwave_long_time',
]


def _canonical_event(canonical_control: str, numeric_id: int, hour: int = 1, duration: float = 1.0) -> BehaviorEvent:
    device, action = canonical_control.split(':', 1)
    return BehaviorEvent(
        day=0,
        hour=hour,
        device_id=device,
        control_id=numeric_id,
        duration=duration,
        raw_fields={
            'raw_control_id': numeric_id,
            'canonical_control': canonical_control,
            'device': device,
            'action': action,
        },
    )


def _full_canonical_sequence() -> BehaviorSequence:
    return BehaviorSequence(
        sequence_id='canonical-normal',
        events=[
            _canonical_event('Light:switch on', 78),
            _canonical_event('Camera:switch on', 37),
            _canonical_event('Television:switch on', 202),
            _canonical_event('SmartLock:lock lock', 170),
            _canonical_event('Camera:switch off', 36),
            _canonical_event('Blind:windowShade open', 28),
            _canonical_event('WaterValve:valve open', 221),
            _canonical_event('Microwave:switch on', 81, duration=2.0),
        ],
        context={'control_encoding': 'canonical'},
        label=0,
    )


def _load_build_script():
    path = PROJECT_ROOT / 'scripts' / 'build_labeled_anomaly_dataset.py'
    spec = importlib.util.spec_from_file_location('build_labeled_anomaly_dataset', path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_required_named_attacks_use_raw_canonical_control() -> None:
    sequence = _full_canonical_sequence()

    for anomaly_type in REQUIRED_NAMED_ATTACKS:
        injected, report = inject_smartguard_style(sequence, anomaly_type)
        assert injected is not None, (anomaly_type, report)
        assert report['status'] == 'injected'
        assert injected.label == 1
        assert injected.anomaly_type == anomaly_type

    flicker, report = inject_smartguard_style(sequence, 'SD_light_flickering')
    assert flicker is not None
    assert report['control_id'] == 'Light:switch on'
    assert any(event.control_id == 'Light:switch on' for event in flicker.events)


def test_numeric_only_sequence_is_skipped_instead_of_fallback() -> None:
    sequence = BehaviorSequence(
        sequence_id='numeric-only',
        events=[
            BehaviorEvent(day=0, hour=1, control_id=78, raw_fields={'raw_control_id': 78}),
        ],
        label=0,
    )

    injected, report = inject_smartguard_style(sequence, 'SD_light_flickering')

    assert injected is None
    assert report['status'] == 'skipped'
    assert report['skipped_reason'] == 'control_not_found:light'


def test_camera_off_while_lock_does_not_use_camera_notification_proxy() -> None:
    sequence = BehaviorSequence(
        sequence_id='camera-notification-only',
        events=[
            _canonical_event('SmartLock:lock lock', 170),
            _canonical_event('Camera:notification', 34),
        ],
        label=0,
    )

    injected, report = inject_smartguard_style(sequence, 'MD_camera_off_while_lock')

    assert injected is None
    assert report['status'] == 'skipped'
    assert report['skipped_reason'] == 'required_control_not_found:lock_or_camera_off'


def test_shower_long_time_skips_when_no_shower_control_exists() -> None:
    injected, report = inject_smartguard_style(_full_canonical_sequence(), 'DD_shower_long_time')

    assert injected is None
    assert report['status'] == 'skipped'
    assert report['skipped_reason'] == 'control_not_found:shower'


def test_dictionary_control_pool_injects_missing_semantic_control() -> None:
    sequence = BehaviorSequence(
        sequence_id='missing-watervalve',
        events=[_canonical_event('Light:switch on', 78)],
        context={'control_encoding': 'canonical'},
        label=0,
    )
    control_to_id = {'WaterValve:valve open': 221}

    injected, report = inject_smartguard_style(
        sequence,
        'DM_watervalve_open_midnight',
        control_pool=control_to_id,
        control_to_id=control_to_id,
    )

    assert injected is not None
    assert report['status'] == 'injected'
    assert report['injection_source'] == DICTIONARY_CONTROL_POOL
    injected_event = injected.events[0]
    assert injected_event.control_id == 'WaterValve:valve open'
    assert injected_event.device_id == 'WaterValve'
    assert injected_event.raw_fields['canonical_control'] == 'WaterValve:valve open'
    assert injected_event.raw_fields['device'] == 'WaterValve'
    assert injected_event.raw_fields['action'] == 'valve open'
    assert injected_event.raw_fields['raw_control_id'] == 221
    assert injected_event.raw_fields['injection_source'] == DICTIONARY_CONTROL_POOL


def test_missing_sequence_and_dictionary_control_is_skipped() -> None:
    sequence = BehaviorSequence(
        sequence_id='missing-watervalve',
        events=[_canonical_event('Light:switch on', 78)],
        context={'control_encoding': 'canonical'},
        label=0,
    )

    injected, report = inject_smartguard_style(
        sequence,
        'DM_watervalve_open_midnight',
        control_pool={'Microwave:switch on': 81},
        control_to_id={'Microwave:switch on': 81},
    )

    assert injected is None
    assert report['status'] == 'skipped'
    assert report['skipped_reason'] == 'control_not_found:watervalve_open'


def test_build_labeled_anomaly_dataset_report_counts(tmp_path: Path) -> None:
    module = _load_build_script()
    input_jsonl = tmp_path / 'normal.jsonl'
    output_jsonl = tmp_path / 'labeled.jsonl'
    report_path = tmp_path / 'report.json'
    input_jsonl.write_text(
        json.dumps(_full_canonical_sequence().to_dict(), ensure_ascii=False) + '\n',
        encoding='utf-8',
    )

    result = module.main(
        [
            '--input-jsonl',
            str(input_jsonl),
            '--output-jsonl',
            str(output_jsonl),
            '--report',
            str(report_path),
            '--anomaly-types',
            'SD_light_flickering',
            'DD_shower_long_time',
            '--per-anomaly-type',
            '1',
            '--seed',
            '7',
        ]
    )

    rows = [json.loads(line) for line in output_jsonl.read_text(encoding='utf-8').splitlines()]
    report = json.loads(report_path.read_text(encoding='utf-8'))

    assert result == 0
    assert sorted(row['label'] for row in rows) == [0, 1]
    assert report['named_injection_success_count'] == 1
    assert report['source_sequence_template_injection_count'] == 1
    assert report['dictionary_control_pool_injection_count'] == 0
    assert report['fallback_numeric_injection_count'] == 0
    assert report['skipped_count'] == 1
    assert report['per_anomaly_type_success']['SD_light_flickering'] == 1
    assert report['per_anomaly_type_success']['DD_shower_long_time'] == 0
    assert report['skipped_reasons']['control_not_found:shower'] == 1


def test_build_labeled_anomaly_dataset_uses_mapping_control_pool(tmp_path: Path) -> None:
    module = _load_build_script()
    input_jsonl = tmp_path / 'normal.jsonl'
    output_jsonl = tmp_path / 'labeled.jsonl'
    report_path = tmp_path / 'report.json'
    mapping_dir = tmp_path / 'mapping'
    mapping_dir.mkdir()
    input_jsonl.write_text(
        json.dumps(
            BehaviorSequence(
                sequence_id='normal-without-microwave',
                events=[_canonical_event('Light:switch on', 78)],
                context={'control_encoding': 'canonical'},
                label=0,
            ).to_dict(),
            ensure_ascii=False,
        )
        + '\n',
        encoding='utf-8',
    )
    (mapping_dir / 'control_to_id.json').write_text(
        json.dumps({'Microwave:switch on': 81}, ensure_ascii=False),
        encoding='utf-8',
    )
    (mapping_dir / 'id_to_control.json').write_text(
        json.dumps({'81': 'Microwave:switch on'}, ensure_ascii=False),
        encoding='utf-8',
    )

    result = module.main(
        [
            '--input-jsonl',
            str(input_jsonl),
            '--output-jsonl',
            str(output_jsonl),
            '--report',
            str(report_path),
            '--mapping-dir',
            str(mapping_dir),
            '--anomaly-types',
            'DD_microwave_long_time',
            '--per-anomaly-type',
            '1',
            '--seed',
            '7',
        ]
    )

    rows = [json.loads(line) for line in output_jsonl.read_text(encoding='utf-8').splitlines()]
    report = json.loads(report_path.read_text(encoding='utf-8'))
    anomaly = next(row for row in rows if row['label'] == 1)
    injected_event = anomaly['events'][0]

    assert result == 0
    assert report['named_injection_success_count'] == 1
    assert report['source_sequence_template_injection_count'] == 0
    assert report['dictionary_control_pool_injection_count'] == 1
    assert report['fallback_numeric_injection_count'] == 0
    assert report['control_pool_size'] == 1
    assert injected_event['control_id'] == 'Microwave:switch on'
    assert injected_event['duration'] == 120.0
    assert injected_event['raw_fields']['raw_control_id'] == 81
    assert injected_event['raw_fields']['injection_source'] == DICTIONARY_CONTROL_POOL
