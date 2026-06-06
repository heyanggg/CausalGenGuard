'''Tests for normalized behavior-event schemas and SmartGuard sample parsing.'''

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / 'src'
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from causal_gen_guard.data.schemas import BehaviorEvent, BehaviorSequence
from causal_gen_guard.data.smartguard_adapter import parse_smartguard_sample
from causal_gen_guard.data.smartguard_dictionary import parse_smartguard_dictionary


def test_behavior_event_and_sequence_roundtrip() -> None:
    event = BehaviorEvent(day=1, hour=8, device_id='kitchen', control_id=3, duration=5, raw_fields={'source': 'unit'})
    sequence = BehaviorSequence(
        sequence_id='seq-1',
        events=[event],
        context={'dataset': 'unit'},
        label=0,
        anomaly_type=None,
    )

    payload = sequence.to_dict()
    json.dumps(payload)
    restored = BehaviorSequence.from_dict(payload)

    assert restored.sequence_id == 'seq-1'
    assert len(restored.events) == 1
    assert restored.events[0].control_id == 3
    assert restored.events[0].raw_fields['source'] == 'unit'


def test_parse_smartguard_sample_flat_length_40() -> None:
    values = list(range(40))
    try:
        import numpy as np
    except ModuleNotFoundError:
        sample = values
    else:
        sample = np.asarray(values)

    sequence = parse_smartguard_sample(sample, vocab_size=128)

    assert sequence.sequence_id == 'smartguard_sample'
    assert len(sequence.events) == 10
    assert sequence.events[0].day == 0
    assert sequence.events[0].hour == 1
    assert sequence.events[0].duration == 2
    assert sequence.events[0].control_id == 3
    assert sequence.events[-1].control_id == 39
    assert sequence.context['vocab_size'] == 128


def test_parse_smartguard_sample_nested_10_by_4() -> None:
    sample = [[row * 4 + col for col in range(4)] for row in range(10)]
    sequence = parse_smartguard_sample(sample)

    assert len(sequence.events) == 10
    assert sequence.events[3].day == 12
    assert sequence.events[3].hour == 13
    assert sequence.events[3].duration == 14
    assert sequence.events[3].control_id == 15


def test_parse_smartguard_sample_emits_canonical_fields(tmp_path: Path) -> None:
    dictionary_path = tmp_path / 'dictionary.py'
    dictionary_path.write_text(
        "device_dict={'Light':13};"
        "device_control_dict={'Light:switch on':78}",
        encoding='utf-8',
    )
    mapping = parse_smartguard_dictionary(dictionary_path, dataset='unit')
    sample = []
    for index in range(10):
        if index == 0:
            sample.extend([0, 0, 13, 78])
        else:
            sample.extend([9, 9, 999, 999])

    sequence = parse_smartguard_sample(sample, smartguard_mapping=mapping, emit_canonical_control=True)
    event = sequence.events[0]
    unknown_event = sequence.events[1]

    assert sequence.context['control_encoding'] == 'canonical'
    assert event.device_id == 'Light'
    assert event.control_id == 'Light:switch on'
    assert event.raw_fields['raw_device_id'] == 13
    assert event.raw_fields['raw_control_id'] == 78
    assert event.raw_fields['device'] == 'Light'
    assert event.raw_fields['canonical_control'] == 'Light:switch on'
    assert event.raw_fields['action'] == 'switch on'
    assert event.raw_fields['day_name'] == 'day:Mon'
    assert event.raw_fields['hour_name'] == 'time:(0~3)'
    assert unknown_event.device_id == 'unknown'
    assert unknown_event.control_id == 'unknown'
    assert unknown_event.raw_fields['device'] == 'unknown'
    assert unknown_event.raw_fields['canonical_control'] == 'unknown'
    assert unknown_event.raw_fields['action'] == 'unknown'
    assert unknown_event.raw_fields['day_name'] == 'unknown'
    assert unknown_event.raw_fields['hour_name'] == 'unknown'


def main() -> None:
    test_behavior_event_and_sequence_roundtrip()
    test_parse_smartguard_sample_flat_length_40()
    test_parse_smartguard_sample_nested_10_by_4()
    with tempfile.TemporaryDirectory() as tmp:
        test_parse_smartguard_sample_emits_canonical_fields(Path(tmp))
    print('test_data_schema.py: all checks passed')


if __name__ == '__main__':
    main()
