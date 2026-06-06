'''Tests for behavior attack injection utilities.'''

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / 'src'
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from causal_gen_guard.data.attack_injector import generate_anomaly_dataset, inject_causal_anomaly
from causal_gen_guard.data.schemas import BehaviorEvent, BehaviorSequence


def _toy_sequence() -> BehaviorSequence:
    return BehaviorSequence(
        sequence_id='toy-normal',
        events=[
            BehaviorEvent(day=0, hour=0, device_id='door', control_id='door_lock'),
            BehaviorEvent(day=0, hour=1, device_id='light', control_id='light_on'),
            BehaviorEvent(day=0, hour=2, device_id='camera', control_id='camera_on'),
        ],
        context={'inverse_vocab': ['door_lock', 'light_on', 'camera_on']},
        label=0,
    )


def test_causal_edge_injection_toy_vocab() -> None:
    injected, report = inject_causal_anomaly(_toy_sequence(), 'causal_edge_injection')

    assert injected is not None
    assert report['status'] == 'injected'
    assert injected.label == 1
    assert injected.anomaly_type == 'causal_edge_injection'
    assert len(injected.events) == 4
    assert any(event.raw_fields.get('causal_attack') == 'edge_injection' for event in injected.events)


def test_generate_anomaly_dataset_sets_labels_and_report() -> None:
    mixed, report = generate_anomaly_dataset(
        [_toy_sequence()],
        [{'family': 'causal', 'type': 'causal_edge_injection'}],
        ratio=1.0,
        seed=7,
    )

    assert report['injected_count'] == 1
    assert report['mixed_count'] == 2
    assert sorted(sequence.label for sequence in mixed) == [0, 1]
    assert any(sequence.anomaly_type == 'causal_edge_injection' for sequence in mixed)


def main() -> None:
    test_causal_edge_injection_toy_vocab()
    test_generate_anomaly_dataset_sets_labels_and_report()
    print('test_attack_injector.py: all checks passed')


if __name__ == '__main__':
    main()
