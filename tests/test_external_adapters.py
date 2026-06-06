'''Smoke tests for external SmartSense and ARGUS adapters.'''

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / 'src'
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from causal_gen_guard.data.argus_adapter import load_argus_dataset, split_argus_sequences
from causal_gen_guard.data.smartsense_adapter import load_smartsense_logs


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding='utf-8')


def test_smartsense_fake_csv_timestamp_device_action() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        write_text(
            root / 'fr_event_log.csv',
            'timestamp,device,action\n'
            '2024-01-01 08:00:00,front_door,open\n'
            '2024-01-01 08:05:00,front_door,close\n',
        )
        sequences = load_smartsense_logs(root, window_size=10)
        assert len(sequences) == 1
        assert len(sequences[0].events) == 2
        assert sequences[0].events[0].device_id == 'front_door'
        assert sequences[0].events[0].control_id == 'front_door:open'
        assert sequences[0].events[0].timestamp == '2024-01-01 08:00:00'
        assert sequences[0].context['source'] == 'SmartSense'
        assert sequences[0].context['country'] == 'FR'


def test_argus_fake_home_csv_timestamp_device_action() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        write_text(
            root / 'Home1' / 'events.csv',
            'timestamp,device,action\n'
            '2024-02-01 09:00:00,kitchen_light,on\n'
            '2024-02-01 09:01:00,kitchen_light,on\n'
            '2024-02-01 09:03:00,kitchen_light,off\n',
        )
        write_text(
            root / 'Home2' / 'events.csv',
            'timestamp,device,action\n'
            '2024-02-02 10:00:00,door,open\n'
            '2024-02-02 10:02:00,door,close\n',
        )
        sequences = load_argus_dataset(root, window_size=10)
        assert len(sequences) == 2
        home1 = [sequence for sequence in sequences if sequence.context['home_id'] == 'Home1'][0]
        assert len(home1.events) == 2
        assert home1.events[0].control_id == 'kitchen_light:on'
        assert home1.events[1].control_id == 'kitchen_light:off'
        splits = split_argus_sequences(sequences, split='leave_one_home', leave_home_id='Home2')
        assert len(splits['test']) == 1
        assert splits['test'][0].context['home_id'] == 'Home2'


def main() -> None:
    test_smartsense_fake_csv_timestamp_device_action()
    test_argus_fake_home_csv_timestamp_device_action()
    print('test_external_adapters.py: all checks passed')


if __name__ == '__main__':
    main()
