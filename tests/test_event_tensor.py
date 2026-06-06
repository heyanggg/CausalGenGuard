'''Tests for behavior-event tensor conversion.'''

from __future__ import annotations

import sys
from pathlib import Path

try:
    import numpy as np
except ModuleNotFoundError:
    np = None
    try:
        import pytest
    except ModuleNotFoundError:
        pytest = None
    if pytest is not None:
        pytest.skip('numpy is required for event tensor tests', allow_module_level=True)
else:
    pytest = None

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / 'src'
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from causal_gen_guard.data.schemas import BehaviorEvent, BehaviorSequence


def _require_numpy() -> None:
    if np is None:
        raise RuntimeError('numpy is required for event tensor tests')


def _toy_sequences() -> list[BehaviorSequence]:
    return [
        BehaviorSequence(
            sequence_id='seq-a',
            events=[
                BehaviorEvent(day=0, hour=0, control_id='light_on', duration=1.0),
                BehaviorEvent(day=0, hour=1, control_id='door_lock', duration=2.0),
                BehaviorEvent(day=0, hour=2, control_id='light_on', duration=3.0),
            ],
            context={'home': 'toy-a'},
            label=0,
        ),
        BehaviorSequence(
            sequence_id='seq-b',
            events=[
                BehaviorEvent(day=1, hour=3, control_id=7, duration=4.0),
                BehaviorEvent(day=1, hour=4, control_id='door_lock', duration=5.0),
            ],
            context={'home': 'toy-b'},
            label=1,
            anomaly_type='toy_attack',
        ),
    ]


def test_build_vocab_remaps_controls_to_contiguous_channels() -> None:
    _require_numpy()
    from causal_gen_guard.data.behavior_event_tensor import build_vocab

    vocab = build_vocab(_toy_sequences())

    assert dict(vocab) == {'light_on': 0, 'door_lock': 1, 7: 2}
    assert vocab.inverse_vocab == ['light_on', 'door_lock', 7]


def test_sequence_to_event_tensor_shape_and_one_hot() -> None:
    _require_numpy()
    from causal_gen_guard.data.behavior_event_tensor import build_vocab, sequence_to_event_tensor

    sequences = _toy_sequences()
    vocab = build_vocab(sequences)
    X, time_features, mask = sequence_to_event_tensor(sequences[0], vocab, max_len=4)

    assert X.shape == (4, 3)
    assert time_features.shape == (4, 3)
    assert mask.shape == (4,)
    np.testing.assert_array_equal(X[0], np.asarray([1.0, 0.0, 0.0], dtype=np.float32))
    np.testing.assert_array_equal(X[1], np.asarray([0.0, 1.0, 0.0], dtype=np.float32))
    np.testing.assert_array_equal(X[2], np.asarray([1.0, 0.0, 0.0], dtype=np.float32))
    np.testing.assert_array_equal(X[3], np.asarray([0.0, 0.0, 0.0], dtype=np.float32))
    np.testing.assert_array_equal(mask, np.asarray([1.0, 1.0, 1.0, 0.0], dtype=np.float32))
    assert time_features[1, 1] == np.float32(1.0 / 23.0)
    assert time_features[2, 2] == np.float32(3.0)


def test_batch_to_event_tensor_shapes_and_metadata() -> None:
    _require_numpy()
    from causal_gen_guard.data.behavior_event_tensor import batch_to_event_tensor, build_vocab

    sequences = _toy_sequences()
    vocab = build_vocab(sequences)
    batch = batch_to_event_tensor(sequences, vocab, max_len=4)

    assert batch['X'].shape == (2, 4, 3)
    assert batch['time_features'].shape == (2, 4, 3)
    assert batch['mask'].shape == (2, 4)
    assert batch['metadata']['sequence_ids'] == ['seq-a', 'seq-b']
    assert batch['metadata']['labels'] == [0, 1]
    assert batch['metadata']['anomaly_types'] == [None, 'toy_attack']


def test_sliding_windows_shape_and_targets() -> None:
    _require_numpy()
    from causal_gen_guard.data.behavior_event_tensor import (
        build_vocab,
        sequence_to_event_tensor,
        sliding_windows_from_tensor,
    )

    sequences = _toy_sequences()
    vocab = build_vocab(sequences)
    X, _, _ = sequence_to_event_tensor(sequences[0], vocab)
    windows, targets = sliding_windows_from_tensor(X, window_size=2, pred_horizon=1)

    assert windows.shape == (1, 2, 3)
    assert targets.shape == (1, 3)
    np.testing.assert_array_equal(windows[0, 0], np.asarray([1.0, 0.0, 0.0], dtype=np.float32))
    np.testing.assert_array_equal(windows[0, 1], np.asarray([0.0, 1.0, 0.0], dtype=np.float32))
    np.testing.assert_array_equal(targets[0], np.asarray([1.0, 0.0, 0.0], dtype=np.float32))


def main() -> None:
    if np is None:
        print('test_event_tensor.py: skipped because numpy is not installed')
        return
    test_build_vocab_remaps_controls_to_contiguous_channels()
    test_sequence_to_event_tensor_shape_and_one_hot()
    test_batch_to_event_tensor_shapes_and_metadata()
    test_sliding_windows_shape_and_targets()
    print('test_event_tensor.py: all checks passed')


if __name__ == '__main__':
    main()
