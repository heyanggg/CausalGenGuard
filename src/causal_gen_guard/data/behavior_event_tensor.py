'''Convert BehaviorSequence objects into GCAD-style behavior event tensors.

The tensors produced here are intended for later causal graph discovery and
shift scoring. This module is deliberately CPU-only and contains no model
training logic.
'''

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import numpy as np

from .schemas import BehaviorSequence


TIME_FEATURE_NAMES = ('normalized_day', 'normalized_hour', 'duration')


class EventVocab(dict):
    '''Mapping from raw control_id values to contiguous channel indices.'''

    def __init__(self, mapping: dict[Any, int] | None = None) -> None:
        super().__init__(mapping or {})
        self.inverse_vocab: list[Any] = [None] * len(self)
        for control_id, index in self.items():
            self.inverse_vocab[index] = control_id


def _as_vocab_key(control_id: Any) -> Any:
    '''Return a hashable vocabulary key for a control identifier.'''
    if hasattr(control_id, 'item'):
        try:
            control_id = control_id.item()
        except Exception:
            pass
    try:
        hash(control_id)
    except TypeError:
        return repr(control_id)
    return control_id


def _control_values(control_id: Any) -> list[Any]:
    '''Normalize a scalar or iterable control id into one or more vocab keys.'''
    if isinstance(control_id, (str, bytes)):
        return [_as_vocab_key(control_id)]
    if isinstance(control_id, dict):
        return [_as_vocab_key(control_id)]
    if isinstance(control_id, Iterable):
        return [_as_vocab_key(item) for item in control_id]
    return [_as_vocab_key(control_id)]


def _as_float(value: Any, default: float = 0.0) -> float:
    '''Convert common scalar values to float with a conservative fallback.'''
    if value is None:
        return default
    if hasattr(value, 'item'):
        try:
            value = value.item()
        except Exception:
            pass
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))


def _time_features_for_event(event: Any) -> list[float]:
    '''Build normalized_day, normalized_hour, duration for one event.'''
    day = _as_float(getattr(event, 'day', None))
    hour = _as_float(getattr(event, 'hour', None))
    duration = _as_float(getattr(event, 'duration', None))
    normalized_day = _clamp01(day / 6.0)
    normalized_hour = _clamp01(hour / 23.0)
    return [normalized_day, normalized_hour, duration]


def build_vocab(sequences: list[BehaviorSequence]) -> EventVocab:
    '''Build a contiguous channel vocabulary from BehaviorSequence controls.

    Raw integer control ids are still remapped to 0..C-1. The first-seen order is
    preserved to avoid brittle sorting across mixed int/string identifiers.
    '''
    mapping: dict[Any, int] = {}
    for sequence in sequences:
        for event in sequence.events:
            for control_id in _control_values(event.control_id):
                if control_id not in mapping:
                    mapping[control_id] = len(mapping)
    return EventVocab(mapping)


def sequence_to_event_tensor(
    sequence: BehaviorSequence,
    vocab: dict[Any, int],
    max_len: int | None = None,
    include_time_features: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    '''Convert one BehaviorSequence to X, time_features, and mask arrays.

    X has shape [T, C]. Each event row is one-hot for scalar control_id values
    and multi-hot when an event carries an iterable of control identifiers.
    '''
    if max_len is None:
        max_len = len(sequence.events)
    if max_len < 0:
        raise ValueError('max_len must be non-negative')

    channel_count = len(vocab)
    X = np.zeros((max_len, channel_count), dtype=np.float32)
    feature_count = len(TIME_FEATURE_NAMES) if include_time_features else 0
    time_features = np.zeros((max_len, feature_count), dtype=np.float32)
    mask = np.zeros((max_len,), dtype=np.float32)

    for row, event in enumerate(sequence.events[:max_len]):
        for control_id in _control_values(event.control_id):
            if control_id not in vocab:
                raise KeyError(f'control_id {control_id!r} is not present in vocab')
            X[row, vocab[control_id]] = 1.0
        if include_time_features:
            time_features[row, :] = np.asarray(_time_features_for_event(event), dtype=np.float32)
        mask[row] = 1.0

    return X, time_features, mask


def batch_to_event_tensor(
    sequences: list[BehaviorSequence],
    vocab: dict[Any, int],
    max_len: int,
) -> dict[str, Any]:
    '''Convert a list of sequences to batch tensors and metadata.'''
    if max_len < 0:
        raise ValueError('max_len must be non-negative')

    batch_size = len(sequences)
    channel_count = len(vocab)
    X = np.zeros((batch_size, max_len, channel_count), dtype=np.float32)
    time_features = np.zeros((batch_size, max_len, len(TIME_FEATURE_NAMES)), dtype=np.float32)
    mask = np.zeros((batch_size, max_len), dtype=np.float32)

    metadata = {
        'sequence_ids': [],
        'labels': [],
        'anomaly_types': [],
    }

    for batch_index, sequence in enumerate(sequences):
        seq_X, seq_time_features, seq_mask = sequence_to_event_tensor(
            sequence,
            vocab,
            max_len=max_len,
            include_time_features=True,
        )
        X[batch_index] = seq_X
        time_features[batch_index] = seq_time_features
        mask[batch_index] = seq_mask
        metadata['sequence_ids'].append(sequence.sequence_id)
        metadata['labels'].append(sequence.label)
        metadata['anomaly_types'].append(sequence.anomaly_type)

    return {
        'X': X,
        'time_features': time_features,
        'mask': mask,
        'metadata': metadata,
    }


def sliding_windows_from_tensor(
    X: np.ndarray,
    window_size: int,
    pred_horizon: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    '''Create causal discovery windows and prediction targets from [T, C].'''
    if window_size <= 0:
        raise ValueError('window_size must be positive')
    if pred_horizon <= 0:
        raise ValueError('pred_horizon must be positive')

    X = np.asarray(X, dtype=np.float32)
    if X.ndim != 2:
        raise ValueError(f'X must have shape [T, C], got {X.shape}')

    time_steps, channel_count = X.shape
    window_count = time_steps - window_size - pred_horizon + 1
    if window_count <= 0:
        return (
            np.zeros((0, window_size, channel_count), dtype=np.float32),
            np.zeros((0, channel_count), dtype=np.float32),
        )

    windows = np.zeros((window_count, window_size, channel_count), dtype=np.float32)
    targets = np.zeros((window_count, channel_count), dtype=np.float32)
    for index in range(window_count):
        windows[index] = X[index : index + window_size]
        target_index = index + window_size + pred_horizon - 1
        targets[index] = X[target_index]

    return windows, targets
