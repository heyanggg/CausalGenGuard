'''Adapter for reading SmartGuard FR/SP behavior-anomaly data.

Assumptions from the audit and user guidance:

- Standard SmartGuard samples are compatible with a 10 x 4 layout.
- Columns are interpreted as day, hour, device, and control.
- The fourth column is the normalized control_id. When a SmartGuard dictionary
  is supplied, it can also be emitted as a canonical ``Device:action`` control.
- The third column is temporarily exposed as duration and preserved as
  persistence_or_device in raw_fields until behavior_event_tensor.py owns the
  final duration and device encoding.

The adapter is read-only with respect to the SmartGuard source project.
'''

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any, Iterable, Iterator

from .schemas import BehaviorEvent, BehaviorSequence
from .smartguard_dictionary import SmartGuardDictionary


SMARTGUARD_SEQUENCE_LENGTH = 10
SMARTGUARD_FIELD_COUNT = 4
UNKNOWN = 'unknown'

DAY_NAMES = {
    0: 'day:Mon',
    1: 'day:Tue',
    2: 'day:Wed',
    3: 'day:Thu',
    4: 'day:Fri',
    5: 'day:Sat',
    6: 'day:Sun',
}

HOUR_NAMES = {
    0: 'time:(0~3)',
    1: 'time:(3~6)',
    2: 'time:(6~9)',
    3: 'time:(9~12)',
    4: 'time:(12~15)',
    5: 'time:(15~18)',
    6: 'time:(18~21)',
    7: 'time:(21~24)',
}


def load_smartguard_array(path: str | Path) -> Any:
    '''Load a SmartGuard pickle file and return its raw object.'''
    path = Path(path)
    try:
        with path.open('rb') as handle:
            return pickle.load(handle)
    except ModuleNotFoundError as exc:
        if exc.name == 'numpy':
            raise RuntimeError(
                'Loading this SmartGuard pickle requires numpy. Install project '
                'requirements before preparing SmartGuard data.'
            ) from exc
        raise


def _scalar(value: Any) -> Any:
    if hasattr(value, 'item'):
        try:
            return value.item()
        except Exception:
            pass
    return value


def _flatten_sample(sample: Any) -> list[Any]:
    '''Flatten one SmartGuard sample without importing numpy.'''
    if isinstance(sample, dict):
        for key in ('sample', 'sequence', 'x', 'data', 'events'):
            if key in sample:
                return _flatten_sample(sample[key])
        raise ValueError(f'Unsupported SmartGuard sample dictionary keys: {sorted(sample)}')

    if hasattr(sample, 'reshape') and hasattr(sample, 'tolist'):
        try:
            return list(sample.reshape(-1).tolist())
        except Exception:
            pass

    if hasattr(sample, 'tolist'):
        sample = sample.tolist()

    if isinstance(sample, (list, tuple)):
        if len(sample) == SMARTGUARD_SEQUENCE_LENGTH and all(
            isinstance(row, (list, tuple)) and len(row) == SMARTGUARD_FIELD_COUNT for row in sample
        ):
            return [value for row in sample for value in row]
        return list(sample)

    raise TypeError(f'Unsupported SmartGuard sample type: {type(sample).__name__}')


def _extract_sample_and_label(item: Any) -> tuple[Any, Any | None, str | None]:
    '''Best-effort handling for labeled wrappers while preserving raw samples.'''
    if isinstance(item, dict):
        label = item.get('label', item.get('y'))
        anomaly_type = item.get('anomaly_type', item.get('attack_type'))
        for key in ('sample', 'sequence', 'x', 'data', 'events'):
            if key in item:
                return item[key], label, anomaly_type
        return item, label, anomaly_type

    if isinstance(item, tuple) and len(item) == 2:
        sample, label = item
        return sample, label, None

    return item, None, None


def _int_key(value: Any) -> int | None:
    value = _scalar(value)
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    if isinstance(value, str):
        try:
            return int(value)
        except ValueError:
            return None
    return None


def _lookup(mapping: dict[int, str], value: Any) -> str:
    key = _int_key(value)
    if key is None:
        return UNKNOWN
    return mapping.get(key, UNKNOWN)


def _action_from_control(canonical_control: str) -> str:
    if canonical_control == UNKNOWN or ':' not in canonical_control:
        return UNKNOWN
    return canonical_control.split(':', 1)[1]


def _canonical_fields(
    day: Any,
    hour: Any,
    raw_device_id: Any,
    raw_control_id: Any,
    smartguard_mapping: SmartGuardDictionary | None,
) -> dict[str, Any]:
    id_to_device = smartguard_mapping.id_to_device if smartguard_mapping is not None else {}
    id_to_control = smartguard_mapping.id_to_control if smartguard_mapping is not None else {}
    canonical_control = _lookup(id_to_control, raw_control_id)
    return {
        'raw_device_id': raw_device_id,
        'raw_control_id': raw_control_id,
        'device': _lookup(id_to_device, raw_device_id),
        'canonical_control': canonical_control,
        'action': _action_from_control(canonical_control),
        'day_name': _lookup(DAY_NAMES, day),
        'hour_name': _lookup(HOUR_NAMES, hour),
    }


def parse_smartguard_sample(
    sample: Any,
    vocab_size: int | None = None,
    smartguard_mapping: SmartGuardDictionary | None = None,
    emit_canonical_control: bool = False,
) -> BehaviorSequence:
    '''Parse one SmartGuard 10 x 4 sample into a BehaviorSequence.'''
    raw_sample, label, anomaly_type = _extract_sample_and_label(sample)
    flat = [_scalar(value) for value in _flatten_sample(raw_sample)]
    expected = SMARTGUARD_SEQUENCE_LENGTH * SMARTGUARD_FIELD_COUNT
    if len(flat) != expected:
        raise ValueError(f'Expected SmartGuard sample with {expected} values, got {len(flat)}')

    events: list[BehaviorEvent] = []
    for index in range(SMARTGUARD_SEQUENCE_LENGTH):
        offset = index * SMARTGUARD_FIELD_COUNT
        day, hour, raw_device_id, raw_control_id = flat[offset : offset + SMARTGUARD_FIELD_COUNT]
        raw_fields = {
            'position': index,
            'day': day,
            'hour': hour,
            'persistence_or_device': raw_device_id,
            'control': raw_control_id,
            'source_format': 'smartguard_10x4',
        }
        device_id = None
        control_id = raw_control_id
        if emit_canonical_control:
            canonical = _canonical_fields(day, hour, raw_device_id, raw_control_id, smartguard_mapping)
            raw_fields.update(canonical)
            device_id = canonical['device']
            control_id = canonical['canonical_control']
        events.append(
            BehaviorEvent(
                day=day,
                hour=hour,
                device_id=device_id,
                control_id=control_id,
                duration=raw_device_id,
                raw_fields=raw_fields,
            )
        )

    context = {
        'source_project': 'SmartGuard',
        'source_format': '10x4',
        'field_order': ['day', 'hour', 'persistence_or_device', 'control'],
    }
    if vocab_size is not None:
        context['vocab_size'] = vocab_size
    if emit_canonical_control:
        context['control_encoding'] = 'canonical'
        if smartguard_mapping is not None:
            context['smartguard_dictionary'] = str(smartguard_mapping.source_path)

    return BehaviorSequence(
        sequence_id='smartguard_sample',
        events=events,
        context=context,
        label=label,
        anomaly_type=anomaly_type,
    )


def _candidate_split_paths(root: Path, dataset_name: str) -> list[tuple[str, Path]]:
    aliases = {
        'train': [
            root / 'data' / f'{dataset_name}_data' / f'{dataset_name}_trn_instance_10.pkl',
            root / 'data' / 'data' / dataset_name / 'trn_instance_10.pkl',
        ],
        'valid': [
            root / 'data' / f'{dataset_name}_data' / f'{dataset_name}_vld_instance_10.pkl',
            root / 'data' / 'data' / dataset_name / 'vld_instance_10.pkl',
        ],
        'test': [
            root / 'data' / f'{dataset_name}_data' / f'{dataset_name}_test_instance_10.pkl',
            root / 'data' / 'data' / dataset_name / 'test_instance_10.pkl',
        ],
    }
    paths: list[tuple[str, Path]] = []
    for split, candidates in aliases.items():
        for candidate in candidates:
            if candidate.exists():
                paths.append((split, candidate))
                break
    return paths


def _dict_payload(raw: dict[str, Any]) -> tuple[Any, Iterable[Any] | None]:
    for data_key in ('data', 'samples', 'x', 'instances', 'sequences'):
        if data_key in raw:
            labels = raw.get('labels', raw.get('y'))
            return raw[data_key], labels
    return raw, None


def _iter_samples(raw: Any) -> Iterator[tuple[Any, Any | None]]:
    labels = None
    if isinstance(raw, dict):
        raw, labels = _dict_payload(raw)

    if labels is not None and hasattr(raw, '__len__') and hasattr(labels, '__len__'):
        for index in range(len(raw)):
            yield raw[index], labels[index]
        return

    if hasattr(raw, 'shape') and hasattr(raw, '__len__'):
        shape = getattr(raw, 'shape')
        if shape in ((SMARTGUARD_SEQUENCE_LENGTH, SMARTGUARD_FIELD_COUNT), (SMARTGUARD_SEQUENCE_LENGTH * SMARTGUARD_FIELD_COUNT,)):
            yield raw, None
            return
        for index in range(len(raw)):
            yield raw[index], None
        return

    if isinstance(raw, (list, tuple)):
        if len(raw) == SMARTGUARD_SEQUENCE_LENGTH * SMARTGUARD_FIELD_COUNT:
            yield raw, None
            return
        if len(raw) == SMARTGUARD_SEQUENCE_LENGTH and all(
            isinstance(row, (list, tuple)) and len(row) == SMARTGUARD_FIELD_COUNT for row in raw
        ):
            yield raw, None
            return
        for item in raw:
            yield item, None
        return

    yield raw, None


def load_smartguard_dataset(
    root: str | Path,
    dataset_name: str,
    smartguard_mapping: SmartGuardDictionary | None = None,
    emit_canonical_control: bool = False,
) -> list[BehaviorSequence]:
    '''Load train, validation, and test SmartGuard samples for one dataset.'''
    root = Path(root).expanduser().resolve()
    dataset_name = dataset_name.lower()
    split_paths = _candidate_split_paths(root, dataset_name)
    if not split_paths:
        raise FileNotFoundError(f'No SmartGuard 10x4 pickle files found for dataset {dataset_name!r} under {root}')

    sequences: list[BehaviorSequence] = []
    for split, path in split_paths:
        raw = load_smartguard_array(path)
        for index, (sample, outer_label) in enumerate(_iter_samples(raw)):
            sequence = parse_smartguard_sample(
                sample,
                smartguard_mapping=smartguard_mapping,
                emit_canonical_control=emit_canonical_control,
            )
            if outer_label is not None and sequence.label is None:
                sequence.label = _scalar(outer_label)
            sequence.sequence_id = f'smartguard_{dataset_name}_{split}_{index:06d}'
            sequence.context.update(
                {
                    'dataset': dataset_name,
                    'split': split,
                    'source_path': str(path),
                    'sample_index': index,
                }
            )
            sequence.validate()
            sequences.append(sequence)
    return sequences
