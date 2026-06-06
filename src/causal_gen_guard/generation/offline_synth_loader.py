'''Offline loader for existing SmartGen synthetic and filtered data.

The loader searches a SmartGen checkout for local synthetic/filter/result files
and converts supported records into BehaviorSequence objects. It never calls an
online LLM and never requires an API key.
'''

from __future__ import annotations

import csv
import json
import pickle
from pathlib import Path
from typing import Any, Iterable, List, Optional

from causal_gen_guard.data.schemas import BehaviorEvent, BehaviorSequence


SUPPORTED_SUFFIXES = {'.pkl', '.pickle', '.txt', '.json', '.csv'}
SEARCH_KEYWORDS = ('synthetic', 'filter_data', 'filter_true', 'generation', 'result', 'results')
TRANSITION_ALIASES = {
    'ST': ('st', 'spring'),
    'TT': ('tt', 'night', 'daytime'),
    'NT': ('nt', 'multiple', 'single'),
}


def _matches_dataset(path: Path, dataset: Optional[str]) -> bool:
    if not dataset:
        return True
    name = path.as_posix().lower()
    dataset = dataset.lower()
    return f'/{dataset}_' in name or f'_{dataset}_' in name or f'/{dataset}/' in name or name.endswith(f'/{dataset}.txt')


def _matches_transition(path: Path, transition: Optional[str]) -> bool:
    if not transition:
        return True
    aliases = TRANSITION_ALIASES.get(transition.upper(), (transition.lower(),))
    name = path.as_posix().lower()
    return any(alias in name for alias in aliases)


def find_smartgen_offline_files(root: str | Path, dataset: Optional[str] = None, transition: Optional[str] = None) -> List[Path]:
    '''Find likely SmartGen offline synthetic/filter/result files.'''
    root = Path(root).expanduser().resolve()
    if not root.exists():
        raise FileNotFoundError(f'SmartGen root does not exist: {root}')
    candidates: List[Path] = []
    for path in root.rglob('*'):
        if not path.is_file() or path.suffix.lower() not in SUPPORTED_SUFFIXES:
            continue
        lowered = path.as_posix().lower()
        if not any(keyword in lowered for keyword in SEARCH_KEYWORDS):
            continue
        if not _matches_dataset(path, dataset):
            continue
        if not _matches_transition(path, transition):
            continue
        candidates.append(path)
    if not candidates and transition:
        for path in root.rglob('*'):
            if not path.is_file() or path.suffix.lower() not in SUPPORTED_SUFFIXES:
                continue
            lowered = path.as_posix().lower()
            if any(keyword in lowered for keyword in SEARCH_KEYWORDS) and _matches_dataset(path, dataset):
                candidates.append(path)
    priority = {'.pkl': 0, '.pickle': 0, '.txt': 1, '.json': 2, '.csv': 3}
    candidates.sort(key=lambda item: (priority.get(item.suffix.lower(), 9), len(item.as_posix()), item.as_posix()))
    return candidates


def _as_list(value: Any) -> list[Any]:
    if hasattr(value, 'tolist'):
        value = value.tolist()
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, list):
        return value
    return [value]


def _parse_flat_sequence(values: Iterable[Any], sequence_id: str, context: dict[str, Any]) -> Optional[BehaviorSequence]:
    flat = _as_list(values)
    if not flat:
        return None
    events: List[BehaviorEvent] = []
    if len(flat) >= 4 and len(flat) % 4 == 0:
        for index in range(0, len(flat), 4):
            day, hour, device, control = flat[index : index + 4]
            events.append(
                BehaviorEvent(
                    day=day,
                    hour=hour,
                    device_id=device,
                    control_id=control,
                    duration=None,
                    raw_fields={
                        'position': index // 4,
                        'day': day,
                        'hour': hour,
                        'device': device,
                        'control': control,
                        'source_format': 'smartgen_flat_4',
                    },
                )
            )
    else:
        for index, control in enumerate(flat):
            events.append(
                BehaviorEvent(
                    day=None,
                    hour=index,
                    device_id=None,
                    control_id=control,
                    duration=None,
                    raw_fields={'position': index, 'source_format': 'smartgen_control_sequence'},
                )
            )
    return BehaviorSequence(sequence_id=sequence_id, events=events, context=dict(context), label=None, anomaly_type=None)


def _load_pickle(path: Path, context: dict[str, Any]) -> List[BehaviorSequence]:
    with path.open('rb') as handle:
        obj = pickle.load(handle)
    records = obj.values() if isinstance(obj, dict) else obj
    sequences: List[BehaviorSequence] = []
    for index, item in enumerate(_as_list(records)):
        if isinstance(item, dict) and 'events' in item:
            payload = dict(item)
            payload.setdefault('sequence_id', f'{path.stem}_{index:06d}')
            payload.setdefault('context', dict(context))
            sequences.append(BehaviorSequence.from_dict(payload))
        else:
            sequence = _parse_flat_sequence(item, f'{path.stem}_{index:06d}', context)
            if sequence is not None:
                sequences.append(sequence)
    return sequences


def _load_txt(path: Path, context: dict[str, Any]) -> List[BehaviorSequence]:
    groups: dict[str, list[Any]] = {}
    fallback: list[Any] = []
    with path.open('r', encoding='utf-8', errors='ignore') as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            parts = line.replace(',', ' ').split()
            if len(parts) >= 2:
                groups.setdefault(parts[0], []).append(parts[1])
            elif parts:
                fallback.append(parts[0])
    sequences: List[BehaviorSequence] = []
    if groups:
        for key, controls in sorted(groups.items()):
            sequence = _parse_flat_sequence(controls, f'{path.stem}_{key}', context)
            if sequence is not None:
                sequences.append(sequence)
    elif fallback:
        sequence = _parse_flat_sequence(fallback, f'{path.stem}_000000', context)
        if sequence is not None:
            sequences.append(sequence)
    return sequences


def _load_json(path: Path, context: dict[str, Any]) -> List[BehaviorSequence]:
    with path.open('r', encoding='utf-8') as handle:
        obj = json.load(handle)
    records = obj.get('sequences', obj.get('data', obj)) if isinstance(obj, dict) else obj
    sequences: List[BehaviorSequence] = []
    for index, item in enumerate(_as_list(records)):
        if isinstance(item, dict) and 'events' in item:
            payload = dict(item)
            payload.setdefault('sequence_id', f'{path.stem}_{index:06d}')
            payload.setdefault('context', dict(context))
            sequences.append(BehaviorSequence.from_dict(payload))
        else:
            sequence = _parse_flat_sequence(item, f'{path.stem}_{index:06d}', context)
            if sequence is not None:
                sequences.append(sequence)
    return sequences


def _load_csv(path: Path, context: dict[str, Any]) -> List[BehaviorSequence]:
    groups: dict[str, list[Any]] = {}
    with path.open('r', encoding='utf-8', errors='ignore', newline='') as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames:
            for row_index, row in enumerate(reader):
                sequence_key = row.get('sequence_id') or row.get('user') or row.get('uid') or 'csv'
                control = row.get('control_id') or row.get('control') or row.get('action') or row.get('item') or row.get('item_id')
                if control is not None:
                    groups.setdefault(str(sequence_key), []).append(control)
    sequences: List[BehaviorSequence] = []
    for key, controls in sorted(groups.items()):
        sequence = _parse_flat_sequence(controls, f'{path.stem}_{key}', context)
        if sequence is not None:
            sequences.append(sequence)
    return sequences


def load_smartgen_offline_sequences(
    root: str | Path,
    dataset: Optional[str] = None,
    transition: Optional[str] = None,
    limit: Optional[int] = None,
) -> List[BehaviorSequence]:
    '''Load SmartGen offline synthetic/filter/result data as BehaviorSequence.'''
    files = find_smartgen_offline_files(root, dataset=dataset, transition=transition)
    if not files:
        raise FileNotFoundError(
            f'No offline SmartGen synthetic/filter/result files found under {Path(root).resolve()} '
            f'for dataset={dataset!r}, transition={transition!r}.'
        )
    sequences: List[BehaviorSequence] = []
    errors: List[str] = []
    for path in files:
        context = {
            'source_project': 'SmartGen',
            'source_path': str(path),
            'dataset': dataset,
            'transition': transition,
            'offline_mode': True,
        }
        try:
            if path.suffix.lower() in ('.pkl', '.pickle'):
                loaded = _load_pickle(path, context)
            elif path.suffix.lower() == '.txt':
                loaded = _load_txt(path, context)
            elif path.suffix.lower() == '.json':
                loaded = _load_json(path, context)
            elif path.suffix.lower() == '.csv':
                loaded = _load_csv(path, context)
            else:
                loaded = []
        except Exception as exc:
            errors.append(f'{path}: {exc}')
            continue
        sequences.extend(loaded)
        if limit is not None and len(sequences) >= limit:
            return sequences[:limit]
    if not sequences:
        raise ValueError('Found SmartGen offline files but could not parse sequences. Errors: ' + '; '.join(errors[:5]))
    return sequences
