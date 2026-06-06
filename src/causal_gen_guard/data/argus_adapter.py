'''ARGUS Home1-Home5 external-data adapter.

The ARGUS adapter expects the user to place the dataset locally. It scans Home*
directories, reads csv/json/jsonl event tables, keeps device state changes, and
maps them into BehaviorSequence windows with home_id stored in context.
'''

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from causal_gen_guard.data.schemas import BehaviorEvent, BehaviorSequence
from causal_gen_guard.data.smartsense_adapter import (
    ACTION_COLUMNS,
    CONTROL_COLUMNS,
    DEVICE_COLUMNS,
    SUPPORTED_EXTENSIONS,
    TIMESTAMP_COLUMNS,
    AdapterFormatError,
    read_records,
)

try:
    import pandas as pd
except Exception:  # pragma: no cover - pandas is optional for timestamp enrichment.
    pd = None


STATE_COLUMNS = ACTION_COLUMNS + ('state_change', 'new_value', 'current_state')


def _path(path: str | Path) -> Path:
    resolved = Path(path).expanduser()
    if not resolved.exists():
        raise FileNotFoundError(
            'ARGUS path does not exist: {}. Place Home1-Home5 locally and pass --argus-root.'.format(resolved)
        )
    return resolved


def _lower_key_map(record: Mapping[str, Any]) -> Dict[str, str]:
    return {str(key).strip().lower(): str(key) for key in record.keys()}


def _find_column(record: Mapping[str, Any], candidates: Sequence[str], mapping: Optional[Mapping[str, str]] = None) -> Optional[str]:
    explicit = dict(mapping or {})
    for candidate in candidates:
        if candidate in explicit:
            return explicit[candidate]
    lowered = _lower_key_map(record)
    for candidate in candidates:
        if candidate in lowered:
            return lowered[candidate]
    for key_lower, original in lowered.items():
        for candidate in candidates:
            if candidate in key_lower:
                return original
    return None


def _timestamp_parts(timestamp: Any) -> Tuple[Any, Any]:
    if pd is not None:
        try:
            parsed = pd.to_datetime(timestamp)
            if not pd.isna(parsed):
                return int(parsed.dayofweek), int(parsed.hour)
        except Exception:
            pass
    text = str(timestamp)
    hour = None
    if ':' in text:
        try:
            hour = int(text.split('T')[-1].split(' ')[-1].split(':')[0])
        except Exception:
            hour = None
    return None, hour


def discover_argus_homes(root: str | Path) -> Dict[str, Path]:
    '''Return Home* directories keyed by home id.'''
    base = _path(root)
    if base.is_file():
        return {base.parent.name: base.parent}
    homes = {path.name: path for path in sorted(base.glob('Home*')) if path.is_dir()}
    if not homes and any(path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS for path in base.iterdir()):
        homes[base.name] = base
    if not homes:
        raise FileNotFoundError('No ARGUS Home* directories or readable event files found under {}'.format(base))
    return homes


def _event_files(home_path: Path) -> List[Path]:
    if home_path.is_file():
        return [home_path]
    files = [path for path in home_path.rglob('*') if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS]
    preferred = [path for path in files if any(token in path.name.lower() for token in ('event', 'log', 'sensor', 'state'))]
    return sorted(preferred or files)


def _records_to_state_events(
    records: List[Mapping[str, Any]],
    home_id: str,
    source_file: Path,
    columns: Optional[Mapping[str, str]] = None,
) -> List[BehaviorEvent]:
    if not records:
        return []
    first = records[0]
    timestamp_col = _find_column(first, TIMESTAMP_COLUMNS, columns)
    device_col = _find_column(first, DEVICE_COLUMNS, columns)
    control_col = _find_column(first, CONTROL_COLUMNS, columns)
    state_col = _find_column(first, STATE_COLUMNS, columns)
    if timestamp_col is None or device_col is None or (control_col is None and state_col is None):
        available = ', '.join(str(key) for key in first.keys())
        raise AdapterFormatError(
            'Cannot map ARGUS rows from {}. Need columns like timestamp, device, and action/control/state. Available columns: {}'.format(
                source_file, available
            )
        )

    events: List[BehaviorEvent] = []
    last_state_by_device: Dict[Any, Any] = {}
    for row in records:
        timestamp = row.get(timestamp_col)
        device = row.get(device_col)
        state = row.get(state_col) if state_col else row.get(control_col)
        control = row.get(control_col) if control_col else None
        if timestamp in (None, '') or device in (None, '') or state in (None, ''):
            continue
        if device in last_state_by_device and str(last_state_by_device[device]) == str(state):
            continue
        last_state_by_device[device] = state
        if control in (None, ''):
            control = '{}:{}'.format(device, state)
        day, hour = _timestamp_parts(timestamp)
        events.append(
            BehaviorEvent(
                timestamp=timestamp,
                day=day,
                hour=hour,
                device_id=device,
                control_id=control,
                duration=None,
                raw_fields={str(key): value for key, value in row.items()},
            )
        )
    return events


def _events_to_sequences(events: List[BehaviorEvent], home_id: str, source_file: Path, window_size: int) -> List[BehaviorSequence]:
    sequences: List[BehaviorSequence] = []
    size = max(1, int(window_size))
    for start in range(0, len(events), size):
        chunk = events[start : start + size]
        if not chunk:
            continue
        sequences.append(
            BehaviorSequence(
                sequence_id='argus:{}:{}:{}'.format(home_id, source_file.stem, start // size),
                events=chunk,
                context={
                    'source': 'ARGUS',
                    'home_id': home_id,
                    'source_file': str(source_file),
                    'window_start': start,
                },
                label=0,
            )
        )
    return sequences


def load_argus_home(
    home_path: str | Path,
    home_id: Optional[str] = None,
    columns: Optional[Mapping[str, str]] = None,
    window_size: int = 50,
) -> List[BehaviorSequence]:
    '''Load one ARGUS home directory or event file.'''
    home = _path(home_path)
    resolved_home_id = home_id or home.name
    sequences: List[BehaviorSequence] = []
    errors: List[str] = []
    for file_path in _event_files(home):
        try:
            records = read_records(file_path)
            events = _records_to_state_events(records, resolved_home_id, file_path, columns=columns)
            sequences.extend(_events_to_sequences(events, resolved_home_id, file_path, window_size=window_size))
        except Exception as exc:
            errors.append('{}: {}'.format(file_path, exc))
    if not sequences:
        raise AdapterFormatError('No ARGUS sequences loaded for {}. Errors: {}'.format(resolved_home_id, '; '.join(errors)))
    return sequences


def load_argus_dataset(
    root: str | Path,
    columns: Optional[Mapping[str, str]] = None,
    window_size: int = 50,
) -> List[BehaviorSequence]:
    '''Load all ARGUS Home* folders under root.'''
    homes = discover_argus_homes(root)
    sequences: List[BehaviorSequence] = []
    errors: List[str] = []
    for home_id, home_path in homes.items():
        try:
            sequences.extend(load_argus_home(home_path, home_id=home_id, columns=columns, window_size=window_size))
        except Exception as exc:
            errors.append('{}: {}'.format(home_id, exc))
    if not sequences:
        raise AdapterFormatError('No ARGUS sequences loaded from {}. Errors: {}'.format(root, '; '.join(errors)))
    return sequences


def _first_timestamp(sequence: BehaviorSequence) -> str:
    if not sequence.events:
        return ''
    return str(sequence.events[0].timestamp)


def split_argus_sequences(
    sequences: List[BehaviorSequence],
    split: str = 'temporal',
    leave_home_id: Optional[str] = None,
    train_ratio: float = 0.7,
    val_ratio: float = 0.1,
) -> Dict[str, List[BehaviorSequence]]:
    '''Split ARGUS sequences for temporal or leave-one-home evaluation.'''
    if split not in ('temporal', 'leave_one_home'):
        raise ValueError('split must be temporal or leave_one_home')
    if not sequences:
        return {'train': [], 'val': [], 'test': []}

    if split == 'leave_one_home':
        homes = sorted({str(sequence.context.get('home_id')) for sequence in sequences})
        heldout = leave_home_id or homes[-1]
        test = [sequence for sequence in sequences if str(sequence.context.get('home_id')) == str(heldout)]
        remaining = [sequence for sequence in sequences if str(sequence.context.get('home_id')) != str(heldout)]
        remaining = sorted(remaining, key=lambda item: (str(item.context.get('home_id')), _first_timestamp(item)))
        val_count = max(1, int(len(remaining) * val_ratio)) if len(remaining) > 1 else 0
        return {'train': remaining[:-val_count] if val_count else remaining, 'val': remaining[-val_count:] if val_count else [], 'test': test}

    ordered = sorted(sequences, key=lambda item: (str(item.context.get('home_id')), _first_timestamp(item)))
    train_end = max(1, int(len(ordered) * train_ratio))
    val_end = min(len(ordered), train_end + max(1, int(len(ordered) * val_ratio))) if len(ordered) > 2 else train_end
    return {'train': ordered[:train_end], 'val': ordered[train_end:val_end], 'test': ordered[val_end:]}
