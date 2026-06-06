'''SmartSense external-data adapter.

SmartSense source dumps are not assumed to have one fixed layout. This adapter
therefore discovers csv/json/txt/pickle files, reads log and routine-like tables,
and maps timestamp/device/control/action columns into the shared BehaviorSequence
schema. Dictionary files are loaded as raw mappings for downstream column or
label normalization.
'''

from __future__ import annotations

import csv
import json
import pickle
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from causal_gen_guard.data.schemas import BehaviorEvent, BehaviorSequence

try:
    import pandas as pd
except Exception:  # pragma: no cover - pandas is optional at adapter import time.
    pd = None

SUPPORTED_EXTENSIONS = {'.csv', '.json', '.jsonl', '.txt', '.pkl', '.pickle'}
TIMESTAMP_COLUMNS = ('timestamp', 'time', 'datetime', 'date_time', 'date', 'ts')
DEVICE_COLUMNS = ('device', 'device_id', 'device_name', 'sensor', 'sensor_id', 'object', 'appliance')
CONTROL_COLUMNS = ('control', 'control_id', 'command', 'event', 'activity')
ACTION_COLUMNS = ('action', 'state', 'status', 'value', 'new_state', 'reading')
COUNTRY_HINTS = {'fr', 'sp', 'us', 'kr'}


class AdapterFormatError(ValueError):
    '''Raised when a source file is readable but cannot be mapped to schema.''' 


def _path(path: str | Path) -> Path:
    resolved = Path(path).expanduser()
    if not resolved.exists():
        raise FileNotFoundError(
            'SmartSense path does not exist: {}. Place the dataset locally and pass --smartsense-root or a concrete file path.'.format(resolved)
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


def _read_with_pandas(path: Path) -> Optional[List[Dict[str, Any]]]:
    if pd is None:
        return None
    if path.suffix.lower() == '.csv':
        return pd.read_csv(path).to_dict(orient='records')
    if path.suffix.lower() == '.txt':
        return pd.read_csv(path, sep=None, engine='python').to_dict(orient='records')
    if path.suffix.lower() in ('.pkl', '.pickle'):
        obj = pd.read_pickle(path)
        if hasattr(obj, 'to_dict'):
            return obj.to_dict(orient='records')
    return None


def _read_csv_or_txt(path: Path) -> List[Dict[str, Any]]:
    pandas_rows = _read_with_pandas(path)
    if pandas_rows is not None:
        return [dict(row) for row in pandas_rows]
    text = path.read_text(encoding='utf-8-sig')
    sample = text[:2048]
    try:
        dialect = csv.Sniffer().sniff(sample)
    except csv.Error:
        dialect = csv.excel
    rows = list(csv.DictReader(text.splitlines(), dialect=dialect))
    if not rows:
        raise AdapterFormatError('No tabular rows found in {}'.format(path))
    return [dict(row) for row in rows]


def _read_json(path: Path) -> Any:
    text = path.read_text(encoding='utf-8-sig').strip()
    if not text:
        return []
    if path.suffix.lower() == '.jsonl':
        return [json.loads(line) for line in text.splitlines() if line.strip()]
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return [json.loads(line) for line in text.splitlines() if line.strip()]


def _records_from_json_payload(payload: Any) -> List[Dict[str, Any]]:
    if isinstance(payload, list):
        return [dict(item) for item in payload if isinstance(item, Mapping)]
    if isinstance(payload, dict):
        for key in ('records', 'events', 'logs', 'data', 'rows', 'routines'):
            value = payload.get(key)
            if isinstance(value, list):
                return [dict(item) for item in value if isinstance(item, Mapping)]
        if all(not isinstance(value, (dict, list)) for value in payload.values()):
            return [dict(payload)]
    raise AdapterFormatError('JSON payload does not contain a list of row dictionaries')


def read_records(path: str | Path) -> List[Dict[str, Any]]:
    '''Read csv/json/jsonl/txt/pickle as a list of row dictionaries.'''
    source = _path(path)
    suffix = source.suffix.lower()
    if suffix not in SUPPORTED_EXTENSIONS:
        raise AdapterFormatError('Unsupported SmartSense file extension {} for {}'.format(suffix, source))
    if suffix in ('.csv', '.txt'):
        return _read_csv_or_txt(source)
    if suffix in ('.json', '.jsonl'):
        return _records_from_json_payload(_read_json(source))
    if suffix in ('.pkl', '.pickle'):
        pandas_rows = _read_with_pandas(source)
        if pandas_rows is not None:
            return [dict(row) for row in pandas_rows]
        with source.open('rb') as handle:
            payload = pickle.load(handle)
        if isinstance(payload, list):
            return [dict(item) for item in payload if isinstance(item, Mapping)]
        if isinstance(payload, dict):
            return _records_from_json_payload(payload)
    raise AdapterFormatError('Could not read SmartSense file {}'.format(source))


def read_dictionary(path: str | Path) -> Dict[str, Any]:
    '''Read a SmartSense dictionary file as a raw mapping.'''
    source = _path(path)
    suffix = source.suffix.lower()
    if suffix in ('.json', '.jsonl'):
        payload = _read_json(source)
        if isinstance(payload, dict):
            return payload
        return {'records': payload}
    if suffix in ('.pkl', '.pickle'):
        with source.open('rb') as handle:
            payload = pickle.load(handle)
        if isinstance(payload, dict):
            return payload
        return {'payload': payload}
    rows = read_records(source)
    mapping: Dict[str, Any] = {}
    for row in rows:
        values = list(row.values())
        if len(values) >= 2:
            mapping[str(values[0])] = values[1] if len(values) == 2 else values[1:]
    return mapping


def discover_smartsense_files(root: str | Path) -> Dict[str, List[Path]]:
    '''Discover SmartSense-like log, dictionary, and routine files.'''
    base = _path(root)
    files = [base] if base.is_file() else [path for path in base.rglob('*') if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS]
    discovered = {'logs': [], 'dictionaries': [], 'routines': [], 'other': []}
    for file_path in sorted(files):
        name = file_path.name.lower()
        if 'dict' in name or 'dictionary' in name or 'mapping' in name:
            discovered['dictionaries'].append(file_path)
        elif 'routine' in name or 'pattern' in name:
            discovered['routines'].append(file_path)
        elif 'log' in name or 'event' in name or 'sensor' in name or base.is_file():
            discovered['logs'].append(file_path)
        else:
            discovered['other'].append(file_path)
    return discovered


def _country_from_path(path: Path) -> Optional[str]:
    for part in path.parts:
        lowered = part.lower()
        if lowered in COUNTRY_HINTS:
            return lowered.upper()
        for hint in COUNTRY_HINTS:
            if lowered.startswith(hint + '_') or lowered.endswith('_' + hint) or ('_' + hint + '_') in lowered:
                return hint.upper()
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


def records_to_sequences(
    records: List[Mapping[str, Any]],
    source_name: str,
    context: Optional[Dict[str, Any]] = None,
    columns: Optional[Mapping[str, str]] = None,
    window_size: int = 50,
) -> List[BehaviorSequence]:
    '''Convert row dictionaries into BehaviorSequence windows.'''
    if not records:
        return []
    first = records[0]
    timestamp_col = _find_column(first, TIMESTAMP_COLUMNS, columns)
    device_col = _find_column(first, DEVICE_COLUMNS, columns)
    control_col = _find_column(first, CONTROL_COLUMNS, columns)
    action_col = _find_column(first, ACTION_COLUMNS, columns)
    if timestamp_col is None or (device_col is None and control_col is None and action_col is None):
        available = ', '.join(str(key) for key in first.keys())
        raise AdapterFormatError(
            'Cannot map SmartSense rows from {}. Need columns like timestamp plus device/control/action. Available columns: {}'.format(
                source_name, available
            )
        )

    events: List[BehaviorEvent] = []
    for row_index, row in enumerate(records):
        timestamp = row.get(timestamp_col)
        device = row.get(device_col) if device_col else None
        control = row.get(control_col) if control_col else None
        action = row.get(action_col) if action_col else None
        if control in (None, ''):
            if device not in (None, '') and action not in (None, ''):
                control = '{}:{}'.format(device, action)
            elif action not in (None, ''):
                control = action
            else:
                control = device
        if control in (None, '') or timestamp in (None, ''):
            continue
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
    if not events:
        raise AdapterFormatError('No valid SmartSense events could be built from {}'.format(source_name))

    size = max(1, int(window_size))
    base_context = dict(context or {})
    sequences: List[BehaviorSequence] = []
    for start in range(0, len(events), size):
        chunk = events[start : start + size]
        if not chunk:
            continue
        sequence_context = dict(base_context)
        sequence_context.update({'source': 'SmartSense', 'source_file': source_name, 'window_start': start})
        sequences.append(
            BehaviorSequence(
                sequence_id='smartsense:{}:{}'.format(Path(source_name).stem, start // size),
                events=chunk,
                context=sequence_context,
                label=0,
            )
        )
    return sequences


def load_smartsense_logs(
    root: str | Path,
    columns: Optional[Mapping[str, str]] = None,
    window_size: int = 50,
) -> List[BehaviorSequence]:
    '''Load SmartSense log/event files into BehaviorSequence objects.'''
    discovered = discover_smartsense_files(root)
    candidate_files = discovered['logs'] or discovered['other']
    if not candidate_files:
        raise FileNotFoundError('No SmartSense log files found under {}. Expected csv/json/txt/pickle files.'.format(root))
    sequences: List[BehaviorSequence] = []
    errors: List[str] = []
    for file_path in candidate_files:
        try:
            records = read_records(file_path)
            context = {'country': _country_from_path(file_path)} if _country_from_path(file_path) else {}
            sequences.extend(records_to_sequences(records, str(file_path), context=context, columns=columns, window_size=window_size))
        except Exception as exc:
            errors.append('{}: {}'.format(file_path, exc))
    if not sequences:
        raise AdapterFormatError('No SmartSense log sequences loaded. Errors: {}'.format('; '.join(errors)))
    return sequences


def load_smartsense_routines(
    root: str | Path,
    columns: Optional[Mapping[str, str]] = None,
    window_size: int = 50,
) -> List[BehaviorSequence]:
    '''Load SmartSense routine files into BehaviorSequence objects.'''
    discovered = discover_smartsense_files(root)
    if not discovered['routines']:
        return []
    sequences: List[BehaviorSequence] = []
    errors: List[str] = []
    for file_path in discovered['routines']:
        try:
            records = read_records(file_path)
            context = {'routine': True}
            country = _country_from_path(file_path)
            if country:
                context['country'] = country
            sequences.extend(records_to_sequences(records, str(file_path), context=context, columns=columns, window_size=window_size))
        except Exception as exc:
            errors.append('{}: {}'.format(file_path, exc))
    if errors and not sequences:
        raise AdapterFormatError('No SmartSense routine sequences loaded. Errors: {}'.format('; '.join(errors)))
    return sequences


def load_smartsense_dictionaries(root: str | Path) -> Dict[str, Dict[str, Any]]:
    '''Load SmartSense dictionary files as raw mappings keyed by file stem.'''
    discovered = discover_smartsense_files(root)
    dictionaries: Dict[str, Dict[str, Any]] = {}
    errors: List[str] = []
    for file_path in discovered['dictionaries']:
        try:
            dictionaries[file_path.stem] = read_dictionary(file_path)
        except Exception as exc:
            errors.append('{}: {}'.format(file_path, exc))
    if errors and not dictionaries:
        raise AdapterFormatError('No SmartSense dictionaries loaded. Errors: {}'.format('; '.join(errors)))
    return dictionaries


def load_smartsense_dataset(
    root: str | Path,
    columns: Optional[Mapping[str, str]] = None,
    window_size: int = 50,
    include_routines: bool = True,
) -> Dict[str, Any]:
    '''Load SmartSense logs, dictionaries, and optionally routines.'''
    logs = load_smartsense_logs(root, columns=columns, window_size=window_size)
    routines = load_smartsense_routines(root, columns=columns, window_size=window_size) if include_routines else []
    dictionaries = load_smartsense_dictionaries(root)
    return {'logs': logs, 'routines': routines, 'dictionaries': dictionaries}
