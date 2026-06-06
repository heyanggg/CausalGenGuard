'''Attack injection utilities for controlled behavior-anomaly benchmarks.

The injector supports SmartGuard-style SD/MD/DM/DD attacks and CausalGenGuard
causal dependency anomalies. It intentionally avoids hard-coded numeric control
ids: targets are found by fuzzy matching control names from the sequence or an
optional inverse_vocab stored in sequence.context.
'''

from __future__ import annotations

import copy
import random
import re
from typing import Any, Dict, Iterable, List, Optional, Tuple

from causal_gen_guard.data.schemas import BehaviorEvent, BehaviorSequence


SMARTGUARD_ANOMALIES = {
    'SD_light_flickering',
    'SD_camera_flickering',
    'SD_tv_flickering',
    'MD_window_open_while_lock',
    'MD_camera_off_while_lock',
    'DM_ac_cool_in_winter',
    'DM_window_open_midnight',
    'DM_watervalve_open_midnight',
    'DD_shower_long_time',
    'DD_microwave_long_time',
}

CAUSAL_ANOMALIES = {
    'causal_edge_break',
    'causal_edge_injection',
    'lag_delay',
    'context_causal_conflict',
    'chain_break',
}


_SPEC_KEYWORDS = {
    'light': [('light', 'lamp')],
    'camera': [('camera', 'cam')],
    'tv': [('tv', 'television')],
    'window_open': [('window', 'blind', 'curtain'), ('open', 'up', 'raise')],
    'lock': [('lock', 'smartlock', 'door_lock'), ('lock', 'locked', 'close')],
    'camera_off': [('camera', 'cam'), ('off', 'close', 'disable')],
    'ac_cool': [('ac', 'airconditioner', 'air_conditioner', 'air'), ('cool', 'cold', 'on')],
    'watervalve_open': [('water', 'watervalve', 'valve'), ('open', 'on')],
    'shower': [('shower', 'bath')],
    'microwave': [('microwave', 'oven')],
}


_SPEC_CANONICAL_CONTROLS = {
    'light': (
        'Light:switch on',
        'Light:switch off',
        'Light:switch toggle',
    ),
    'camera': (
        'Camera:switch on',
        'Camera:switch off',
    ),
    'tv': (
        'Television:switch on',
        'Television:switch off',
        'Television:switch toggle',
    ),
    'window_open': (
        'Blind:windowShade open',
        'Other:windowShade open',
        'Switch:windowShade open',
    ),
    'lock': (
        'SmartLock:lock lock',
        'ContactSensor:lock lock',
    ),
    'camera_off': (
        'Camera:switch off',
    ),
    'ac_cool': (
        'AirConditioner:switch on',
        'AirConditioner:setCoolingSetpoint',
        'AirConditioner:setAirConditionerMode',
    ),
    'watervalve_open': (
        'WaterValve:valve open',
    ),
    'shower': (
        'Shower:switch on',
        'Shower:water on',
        'Bathroom:shower on',
    ),
    'microwave': (
        'Microwave:switch on',
    ),
}


def _normalize_text(value: Any) -> str:
    text = str(value).lower()
    return re.sub(r'[^a-z0-9]+', ' ', text)


def _as_float(value: Any, default: float = 0.0) -> float:
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


def _control_key(control_id: Any) -> Any:
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


def _is_unknown(value: Any) -> bool:
    if value is None:
        return True
    text = str(value).strip()
    return not text or text.lower() == 'unknown'


def _raw_canonical_control(event: BehaviorEvent) -> Optional[Any]:
    raw_fields = event.raw_fields or {}
    canonical_control = raw_fields.get('canonical_control')
    if _is_unknown(canonical_control):
        return None
    return canonical_control


def _event_named_controls(event: BehaviorEvent) -> List[Any]:
    controls: List[Any] = []
    canonical_control = _raw_canonical_control(event)
    if canonical_control is not None:
        controls.append(canonical_control)
    if not _is_unknown(event.control_id):
        controls.append(event.control_id)

    seen = set()
    unique: List[Any] = []
    for control in controls:
        key = repr(_control_key(control))
        if key not in seen:
            seen.add(key)
            unique.append(control)
    return unique


def _canonical_parts(control: Any) -> tuple[str | None, str | None]:
    if _is_unknown(control):
        return None, None
    text = str(control)
    if ':' not in text:
        return None, None
    device, action = text.split(':', 1)
    if not device or not action:
        return None, None
    return device, action


def _candidate_controls(sequence: BehaviorSequence) -> List[Any]:
    controls: List[Any] = []
    for event in sequence.events:
        controls.append(_control_key(event.control_id))
    inverse_vocab = sequence.context.get('inverse_vocab') or sequence.context.get('control_inverse_vocab')
    if isinstance(inverse_vocab, dict):
        controls.extend(inverse_vocab.values())
    elif isinstance(inverse_vocab, (list, tuple)):
        controls.extend(inverse_vocab)
    seen = set()
    unique: List[Any] = []
    for control in controls:
        key = repr(control)
        if key not in seen:
            seen.add(key)
            unique.append(control)
    return unique


def _matches_groups(control: Any, groups: List[tuple[str, ...]]) -> bool:
    text = _normalize_text(control)
    return all(any(keyword in text for keyword in group) for group in groups)


def _find_control(sequence: BehaviorSequence, spec_name: str) -> Optional[Any]:
    canonical_controls = _SPEC_CANONICAL_CONTROLS.get(spec_name)
    if canonical_controls:
        targets = {_control_key(control) for control in canonical_controls}
        for event in sequence.events:
            for control in _event_named_controls(event):
                if _control_key(control) in targets:
                    return control
        return None

    groups = _SPEC_KEYWORDS[spec_name]
    for event in sequence.events:
        for control in _event_named_controls(event):
            if _matches_groups(control, groups):
                return control
    return None


def _event_index_for_control(sequence: BehaviorSequence, control: Any) -> Optional[int]:
    for index, event in enumerate(sequence.events):
        for candidate in _event_named_controls(event):
            if _control_key(candidate) == _control_key(control):
                return index
    return None


def _clone_event(event: BehaviorEvent, control_id: Any = None, **updates: Any) -> BehaviorEvent:
    cloned = copy.deepcopy(event)
    raw_fields = dict(cloned.raw_fields)
    if control_id is not None:
        cloned.control_id = control_id
        device, action = _canonical_parts(control_id)
        if device is not None and action is not None:
            cloned.device_id = device
            raw_fields['device'] = device
            raw_fields['action'] = action
            raw_fields['canonical_control'] = str(control_id)
            raw_fields['injected_canonical_control'] = str(control_id)
    for key, value in updates.items():
        setattr(cloned, key, value)
        if key in ('day', 'hour', 'duration'):
            raw_fields[key] = value
    raw_fields['attack_injected'] = True
    cloned.raw_fields = raw_fields
    return cloned


def _make_event_like(sequence: BehaviorSequence, control_id: Any, anchor_index: int = 0, **updates: Any) -> BehaviorEvent:
    if sequence.events:
        anchor = sequence.events[max(0, min(anchor_index, len(sequence.events) - 1))]
        return _clone_event(anchor, control_id=control_id, **updates)
    return BehaviorEvent(control_id=control_id, day=updates.get('day', 0), hour=updates.get('hour', 0), raw_fields={'attack_injected': True})


def _make_sequence(sequence: BehaviorSequence, events: List[BehaviorEvent], anomaly_type: str, operations: List[str]) -> BehaviorSequence:
    context = dict(sequence.context)
    context.update({'attack_injected': True, 'attack_operations': operations, 'source_sequence_id': sequence.sequence_id})
    return BehaviorSequence(
        sequence_id=f'{sequence.sequence_id}::{anomaly_type}',
        events=events,
        context=context,
        label=1,
        anomaly_type=anomaly_type,
    )


def _report(sequence: BehaviorSequence, anomaly_type: str, status: str, **extra: Any) -> Dict[str, Any]:
    payload = {'sequence_id': sequence.sequence_id, 'anomaly_type': anomaly_type, 'status': status}
    payload.update(extra)
    return payload


def _skip(sequence: BehaviorSequence, anomaly_type: str, reason: str) -> Tuple[None, Dict[str, Any]]:
    return None, _report(sequence, anomaly_type, 'skipped', skipped_reason=reason)


def _insert_after(events: List[BehaviorEvent], index: int, new_events: Iterable[BehaviorEvent]) -> List[BehaviorEvent]:
    return list(events[: index + 1]) + list(new_events) + list(events[index + 1 :])


def _flicker(sequence: BehaviorSequence, anomaly_type: str, spec_name: str) -> Tuple[Optional[BehaviorSequence], Dict[str, Any]]:
    control = _find_control(sequence, spec_name)
    if control is None:
        return _skip(sequence, anomaly_type, f'control_not_found:{spec_name}')
    index = _event_index_for_control(sequence, control)
    if index is None:
        index = 0
    repeats = [_make_event_like(sequence, control, anchor_index=index) for _ in range(3)]
    events = _insert_after(copy.deepcopy(sequence.events), index, repeats)
    injected = _make_sequence(sequence, events, anomaly_type, [f'flicker:{control}'])
    return injected, _report(sequence, anomaly_type, 'injected', operation='flicker', control_id=control)


def inject_smartguard_style(sequence: BehaviorSequence, anomaly_type: str) -> Tuple[Optional[BehaviorSequence], Dict[str, Any]]:
    '''Inject SmartGuard-style SD/MD/DM/DD anomalies into one sequence.'''
    if anomaly_type not in SMARTGUARD_ANOMALIES:
        return _skip(sequence, anomaly_type, 'unsupported_smartguard_anomaly')
    if not sequence.events:
        return _skip(sequence, anomaly_type, 'empty_sequence')

    if anomaly_type == 'SD_light_flickering':
        return _flicker(sequence, anomaly_type, 'light')
    if anomaly_type == 'SD_camera_flickering':
        return _flicker(sequence, anomaly_type, 'camera')
    if anomaly_type == 'SD_tv_flickering':
        return _flicker(sequence, anomaly_type, 'tv')

    events = copy.deepcopy(sequence.events)
    if anomaly_type == 'MD_window_open_while_lock':
        lock = _find_control(sequence, 'lock')
        window = _find_control(sequence, 'window_open')
        if lock is None or window is None:
            return _skip(sequence, anomaly_type, 'required_control_not_found:lock_or_window_open')
        index = _event_index_for_control(sequence, lock) or 0
        injected_event = _make_event_like(sequence, window, anchor_index=index)
        return _make_sequence(sequence, _insert_after(events, index, [injected_event]), anomaly_type, ['window_open_after_lock']), _report(sequence, anomaly_type, 'injected')

    if anomaly_type == 'MD_camera_off_while_lock':
        lock = _find_control(sequence, 'lock')
        camera_off = _find_control(sequence, 'camera_off')
        if lock is None or camera_off is None:
            return _skip(sequence, anomaly_type, 'required_control_not_found:lock_or_camera_off')
        index = _event_index_for_control(sequence, lock) or 0
        injected_event = _make_event_like(sequence, camera_off, anchor_index=index)
        return _make_sequence(sequence, _insert_after(events, index, [injected_event]), anomaly_type, ['camera_off_after_lock']), _report(sequence, anomaly_type, 'injected')

    if anomaly_type == 'DM_ac_cool_in_winter':
        ac_cool = _find_control(sequence, 'ac_cool')
        if ac_cool is None:
            return _skip(sequence, anomaly_type, 'control_not_found:ac_cool')
        event = _make_event_like(sequence, ac_cool, anchor_index=0)
        new_sequence = _make_sequence(sequence, [event] + events, anomaly_type, ['ac_cool_forced_in_winter'])
        new_sequence.context['context_id'] = 'winter'
        new_sequence.context['season'] = 'winter'
        return new_sequence, _report(sequence, anomaly_type, 'injected')

    if anomaly_type == 'DM_window_open_midnight':
        control = _find_control(sequence, 'window_open')
        if control is None:
            return _skip(sequence, anomaly_type, 'control_not_found:window_open')
        event = _make_event_like(sequence, control, anchor_index=0, hour=0)
        return _make_sequence(sequence, [event] + events, anomaly_type, ['window_open_at_midnight']), _report(sequence, anomaly_type, 'injected')

    if anomaly_type == 'DM_watervalve_open_midnight':
        control = _find_control(sequence, 'watervalve_open')
        if control is None:
            return _skip(sequence, anomaly_type, 'control_not_found:watervalve_open')
        event = _make_event_like(sequence, control, anchor_index=0, hour=0)
        return _make_sequence(sequence, [event] + events, anomaly_type, ['watervalve_open_at_midnight']), _report(sequence, anomaly_type, 'injected')

    if anomaly_type in ('DD_shower_long_time', 'DD_microwave_long_time'):
        spec = 'shower' if 'shower' in anomaly_type else 'microwave'
        control = _find_control(sequence, spec)
        if control is None:
            return _skip(sequence, anomaly_type, f'control_not_found:{spec}')
        index = _event_index_for_control(sequence, control)
        if index is None:
            return _skip(sequence, anomaly_type, f'control_not_present_in_sequence:{spec}')
        events[index].duration = max(_as_float(events[index].duration, 1.0) * 5.0, 120.0)
        events[index].raw_fields = {**events[index].raw_fields, 'attack_injected': True, 'duration_attack': 'long_time'}
        return _make_sequence(sequence, events, anomaly_type, [f'long_duration:{control}']), _report(sequence, anomaly_type, 'injected')

    return _skip(sequence, anomaly_type, 'unhandled_smartguard_anomaly')


def _unique_controls(sequence: BehaviorSequence) -> List[Any]:
    controls = []
    seen = set()
    for control in _candidate_controls(sequence):
        key = repr(control)
        if key not in seen:
            seen.add(key)
            controls.append(control)
    return controls


def inject_causal_anomaly(sequence: BehaviorSequence, anomaly_type: str) -> Tuple[Optional[BehaviorSequence], Dict[str, Any]]:
    '''Inject dependency-structure anomalies into one sequence.'''
    if anomaly_type not in CAUSAL_ANOMALIES:
        return _skip(sequence, anomaly_type, 'unsupported_causal_anomaly')
    if not sequence.events:
        return _skip(sequence, anomaly_type, 'empty_sequence')

    events = copy.deepcopy(sequence.events)
    controls = _unique_controls(sequence)

    if anomaly_type == 'causal_edge_injection':
        if len(controls) < 2:
            return _skip(sequence, anomaly_type, 'need_at_least_two_controls')
        src = _control_key(sequence.events[0].control_id)
        dst = next((control for control in controls if _control_key(control) != src), None)
        if dst is None:
            return _skip(sequence, anomaly_type, 'no_distinct_target_control')
        injected_event = _make_event_like(sequence, dst, anchor_index=0)
        injected_event.raw_fields['causal_attack'] = 'edge_injection'
        new_events = _insert_after(events, 0, [injected_event])
        return _make_sequence(sequence, new_events, anomaly_type, [f'injected_edge:{src}->{dst}']), _report(sequence, anomaly_type, 'injected', injected_edge=[src, dst])

    if anomaly_type == 'causal_edge_break':
        if len(events) < 2:
            return _skip(sequence, anomaly_type, 'need_at_least_two_events')
        removed = events.pop(1)
        return _make_sequence(sequence, events, anomaly_type, [f'removed_dependency_event:{removed.control_id}']), _report(sequence, anomaly_type, 'injected', removed_control=removed.control_id)

    if anomaly_type == 'lag_delay':
        if len(events) < 2:
            return _skip(sequence, anomaly_type, 'need_at_least_two_events')
        index = 1
        events[index].hour = _as_float(events[index].hour, 0.0) + 12.0
        events[index].duration = _as_float(events[index].duration, 0.0) + 12.0
        events[index].raw_fields = {**events[index].raw_fields, 'causal_attack': 'lag_delay'}
        return _make_sequence(sequence, events, anomaly_type, [f'delayed_event:{events[index].control_id}']), _report(sequence, anomaly_type, 'injected')

    if anomaly_type == 'context_causal_conflict':
        context = dict(sequence.context)
        context['context_id'] = 'causal_conflict'
        context['context_causal_conflict'] = True
        injected = _make_sequence(sequence, events, anomaly_type, ['context_id_forced_to_causal_conflict'])
        injected.context.update(context)
        return injected, _report(sequence, anomaly_type, 'injected')

    if anomaly_type == 'chain_break':
        if len(events) < 3:
            return _skip(sequence, anomaly_type, 'need_at_least_three_events')
        removed = events.pop(len(events) // 2)
        return _make_sequence(sequence, events, anomaly_type, [f'chain_middle_removed:{removed.control_id}']), _report(sequence, anomaly_type, 'injected', removed_control=removed.control_id)

    return _skip(sequence, anomaly_type, 'unhandled_causal_anomaly')


def _normal_copy(sequence: BehaviorSequence) -> BehaviorSequence:
    copied = copy.deepcopy(sequence)
    copied.label = 0
    copied.anomaly_type = None
    return copied


def _choose_spec(specs: List[Any], rng: random.Random) -> tuple[str, str]:
    spec = rng.choice(specs)
    if isinstance(spec, str):
        anomaly_type = spec
        family = 'causal' if anomaly_type in CAUSAL_ANOMALIES else 'smartguard'
        return family, anomaly_type
    if isinstance(spec, dict):
        anomaly_type = spec.get('anomaly_type') or spec.get('type')
        family = spec.get('family') or ('causal' if anomaly_type in CAUSAL_ANOMALIES else 'smartguard')
        return str(family), str(anomaly_type)
    raise ValueError(f'Unsupported anomaly spec: {spec!r}')


def generate_anomaly_dataset(
    normal_sequences: List[BehaviorSequence],
    anomaly_specs: List[Any],
    ratio: float,
    seed: int = 42,
) -> Tuple[List[BehaviorSequence], Dict[str, Any]]:
    '''Return normal+injected mixed sequences and a generation report.'''
    if ratio < 0:
        raise ValueError('ratio must be non-negative')
    if not anomaly_specs:
        raise ValueError('anomaly_specs must not be empty')

    rng = random.Random(seed)
    mixed = [_normal_copy(sequence) for sequence in normal_sequences]
    target_count = int(round(len(normal_sequences) * ratio)) if ratio <= 1.0 else int(round(ratio))
    injected_count = 0
    skipped: List[Dict[str, Any]] = []
    injected_reports: List[Dict[str, Any]] = []

    attempts = 0
    max_attempts = max(target_count * 5, len(normal_sequences), 1)
    while injected_count < target_count and attempts < max_attempts and normal_sequences:
        attempts += 1
        source = rng.choice(normal_sequences)
        family, anomaly_type = _choose_spec(anomaly_specs, rng)
        if family == 'causal':
            injected, report = inject_causal_anomaly(source, anomaly_type)
        else:
            injected, report = inject_smartguard_style(source, anomaly_type)
        if injected is None:
            skipped.append(report)
            continue
        mixed.append(injected)
        injected_reports.append(report)
        injected_count += 1

    rng.shuffle(mixed)
    report = {
        'normal_count': len(normal_sequences),
        'target_anomaly_count': target_count,
        'injected_count': injected_count,
        'skipped_count': len(skipped),
        'mixed_count': len(mixed),
        'attempts': attempts,
        'seed': seed,
        'injected': injected_reports,
        'skipped': skipped,
    }
    return mixed, report
