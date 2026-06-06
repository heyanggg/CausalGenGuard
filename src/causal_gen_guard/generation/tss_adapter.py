'''Temporal/semantic sequence splitting adapter inspired by SmartGen TSS.

This is a lightweight offline implementation. It follows the SmartGen split.py
idea of breaking long behavior traces by time gap and total span, while keeping
obvious same-device on/off or open/close pairs together when possible.

TODO: Align interval units and action dictionaries with the original SmartGen
split.py once the exact dataset dictionaries are migrated.
'''

from __future__ import annotations

from typing import Any, List, Optional

from causal_gen_guard.data.schemas import BehaviorEvent, BehaviorSequence


_OPEN_ACTIONS = ('open', 'on', 'unlock', 'start', 'enable')
_CLOSE_ACTIONS = ('close', 'off', 'lock', 'stop', 'disable')


def _as_float(value: Any, default: Optional[float] = None) -> Optional[float]:
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


def _event_time(event: BehaviorEvent, fallback_index: int) -> float:
    timestamp = _as_float(event.timestamp)
    if timestamp is not None:
        return timestamp
    day = _as_float(event.day)
    hour = _as_float(event.hour)
    if day is not None and hour is not None:
        return day * 24.0 + hour
    return float(fallback_index)


def _device_key(event: BehaviorEvent) -> Any:
    if event.device_id is not None:
        return event.device_id
    for key in ('device', 'device_id', 'persistence_or_device'):
        if key in event.raw_fields:
            return event.raw_fields[key]
    return None


def _action_text(event: BehaviorEvent) -> str:
    return str(event.control_id).lower()


def _has_word(text: str, words: tuple[str, ...]) -> bool:
    return any(word in text for word in words)


def _is_same_device_pair(prev_event: BehaviorEvent, event: BehaviorEvent) -> bool:
    prev_device = _device_key(prev_event)
    cur_device = _device_key(event)
    if prev_device is not None and cur_device is not None and prev_device != cur_device:
        return False
    prev_action = _action_text(prev_event)
    cur_action = _action_text(event)
    return (
        _has_word(prev_action, _OPEN_ACTIONS) and _has_word(cur_action, _CLOSE_ACTIONS)
    ) or (
        _has_word(prev_action, _CLOSE_ACTIONS) and _has_word(cur_action, _OPEN_ACTIONS)
    )


def _make_segment(sequence: BehaviorSequence, events: List[BehaviorEvent], segment_index: int) -> BehaviorSequence:
    context = dict(sequence.context)
    context.update(
        {
            'source_sequence_id': sequence.sequence_id,
            'segment_index': segment_index,
            'split_adapter': 'smartgen_tss_simplified',
        }
    )
    return BehaviorSequence(
        sequence_id=f'{sequence.sequence_id}::tss_{segment_index:03d}',
        events=list(events),
        context=context,
        label=sequence.label,
        anomaly_type=sequence.anomaly_type,
    )


def split_sequences_by_time_and_semantics(
    sequences: List[BehaviorSequence],
    max_gap: float,
    max_span: float,
) -> List[BehaviorSequence]:
    '''Split sequences by gap/span while preserving obvious action pairs.'''
    if max_gap < 0:
        raise ValueError('max_gap must be non-negative')
    if max_span < 0:
        raise ValueError('max_span must be non-negative')

    segments: List[BehaviorSequence] = []
    for sequence in sequences:
        if not sequence.events:
            continue
        current_events: List[BehaviorEvent] = [sequence.events[0]]
        segment_start_time = _event_time(sequence.events[0], 0)
        prev_time = segment_start_time
        segment_index = 0

        for event_index, event in enumerate(sequence.events[1:], start=1):
            cur_time = _event_time(event, event_index)
            if cur_time < prev_time:
                cur_time += 168.0
            gap = cur_time - prev_time
            span = cur_time - segment_start_time
            should_split = (gap > max_gap) or (span > max_span)
            if should_split and not _is_same_device_pair(current_events[-1], event):
                segments.append(_make_segment(sequence, current_events, segment_index))
                segment_index += 1
                current_events = [event]
                segment_start_time = cur_time
            else:
                current_events.append(event)
            prev_time = cur_time

        if current_events:
            segments.append(_make_segment(sequence, current_events, segment_index))
    return segments
