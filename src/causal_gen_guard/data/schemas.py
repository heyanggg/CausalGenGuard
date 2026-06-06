'''Shared schemas for behavior events, contexts, labels, and anomaly scores.

The schema keeps raw source fields next to normalized fields so adapters can
preserve SmartGuard, SmartGen, and GCAD provenance while the tensor format is
still settling.
'''

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping


def _json_safe(value: Any) -> Any:
    '''Convert common scalar and array-like objects into JSON-safe values.'''
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if hasattr(value, 'item'):
        try:
            return value.item()
        except Exception:
            pass
    if hasattr(value, 'tolist'):
        try:
            return _json_safe(value.tolist())
        except Exception:
            pass
    return value


@dataclass
class BehaviorEvent:
    '''One normalized IoT behavior event.'''

    control_id: Any
    timestamp: Any | None = None
    day: Any | None = None
    hour: Any | None = None
    device_id: Any | None = None
    duration: Any | None = None
    raw_fields: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        '''Run lightweight structural validation.'''
        if self.control_id is None:
            raise ValueError('BehaviorEvent.control_id is required')
        has_timestamp = self.timestamp is not None
        has_day_hour = self.day is not None and self.hour is not None
        if not has_timestamp and not has_day_hour:
            raise ValueError('BehaviorEvent requires timestamp or both day and hour')
        if self.raw_fields is None:
            raise ValueError('BehaviorEvent.raw_fields must be a dictionary')
        if not isinstance(self.raw_fields, dict):
            raise TypeError('BehaviorEvent.raw_fields must be a dictionary')

    def to_dict(self) -> dict[str, Any]:
        '''Serialize this event to a JSON-friendly dictionary.'''
        self.validate()
        return {
            'timestamp': _json_safe(self.timestamp),
            'day': _json_safe(self.day),
            'hour': _json_safe(self.hour),
            'device_id': _json_safe(self.device_id),
            'control_id': _json_safe(self.control_id),
            'duration': _json_safe(self.duration),
            'raw_fields': _json_safe(self.raw_fields),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> BehaviorEvent:
        '''Create an event from a dictionary and validate it.'''
        event = cls(
            timestamp=data.get('timestamp'),
            day=data.get('day'),
            hour=data.get('hour'),
            device_id=data.get('device_id'),
            control_id=data.get('control_id'),
            duration=data.get('duration'),
            raw_fields=dict(data.get('raw_fields') or {}),
        )
        event.validate()
        return event


@dataclass
class BehaviorSequence:
    '''A normalized sequence of behavior events plus context and optional label.'''

    sequence_id: str
    events: list[BehaviorEvent]
    context: dict[str, Any] = field(default_factory=dict)
    label: Any | None = None
    anomaly_type: str | None = None

    def validate(self) -> None:
        '''Run lightweight structural validation.'''
        if not self.sequence_id:
            raise ValueError('BehaviorSequence.sequence_id is required')
        if not isinstance(self.events, list):
            raise TypeError('BehaviorSequence.events must be a list')
        if not self.events:
            raise ValueError('BehaviorSequence.events must not be empty')
        if not isinstance(self.context, dict):
            raise TypeError('BehaviorSequence.context must be a dictionary')
        for event in self.events:
            if not isinstance(event, BehaviorEvent):
                raise TypeError('BehaviorSequence.events must contain BehaviorEvent objects')
            event.validate()

    def to_dict(self) -> dict[str, Any]:
        '''Serialize this sequence to a JSON-friendly dictionary.'''
        self.validate()
        return {
            'sequence_id': self.sequence_id,
            'events': [event.to_dict() for event in self.events],
            'context': _json_safe(self.context),
            'label': _json_safe(self.label),
            'anomaly_type': self.anomaly_type,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> BehaviorSequence:
        '''Create a sequence from a dictionary and validate it.'''
        sequence = cls(
            sequence_id=str(data.get('sequence_id') or ''),
            events=[BehaviorEvent.from_dict(item) for item in data.get('events', [])],
            context=dict(data.get('context') or {}),
            label=data.get('label'),
            anomaly_type=data.get('anomaly_type'),
        )
        sequence.validate()
        return sequence
