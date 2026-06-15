"""Unified data schema for CausalGenGuard.

This file is intentionally small and explicit.  Every adapter in the project
should convert its raw source data into these two dataclasses before model code
is called:

* BehaviorEvent    - one device/control event.
* BehaviorSequence - one ordered list of BehaviorEvent objects.

Why this matters:
SmartGuard stores behavior samples as numeric 10 x 4 arrays, SmartGen often
uses textual Device:action controls, and GCAD-style causal analysis needs an
event-channel tensor.  A shared schema prevents every module from inventing its
own input format.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping


def _json_safe(value: Any) -> Any:
    """Convert common Python / numpy objects into JSON-serializable values."""
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    if hasattr(value, "tolist"):
        try:
            return _json_safe(value.tolist())
        except Exception:
            pass
    return value


@dataclass
class BehaviorEvent:
    """One normalized IoT behavior event.

    Parameters
    ----------
    control_id:
        The main behavior token used by detectors.  It can be a numeric
        SmartGuard control id or a canonical string such as
        ``"Camera:switch off"``.
    timestamp / day / hour:
        At least one timestamp representation should exist.  SmartGuard data
        usually has day/hour ids instead of absolute timestamps.
    device_id:
        Optional device identifier.  It may be numeric or canonical text.
    duration:
        Optional duration or persistence feature.
    raw_fields:
        Lossless metadata from the source project.  Store non-standard fields
        here instead of changing the schema each time a new adapter appears.
    """

    control_id: Any
    timestamp: Any | None = None
    day: Any | None = None
    hour: Any | None = None
    device_id: Any | None = None
    duration: Any | None = None
    raw_fields: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        """Run lightweight structural validation.

        Validation is deliberately permissive because different source projects
        store time and ids differently.  It checks only requirements that every
        downstream module depends on.
        """
        if self.control_id is None:
            raise ValueError("BehaviorEvent.control_id is required")
        has_timestamp = self.timestamp is not None
        has_day_hour = self.day is not None and self.hour is not None
        if not has_timestamp and not has_day_hour:
            raise ValueError("BehaviorEvent requires timestamp or both day and hour")
        if self.raw_fields is None:
            raise ValueError("BehaviorEvent.raw_fields must be a dictionary")
        if not isinstance(self.raw_fields, dict):
            raise TypeError("BehaviorEvent.raw_fields must be a dictionary")

    @property
    def canonical_control(self) -> Any:
        """Return the best semantic control name for display/explanation."""
        value = self.raw_fields.get("canonical_control")
        if value not in (None, "", "unknown"):
            return value
        return self.control_id

    @property
    def action(self) -> Any | None:
        """Return the action part after ``Device:`` when available."""
        value = self.raw_fields.get("action")
        if value not in (None, "", "unknown"):
            return value
        canonical = self.canonical_control
        if isinstance(canonical, str) and ":" in canonical:
            return canonical.split(":", 1)[1]
        return None

    def with_raw_fields(self, **updates: Any) -> "BehaviorEvent":
        """Return a copy with extra raw fields.

        This helper is useful for attack injection and explanation code, where
        we want to record why an event was modified without losing source data.
        """
        data = self.to_dict(validate=False)
        raw_fields = dict(data.get("raw_fields") or {})
        raw_fields.update(updates)
        data["raw_fields"] = raw_fields
        return BehaviorEvent.from_dict(data, validate=False)

    def to_dict(self, validate: bool = True) -> dict[str, Any]:
        """Serialize this event to a JSON-friendly dictionary."""
        if validate:
            self.validate()
        return {
            "timestamp": _json_safe(self.timestamp),
            "day": _json_safe(self.day),
            "hour": _json_safe(self.hour),
            "device_id": _json_safe(self.device_id),
            "control_id": _json_safe(self.control_id),
            "duration": _json_safe(self.duration),
            "raw_fields": _json_safe(self.raw_fields),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any], validate: bool = True) -> "BehaviorEvent":
        """Create an event from a dictionary."""
        event = cls(
            timestamp=data.get("timestamp"),
            day=data.get("day"),
            hour=data.get("hour"),
            device_id=data.get("device_id"),
            control_id=data.get("control_id"),
            duration=data.get("duration"),
            raw_fields=dict(data.get("raw_fields") or {}),
        )
        if validate:
            event.validate()
        return event


@dataclass
class BehaviorSequence:
    """A normalized sequence of behavior events plus context and label."""

    sequence_id: str
    events: list[BehaviorEvent]
    context: dict[str, Any] = field(default_factory=dict)
    label: Any | None = None
    anomaly_type: str | None = None

    def validate(self) -> None:
        """Run lightweight structural validation."""
        if not self.sequence_id:
            raise ValueError("BehaviorSequence.sequence_id is required")
        if not isinstance(self.events, list):
            raise TypeError("BehaviorSequence.events must be a list")
        if not self.events:
            raise ValueError("BehaviorSequence.events must not be empty")
        if not isinstance(self.context, dict):
            raise TypeError("BehaviorSequence.context must be a dictionary")
        for event in self.events:
            if not isinstance(event, BehaviorEvent):
                raise TypeError("BehaviorSequence.events must contain BehaviorEvent objects")
            event.validate()

    def controls(self) -> list[Any]:
        """Return event control ids in sequence order."""
        return [event.control_id for event in self.events]

    def canonical_controls(self) -> list[Any]:
        """Return semantic control names when present, otherwise control ids."""
        return [event.canonical_control for event in self.events]

    def to_dict(self, validate: bool = True) -> dict[str, Any]:
        """Serialize this sequence to a JSON-friendly dictionary."""
        if validate:
            self.validate()
        return {
            "sequence_id": self.sequence_id,
            "events": [event.to_dict(validate=validate) for event in self.events],
            "context": _json_safe(self.context),
            "label": _json_safe(self.label),
            "anomaly_type": self.anomaly_type,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any], validate: bool = True) -> "BehaviorSequence":
        """Create a sequence from a dictionary."""
        sequence = cls(
            sequence_id=str(data.get("sequence_id") or ""),
            events=[BehaviorEvent.from_dict(item, validate=validate) for item in data.get("events", [])],
            context=dict(data.get("context") or {}),
            label=data.get("label"),
            anomaly_type=data.get("anomaly_type"),
        )
        if validate:
            sequence.validate()
        return sequence

    def copy_with(
        self,
        *,
        sequence_id: str | None = None,
        events: list[BehaviorEvent] | None = None,
        context_updates: Mapping[str, Any] | None = None,
        label: Any | None = None,
        anomaly_type: str | None = None,
    ) -> "BehaviorSequence":
        """Return a modified copy while preserving existing metadata."""
        new_context = dict(self.context)
        if context_updates:
            new_context.update(dict(context_updates))
        copied = BehaviorSequence(
            sequence_id=sequence_id or self.sequence_id,
            events=events if events is not None else list(self.events),
            context=new_context,
            label=self.label if label is None else label,
            anomaly_type=self.anomaly_type if anomaly_type is None else anomaly_type,
        )
        copied.validate()
        return copied
