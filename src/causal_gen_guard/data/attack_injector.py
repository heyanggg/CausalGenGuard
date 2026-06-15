"""Attack injection utilities for controlled behavior-anomaly benchmarks."""
from __future__ import annotations

import copy
import random
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple

from causal_gen_guard.data.schemas import BehaviorEvent, BehaviorSequence

SOURCE_SEQUENCE_TEMPLATE = "source_sequence_template"
DICTIONARY_CONTROL_POOL = "dictionary_control_pool"
FALLBACK_NUMERIC = "fallback_numeric"

SMARTGUARD_ANOMALIES = {
    "SD_light_flickering",
    "SD_camera_flickering",
    "SD_tv_flickering",
    "MD_window_open_while_lock",
    "MD_camera_off_while_lock",
    "DM_ac_cool_in_winter",
    "DM_window_open_midnight",
    "DM_watervalve_open_midnight",
    "DD_shower_long_time",
    "DD_microwave_long_time",
}

CAUSAL_ANOMALIES = {
    "causal_edge_break",
    "causal_edge_injection",
    "lag_delay",
    "context_causal_conflict",
    "chain_break",
}

_SPEC_CANONICAL_CONTROLS = {
    "light": ("Light:switch on", "Light:switch off", "Light:switch toggle"),
    "camera": ("Camera:switch on", "Camera:switch off"),
    "tv": ("Television:switch on", "Television:switch off", "Television:switch toggle"),
    "window_open": ("Blind:windowShade open", "Other:windowShade open", "Switch:windowShade open"),
    "lock": ("SmartLock:lock lock", "ContactSensor:lock lock"),
    "camera_off": ("Camera:switch off",),
    "ac_cool": (
        "AirConditioner:switch on",
        "AirConditioner:setCoolingSetpoint",
        "AirConditioner:setAirConditionerMode",
    ),
    "watervalve_open": ("WaterValve:valve open",),
    "shower": ("Shower:switch on", "Shower:water on", "Bathroom:shower on"),
    "microwave": ("Microwave:switch on",),
}

_SPEC_KEYWORDS = {
    "light": [("light", "lamp"), ("on", "off", "switch")],
    "camera": [("camera", "cam"), ("on", "off", "switch")],
    "tv": [("tv", "television"), ("on", "off", "switch")],
    "window_open": [("window", "blind", "curtain"), ("open", "up", "raise")],
    "lock": [("lock", "smartlock", "doorlock"), ("lock", "locked")],
    "camera_off": [("camera", "cam"), ("off", "disable")],
    "ac_cool": [("ac", "airconditioner", "air"), ("cool", "cold", "on")],
    "watervalve_open": [("water", "watervalve", "valve"), ("open", "on")],
    "shower": [("shower", "bath")],
    "microwave": [("microwave", "oven")],
}


@dataclass(frozen=True)
class _ResolvedControl:
    control: Any
    source: str
    raw_control_id: Optional[int] = None


_UNSET = object()


def _normalize_text(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(value).lower()).strip()


def _control_key(control_id: Any) -> Any:
    if hasattr(control_id, "item"):
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
    return not text or text.lower() == "unknown"


def _looks_numeric_only(value: Any) -> bool:
    return isinstance(value, (int, float)) or str(value).strip().isdigit()


def _raw_canonical_control(event: BehaviorEvent) -> Optional[Any]:
    raw_fields = event.raw_fields or {}
    canonical_control = raw_fields.get("canonical_control")
    if _is_unknown(canonical_control):
        return None
    return canonical_control


def _event_named_controls(event: BehaviorEvent) -> List[Any]:
    controls: List[Any] = []
    canonical_control = _raw_canonical_control(event)
    if canonical_control is not None:
        controls.append(canonical_control)
    # Avoid numeric fallback for SmartGuard-style named attacks.
    if not _is_unknown(event.control_id) and not _looks_numeric_only(event.control_id):
        controls.append(event.control_id)
    seen = set()
    unique: List[Any] = []
    for control in controls:
        key = repr(_control_key(control))
        if key not in seen:
            seen.add(key)
            unique.append(control)
    return unique


def _candidate_controls(sequence: BehaviorSequence) -> List[Any]:
    controls: List[Any] = []
    for event in sequence.events:
        controls.append(_control_key(event.control_id))
    inverse_vocab = sequence.context.get("inverse_vocab") or sequence.context.get("control_inverse_vocab")
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


def _matches_groups(control: Any, groups: List[Tuple[str, ...]]) -> bool:
    text = _normalize_text(control)
    return all(any(keyword in text for keyword in group) for group in groups)


def _find_control(sequence: BehaviorSequence, spec_name: str) -> Optional[Any]:
    resolved = _resolve_control(sequence, spec_name)
    return resolved.control if resolved is not None else None


def _find_sequence_control(sequence: BehaviorSequence, spec_name: str) -> Optional[Any]:
    canonical_controls = _SPEC_CANONICAL_CONTROLS.get(spec_name, ())
    canonical_targets = {_control_key(control) for control in canonical_controls}
    for event in sequence.events:
        for control in _event_named_controls(event):
            if _control_key(control) in canonical_targets:
                return control
    groups = _SPEC_KEYWORDS.get(spec_name)
    if groups is None:
        return None
    for event in sequence.events:
        for control in _event_named_controls(event):
            if _matches_groups(control, groups):
                return control
    return None


def _to_int_or_none(value: Any) -> Optional[int]:
    if value is None:
        return None
    if hasattr(value, "item"):
        try:
            value = value.item()
        except Exception:
            pass
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _is_canonical_control_name(value: Any) -> bool:
    if not isinstance(value, str):
        return False
    if _is_unknown(value) or _looks_numeric_only(value):
        return False
    device, action = _canonical_parts(value)
    return device is not None and action is not None


def _normalize_control_to_id(
    control_pool: Any = None,
    control_to_id: Optional[Mapping[Any, Any]] = None,
) -> Dict[str, Optional[int]]:
    normalized: Dict[str, Optional[int]] = {}
    if isinstance(control_pool, Mapping):
        for key, value in control_pool.items():
            control = str(key)
            if _is_canonical_control_name(control):
                normalized[control] = _to_int_or_none(value)
    elif control_pool is not None:
        for value in control_pool:
            control = str(value)
            if _is_canonical_control_name(control):
                normalized.setdefault(control, None)

    if control_to_id:
        for key, value in control_to_id.items():
            control = str(key)
            if _is_canonical_control_name(control):
                normalized[control] = _to_int_or_none(value)
    return normalized


def _find_pool_control(
    spec_name: str,
    control_pool: Any = None,
    control_to_id: Optional[Mapping[Any, Any]] = None,
) -> Optional[_ResolvedControl]:
    pool = _normalize_control_to_id(control_pool=control_pool, control_to_id=control_to_id)
    if not pool:
        return None

    for control in _SPEC_CANONICAL_CONTROLS.get(spec_name, ()):
        if control in pool:
            return _ResolvedControl(control=control, source=DICTIONARY_CONTROL_POOL, raw_control_id=pool[control])

    groups = _SPEC_KEYWORDS.get(spec_name)
    if groups is None:
        return None
    for control in sorted(pool):
        if _matches_groups(control, groups):
            return _ResolvedControl(control=control, source=DICTIONARY_CONTROL_POOL, raw_control_id=pool[control])
    return None


def _resolve_control(
    sequence: BehaviorSequence,
    spec_name: str,
    control_pool: Any = None,
    control_to_id: Optional[Mapping[Any, Any]] = None,
) -> Optional[_ResolvedControl]:
    control = _find_sequence_control(sequence, spec_name)
    if control is not None:
        return _ResolvedControl(control=control, source=SOURCE_SEQUENCE_TEMPLATE)
    return _find_pool_control(spec_name, control_pool=control_pool, control_to_id=control_to_id)


def _event_index_for_control(sequence: BehaviorSequence, control: Any) -> Optional[int]:
    for index, event in enumerate(sequence.events):
        for candidate in _event_named_controls(event):
            if _control_key(candidate) == _control_key(control):
                return index
    return None


def _canonical_parts(control: Any) -> Tuple[Optional[str], Optional[str]]:
    if _is_unknown(control):
        return None, None
    text = str(control)
    if ":" not in text:
        return None, None
    device, action = text.split(":", 1)
    if not device or not action:
        return None, None
    return device, action


def _as_float(value: Any, default: float = 0.0) -> float:
    if value is None:
        return default
    if hasattr(value, "item"):
        try:
            value = value.item()
        except Exception:
            pass
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _clone_event(
    event: BehaviorEvent,
    control_id: Any = None,
    raw_control_id: Any = _UNSET,
    injection_source: Optional[str] = None,
    **updates: Any,
) -> BehaviorEvent:
    cloned = copy.deepcopy(event)
    raw_fields = dict(cloned.raw_fields)
    if control_id is not None:
        cloned.control_id = control_id
        device, action = _canonical_parts(control_id)
        if device is not None and action is not None:
            cloned.device_id = device
            raw_fields["device"] = device
            raw_fields["action"] = action
            raw_fields["canonical_control"] = str(control_id)
            raw_fields["injected_canonical_control"] = str(control_id)
        if raw_control_id is not _UNSET:
            raw_fields["raw_control_id"] = raw_control_id
    for key, value in updates.items():
        setattr(cloned, key, value)
        if key in ("day", "hour", "duration"):
            raw_fields[key] = value
    if injection_source is not None:
        raw_fields["injection_source"] = injection_source
    raw_fields["attack_injected"] = True
    cloned.raw_fields = raw_fields
    return cloned


def _make_event_like(
    sequence: BehaviorSequence,
    control_id: Any,
    anchor_index: int = 0,
    raw_control_id: Any = _UNSET,
    injection_source: Optional[str] = None,
    **updates: Any,
) -> BehaviorEvent:
    if sequence.events:
        anchor = sequence.events[max(0, min(anchor_index, len(sequence.events) - 1))]
        return _clone_event(
            anchor,
            control_id=control_id,
            raw_control_id=raw_control_id,
            injection_source=injection_source,
            **updates,
        )
    raw_fields = {"attack_injected": True}
    device, action = _canonical_parts(control_id)
    if device is not None and action is not None:
        raw_fields.update({"canonical_control": str(control_id), "device": device, "action": action})
    if raw_control_id is not _UNSET:
        raw_fields["raw_control_id"] = raw_control_id
    if injection_source is not None:
        raw_fields["injection_source"] = injection_source
    return BehaviorEvent(control_id=control_id, raw_fields=raw_fields)


def _event_template_index(sequence: BehaviorSequence, resolved: _ResolvedControl) -> Optional[int]:
    if resolved.source != SOURCE_SEQUENCE_TEMPLATE:
        return None
    return _event_index_for_control(sequence, resolved.control)


def _make_resolved_event(
    sequence: BehaviorSequence,
    resolved: _ResolvedControl,
    default_anchor_index: int = 0,
    **updates: Any,
) -> BehaviorEvent:
    if resolved.source == DICTIONARY_CONTROL_POOL:
        return _make_dictionary_event(sequence, resolved, default_anchor_index=default_anchor_index, **updates)
    template_index = _event_template_index(sequence, resolved)
    anchor_index = template_index if template_index is not None else default_anchor_index
    return _make_event_like(
        sequence,
        resolved.control,
        anchor_index=anchor_index,
        injection_source=resolved.source,
        **updates,
    )


def _make_dictionary_event(
    sequence: BehaviorSequence,
    resolved: _ResolvedControl,
    default_anchor_index: int = 0,
    **updates: Any,
) -> BehaviorEvent:
    anchor: Optional[BehaviorEvent] = None
    if sequence.events:
        anchor = sequence.events[max(0, min(default_anchor_index, len(sequence.events) - 1))]
    device, action = _canonical_parts(resolved.control)
    raw_fields = {
        "canonical_control": str(resolved.control),
        "injected_canonical_control": str(resolved.control),
        "device": device,
        "action": action,
        "raw_control_id": resolved.raw_control_id,
        "attack_injected": True,
        "injection_source": resolved.source,
    }
    data = {
        "timestamp": anchor.timestamp if anchor is not None else None,
        "day": anchor.day if anchor is not None else None,
        "hour": anchor.hour if anchor is not None else None,
        "device_id": device,
        "control_id": resolved.control,
        "duration": anchor.duration if anchor is not None else None,
    }
    for key, value in updates.items():
        data[key] = value
        if key in ("day", "hour", "duration"):
            raw_fields[key] = value
    return BehaviorEvent(raw_fields=raw_fields, **data)


def _make_sequence(sequence: BehaviorSequence, events: List[BehaviorEvent], anomaly_type: str, operations: List[str]) -> BehaviorSequence:
    context = dict(sequence.context)
    context.update({"attack_injected": True, "attack_operations": operations, "source_sequence_id": sequence.sequence_id})
    return BehaviorSequence(
        sequence_id=f"{sequence.sequence_id}::{anomaly_type}",
        events=events,
        context=context,
        label=1,
        anomaly_type=anomaly_type,
    )


def _report(sequence: BehaviorSequence, anomaly_type: str, status: str, **extra: Any) -> Dict[str, Any]:
    payload = {"source_sequence_id": sequence.sequence_id, "anomaly_type": anomaly_type, "status": status}
    payload.update(extra)
    return payload


def _skip(sequence: BehaviorSequence, anomaly_type: str, reason: str) -> Tuple[None, Dict[str, Any]]:
    return None, _report(sequence, anomaly_type, "skipped", skipped_reason=reason)


def _insert_after(events: List[BehaviorEvent], index: int, new_events: Iterable[BehaviorEvent]) -> List[BehaviorEvent]:
    return list(events[: index + 1]) + list(new_events) + list(events[index + 1 :])


def _combined_injection_source(*resolved_controls: _ResolvedControl) -> str:
    if any(resolved.source == DICTIONARY_CONTROL_POOL for resolved in resolved_controls):
        return DICTIONARY_CONTROL_POOL
    return SOURCE_SEQUENCE_TEMPLATE


def _flicker(
    sequence: BehaviorSequence,
    anomaly_type: str,
    spec_name: str,
    control_pool: Any = None,
    control_to_id: Optional[Mapping[Any, Any]] = None,
) -> Tuple[Optional[BehaviorSequence], Dict[str, Any]]:
    resolved = _resolve_control(sequence, spec_name, control_pool=control_pool, control_to_id=control_to_id)
    if resolved is None:
        return _skip(sequence, anomaly_type, f"control_not_found:{spec_name}")
    control = resolved.control
    index = _event_template_index(sequence, resolved)
    if index is None:
        index = 0
    repeats = [_make_resolved_event(sequence, resolved, default_anchor_index=index) for _ in range(3)]
    events = _insert_after(copy.deepcopy(sequence.events), index, repeats)
    injected = _make_sequence(sequence, events, anomaly_type, [f"flicker:{control}"])
    return injected, _report(
        sequence,
        anomaly_type,
        "injected",
        operation="flicker",
        control_id=control,
        injection_source=resolved.source,
    )


def inject_smartguard_style(
    sequence: BehaviorSequence,
    anomaly_type: str,
    control_pool: Any = None,
    control_to_id: Optional[Mapping[Any, Any]] = None,
) -> Tuple[Optional[BehaviorSequence], Dict[str, Any]]:
    if anomaly_type not in SMARTGUARD_ANOMALIES:
        return _skip(sequence, anomaly_type, "unsupported_smartguard_anomaly")
    if not sequence.events:
        return _skip(sequence, anomaly_type, "empty_sequence")

    if anomaly_type == "SD_light_flickering":
        return _flicker(sequence, anomaly_type, "light", control_pool=control_pool, control_to_id=control_to_id)
    if anomaly_type == "SD_camera_flickering":
        return _flicker(sequence, anomaly_type, "camera", control_pool=control_pool, control_to_id=control_to_id)
    if anomaly_type == "SD_tv_flickering":
        return _flicker(sequence, anomaly_type, "tv", control_pool=control_pool, control_to_id=control_to_id)

    events = copy.deepcopy(sequence.events)

    if anomaly_type == "MD_window_open_while_lock":
        lock = _resolve_control(sequence, "lock", control_pool=control_pool, control_to_id=control_to_id)
        window = _resolve_control(sequence, "window_open", control_pool=control_pool, control_to_id=control_to_id)
        if lock is None or window is None:
            return _skip(sequence, anomaly_type, "required_control_not_found:lock_or_window_open")
        source = _combined_injection_source(lock, window)
        index = _event_template_index(sequence, lock)
        if index is None:
            lock_event = _make_resolved_event(sequence, lock, default_anchor_index=0)
            window_event = _make_resolved_event(sequence, window, default_anchor_index=0)
            return (
                _make_sequence(sequence, [lock_event, window_event] + events, anomaly_type, ["window_open_after_lock"]),
                _report(sequence, anomaly_type, "injected", injection_source=source),
            )
        injected_event = _make_resolved_event(sequence, window, default_anchor_index=index)
        return (
            _make_sequence(sequence, _insert_after(events, index, [injected_event]), anomaly_type, ["window_open_after_lock"]),
            _report(sequence, anomaly_type, "injected", injection_source=source),
        )

    if anomaly_type == "MD_camera_off_while_lock":
        lock = _resolve_control(sequence, "lock", control_pool=control_pool, control_to_id=control_to_id)
        camera_off = _resolve_control(sequence, "camera_off", control_pool=control_pool, control_to_id=control_to_id)
        if lock is None or camera_off is None:
            return _skip(sequence, anomaly_type, "required_control_not_found:lock_or_camera_off")
        source = _combined_injection_source(lock, camera_off)
        index = _event_template_index(sequence, lock)
        if index is None:
            lock_event = _make_resolved_event(sequence, lock, default_anchor_index=0)
            camera_event = _make_resolved_event(sequence, camera_off, default_anchor_index=0)
            return (
                _make_sequence(sequence, [lock_event, camera_event] + events, anomaly_type, ["camera_off_after_lock"]),
                _report(sequence, anomaly_type, "injected", injection_source=source),
            )
        injected_event = _make_resolved_event(sequence, camera_off, default_anchor_index=index)
        return (
            _make_sequence(sequence, _insert_after(events, index, [injected_event]), anomaly_type, ["camera_off_after_lock"]),
            _report(sequence, anomaly_type, "injected", injection_source=source),
        )

    if anomaly_type == "DM_ac_cool_in_winter":
        ac_cool = _resolve_control(sequence, "ac_cool", control_pool=control_pool, control_to_id=control_to_id)
        if ac_cool is None:
            return _skip(sequence, anomaly_type, "control_not_found:ac_cool")
        event = _make_resolved_event(sequence, ac_cool, default_anchor_index=0)
        new_sequence = _make_sequence(sequence, [event] + events, anomaly_type, ["ac_cool_forced_in_winter"])
        new_sequence.context["context_id"] = "winter"
        new_sequence.context["season"] = "winter"
        return new_sequence, _report(sequence, anomaly_type, "injected", injection_source=ac_cool.source)

    if anomaly_type == "DM_window_open_midnight":
        resolved = _resolve_control(sequence, "window_open", control_pool=control_pool, control_to_id=control_to_id)
        if resolved is None:
            return _skip(sequence, anomaly_type, "control_not_found:window_open")
        event = _make_resolved_event(sequence, resolved, default_anchor_index=0, hour=0)
        return (
            _make_sequence(sequence, [event] + events, anomaly_type, ["window_open_at_midnight"]),
            _report(sequence, anomaly_type, "injected", injection_source=resolved.source),
        )

    if anomaly_type == "DM_watervalve_open_midnight":
        resolved = _resolve_control(sequence, "watervalve_open", control_pool=control_pool, control_to_id=control_to_id)
        if resolved is None:
            return _skip(sequence, anomaly_type, "control_not_found:watervalve_open")
        event = _make_resolved_event(sequence, resolved, default_anchor_index=0, hour=0)
        return (
            _make_sequence(sequence, [event] + events, anomaly_type, ["watervalve_open_at_midnight"]),
            _report(sequence, anomaly_type, "injected", injection_source=resolved.source),
        )

    if anomaly_type == "DD_shower_long_time":
        resolved = _resolve_control(sequence, "shower", control_pool=control_pool, control_to_id=control_to_id)
        if resolved is None:
            return _skip(sequence, anomaly_type, "control_not_found:shower")
        control = resolved.control
        index = _event_template_index(sequence, resolved)
        if index is None:
            event = _make_resolved_event(sequence, resolved, default_anchor_index=0, duration=120.0)
            return (
                _make_sequence(sequence, [event] + events, anomaly_type, [f"long_duration:{control}"]),
                _report(sequence, anomaly_type, "injected", injection_source=resolved.source),
            )
        events[index].duration = max(_as_float(events[index].duration, 1.0) * 5.0, 120.0)
        events[index].raw_fields = {**events[index].raw_fields, "attack_injected": True, "duration_attack": "long_time", "injection_source": resolved.source}
        return (
            _make_sequence(sequence, events, anomaly_type, [f"long_duration:{control}"]),
            _report(sequence, anomaly_type, "injected", injection_source=resolved.source),
        )

    if anomaly_type == "DD_microwave_long_time":
        resolved = _resolve_control(sequence, "microwave", control_pool=control_pool, control_to_id=control_to_id)
        if resolved is None:
            return _skip(sequence, anomaly_type, "control_not_found:microwave")
        control = resolved.control
        index = _event_template_index(sequence, resolved)
        if index is None:
            event = _make_resolved_event(sequence, resolved, default_anchor_index=0, duration=120.0)
            return (
                _make_sequence(sequence, [event] + events, anomaly_type, [f"long_duration:{control}"]),
                _report(sequence, anomaly_type, "injected", injection_source=resolved.source),
            )
        events[index].duration = max(_as_float(events[index].duration, 1.0) * 5.0, 120.0)
        events[index].raw_fields = {**events[index].raw_fields, "attack_injected": True, "duration_attack": "long_time", "injection_source": resolved.source}
        return (
            _make_sequence(sequence, events, anomaly_type, [f"long_duration:{control}"]),
            _report(sequence, anomaly_type, "injected", injection_source=resolved.source),
        )

    return _skip(sequence, anomaly_type, "unhandled_smartguard_anomaly")


def _unique_controls(sequence: BehaviorSequence) -> List[Any]:
    controls = []
    seen = set()
    for control in _candidate_controls(sequence):
        key = repr(control)
        if key not in seen:
            seen.add(key)
            controls.append(control)
    return controls


def inject_causal_anomaly(
    sequence: BehaviorSequence,
    anomaly_type: str,
    control_pool: Any = None,
    control_to_id: Optional[Mapping[Any, Any]] = None,
) -> Tuple[Optional[BehaviorSequence], Dict[str, Any]]:
    del control_pool, control_to_id
    if anomaly_type not in CAUSAL_ANOMALIES:
        return _skip(sequence, anomaly_type, "unsupported_causal_anomaly")
    if not sequence.events:
        return _skip(sequence, anomaly_type, "empty_sequence")
    events = copy.deepcopy(sequence.events)
    controls = _unique_controls(sequence)

    if anomaly_type == "causal_edge_injection":
        if len(controls) < 2:
            return _skip(sequence, anomaly_type, "need_at_least_two_controls")
        src = _control_key(sequence.events[0].control_id)
        dst = next((control for control in controls if _control_key(control) != src), None)
        if dst is None:
            return _skip(sequence, anomaly_type, "no_distinct_target_control")
        injected_event = _make_event_like(sequence, dst, anchor_index=0)
        injected_event.raw_fields["causal_attack"] = "edge_injection"
        return _make_sequence(sequence, _insert_after(events, 0, [injected_event]), anomaly_type, [f"injected_edge:{src}->{dst}"]), _report(sequence, anomaly_type, "injected", injected_edge=[src, dst])

    if anomaly_type == "causal_edge_break":
        if len(events) < 2:
            return _skip(sequence, anomaly_type, "need_at_least_two_events")
        removed = events.pop(1)
        return _make_sequence(sequence, events, anomaly_type, [f"removed_dependency_event:{removed.control_id}"]), _report(sequence, anomaly_type, "injected", removed_control=removed.control_id)

    if anomaly_type == "lag_delay":
        if len(events) < 2:
            return _skip(sequence, anomaly_type, "need_at_least_two_events")
        index = 1
        events[index].hour = _as_float(events[index].hour, 0.0) + 12.0
        events[index].duration = _as_float(events[index].duration, 0.0) + 12.0
        events[index].raw_fields = {**events[index].raw_fields, "causal_attack": "lag_delay"}
        return _make_sequence(sequence, events, anomaly_type, [f"delayed_event:{events[index].control_id}"]), _report(sequence, anomaly_type, "injected")

    if anomaly_type == "context_causal_conflict":
        injected = _make_sequence(sequence, events, anomaly_type, ["context_id_forced_to_causal_conflict"])
        injected.context["context_id"] = "causal_conflict"
        injected.context["context_causal_conflict"] = True
        return injected, _report(sequence, anomaly_type, "injected")

    if anomaly_type == "chain_break":
        if len(events) < 3:
            return _skip(sequence, anomaly_type, "need_at_least_three_events")
        removed = events.pop(len(events) // 2)
        return _make_sequence(sequence, events, anomaly_type, [f"chain_middle_removed:{removed.control_id}"]), _report(sequence, anomaly_type, "injected", removed_control=removed.control_id)

    return _skip(sequence, anomaly_type, "unhandled_causal_anomaly")


def _normal_copy(sequence: BehaviorSequence) -> BehaviorSequence:
    copied = copy.deepcopy(sequence)
    copied.label = 0
    copied.anomaly_type = None
    return copied


def _choose_spec(specs: List[Any], rng: random.Random) -> Tuple[str, str]:
    spec = rng.choice(specs)
    if isinstance(spec, str):
        anomaly_type = spec
        family = "causal" if anomaly_type in CAUSAL_ANOMALIES else "smartguard"
        return family, anomaly_type
    if isinstance(spec, dict):
        anomaly_type = spec.get("anomaly_type") or spec.get("type")
        family = spec.get("family") or ("causal" if anomaly_type in CAUSAL_ANOMALIES else "smartguard")
        return str(family), str(anomaly_type)
    raise ValueError(f"Unsupported anomaly spec: {spec!r}")


def generate_anomaly_dataset(
    normal_sequences: List[BehaviorSequence],
    anomaly_specs: List[Any],
    ratio: float,
    seed: int = 42,
) -> Tuple[List[BehaviorSequence], Dict[str, Any]]:
    if ratio < 0:
        raise ValueError("ratio must be non-negative")
    if not anomaly_specs:
        raise ValueError("anomaly_specs must not be empty")
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
        if family == "causal":
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
        "normal_count": len(normal_sequences),
        "target_anomaly_count": target_count,
        "injected_count": injected_count,
        "skipped_count": len(skipped),
        "mixed_count": len(mixed),
        "attempts": attempts,
        "seed": seed,
        "injected": injected_reports,
        "skipped": skipped,
    }
    return mixed, report
