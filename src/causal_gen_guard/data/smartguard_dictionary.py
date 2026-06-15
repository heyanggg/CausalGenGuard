"""Utilities for loading SmartGuard device/control dictionaries.

The original SmartGuard dictionaries are Python files.  This module parses their
literal assignments with ``ast`` instead of importing the source project.  That
keeps SmartGuard read-only and makes the mapping generation deterministic.
"""
from __future__ import annotations

import ast
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional

KEY_CONTROLS = [
    "Light:switch on",
    "Light:switch off",
    "Camera:switch on",
    "Camera:switch off",
    "Television:switch on",
    "Television:switch off",
    "Blind:windowShade open",
    "SmartLock:lock lock",
    "SmartLock:lock unlock",
    "WaterValve:valve open",
    "WaterValve:valve close",
    "AirConditioner:setCoolingSetpoint",
    "Microwave:switch on",
]

DEFAULT_DAYOFWEEK_DICT = {
    "day:Mon": 0,
    "day:Tue": 1,
    "day:Wed": 2,
    "day:Thu": 3,
    "day:Fri": 4,
    "day:Sat": 5,
    "day:Sun": 6,
}

DEFAULT_HOUR_DICT = {
    "time:(0~3)": 0,
    "time:(3~6)": 1,
    "time:(6~9)": 2,
    "time:(9~12)": 3,
    "time:(12~15)": 4,
    "time:(15~18)": 5,
    "time:(18~21)": 6,
    "time:(21~24)": 7,
}


def _literal_assignments(path: Path) -> Dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    tree = ast.parse(text, filename=str(path))
    values: Dict[str, Any] = {}
    for node in tree.body:
        if isinstance(node, ast.Assign):
            try:
                value = ast.literal_eval(node.value)
            except Exception:
                continue
            for target in node.targets:
                if isinstance(target, ast.Name):
                    values[target.id] = value
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            try:
                values[node.target.id] = ast.literal_eval(node.value)
            except Exception:
                continue
    return values


def _to_int(value: Any) -> int:
    if hasattr(value, "item"):
        value = value.item()
    return int(value)


def _normalize_mapping(mapping: Optional[Mapping[Any, Any]]) -> Dict[str, int]:
    if not mapping:
        return {}
    out: Dict[str, int] = {}
    for key, value in mapping.items():
        out[str(key)] = _to_int(value)
    return out


def invert_dict(mapping: Mapping[Any, Any]) -> Dict[int, str]:
    """Invert a name->id mapping into int id->name."""
    return {_to_int(value): str(key) for key, value in mapping.items()}


def _sort_name_to_id(mapping: Mapping[str, int]) -> Dict[str, int]:
    return {key: int(mapping[key]) for key in sorted(mapping)}


def _sort_id_to_name(mapping: Mapping[int, str]) -> Dict[str, str]:
    return {str(key): mapping[key] for key in sorted(mapping)}


@dataclass(frozen=True)
class SmartGuardDictionary:
    """Bidirectional SmartGuard mappings for one dataset."""

    dataset: str
    source_path: Path
    device_to_id: Dict[str, int]
    id_to_device: Dict[int, str]
    control_to_id: Dict[str, int]
    id_to_control: Dict[int, str]
    day_to_id: Dict[str, int]
    id_to_day: Dict[int, str]
    hour_to_id: Dict[str, int]
    id_to_hour: Dict[int, str]

    @property
    def dictionary_path(self) -> Path:
        """Backward-compatible alias used by older scripts/tests."""
        return self.source_path

    def json_payloads(self) -> Dict[str, Dict[str, Any]]:
        return {
            "device_to_id": _sort_name_to_id(self.device_to_id),
            "id_to_device": _sort_id_to_name(self.id_to_device),
            "control_to_id": _sort_name_to_id(self.control_to_id),
            "id_to_control": _sort_id_to_name(self.id_to_control),
            "day_to_id": _sort_name_to_id(self.day_to_id),
            "id_to_day": _sort_id_to_name(self.id_to_day),
            "hour_to_id": _sort_name_to_id(self.hour_to_id),
            "id_to_hour": _sort_id_to_name(self.id_to_hour),
        }


def load_dictionary_py(path: str | Path) -> Dict[str, Dict[str, int]]:
    """Load literal dictionaries from a SmartGuard dictionary.py file."""
    path = Path(path)
    values = _literal_assignments(path)
    return {
        "dayofweek_dict": _normalize_mapping(values.get("dayofweek_dict") or DEFAULT_DAYOFWEEK_DICT),
        "hour_dict": _normalize_mapping(values.get("hour_dict") or DEFAULT_HOUR_DICT),
        "device_dict": _normalize_mapping(values.get("device_dict")),
        "device_control_dict": _normalize_mapping(values.get("device_control_dict")),
    }


def parse_smartguard_dictionary(path: str | Path, dataset: str = "unknown") -> SmartGuardDictionary:
    path = Path(path)
    raw = load_dictionary_py(path)
    device_to_id = raw["device_dict"]
    control_to_id = raw["device_control_dict"]
    day_to_id = raw["dayofweek_dict"] or DEFAULT_DAYOFWEEK_DICT
    hour_to_id = raw["hour_dict"] or DEFAULT_HOUR_DICT
    return SmartGuardDictionary(
        dataset=dataset,
        source_path=path,
        device_to_id=device_to_id,
        id_to_device=invert_dict(device_to_id),
        control_to_id=control_to_id,
        id_to_control=invert_dict(control_to_id),
        day_to_id=day_to_id,
        id_to_day=invert_dict(day_to_id),
        hour_to_id=hour_to_id,
        id_to_hour=invert_dict(hour_to_id),
    )


def load_smartguard_dictionary(smartguard_root: str | Path, dataset: str) -> SmartGuardDictionary:
    dataset = dataset.lower()
    path = Path(smartguard_root) / "data" / "data" / dataset / "dictionary.py"
    if not path.exists():
        raise FileNotFoundError(f"SmartGuard dictionary not found: {path}")
    return parse_smartguard_dictionary(path, dataset=dataset)


def load_smartguard_mappings(smartguard_root: str | Path, dataset: str) -> SmartGuardDictionary:
    return load_smartguard_dictionary(smartguard_root, dataset)


def parse_control_name(control_name: str) -> Dict[str, str]:
    text = str(control_name)
    if ":" in text:
        device, action = text.split(":", 1)
    else:
        device, action = "unknown", text
    return {
        "device": device,
        "action": action,
        "canonical_control": f"{device}:{action}" if device != "unknown" else text,
    }


def _norm(text: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(text).lower())


def lookup_control_id(control_to_id: Mapping[str, int], canonical_control: str, fuzzy: bool = False) -> Dict[str, Any]:
    if canonical_control in control_to_id:
        return {
            "control_id": int(control_to_id[canonical_control]),
            "matched_control": canonical_control,
            "confidence": "exact",
        }
    if fuzzy:
        target = _norm(canonical_control)
        for name, value in control_to_id.items():
            if _norm(name) == target:
                return {
                    "control_id": int(value),
                    "matched_control": name,
                    "confidence": "normalized",
                }
    return {"control_id": None, "matched_control": None, "confidence": "missing"}


def build_mapping_report(
    mapping: SmartGuardDictionary,
    key_controls: Optional[Iterable[str]] = None,
) -> Dict[str, Any]:
    controls = list(key_controls or KEY_CONTROLS)
    key_payload: Dict[str, Dict[str, Any]] = {}
    missing: List[str] = []
    for control in controls:
        exists = control in mapping.control_to_id
        if not exists:
            missing.append(control)
        key_payload[control] = {
            "exists": exists,
            "id": int(mapping.control_to_id[control]) if exists else None,
        }
    all_present = len(missing) == 0
    return {
        "dataset": mapping.dataset,
        "dictionary_path": str(mapping.source_path),
        "device_count": len(mapping.device_to_id),
        "control_count": len(mapping.control_to_id),
        "day_count": len(mapping.day_to_id),
        "hour_count": len(mapping.hour_to_id),
        "key_controls": key_payload,
        "missing_key_controls": missing,
        "all_key_controls_present": all_present,
        "can_run_named_smartguard_attacks": all_present,
    }


def write_mapping_files(
    mapping: SmartGuardDictionary,
    output_dir: str | Path,
    include_time_mappings: bool = False,
) -> Dict[str, Any]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    payloads = mapping.json_payloads()
    names = ["device_to_id", "id_to_device", "control_to_id", "id_to_control"]
    if include_time_mappings:
        names.extend(["day_to_id", "id_to_day", "hour_to_id", "id_to_hour"])
    for name in names:
        (output_dir / f"{name}.json").write_text(
            json.dumps(payloads[name], ensure_ascii=False, indent=2, sort_keys=True),
            encoding="utf-8",
        )
    report = build_mapping_report(mapping)
    (output_dir / "mapping_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return report
