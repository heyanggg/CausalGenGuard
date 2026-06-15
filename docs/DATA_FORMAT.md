# CausalGenGuard Data Format

All adapters should output JSONL files.  Each line is one `BehaviorSequence`.

## BehaviorEvent

```json
{
  "timestamp": null,
  "day": 1,
  "hour": 12,
  "device_id": 29,
  "control_id": 202,
  "duration": 3,
  "raw_fields": {
    "raw_device_id": 29,
    "raw_control_id": 202,
    "device": "Television",
    "canonical_control": "Television:switch on",
    "action": "switch on",
    "day_name": "weekday",
    "hour_name": "time:(12~15)"
  }
}
```

## BehaviorSequence

```json
{
  "sequence_id": "fr_000001",
  "events": [
    {"day": 1, "hour": 12, "device_id": 29, "control_id": 202, "duration": 3, "raw_fields": {}}
  ],
  "context": {
    "dataset": "fr",
    "source": "SmartGuard",
    "context_id": "winter"
  },
  "label": 0,
  "anomaly_type": "normal"
}
```

## Label convention

```text
label = 0: normal
label = 1: anomaly
anomaly_type = normal | SD_light_flickering | causal_edge_injection | ...
```

## Mapping files

`scripts/build_smartguard_mapping.py` writes:

```text
device_to_id.json
id_to_device.json
control_to_id.json
id_to_control.json
day_to_id.json
id_to_day.json
hour_to_id.json
id_to_hour.json
mapping_report.json
```

These files make numeric SmartGuard ids readable and allow SmartGen textual
controls to align with SmartGuard ids.
