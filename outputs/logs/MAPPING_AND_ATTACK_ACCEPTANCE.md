# Mapping And Attack Acceptance

Date: 2026-06-06
Project path: /home/heyang/projects/CausalGenGuard

## Mapping

### FR

- mapping_report_exists: True
- device_count: 33
- control_count: 222
- missing_key_controls: []
- can_run_named_smartguard_attacks: True

### SP

- mapping_report_exists: True
- device_count: 34
- control_count: 234
- missing_key_controls: []
- can_run_named_smartguard_attacks: True

## Canonical FR Data

- fr_sequences_canonical_exists: True
- line_count: 100
- checked_sequence_ids: ['smartguard_fr_train_000000', 'smartguard_fr_train_000001', 'smartguard_fr_train_000002']
- required_fields: ['raw_device_id', 'raw_control_id', 'device', 'canonical_control', 'action', 'day_name', 'hour_name']
- first_3_required_fields_present: True
- missing_examples: []

## Labeled FR SmartGuard-Style Data

- fr_smartguard_style_labeled_jsonl_exists: True
- fr_smartguard_style_labeled_report_exists: True
- labeled_jsonl_line_count: 200
- normal_count: 100
- anomaly_count: 100
- named_injection_success_count: 100
- fallback_numeric_injection_count: 0
- skipped_count: 793
- per_anomaly_type_success: {'SD_light_flickering': 0, 'SD_camera_flickering': 0, 'SD_tv_flickering': 59, 'MD_camera_off_while_lock': 0, 'MD_window_open_while_lock': 0, 'DM_window_open_midnight': 41, 'DM_watervalve_open_midnight': 0, 'DD_microwave_long_time': 0}
- skipped_reasons: {'control_not_found:camera': 100, 'control_not_found:light': 100, 'control_not_found:microwave': 100, 'control_not_found:tv': 96, 'control_not_found:watervalve_open': 100, 'control_not_found:window_open': 97, 'required_control_not_found:lock_or_camera_off': 100, 'required_control_not_found:lock_or_window_open': 100}

## Tests

Command:

```bash
PYTHONPATH=src /home/heyang/miniconda3/envs/smartguard_env/bin/python -m pytest tests/test_smartguard_dictionary.py tests/test_named_attack_injection.py
```

Result: 9 passed in 0.10s

## Acceptance Decision

- passed: True
- conditions:
  - FR can_run_named_smartguard_attacks=True: True
  - SP mapping_report exists: True
  - labeled jsonl exists: True
  - labeled report exists: True
  - named_injection_success_count > 0: True
  - fallback_numeric_injection_count == 0: True
  - pytest passed: True
