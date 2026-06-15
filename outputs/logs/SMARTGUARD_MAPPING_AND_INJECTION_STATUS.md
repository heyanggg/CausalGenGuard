# SmartGuard Mapping and Named Injection Status

## Mapping

FR:
- device_count: 33
- control_count: 222
- all_key_controls_present: true
- can_run_named_smartguard_attacks: true

SP:
- device_count: 34
- control_count: 234
- all_key_controls_present: true
- can_run_named_smartguard_attacks: true

AN:
- device_count: 37
- control_count: 141
- all_key_controls_present: false
- can_run_named_smartguard_attacks: false
- Note: AN dictionary is parsed successfully, but it does not contain the FR/SP SmartSense-style key controls required by the current named attack set.

## FR Canonical and Named Injection

- normal_count: 3196
- anomaly_count: 250
- named_injection_success_count: 250
- fallback_numeric_injection_count: 0
- successful anomaly types:
  - SD_light_flickering: 50
  - SD_camera_flickering: 50
  - SD_tv_flickering: 50
  - DM_ac_cool_in_winter: 50
  - DM_window_open_midnight: 50
- skipped or unsupported under source-normal-only injection:
  - MD_window_open_while_lock
  - MD_camera_off_while_lock
  - DM_watervalve_open_midnight
  - DD_shower_long_time
  - DD_microwave_long_time

## SP Canonical and Named Injection

- normal_count: 11062
- anomaly_count: 201
- named_injection_success_count: 201
- fallback_numeric_injection_count: 0
- successful anomaly types:
  - SD_light_flickering: 50
  - SD_camera_flickering: 42
  - SD_tv_flickering: 50
  - DM_ac_cool_in_winter: 50
  - DD_microwave_long_time: 9
- skipped or unsupported under source-normal-only injection:
  - MD_window_open_while_lock
  - MD_camera_off_while_lock
  - DM_window_open_midnight
  - DM_watervalve_open_midnight
  - DD_shower_long_time

## Conclusion

The semantic mapping pipeline is fixed. FR/SP numeric SmartGuard controls are now converted to canonical Device:action controls. Named SmartGuard-style semantic injection works without numeric fallback.

The current labeled datasets are suitable for smoke tests and feasibility experiments. For the final paper-scale experiment, we should either:
1. use only semantically supported anomaly types per dataset, or
2. implement dictionary-based semantic control-pool injection so that legal controls from dictionary.py can be injected even when they do not appear in a sampled normal sequence.
