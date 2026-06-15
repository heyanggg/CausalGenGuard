# Route B Dictionary-based Semantic Injection Status

## Goal

Upgrade SmartGuard-style anomaly injection from source-sequence-only semantic injection to dictionary-based semantic control-pool injection.

## Injection result

FR:
- normal_count: 3196
- anomaly_count: 450
- named_injection_success_count: 450
- source_sequence_template_injection_count: 44
- dictionary_control_pool_injection_count: 406
- fallback_numeric_injection_count: 0
- successful anomaly types: 9/10
- unsupported anomaly type: DD_shower_long_time
- skipped reason: control_not_found:shower

SP:
- normal_count: 11062
- anomaly_count: 450
- named_injection_success_count: 450
- source_sequence_template_injection_count: 44
- dictionary_control_pool_injection_count: 406
- fallback_numeric_injection_count: 0
- successful anomaly types: 9/10
- unsupported anomaly type: DD_shower_long_time
- skipped reason: control_not_found:shower

## Successful anomaly types

- DD_microwave_long_time: 50
- DM_ac_cool_in_winter: 50
- DM_watervalve_open_midnight: 50
- DM_window_open_midnight: 50
- MD_camera_off_while_lock: 50
- MD_window_open_while_lock: 50
- SD_camera_flickering: 50
- SD_light_flickering: 50
- SD_tv_flickering: 50

## Context-shift tiny smoke

Output:
- outputs/results/route_b_context_shift_tiny/summary.csv

Run status:
- all rows status=success
- missing_reason is empty for all rows

Tiny smoke bounds:
- target_normal_count: 50
- target_anomaly_count: 20
- synthetic_count: 20
- epochs: 1

Observed tiny-smoke FPR:
- source_only: 0.20
- source_plus_raw_synthetic: 0.06
- source_plus_tof_synthetic: 0.44
- source_plus_causal_tof_synthetic: 0.00
- oracle_target: 0.02

## Conclusion

Route B successfully enables dictionary-based semantic anomaly injection while keeping fallback_numeric_injection_count at 0. The supported SmartGuard-style anomaly set contains 9 semantic anomaly types for FR/SP. The Route B tiny context-shift smoke also runs end-to-end successfully.
