# Experiment Summary

## Route B Semantic Injection

FR:
- normal_count: 3196
- anomaly_count: 450
- named_injection_success_count: 450
- source_sequence_template_injection_count: 44
- dictionary_control_pool_injection_count: 406
- fallback_numeric_injection_count: 0
- supported anomaly types: 9/10
- unsupported: DD_shower_long_time

SP:
- normal_count: 11062
- anomaly_count: 450
- named_injection_success_count: 450
- source_sequence_template_injection_count: 44
- dictionary_control_pool_injection_count: 406
- fallback_numeric_injection_count: 0
- supported anomaly types: 9/10
- unsupported: DD_shower_long_time

## Three-seed FR/SP Seasonal Context-shift Results

Setting:
- max-normal: 600
- max-anomaly: 200
- max-synthetic: 200
- epochs: 2
- seeds: 42, 43, 44

FR FPR:
- source_only: 0.4193 ± 0.0893
- source_plus_raw_synthetic: 0.3505 ± 0.0502
- source_plus_tof_synthetic: 0.3175 ± 0.0525
- source_plus_causal_tof_synthetic: 0.3214 ± 0.0757
- oracle_target: 0.0529 ± 0.0165

SP FPR:
- source_only: 0.8507 ± 0.0431
- source_plus_raw_synthetic: 0.4993 ± 0.1212
- source_plus_tof_synthetic: 0.5687 ± 0.2470
- source_plus_causal_tof_synthetic: 0.6033 ± 0.0514
- oracle_target: 0.0093 ± 0.0050

## Filtering Diagnostic

Vanilla TOF with iqr_1.5 rejects very few synthetic samples:
- FR: keeps 199/200, 199/200, 200/200 across seeds.
- SP: keeps 198/200 across seeds.

Causal-TOF uses relaxed keep-ratio filtering and rejects about 10%:
- FR: keeps 180/200 across seeds.
- SP: keeps 179/200 across seeds.

## Interpretation

The stable conclusion is that source-only models suffer from high target-context false positive rates under seasonal context shift, while target-context synthetic adaptation reduces FPR on both FR and SP.

TOF and Causal-TOF are filtering ablations with dataset-dependent behavior. Current evidence does not support claiming that Causal-TOF is uniformly best.
