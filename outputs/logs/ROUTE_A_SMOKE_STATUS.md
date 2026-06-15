# Route A Smoke Status

## Goal

Use existing FR/SP canonical and labeled anomaly datasets to verify that the context-shift pipeline can run end-to-end without modifying the injection logic.

## Semantic Injection Data

FR:
- canonical: outputs/processed/fr_sequences_canonical.jsonl
- labeled: outputs/labels/fr_smartguard_style_labeled.jsonl
- normal_count: 3196
- anomaly_count: 250
- named_injection_success_count: 250
- fallback_numeric_injection_count: 0

SP:
- canonical: outputs/processed/sp_sequences_canonical.jsonl
- labeled: outputs/labels/sp_smartguard_style_labeled.jsonl
- normal_count: 11062
- anomaly_count: 201
- named_injection_success_count: 201
- fallback_numeric_injection_count: 0

## Context-shift tiny smoke

Output:
- outputs/results/route_a_context_shift_tiny/summary.csv
- outputs/results/route_a_context_shift_tiny/summary.json

Run status:
- successful_runs: 1
- missing_runs: 0

Bounds:
- max_source_train: 50
- max_source_val: 50
- max_target_normal: 50
- max_target_anomaly: 20
- max_synthetic: 20
- epochs: 1

Observed tiny-smoke result:
- source_only target_normal_fpr: 0.20
- source_plus_raw_synthetic target_normal_fpr: 0.06
- source_plus_tof_synthetic target_normal_fpr: 0.44
- source_plus_causal_tof_synthetic target_normal_fpr: 0.00
- oracle_target target_normal_fpr: 0.02

## Interpretation

Route A tiny smoke successfully verifies that the context-shift pipeline can read the semantic SmartGuard data, generate/adapt synthetic target-context data, run filtering baselines, evaluate target-context anomalies, and write summary outputs.

The tiny-smoke metrics are not suitable as final paper results because the run uses very small bounds. They should only be used as a pipeline validation record.
