# Final Context Shift Run Report

Date: 2026-06-06
Project path: `/home/heyang/projects/CausalGenGuard`

## Run Status

- Ran through: `True`
- Command:

```bash
PYTHONPATH=src /home/heyang/miniconda3/envs/smartguard_env/bin/python scripts/run_context_shift_final.py --config configs/context_shift_fr.yaml
```

## Contexts

- Source context: `winter`
- Target context: `spring`

## Synthetic Data

- synthetic_count: `500`

## Selected Method Results

| method | filter_strategy | kept_count | rejected_count | target_normal_fpr | anomaly_f1 | auroc | auprc |
| --- | --- | ---: | ---: | ---: | --- | --- | --- |
| source_only |  | 0 | 0 | 0.599206 | skipped | skipped | skipped |
| source_plus_raw_synthetic | no_filter | 500 | 0 | 0.305556 | skipped | skipped | skipped |
| source_plus_tof_synthetic | iqr_1.5 | 490 | 10 | 0.142857 | skipped | skipped | skipped |
| source_plus_causal_tof_synthetic | relaxed_causal_keep_90_percent | 441 | 59 | 0.138889 | skipped | skipped | skipped |
| oracle_target |  | 0 | 0 | 0.083333 | skipped | skipped | skipped |

## Filter Strategy Sweep

| method | filter_strategy | selected | kept_count | rejected_count | target_normal_fpr |
| --- | --- | --- | ---: | ---: | ---: |
| source_plus_tof_synthetic | no_filter | no | 500 | 0 | 0.305556 |
| source_plus_tof_synthetic | iqr_1.5 | yes | 490 | 10 | 0.142857 |
| source_plus_tof_synthetic | iqr_3.0 | no | 490 | 10 | 0.142857 |
| source_plus_tof_synthetic | keep_top_80_percent_by_low_loss | no | 392 | 108 | 0.384921 |
| source_plus_tof_synthetic | keep_top_90_percent_by_low_loss | no | 441 | 59 | 0.392857 |
| source_plus_causal_tof_synthetic | tof_only | no | 490 | 10 | 0.142857 |
| source_plus_causal_tof_synthetic | relaxed_causal_keep_90_percent | yes | 441 | 59 | 0.138889 |
| source_plus_causal_tof_synthetic | relaxed_causal_keep_95_percent | no | 466 | 34 | 0.365079 |
| source_plus_causal_tof_synthetic | causal_filter_disabled | no | 490 | 10 | 0.142857 |

## Metric Availability

- `target_normal_fpr`: available for all methods.
- `anomaly_f1`: skipped.
- `auroc`: skipped.
- `auprc`: skipped.
- Skip reason: no target-context anomaly data was found for the selected target context, so anomaly-label metrics were not fabricated.

## Best Method By Target-Normal FPR

- Best non-oracle method: `source_plus_causal_tof_synthetic`
- Best non-oracle filter_strategy: `relaxed_causal_keep_90_percent`
- Best non-oracle target_normal_fpr: `0.138889`
- Source-only target_normal_fpr: `0.599206`

## Output Files

- `outputs/results/context_shift_final_fr.csv`
- `outputs/results/context_shift_final_fr.json`
- `outputs/logs/TOF_FILTER_ANALYSIS.md`
- `outputs/logs/FINAL_CONTEXT_SHIFT_RUN_REPORT.md`
