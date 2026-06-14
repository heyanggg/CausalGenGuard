# Final Context Shift Run Report

Date: 2026-06-11
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

## Target Labeled Anomaly Set

- target_labeled_jsonl: `/home/heyang/projects/CausalGenGuard/outputs/labels/fr_target_context_labeled.jsonl`
- target_labeled_report: `/home/heyang/projects/CausalGenGuard/outputs/labels/fr_target_context_labeled_report.json`
- labeled_normal_count: `500`
- labeled_anomaly_count: `234`
- target_anomaly_source: `target_context_labeled_jsonl`
- injectable_anomaly_types: `['SD_light_flickering', 'SD_tv_flickering', 'DM_window_open_midnight', 'DM_watervalve_open_midnight']`

| anomaly_type | candidate_count | injected_count |
| --- | ---: | ---: |
| SD_light_flickering | 37 | 37 |
| SD_camera_flickering | 0 | 0 |
| SD_tv_flickering | 6 | 6 |
| MD_camera_off_while_lock | 0 | 0 |
| MD_window_open_while_lock | 0 | 0 |
| DM_window_open_midnight | 188 | 188 |
| DM_watervalve_open_midnight | 3 | 3 |
| DD_microwave_long_time | 0 | 0 |

## Selected Method Results

| method | filter_strategy | kept_count | rejected_count | target_normal_fpr | precision | recall | f1 | auroc | auprc | adaptation_gain |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| source_only |  | 0 | 0 | 0.599206 | 0.346320 | 0.341880 | 0.344086 | 0.314204 | 0.366095 | skipped |
| source_plus_raw_synthetic | no_filter | 500 | 0 | 0.305556 | 0.189474 | 0.076923 | 0.109422 | 0.312899 | 0.374589 | 0.293651 |
| source_plus_tof_synthetic | iqr_1.5 | 490 | 10 | 0.142857 | 0.428571 | 0.115385 | 0.181818 | 0.366012 | 0.407284 | 0.456349 |
| source_plus_causal_tof_synthetic | relaxed_causal_keep_90_percent | 441 | 59 | 0.138889 | 0.406780 | 0.102564 | 0.163823 | 0.356397 | 0.403828 | 0.460317 |
| oracle_target |  | 0 | 0 | 0.083333 | 0.631579 | 0.153846 | 0.247423 | 0.651082 | 0.582548 | 0.515873 |

## Filter Strategy Sweep

| method | filter_strategy | selected | kept_count | rejected_count | target_normal_fpr | precision | recall | f1 | auroc | auprc | adaptation_gain |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| source_plus_tof_synthetic | no_filter | no | 500 | 0 | 0.305556 | 0.189474 | 0.076923 | 0.109422 | 0.312899 | 0.374589 | 0.293651 |
| source_plus_tof_synthetic | iqr_1.5 | yes | 490 | 10 | 0.142857 | 0.428571 | 0.115385 | 0.181818 | 0.366012 | 0.407284 | 0.456349 |
| source_plus_tof_synthetic | iqr_3.0 | no | 490 | 10 | 0.142857 | 0.409836 | 0.106838 | 0.169492 | 0.347375 | 0.395990 | 0.456349 |
| source_plus_tof_synthetic | keep_top_80_percent_by_low_loss | no | 392 | 108 | 0.384921 | 0.198347 | 0.102564 | 0.135211 | 0.293329 | 0.361412 | 0.214286 |
| source_plus_tof_synthetic | keep_top_90_percent_by_low_loss | no | 441 | 59 | 0.392857 | 0.188525 | 0.098291 | 0.129213 | 0.296992 | 0.359807 | 0.206349 |
| source_plus_causal_tof_synthetic | tof_only | no | 490 | 10 | 0.142857 | 0.428571 | 0.115385 | 0.181818 | 0.366012 | 0.407284 | 0.456349 |
| source_plus_causal_tof_synthetic | relaxed_causal_keep_90_percent | yes | 441 | 59 | 0.138889 | 0.406780 | 0.102564 | 0.163823 | 0.356397 | 0.403828 | 0.460317 |
| source_plus_causal_tof_synthetic | relaxed_causal_keep_95_percent | no | 466 | 34 | 0.365079 | 0.220339 | 0.111111 | 0.147727 | 0.309269 | 0.368646 | 0.234127 |
| source_plus_causal_tof_synthetic | causal_filter_disabled | no | 490 | 10 | 0.142857 | 0.428571 | 0.115385 | 0.181818 | 0.366012 | 0.407284 | 0.456349 |

## Metric Availability

- `target_normal_fpr`: available for all methods.
- `precision`: available.
- `recall`: available.
- `f1`: available.
- `auroc`: available.
- `auprc`: available.

## Best Method By Target-Normal FPR

- Best non-oracle method: `source_plus_causal_tof_synthetic`
- Best non-oracle filter_strategy: `relaxed_causal_keep_90_percent`
- Best non-oracle target_normal_fpr: `0.138889`
- Source-only target_normal_fpr: `0.599206`

## Output Files

- `outputs/labels/fr_target_context_labeled.jsonl`
- `outputs/labels/fr_target_context_labeled_report.json`
- `outputs/results/context_shift_final_fr.csv`
- `outputs/results/context_shift_final_fr.json`
- `outputs/logs/TOF_FILTER_ANALYSIS.md`
- `outputs/logs/FINAL_CONTEXT_SHIFT_RUN_REPORT.md`
