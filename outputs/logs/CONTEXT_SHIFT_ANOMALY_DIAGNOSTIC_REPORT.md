# Context Shift Anomaly Diagnostic Report

- Source context: `winter`
- Target context: `spring`
- Target labeled JSONL: `/home/heyang/projects/CausalGenGuard/outputs/labels/fr_target_context_labeled.jsonl`
- Normal bound: `<=500`; anomaly bound: `<=500`; epochs: `<=3`

## Direct Answers

1. Lowest target-normal FPR: `oracle_target` (`0.083333`).
2. Highest overall F1: `source_only` (`0.344086`).
3. Highest balanced F1: `oracle_target` (`0.139535`).
4. Recall diagnosis: score separation is the main bottleneck: many per-type score margins are non-positive or near zero, so lowering the threshold cannot reliably recover recall without increasing false positives.
5. Easiest anomaly types: `SD_light_flickering` (mean_f1=0.238483, margin=0.334794), `DM_window_open_midnight` (mean_f1=0.136652, margin=-0.771846), `SD_tv_flickering` (mean_f1=0.041997, margin=0.211156)
6. Hardest anomaly types: `SD_tv_flickering` (mean_f1=0.041997, margin=0.211156), `DM_watervalve_open_midnight` (mean_f1=0.081856, margin=1.327173, low_support), `DM_window_open_midnight` (mean_f1=0.136652, margin=-0.771846)
7. Causal-TOF vs TOF: `FPR/F1 trade-off`. TOF FPR/F1 = `0.142857` / `0.181818`; Causal-TOF FPR/F1 = `0.138889` / `0.163823`.
8. Recommendation: write Causal-TOF as a low-false-alarm deployment option, not the sole main method.

## Multi-Dataset Readiness

Current FR winter->spring results are sufficient to enter bounded FR/SP/US multi-dataset experiments as a diagnostic/adaptation pipeline, but not yet sufficient to claim anomaly-detection superiority: recall and score separation remain weak on several target-context anomaly types.

## Overall Metrics

| method | filter_strategy | target_normal_fpr | precision | recall | f1 | auroc | auprc | kept_count | rejected_count | adaptation_gain |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| source_only |  | 0.599206 | 0.346320 | 0.341880 | 0.344086 | 0.314204 | 0.366095 | 0 | 0 | skipped |
| source_plus_raw_synthetic | no_filter | 0.305556 | 0.189474 | 0.076923 | 0.109422 | 0.312899 | 0.374589 | 500 | 0 | 0.293651 |
| source_plus_tof_synthetic | iqr_1.5 | 0.142857 | 0.428571 | 0.115385 | 0.181818 | 0.366012 | 0.407284 | 490 | 10 | 0.456349 |
| source_plus_causal_tof_synthetic | relaxed_causal_keep_90_percent | 0.138889 | 0.406780 | 0.102564 | 0.163823 | 0.356397 | 0.403828 | 441 | 59 | 0.460317 |
| oracle_target |  | 0.083333 | 0.631579 | 0.153846 | 0.247423 | 0.651082 | 0.582548 | 0 | 0 | 0.515873 |

## Balanced Subset

- Balanced normal_count: `500`
- Balanced anomaly_count: `12`
- take_per_type: `3`
- low_support: `{'DM_watervalve_open_midnight': 3}`

| method | target_normal_fpr | precision | recall | f1 | auroc | auprc |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| source_only | 0.548000 | 0.035211 | 0.833333 | 0.067568 | 0.590833 | 0.028678 |
| source_plus_raw_synthetic | 0.192000 | 0.000000 | 0.000000 | 0.000000 | 0.528333 | 0.025695 |
| source_plus_tof_synthetic | 0.108000 | 0.052632 | 0.250000 | 0.086957 | 0.635833 | 0.040540 |
| source_plus_causal_tof_synthetic | 0.106000 | 0.070175 | 0.333333 | 0.115942 | 0.681000 | 0.050870 |
| oracle_target | 0.136000 | 0.081081 | 0.500000 | 0.139535 | 0.800833 | 0.137626 |

## Output Files

- `outputs/results/context_shift_final_fr_per_anomaly.csv`
- `outputs/results/context_shift_final_fr_per_anomaly.json`
- `outputs/results/context_shift_final_fr_threshold_sweep.csv`
- `outputs/labels/fr_target_context_labeled_balanced.jsonl`
- `outputs/labels/fr_target_context_labeled_balanced_report.json`
- `outputs/results/context_shift_final_fr_balanced.csv`
- `outputs/results/context_shift_final_fr_balanced.json`
