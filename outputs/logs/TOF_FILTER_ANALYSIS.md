# TOF Filter Analysis

## Summary

- Source context: `winter`
- Target context: `spring`
- Synthetic count: `500`
- Raw synthetic target_normal_fpr: `0.305556`
- Selected TOF strategy: `iqr_1.5` with target_normal_fpr `0.142857`
- Selected Causal-TOF strategy: `relaxed_causal_keep_90_percent` with target_normal_fpr `0.138889`

## Why Raw Synthetic Is Currently Strong

Raw target-context synthetic normal keeps the full generated target-context diversity. The stricter filters rank generated sequences by source-trained reconstruction or causal scores, so they can reject normal target-context patterns precisely because those patterns differ from the source context.

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

## Over-Rejection Notes

The following strategies rejected synthetic data and performed worse than raw synthetic:

- `source_plus_tof_synthetic` / `keep_top_80_percent_by_low_loss` rejected `108` and produced FPR `0.384921`.
- `source_plus_tof_synthetic` / `keep_top_90_percent_by_low_loss` rejected `59` and produced FPR `0.392857`.
- `source_plus_causal_tof_synthetic` / `relaxed_causal_keep_95_percent` rejected `34` and produced FPR `0.365079`.

## Final Choice

- `source_plus_tof_synthetic` uses `iqr_1.5`.
- `source_plus_causal_tof_synthetic` uses `relaxed_causal_keep_90_percent`.
- Causal information remains part of Causal-TOF filtering/explanation only; it is not promoted to a detector branch in the final route.
