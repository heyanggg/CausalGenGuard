# Route B FR/SP Context-shift Seed Stability Status

## Goal

Summarize the three-seed medium-scale FR/SP seasonal context-shift experiment after Route B dictionary-based semantic anomaly injection.

This report is intended to support the final paper claim about context-shift false positives and target-context synthetic adaptation.

## Setting

Datasets:
- FR
- SP

Transition:
- seasonal

Bounds:
- max-normal: 600
- max-anomaly: 200
- max-synthetic: 200
- epochs: 2
- seeds: 42, 43, 44

Methods:
- source_only
- source_plus_raw_synthetic
- source_plus_tof_synthetic
- source_plus_causal_tof_synthetic
- oracle_target

Primary metric:
- target-normal false positive rate, abbreviated as FPR.

## FR results

| Method | FPR mean ± std | Interpretation |
|---|---:|---|
| source_only | 0.4193 ± 0.0893 | High target-context false positive rate under source-only training. |
| source_plus_raw_synthetic | 0.3505 ± 0.0502 | Synthetic target-context adaptation reduces FPR. |
| source_plus_tof_synthetic | 0.3175 ± 0.0525 | Best mean FR FPR among non-oracle synthetic variants in this run. |
| source_plus_causal_tof_synthetic | 0.3214 ± 0.0757 | Similar to TOF on FR, but not clearly dominant. |
| oracle_target | 0.0529 ± 0.0165 | Lower-bound reference when real target-context normal data is available. |

## SP results

| Method | FPR mean ± std | Interpretation |
|---|---:|---|
| source_only | 0.8507 ± 0.0431 | Very high target-context false positive rate under source-only training. |
| source_plus_raw_synthetic | 0.4993 ± 0.1212 | Synthetic target-context adaptation strongly reduces FPR. |
| source_plus_tof_synthetic | 0.5687 ± 0.2470 | Filtering is unstable on SP and is weaker than raw synthetic on average. |
| source_plus_causal_tof_synthetic | 0.6033 ± 0.0514 | Active filtering, but not better than raw synthetic on SP. |
| oracle_target | 0.0093 ± 0.0050 | Lower-bound reference when real target-context normal data is available. |

## Cross-dataset interpretation

The stable conclusion is:

1. Source-only models suffer from high target-context FPR under seasonal context shift.
2. Target-context synthetic adaptation reduces FPR on both FR and SP.
3. Oracle target training provides the expected lower-bound reference and confirms that target-context adaptation is meaningful.
4. TOF and Causal-TOF should be treated as filtering ablations with dataset-dependent behavior.
5. Current evidence does not support claiming that Causal-TOF is uniformly best.

## Recommended paper wording

A safe wording is:

> Across FR and SP seasonal context-shift settings, source-only training produces high false positive rates on target-context normal behavior. Adding target-context synthetic normal behavior consistently reduces FPR, demonstrating the value of synthetic adaptation for context-shift-aware IoT anomaly detection. Filtering variants such as TOF and Causal-TOF provide diagnostic ablations, but their effect is dataset-dependent.

Avoid claiming:

- Causal-TOF is always the best variant.
- The causal branch alone is the main source of performance gain.
- The method fully reproduces or outperforms SmartGen or GCAD.

## Relation to other reports

This report complements:

- `ROUTE_B_INJECTION_STATUS.md` for semantic anomaly injection correctness.
- `ROUTE_B_CONTEXT_SHIFT_FRSP_MAIN_STATUS.md` for the single main-scale run.
- `ROUTE_B_FILTER_DIAGNOSTIC_STATUS.md` for TOF and Causal-TOF filtering behavior.
- `EXPERIMENT_SUMMARY.md` for the compact overall summary.
