# Route B FR/SP Context-shift Smoke Status

## Goal

Run a small FR/SP seasonal context-shift smoke experiment after enabling dictionary-based SmartGuard semantic anomaly injection.

## Run

Output:
- outputs/results/route_b_context_shift_frsp_smoke/summary.csv

Datasets:
- fr
- sp

Transition:
- seasonal

Status:
- rows: 10
- all rows: success

## Key FPR results

FR:
- source_only FPR: 0.2897
- source_plus_raw_synthetic FPR: 0.2460
- source_plus_tof_synthetic FPR: 0.2579
- source_plus_causal_tof_synthetic FPR: 0.2540
- oracle_target FPR: 0.0357

SP:
- source_only FPR: 0.6200
- source_plus_raw_synthetic FPR: 0.4967
- source_plus_tof_synthetic FPR: 0.5800
- source_plus_causal_tof_synthetic FPR: 0.4900
- oracle_target FPR: 0.0100

## Interpretation

The smoke run confirms the main context-shift pipeline works for both FR and SP after Route B dictionary-based semantic injection.

The expected pattern appears:
- source-only has high target-context false positive rate;
- raw synthetic target-context adaptation reduces FPR;
- causal-TOF is more stable than vanilla TOF, especially on SP;
- oracle target gives the lower-bound reference FPR.

These results are still smoke-scale and should not be used as final paper results.
