# Route B FR/SP Context-shift Main-scale Single-run Status

## Goal

Run a main-scale single FR/SP seasonal context-shift experiment after enabling dictionary-based SmartGuard semantic anomaly injection.

## Run

Output:
- outputs/results/route_b_context_shift_frsp_main/summary.csv

Datasets:
- fr
- sp

Transition:
- seasonal

Status:
- rows: 10
- all rows: success

## FR Results

| Method | FPR | F1 | AUROC | AUPRC | FPR Gain |
|---|---:|---:|---:|---:|---:|
| source_only | 0.5714 | 0.0985 | 0.1988 | 0.3815 | - |
| source_plus_raw_synthetic | 0.4048 | 0.0485 | 0.2341 | 0.3920 | 0.1667 |
| source_plus_tof_synthetic | 0.4048 | 0.1302 | 0.2557 | 0.4020 | 0.1667 |
| source_plus_causal_tof_synthetic | 0.4444 | 0.0703 | 0.2418 | 0.3968 | 0.1270 |
| oracle_target | 0.0794 | 0.0547 | 0.5924 | 0.5657 | 0.4921 |

## SP Results

| Method | FPR | F1 | AUROC | AUPRC | FPR Gain |
|---|---:|---:|---:|---:|---:|
| source_only | 0.8920 | 0.0403 | 0.0698 | 0.0414 | - |
| source_plus_raw_synthetic | 0.4160 | 0.0700 | 0.3082 | 0.0613 | 0.4760 |
| source_plus_tof_synthetic | 0.5540 | 0.0671 | 0.2747 | 0.0507 | 0.3380 |
| source_plus_causal_tof_synthetic | 0.4920 | 0.0676 | 0.2907 | 0.0580 | 0.4000 |
| oracle_target | 0.0120 | 0.0426 | 0.8527 | 0.2638 | 0.8800 |

## Interpretation

This main-scale single run supports the core context-shift claim:

1. Source-only models suffer from high target-context false positive rates under seasonal shift.
2. Target-context synthetic adaptation substantially reduces FPR on both FR and SP.
3. Vanilla TOF is not consistently helpful; it improves FR but remains weaker than raw synthetic on SP.
4. Causal-TOF improves over vanilla TOF on SP, but is not consistently best across datasets in this single run.
5. Oracle target provides the expected lower-bound reference for target-context FPR.

These results are suitable as an initial main table, but final claims should be based on additional seeds or repeated runs.
