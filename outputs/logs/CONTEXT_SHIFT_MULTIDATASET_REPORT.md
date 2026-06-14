# Context Shift Multi-Dataset Report

## Scope

- Datasets requested: `fr, sp, us`
- Transition requested: `seasonal` (`winter -> spring`, falling back to `summer` if needed)
- Bounds: normal <= 500, anomaly <= 500, synthetic <= 500, epochs <= 3

## Direct Answers

1. Successful dataset/transitions: `fr:seasonal`, `sp:seasonal`, `us:seasonal`
2. Missing dataset/transitions: none
3. US mapping found: `True`.
4. US mapping type: `smartgen_textual`.
5. US seasonal success: `True`.
6. Source-only consistently high FPR: `True` (3/3 successful runs >= 0.3).
7. Raw synthetic consistently lower FPR than source-only: `True` (3/3).
8. TOF/Causal-TOF consistently lower FPR than raw synthetic: `False` (1/3).
9. Causal-TOF remains FPR/F1 trade-off: `False` (1/3).
10. Lowest average target-normal FPR: `oracle_target` (`0.045111`).
11. Highest average balanced F1: `oracle_target` (`0.100566`).
12. Causal-TOF average FPR is lowest among non-oracle methods: `True` (`0.318296`). Including oracle, oracle remains lower.
13. Causal-TOF average balanced F1 is highest among non-oracle methods: `True` (`0.073039`). Including oracle, `oracle_target` is highest.
14. Paper-main-experiment readiness: bounded seasonal results are a usable prototype, but weak/variable anomaly F1 means this is not yet enough as the final paper main experiment.
15. Next transitions: yes, extend to `daytime -> night` and `single -> multiple`; those transitions test different context axes than seasonal weather.

## Average Metrics On Successful Runs

| method | avg_target_normal_fpr | avg_f1 | avg_balanced_f1 |
| --- | ---: | ---: | ---: |
| source_only | 0.715069 | 0.262395 | 0.052148 |
| source_plus_raw_synthetic | 0.327185 | 0.148972 | 0.038524 |
| source_plus_tof_synthetic | 0.371619 | 0.145666 | 0.050067 |
| source_plus_causal_tof_synthetic | 0.318296 | 0.175230 | 0.073039 |
| oracle_target | 0.045111 | 0.095546 | 0.100566 |

## Mapping Types

| dataset | transition | status | mapping_type |
| --- | --- | --- | --- |
| fr | seasonal | success | smartguard_numeric |
| sp | seasonal | success | smartguard_numeric |
| us | seasonal | success | smartgen_textual |

## Missing Details

- none

## Output Files

- `outputs/results/context_shift_multidataset/summary.csv`
- `outputs/results/context_shift_multidataset/summary.json`
- per-dataset CSV/JSON, per-anomaly CSV, and threshold-sweep CSV under the same directory
