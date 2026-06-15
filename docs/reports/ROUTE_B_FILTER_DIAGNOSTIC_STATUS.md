# Route B Filtering Diagnostic Status

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

## Observation

The multi-seed diagnostic shows that the synthetic adaptation branch is effective, but the filtering branches are not consistently superior.

### TOF filtering behavior

Vanilla TOF with `iqr_1.5` rejects very few synthetic samples:

FR:
- seed 42: kept 199 / rejected 1
- seed 43: kept 199 / rejected 1
- seed 44: kept 200 / rejected 0

SP:
- seed 42: kept 198 / rejected 2
- seed 43: kept 198 / rejected 2
- seed 44: kept 198 / rejected 2

This suggests that the current TOF threshold is too permissive for this synthetic pool.

### Causal-TOF filtering behavior

Causal-TOF uses a relaxed keep-ratio strategy and rejects about 10% of synthetic samples:

FR:
- seed 42: kept 180 / rejected 20
- seed 43: kept 180 / rejected 20
- seed 44: kept 180 / rejected 20

SP:
- seed 42: kept 179 / rejected 21
- seed 43: kept 179 / rejected 21
- seed 44: kept 179 / rejected 21

This filtering is more active than TOF, but it is not consistently better than raw synthetic adaptation.

## Mean/std FPR summary

FR:
- source_only: 0.4193 ± 0.0893
- raw synthetic: 0.3505 ± 0.0502
- TOF synthetic: 0.3175 ± 0.0525
- Causal-TOF synthetic: 0.3214 ± 0.0757
- oracle target: 0.0529 ± 0.0165

SP:
- source_only: 0.8507 ± 0.0431
- raw synthetic: 0.4993 ± 0.1212
- TOF synthetic: 0.5687 ± 0.2470
- Causal-TOF synthetic: 0.6033 ± 0.0514
- oracle target: 0.0093 ± 0.0050

## Interpretation

The stable paper claim should be:

1. Source-only models suffer from high target-context false positive rates under seasonal context shift.
2. Target-context synthetic adaptation substantially reduces FPR on both FR and SP.
3. TOF and Causal-TOF are useful ablation branches, but current filtering settings are dataset-dependent.
4. The current evidence does not support claiming that Causal-TOF is uniformly best.
5. Future tuning should explore stricter TOF thresholds and multiple Causal-TOF keep ratios.

## Recommended next step

For paper writing, present TOF and Causal-TOF as filtering ablations rather than as universally dominant components.

For further method optimization, run a small grid:
- TOF IQR factor: 0.5, 1.0, 1.5
- Causal-TOF keep ratio: 0.6, 0.7, 0.8, 0.9
