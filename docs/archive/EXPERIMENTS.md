# Experiment Plan

## Main objective

Evaluate whether target-context synthetic normal data reduces false positives
under context shift.

## Main hypothesis

A model trained only on source-context normal sequences will over-report
anomalies on real target-context normal sequences.  Adding target-context
synthetic normal data, especially after TOF or Causal-TOF filtering, should
reduce target-context normal FPR.

## Compared methods

1. `source_only`: train on source-context normal data only.
2. `source_plus_raw_synthetic`: add raw SmartGen/CausalGenGuard synthetic data.
3. `source_plus_tof_synthetic`: add reconstruction-filtered synthetic data.
4. `source_plus_causal_tof_synthetic`: add legality + reconstruction + causal
   filtered synthetic data.
5. `oracle_target`: upper bound trained with real target-context normal data.

## Variables

Independent variables:

```text
training method
context transition
dataset
synthetic filtering strategy
target-data percentage in low-data experiments
```

Dependent variables:

```text
target_normal_fpr
precision
recall
f1
auroc
auprc
synthetic_keep_rate
runtime
```

Controlled variables:

```text
same train/validation/test split policy
same SmartGuard backbone architecture
same threshold calibration rule
same random seeds
same maximum sample bounds for smoke tests
```

## Recommended statistics

Run at least 3 seeds, preferably 5.  Report mean and standard deviation.  For
method comparisons, use paired comparisons by seed and dataset.  Do not report
missing anomaly metrics when target-context anomaly labels are absent.
