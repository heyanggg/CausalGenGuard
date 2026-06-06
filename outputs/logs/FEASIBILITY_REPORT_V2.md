# Feasibility Report V2

Project path: `/home/heyang/projects/CausalGenGuard`
Environment: `/home/heyang/miniconda3/envs/smartguard_env/bin/python`

## Run Bounds

- Max normal sequences: 100
- Epochs: 2
- Train/val/test normal split: train=60, val=20, test=20
- Vocab size: 29
- Window size: 4

## Input Files

- `/home/heyang/projects/CausalGenGuard/outputs/processed/fr_sequences_canonical.jsonl`
- `/home/heyang/projects/CausalGenGuard/outputs/labels/fr_smartguard_style_labeled.jsonl`
- `/home/heyang/projects/CausalGenGuard/outputs/labels/fr_smartguard_style_labeled_report.json`
- `/home/heyang/projects/CausalGenGuard/outputs/mappings/smartguard/fr/id_to_control.json`

## Labeled Injection Report

- named_injection_success_count: 100
- fallback_numeric_injection_count: 0
- skipped_count: 793

## Experiment 1: SmartGuard-Style Named Semantic Anomaly Sanity

| method | precision | recall | f1 | fpr | auroc | auprc |
| --- | --- | --- | --- | --- | --- | --- |
| reconstruction_only | 0.7018 | 0.4000 | 0.5096 | 0.1700 | 0.8423 | 0.8121 |
| causal_only | 0.0000 | 0.0000 | 0.0000 | 0.0600 | 0.3465 | 0.4479 |
| fusion | 0.5952 | 0.2500 | 0.3521 | 0.1700 | 0.7168 | 0.6704 |

Output: `outputs/results/sanity_fr_v2.json`

## Experiment 2: A_norm Top-Edge Causal Anomaly Smoke

| method | precision | recall | f1 | fpr | auroc | auprc |
| --- | --- | --- | --- | --- | --- | --- |
| reconstruction_only | 0.7826 | 0.6923 | 0.7347 | 0.1923 | 0.8905 | 0.8935 |
| causal_only | 0.7143 | 0.1923 | 0.3030 | 0.0769 | 0.6538 | 0.6870 |
| fusion | 0.7826 | 0.6923 | 0.7347 | 0.1923 | 0.8314 | 0.8315 |

Output: `outputs/results/causal_anomaly_smoke_v2.json`

## Signals

- Experiment 2 causal branch better than reconstruction_only: `False`
- Experiment 2 fusion better than reconstruction_only: `False`
- Recommend SmartGen alignment / Causal-TOF v2: `True`
- Recommend formal main experiments: `False`

## Notes

- Bounded v2 run used canonical FR data only; no source projects were modified.
- Normal training was limited to 60 sequences and 2 epochs.
- Formal main experiments should wait for broader semantic coverage if many named attack types remain skipped.
