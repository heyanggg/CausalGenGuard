# Feasibility Report

Project path: `/home/heyang/projects/CausalGenGuard`
Environment: `/home/heyang/miniconda3/envs/smartguard_env/bin/python`

## Run Bounds

- Dataset: `fr`
- Max sequences loaded: 500
- Epochs per model: 2
- Normal split: train=300, val=100, test=100
- Vocab size: 50
- Full training was not run.

## Experiment 1: SmartGuard Standard Sanity Check

Status: ran through.

| method | precision | recall | f1 | fpr | auroc | auprc |
| --- | --- | --- | --- | --- | --- | --- |
| reconstruction_only | 0.5000 | 0.0400 | 0.0741 | 0.0400 | 0.5048 | 0.5097 |
| fusion | 0.5556 | 0.0500 | 0.0917 | 0.0400 | 0.4966 | 0.4999 |

Output: `outputs/results/sanity_fr.json`

Injection summary:

- Named SmartGuard injections: 0
- Fallback numeric-control injections: 100
- Reason: SmartGuard 10x4 controls are numeric here, so semantic device/action matching was unavailable.

## Experiment 2: Causal Anomaly Smoke Test

Status: ran through.

| method | precision | recall | f1 | fpr | auroc | auprc |
| --- | --- | --- | --- | --- | --- | --- |
| reconstruction_only | 0.5000 | 0.0404 | 0.0748 | 0.0404 | 0.5745 | 0.5397 |
| causal_only | 0.3636 | 0.0404 | 0.0727 | 0.0707 | 0.5172 | 0.5032 |
| fusion | 0.5000 | 0.0404 | 0.0748 | 0.0404 | 0.5551 | 0.5232 |

Output: `outputs/results/causal_anomaly_smoke.json`

- Injected anomaly types: `causal_edge_break`, `causal_edge_injection`, `lag_delay`
- Causal branch improves over reconstruction: `False`
- Best method by F1: `reconstruction_only`
- Interpretation: no causal-branch gain was observed in this bounded 2-epoch run; this is a negative/uncertain feasibility signal, not a final conclusion.

## Experiment 3: Synthetic Data Pipeline Smoke Test

Status: ran through.

- Synthetic source: `SmartGen offline`
- Candidate count: 80
- Kept count: 0
- Rejected count: 80
- Proxy target-normal FPR before Causal-TOF: None
- Proxy target-normal FPR after Causal-TOF: None
- Observed SmartGen/Causal-TOF target-normal FPR reduction: `None`

Causal-TOF stage counts:

- Legality: input=80, kept=78, rejected=2
- Reconstruction: input=78, kept=5, rejected=73
- Causal: input=5, kept=0, rejected=5
- Utility: input=0, kept=0, rejected=0, mode=`no_op_interface`

Outputs:

- Candidates: `outputs/synthetic/feasibility_tof/candidates.jsonl`
- Kept: `outputs/synthetic/feasibility_tof/kept.jsonl`
- Rejected: `outputs/synthetic/feasibility_tof/rejected.jsonl`
- Report: `outputs/synthetic/feasibility_tof/report.json`
- Summary: `outputs/results/synthetic_pipeline_smoke.json`

Interpretation:

- SmartGen offline files were found and parsed.
- No FPR reduction conclusion can be drawn because all candidates were rejected and raw scoring hit vocab mismatch for some SmartGen controls.
- This points to a required SmartGen-to-SmartGuard control mapping step before target-normal FPR claims are meaningful.

## Fixes Applied During Feasibility Run

- `scripts/run_feasibility_experiments.py`: vocabulary is now built from the bounded 500-sequence experiment pool instead of train-only split, preventing validation/test OOV controls in smoke scoring.
- `src/causal_gen_guard/data/attack_injector.py`: added missing `_as_float` helper used by `lag_delay` causal anomaly injection.
- Verified after fixes: `python -m pytest tests` -> 18 passed.

## Current Biggest Issues

- SmartGuard FR controls are numeric in the prepared 10x4 data, so semantic SmartGuard-style attacks need a device/action dictionary or mapped control names.
- In this small run, the causal branch did not improve causal anomaly F1 over reconstruction-only.
- SmartGen offline synthetic controls are not yet aligned with the SmartGuard FR vocab; Causal-TOF rejected all candidates.
- Target-normal FPR reduction could not be evaluated from the current synthetic run.
- These are feasibility metrics from one small seed and 2 epochs, not final paper results.
