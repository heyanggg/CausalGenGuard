# Final Experiment Route

Date: 2026-06-06
Project path: `/home/heyang/projects/CausalGenGuard`

## Route Decision

The final experimental route is **context shift adaptation**, not repeated smoke tests and not proving that `causal_only` or `fusion` independently beats the SmartGuard reconstruction baseline.

SmartGuard standard experiments remain useful as sanity checks: they verify that the canonical mapping, labeled anomaly construction, and detector pipeline can run. They are not the core claim.

## Why Not Make Causal-Only/Fusion The Main Goal

Feasibility v2 and the causal branch diagnostic show that the causal implementation can pass a toy sanity check, but real FR causal scores do not yet separate constructed anomalies reliably. The bounded FR run showed reconstruction-only as the best method, while causal-only and fusion did not provide positive signal.

This means the weak result is not primarily a reason to keep re-running smoke tests. It points to a mismatch between current FR coverage, constructed anomaly types, causal score calibration, and the amount of context-specific normal data. Treating causal-only as the main detector would turn the project into a fragile head-to-head baseline contest rather than the adaptation problem it is better suited for.

## Final Main Experiment

The main question becomes:

Can target-context synthetic normal data reduce false positives when a SmartGuard-style detector trained on a source context is evaluated on target-context normal behavior?

The final runner is:

```bash
PYTHONPATH=src /home/heyang/miniconda3/envs/smartguard_env/bin/python scripts/run_context_shift_final.py --config configs/context_shift_fr.yaml
```

The script auto-selects local FR contexts:

- Source context candidates: `winter`, then `single`
- Target context candidates: `spring`, then `multiple`, then `night`

It reads local SmartGen FR context pkl files and aligns generated controls with `outputs/mappings/smartguard/fr/control_to_id.json`. Numeric SmartGen flat 4-tuples and textual `Device:action` generated pkl records are both supported.

## Methods

The final table includes these rows:

| method | role |
| --- | --- |
| `source_only` | Train only on source-context normal data. |
| `source_plus_raw_synthetic` | Add raw SmartGen target-context synthetic normal data. |
| `source_plus_tof_synthetic` | Add SmartGen synthetic normal data after legality and reconstruction TOF filtering. |
| `source_plus_causal_tof_synthetic` | Add SmartGen synthetic normal data after legality, reconstruction, and causal-deviation filtering. |
| `oracle_target` | Optional upper bound trained with target normal data if local target normal data exists. |

Causal information is used only inside `source_plus_causal_tof_synthetic` and for explanation-oriented diagnostics. It is not the detection main branch.

## Metrics

The final CSV/JSON outputs are:

- `outputs/results/context_shift_final_fr.csv`
- `outputs/results/context_shift_final_fr.json`

The table reports:

| metric | meaning |
| --- | --- |
| `target_normal_fpr` | Primary metric: false positive rate on target-context normal data. |
| `anomaly_f1` | Optional, only when real target-context anomaly data exists. |
| `auroc` | Optional, only when target-context anomaly labels exist. |
| `auprc` | Optional, only when target-context anomaly labels exist. |
| `synthetic_count` | Number of raw target-context synthetic normal sequences loaded. |
| `kept_count` | Number of synthetic sequences kept after the method's filter. |
| `rejected_count` | Number of synthetic sequences rejected by TOF or Causal-TOF. |
| `adaptation_gain` | `source_only.target_normal_fpr - method.target_normal_fpr`, when both FPR values are available. |

## Data Gaps

Some metrics are skipped rather than fabricated:

- If target-context anomaly data is not found under the selected SmartGen FR context directory, `anomaly_f1`, `auroc`, and `auprc` are left empty.
- If no generated target-context SmartGen pkl files can be parsed, synthetic methods still run structurally but will report `synthetic_count=0`.
- If `oracle_target` target normal data is unavailable, the oracle row is skipped.

Current local data appears strongest for target normal and generated synthetic context-shift evaluation. The formal claim should therefore emphasize target-normal FPR reduction and synthetic quality filtering before making any anomaly-detection claims under target context labels.

## Implementation Scope

Only the final route files were changed:

- `configs/context_shift_fr.yaml`
- `scripts/run_context_shift_final.py`
- `outputs/logs/FINAL_EXPERIMENT_ROUTE.md`

No SmartGuard, SmartGen, or GCAD source project files are modified by this route.
