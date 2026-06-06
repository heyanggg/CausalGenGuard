# CausalGenGuard Acceptance Report

Date: 2026-06-05
Project path: `/home/heyang/projects/CausalGenGuard`
Python environment: `/home/heyang/miniconda3/envs/smartguard_env/bin/python`

## Scope And Constraints

- Original projects were not modified:
  - `/home/heyang/projects/SmartGuard`
  - `/home/heyang/projects/SmartGen`
  - `/home/heyang/projects/GCAD`
- No full training was run.
- No network access was used during acceptance.
- All reads, runs, generated outputs, and this report stayed under `/home/heyang/projects/CausalGenGuard` except read-only access to `../SmartGuard` data.

## File Tree And Required Directories

Required project entries are present:

```text
README.md: present
configs/: present
scripts/: present
src/: present
tests/: present
outputs/: present
```

Top-level implementation areas checked:

```text
configs/default.yaml
configs/context_shift_fr.yaml
configs/context_shift_sp.yaml
configs/context_shift_us.yaml
scripts/prepare_smartguard_data.py
scripts/prepare_smartgen_data.py
scripts/prepare_argus_data.py
scripts/run_standard_repro.sh
scripts/run_context_shift.sh
scripts/run_ablation.sh
scripts/run_causal_anomaly.sh
src/causal_gen_guard/data/
src/causal_gen_guard/models/
src/causal_gen_guard/generation/
src/causal_gen_guard/training/
src/causal_gen_guard/evaluation/
src/causal_gen_guard/utils/
tests/
outputs/
```

## Unit Tests

Command:

```bash
PYTHONPATH=src /home/heyang/miniconda3/envs/smartguard_env/bin/python -m pytest tests
```

Result:

```text
18 passed in 10.31s
```

### Passed Tests

```text
tests/test_attack_injector.py::test_inject_causal_edge_injection_toy_vocab
tests/test_attack_injector.py::test_generate_anomaly_dataset_reports_skips
tests/test_causal_graph.py::test_compute_gradient_causality_shape
tests/test_causal_graph.py::test_sparsify_non_negative
tests/test_causal_graph.py::test_causal_deviation_score_ordering
tests/test_data_schema.py::test_behavior_event_and_sequence_roundtrip
tests/test_data_schema.py::test_parse_smartguard_sample_flat_length_40
tests/test_data_schema.py::test_parse_smartguard_sample_nested_10_by_4
tests/test_event_tensor.py::test_build_vocab_remaps_controls
tests/test_event_tensor.py::test_sequence_to_event_tensor_shape_and_one_hot
tests/test_event_tensor.py::test_batch_to_event_tensor_metadata
tests/test_event_tensor.py::test_sliding_windows_from_tensor_shape
tests/test_external_adapters.py::test_smartsense_fake_csv_timestamp_device_action
tests/test_external_adapters.py::test_argus_fake_home_csv_timestamp_device_action
tests/test_prompt_builder.py::test_prompt_builder_contains_context_and_causal_hints
tests/test_scoring.py::test_noise_aware_weight_high_loss_lower_weight
tests/test_scoring.py::test_causal_aware_weight_high_centrality_lifts_weight
tests/test_scoring.py::test_causal_aware_weight_clamped
```

### Failed Tests

No tests failed.

### Failure Reasons

No failure reasons. Pytest is available in the SmartGuard environment and ran successfully.

## Smoke Test Commands And Results

### 1. Prepare SmartGuard Data, Limit 100

Command:

```bash
PYTHONPATH=src /home/heyang/miniconda3/envs/smartguard_env/bin/python scripts/prepare_smartguard_data.py \
  --smartguard-root ../SmartGuard \
  --dataset fr \
  --output outputs/processed/fr_sequences_acceptance.jsonl \
  --limit 100
```

Result:

```text
Wrote 100 SmartGuard sequences to outputs/processed/fr_sequences_acceptance.jsonl
```

### 2. Train Backbone, 1 Epoch, Limit 100

Command:

```bash
PYTHONPATH=src /home/heyang/miniconda3/envs/smartguard_env/bin/python -m causal_gen_guard.training.train_backbone \
  --input-jsonl outputs/processed/fr_sequences_acceptance.jsonl \
  --checkpoint-path outputs/checkpoints/fr_acceptance_smartguard_backbone.pt \
  --loss-vector-output outputs/results/fr_acceptance_loss_vector.npy \
  --epochs 1 \
  --max-sequences 100 \
  --batch-size 32 \
  --hidden-dim 64 \
  --num-layers 1
```

Result:

```text
Epoch 1/1 train_loss=2.890390
Saved checkpoint to outputs/checkpoints/fr_acceptance_smartguard_backbone.pt
Saved loss vector to outputs/results/fr_acceptance_loss_vector.npy
```

### 3. Train Causal Predictor, 1 Epoch, Limit 100

Command:

```bash
PYTHONPATH=src /home/heyang/miniconda3/envs/smartguard_env/bin/python -m causal_gen_guard.training.train_causal \
  --input-jsonl outputs/processed/fr_sequences_acceptance.jsonl \
  --checkpoint-path outputs/checkpoints/fr_acceptance_causal_predictor.pt \
  --graph-output outputs/results/fr_acceptance_A_norm.npy \
  --centrality-output outputs/results/fr_acceptance_causal_centrality.npy \
  --epochs 1 \
  --max-sequences 100 \
  --batch-size 32 \
  --hidden-dim 64 \
  --window-size 4 \
  --causality-samples 32 \
  --include-time-features
```

Result:

```text
Epoch 1/1 causal_loss=0.496915
Saved causal predictor checkpoint to outputs/checkpoints/fr_acceptance_causal_predictor.pt
Saved normal causal graph to outputs/results/fr_acceptance_A_norm.npy
Saved causal centrality to outputs/results/fr_acceptance_causal_centrality.npy
```

### 4. Evaluate, Input Limited To 100 Sequences

Command:

```bash
PYTHONPATH=src /home/heyang/miniconda3/envs/smartguard_env/bin/python -m causal_gen_guard.evaluation.evaluate \
  --sequences outputs/processed/fr_sequences_acceptance.jsonl \
  --checkpoint-backbone outputs/checkpoints/fr_acceptance_smartguard_backbone.pt \
  --checkpoint-causal outputs/checkpoints/fr_acceptance_causal_predictor.pt \
  --a-norm outputs/results/fr_acceptance_A_norm.npy \
  --threshold 0.5 \
  --output outputs/results/fr_acceptance_eval.json
```

Result:

```text
Wrote evaluation results to outputs/results/fr_acceptance_eval.json
sequence_count=100
metrics=None
warnings=[]
```

`metrics=None` is expected because this FR smoke input contains converted normal sequences without anomaly labels.

### 5. Evaluate With Explanations

Command:

```bash
PYTHONPATH=src /home/heyang/miniconda3/envs/smartguard_env/bin/python -m causal_gen_guard.evaluation.evaluate \
  --sequences outputs/processed/fr_sequences_acceptance.jsonl \
  --checkpoint-backbone outputs/checkpoints/fr_acceptance_smartguard_backbone.pt \
  --checkpoint-causal outputs/checkpoints/fr_acceptance_causal_predictor.pt \
  --a-norm outputs/results/fr_acceptance_A_norm.npy \
  --threshold 0.5 \
  --output outputs/results/fr_acceptance_eval_explain.json \
  --save-explanations \
  --num-explanations 1 \
  --explanation-dir outputs/results/explanations/fr_acceptance \
  --figure-dir outputs/figures/fr_acceptance
```

Result:

```text
Wrote evaluation results to outputs/results/fr_acceptance_eval_explain.json
Generated 1 explanation JSON
Generated 1 causal graph diff PNG
warnings=['Labels missing; score distribution normal/anomaly plot skipped']
```

The warning is expected because the acceptance input has no anomaly labels, so a normal-vs-anomaly score distribution cannot be drawn.

## Required Output Files

All required acceptance outputs exist and are non-empty:

```text
outputs/processed/fr_sequences_acceptance.jsonl
outputs/checkpoints/fr_acceptance_smartguard_backbone.pt
outputs/checkpoints/fr_acceptance_causal_predictor.pt
outputs/results/fr_acceptance_A_norm.npy
outputs/results/fr_acceptance_causal_centrality.npy
outputs/results/fr_acceptance_eval.json
outputs/results/fr_acceptance_eval_explain.json
outputs/results/explanations/fr_acceptance/smartguard_fr_train_000027_001.json
outputs/figures/fr_acceptance/smartguard_fr_train_000027_001_causal_diff.png
```

## Error Checks

No blocking errors were observed:

```text
import error: none
shape mismatch: none
dtype error: none
device error: none
path error: none
```

Non-blocking warnings:

```text
PyTorch nested tensor API prototype warning during Transformer forward.
transformers torch.utils._pytree deprecation warning from installed environment.
Labels missing; score distribution normal/anomaly plot skipped.
```

These warnings do not block small-scale experiments.

## Fixed Issues During Acceptance

No code fixes were required during this acceptance run.

Previously completed project setup already included:

```text
pytest installed in smartguard_env
matplotlib installed in smartguard_env
--limit available in scripts/prepare_smartguard_data.py
```

## Unfixed TODO

- Add labeled anomaly JSONL for acceptance runs that need precision, recall, F1, AUROC, and AUPRC from real labels.
- Run context-shift and ablation smoke again after final real FR/SP/US synthetic target files are fixed.
- Add runtime timing fields if runtime tables are needed in paper summaries.
- Keep full experiments disabled until data paths, labels, and output naming are frozen.

## Acceptance Decision

Current status: can enter small-scale experiments.

Rationale:

- Complete package structure is present.
- All unit tests pass under the SmartGuard virtual environment.
- SmartGuard data conversion works with `--limit 100`.
- Backbone training smoke completes on CPU for 1 epoch and 100 samples.
- Causal predictor training smoke completes on CPU for 1 epoch and 100 samples.
- Evaluation works on the 100-sequence smoke input.
- Explanation JSON and causal graph difference figure generation work.
- No import, shape, dtype, device, or path errors were observed.
