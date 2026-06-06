# CausalGenGuard Smoke Test Report

Date: 2026-06-05
Host project path: `/home/heyang/projects/CausalGenGuard`

## Scope And Constraints

- No full training was run.
- No network access or dependency installation was used.
- Source projects were treated as read-only:
  - `/home/heyang/projects/SmartGuard`
  - `/home/heyang/projects/SmartGen`
  - `/home/heyang/projects/GCAD`
- All generated artifacts and fixes were kept under `/home/heyang/projects/CausalGenGuard`.
- The server does not expose a default `python` command, so smoke commands used `/home/heyang/miniconda3/envs/smartguard_env/bin/python`.

## Unit Tests

`python -m pytest tests` could not run because pytest is not installed in `smartguard_env`:

```text
No module named pytest
```

Fallback direct execution was used, as requested. All direct test files passed:

```text
tests/test_attack_injector.py: passed
tests/test_causal_graph.py: passed
tests/test_data_schema.py: passed
tests/test_event_tensor.py: passed
tests/test_external_adapters.py: passed
tests/test_prompt_builder.py: passed
tests/test_scoring.py: passed
```

## Data Preparation Smoke

Command:

```bash
PYTHONPATH=src /home/heyang/miniconda3/envs/smartguard_env/bin/python scripts/prepare_smartguard_data.py \
  --smartguard-root ../SmartGuard \
  --dataset fr \
  --output outputs/processed/fr_sequences_smoke.jsonl \
  --limit 100
```

Result:

```text
Wrote 100 SmartGuard sequences to outputs/processed/fr_sequences_smoke.jsonl
```

Generated file:

```text
outputs/processed/fr_sequences_smoke.jsonl
```

## Backbone Smoke Training

Command:

```bash
PYTHONPATH=src /home/heyang/miniconda3/envs/smartguard_env/bin/python -m causal_gen_guard.training.train_backbone \
  --input-jsonl outputs/processed/fr_sequences_smoke.jsonl \
  --checkpoint-path outputs/checkpoints/fr_smoke_smartguard_backbone.pt \
  --loss-vector-output outputs/results/fr_smoke_loss_vector.npy \
  --epochs 1 \
  --max-sequences 100 \
  --batch-size 32 \
  --hidden-dim 64 \
  --num-layers 1
```

Result:

```text
Epoch 1/1 train_loss=2.890390
Saved checkpoint to outputs/checkpoints/fr_smoke_smartguard_backbone.pt
Saved loss vector to outputs/results/fr_smoke_loss_vector.npy
```

Generated files:

```text
outputs/checkpoints/fr_smoke_smartguard_backbone.pt
outputs/results/fr_smoke_loss_vector.npy
```

## Causal Predictor Smoke Training

Command:

```bash
PYTHONPATH=src /home/heyang/miniconda3/envs/smartguard_env/bin/python -m causal_gen_guard.training.train_causal \
  --input-jsonl outputs/processed/fr_sequences_smoke.jsonl \
  --checkpoint-path outputs/checkpoints/fr_smoke_causal_predictor.pt \
  --graph-output outputs/results/fr_smoke_A_norm.npy \
  --centrality-output outputs/results/fr_smoke_causal_centrality.npy \
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
Saved causal predictor checkpoint to outputs/checkpoints/fr_smoke_causal_predictor.pt
Saved normal causal graph to outputs/results/fr_smoke_A_norm.npy
Saved causal centrality to outputs/results/fr_smoke_causal_centrality.npy
```

Generated files:

```text
outputs/checkpoints/fr_smoke_causal_predictor.pt
outputs/results/fr_smoke_A_norm.npy
outputs/results/fr_smoke_causal_centrality.npy
```

## Evaluation Smoke

Command:

```bash
PYTHONPATH=src /home/heyang/miniconda3/envs/smartguard_env/bin/python -m causal_gen_guard.evaluation.evaluate \
  --sequences outputs/processed/fr_sequences_smoke.jsonl \
  --checkpoint-backbone outputs/checkpoints/fr_smoke_smartguard_backbone.pt \
  --checkpoint-causal outputs/checkpoints/fr_smoke_causal_predictor.pt \
  --a-norm outputs/results/fr_smoke_A_norm.npy \
  --threshold 0.5 \
  --output outputs/results/fr_smoke_eval.json
```

Result:

```text
Wrote evaluation results to outputs/results/fr_smoke_eval.json
sequence_count=100
threshold=0.5
metrics=None
warnings=[]
```

`metrics=None` is expected for this smoke run because the converted SmartGuard normal JSONL does not contain anomaly labels.

Generated file:

```text
outputs/results/fr_smoke_eval.json
```

## Explanation Smoke

Command:

```bash
PYTHONPATH=src /home/heyang/miniconda3/envs/smartguard_env/bin/python -m causal_gen_guard.evaluation.evaluate \
  --sequences outputs/processed/fr_sequences_smoke.jsonl \
  --checkpoint-backbone outputs/checkpoints/fr_smoke_smartguard_backbone.pt \
  --checkpoint-causal outputs/checkpoints/fr_smoke_causal_predictor.pt \
  --a-norm outputs/results/fr_smoke_A_norm.npy \
  --threshold 0.5 \
  --output outputs/results/fr_smoke_eval_explain.json \
  --save-explanations \
  --num-explanations 1 \
  --explanation-dir outputs/results/explanations/fr_smoke \
  --figure-dir outputs/figures/fr_smoke
```

Result:

```text
Wrote evaluation results to outputs/results/fr_smoke_eval_explain.json
Generated 1 explanation JSON
```

Warnings recorded gracefully:

```text
Labels missing; score distribution normal/anomaly plot skipped
causal graph diff plot for smartguard_fr_train_000027 failed: No module named 'matplotlib'
```

The image warning did not fail evaluation. This confirms the visualization fallback path works when matplotlib is unavailable.

Generated files:

```text
outputs/results/fr_smoke_eval_explain.json
outputs/results/explanations/fr_smoke/smartguard_fr_train_000027_001.json
```

No figure PNG was generated because matplotlib is not installed in the active server environment.

## Generated Smoke Artifacts

```text
outputs/processed/fr_sequences_smoke.jsonl
outputs/checkpoints/fr_smoke_smartguard_backbone.pt
outputs/checkpoints/fr_smoke_causal_predictor.pt
outputs/results/fr_smoke_loss_vector.npy
outputs/results/fr_smoke_A_norm.npy
outputs/results/fr_smoke_causal_centrality.npy
outputs/results/fr_smoke_eval.json
outputs/results/fr_smoke_eval_explain.json
outputs/results/explanations/fr_smoke/smartguard_fr_train_000027_001.json
outputs/logs/SMOKE_TEST_REPORT.md
```

## Fixes Applied During Smoke Test

- Added `--limit` to `scripts/prepare_smartguard_data.py` so SmartGuard conversion can run in bounded smoke-test mode.

## Remaining TODO

- Install or select an environment with `pytest` if the preferred `python -m pytest tests` command should be used instead of direct test-file execution.
- Install or select an environment with `matplotlib` if figure PNG generation is required during smoke tests. Current behavior is a graceful warning and JSON-only explanations.
- Add labels or anomaly-injected JSONL for evaluation smoke runs that need non-null precision, recall, F1, AUROC, and AUPRC.
- Keep using small `--limit`, `--max-sequences`, and `--epochs 1` settings for smoke tests before any full experiment.
