# CausalGenGuard

## Project Overview

CausalGenGuard is a context-adaptive anomaly detection framework for behavior-based IoT systems.

It combines three complementary ideas:

- SmartGuard reconstruction: learn normal behavior sequences and score events by reconstruction difficulty.
- SmartGen context generation: use offline synthetic or generated normal behavior to cover target-context shifts.
- GCAD-style causal graph deviation: model behavior dependencies as event-channel causal graphs and score deviations from normal causal patterns.

In short:

```text
CausalGenGuard = SmartGuard reconstruction
               + SmartGen context generation
               + GCAD-style causal graph deviation
```

The goal is not only to reproduce standard injected-anomaly benchmarks, but to study whether behavior detectors remain reliable when context, home layout, country, or behavior dependencies shift.

## Why Not Only Improve SmartGuard Standard Benchmark

The original SmartGuard-style FR/SP/AN injected-anomaly setting is useful for validating reconstruction-based detection, but it is close to saturated for many standard attacks. Further gains on that benchmark alone may not reveal whether a detector generalizes under realistic deployment shifts.

CausalGenGuard therefore focuses on harder settings:

- Context shift: source context to target context, such as winter to spring, daytime to night, or single-person to multi-person behavior.
- Cross-home generalization: train on some homes and evaluate on held-out ARGUS homes.
- Cross-country generalization: use SmartSense-style FR/SP/US/KR data when locally available.
- Causal dependency anomaly: detect attacks that preserve marginal event frequencies but break normal behavior dependencies.

Standard SmartGuard reproduction remains a sanity check, not the main contribution.

## Data Preparation

No script downloads data automatically. Put source datasets on the server first, then run preparation scripts from the CausalGenGuard root.

### SmartGuard Data

SmartGuard FR/SP data should be prepared from the local source project, expected at a path such as `/home/heyang/projects/SmartGuard`.

```bash
PYTHONPATH=src python scripts/prepare_smartguard_data.py \
  --smartguard-root ../SmartGuard \
  --dataset fr \
  --output outputs/processed/fr_sequences.jsonl
```

Preparation is a conversion step only. It does not train a model. For a smoke-style check, point `--output` to a temporary JSONL and use a tiny source fixture if available.

### SmartGen Synthetic / Offline Data

SmartGen synthetic data is loaded offline from local files. No online LLM call is made and no API key is required.

```bash
PYTHONPATH=src python scripts/prepare_smartgen_data.py \
  --smartgen-root ../SmartGen \
  --dataset fr \
  --transition ST \
  --output outputs/synthetic/fr_ST.jsonl
```

Optional Causal-TOF filtering can be enabled after local backbone and causal artifacts exist:

```bash
PYTHONPATH=src python scripts/prepare_smartgen_data.py \
  --smartgen-root ../SmartGen \
  --dataset fr \
  --transition ST \
  --output outputs/synthetic/fr_ST_tof.jsonl \
  --apply-causal-tof
```

### SmartSense Data

SmartSense is intended for FR/SP/US/KR cross-country evaluation. The adapter auto-detects `csv`, `json`, `jsonl`, `txt`, `pickle`, log datasets, dictionaries, and routine datasets.

Expected event information is equivalent to:

```text
timestamp, device, action/control/state
```

If source columns differ, pass a columns mapping when calling the adapter in Python, for example:

```python
from causal_gen_guard.data.smartsense_adapter import load_smartsense_dataset

dataset = load_smartsense_dataset(
    '/path/to/SmartSense',
    columns={'timestamp': 'event_time', 'device': 'sensor_name', 'action': 'state'},
    window_size=50,
)
```

### ARGUS Data

ARGUS is intended for Home1-Home5 cross-home evaluation. The adapter scans `Home*` folders, reads local `csv/json/jsonl` event tables, converts device state changes into `BehaviorEvent`, and records `home_id` in sequence context.

```bash
PYTHONPATH=src python scripts/prepare_argus_data.py \
  --argus-root /path/to/ARGUS \
  --output outputs/processed/argus_sequences.jsonl \
  --split leave_one_home \
  --window-size 50
```

If ARGUS columns differ:

```bash
PYTHONPATH=src python scripts/prepare_argus_data.py \
  --argus-root /path/to/ARGUS \
  --output outputs/processed/argus_sequences.jsonl \
  --split temporal \
  --window-size 50 \
  --timestamp-column event_time \
  --device-column sensor_name \
  --action-column state
```

## Training

Training entry points are designed to support small smoke runs first. Use the smoke versions before launching full experiments.

### Train Backbone

```bash
PYTHONPATH=src python -m causal_gen_guard.training.train_backbone \
  --input-jsonl outputs/processed/fr_sequences.jsonl \
  --checkpoint-path outputs/checkpoints/fr_smartguard_backbone.pt \
  --loss-vector-output outputs/results/fr_loss_vector.npy \
  --epochs 1 \
  --max-sequences 8
```

Smoke-test style:

```bash
PYTHONPATH=src python -m causal_gen_guard.training.train_backbone \
  --input-jsonl outputs/processed/toy_sequences.jsonl \
  --checkpoint-path outputs/checkpoints/toy_smartguard_backbone.pt \
  --loss-vector-output outputs/results/toy_loss_vector.npy \
  --epochs 1 \
  --max-sequences 4
```

### Train Causal Predictor

```bash
PYTHONPATH=src python -m causal_gen_guard.training.train_causal \
  --input-jsonl outputs/processed/fr_sequences.jsonl \
  --checkpoint-path outputs/checkpoints/fr_causal_predictor.pt \
  --graph-output outputs/results/fr_A_norm.npy \
  --centrality-output outputs/results/fr_causal_centrality.npy \
  --epochs 1 \
  --max-sequences 8
```

Smoke-test style:

```bash
PYTHONPATH=src python -m causal_gen_guard.training.train_causal \
  --input-jsonl outputs/processed/toy_sequences.jsonl \
  --checkpoint-path outputs/checkpoints/toy_causal_predictor.pt \
  --graph-output outputs/results/toy_A_norm.npy \
  --centrality-output outputs/results/toy_causal_centrality.npy \
  --epochs 1 \
  --max-sequences 4
```

### Train Fusion Detector

```bash
PYTHONPATH=src python -m causal_gen_guard.training.train_fusion \
  --config configs/context_shift_fr.yaml \
  --smoke-test
```

Full training should be launched only after paths and smoke outputs are checked:

```bash
PYTHONPATH=src python -m causal_gen_guard.training.train_fusion \
  --config configs/context_shift_fr.yaml
```

## Experiments

### Standard Reproduction

Use SmartGuard-format prepared data to verify that the reconstruction backbone and scoring pipeline run end to end.

```bash
bash scripts/run_standard_repro.sh
```

Smoke-test equivalent:

```bash
PYTHONPATH=src python -m causal_gen_guard.training.train_backbone \
  --input-jsonl outputs/processed/toy_sequences.jsonl \
  --epochs 1 \
  --max-sequences 4
```

### Context Shift

FR/SP/US context-shift configs live under `configs/`.

```bash
scripts/run_context_shift.sh configs/context_shift_fr.yaml smoke
```

Direct smoke-test command:

```bash
PYTHONPATH=src python -m causal_gen_guard.training.train_fusion \
  --config configs/context_shift_fr.yaml \
  --smoke-test
```

### Low-Data Target Adaptation

Low-data target adaptation can be represented by combining a source normal set with a small target or synthetic target JSONL in the context-shift config. Use `--smoke-test` first:

```bash
PYTHONPATH=src python -m causal_gen_guard.training.train_fusion \
  --config configs/context_shift_fr.yaml \
  --smoke-test \
  --run-name low_data_target_smoke
```

### Cross-Home / Cross-Country

ARGUS cross-home starts with leave-one-home preparation:

```bash
PYTHONPATH=src python scripts/prepare_argus_data.py \
  --argus-root /path/to/ARGUS \
  --output outputs/processed/argus_sequences.jsonl \
  --split leave_one_home \
  --window-size 50
```

Then point a config at the generated train/val/test files and run smoke fusion training:

```bash
PYTHONPATH=src python -m causal_gen_guard.training.train_fusion \
  --config configs/context_shift_fr.yaml \
  --smoke-test \
  --run-name cross_home_smoke
```

SmartSense cross-country follows the same JSONL schema after adapter conversion.

### Causal Anomaly

Causal dependency anomalies are generated under CausalGenGuard only.

```bash
bash scripts/run_causal_anomaly.sh
```

Smoke-test evaluation after generating a tiny anomaly JSONL:

```bash
PYTHONPATH=src python -m causal_gen_guard.evaluation.evaluate \
  --sequences outputs/processed/toy_sequences.jsonl \
  --checkpoint-backbone outputs/checkpoints/toy_smartguard_backbone.pt \
  --checkpoint-causal outputs/checkpoints/toy_causal_predictor.pt \
  --a-norm outputs/results/toy_A_norm.npy \
  --threshold 0.5 \
  --output outputs/results/toy_eval.json \
  --save-explanations \
  --num-explanations 2
```

### Ablation

Ablation methods include:

- SmartGuard only
- SmartGuard + SmartGen
- SmartGuard + Causal
- SmartGuard + SmartGen + Causal
- Full CausalGenGuard + CausalTOF + causal-aware NWRL

```bash
PYTHONPATH=src python -m causal_gen_guard.evaluation.ablation \
  --config configs/context_shift_fr.yaml \
  --smoke-test
```

## Example Commands

All experiment and training examples should be smoke-tested before full runs.

```bash
cd /home/heyang/projects/CausalGenGuard

PYTHONPATH=src python -m causal_gen_guard.training.train_fusion \
  --config configs/context_shift_fr.yaml \
  --smoke-test

PYTHONPATH=src python -m causal_gen_guard.evaluation.ablation \
  --config configs/context_shift_fr.yaml \
  --smoke-test

scripts/run_context_shift.sh configs/context_shift_fr.yaml smoke

PYTHONPATH=src python -m causal_gen_guard.evaluation.evaluate \
  --sequences outputs/processed/toy_sequences.jsonl \
  --checkpoint-backbone outputs/checkpoints/toy_smartguard_backbone.pt \
  --checkpoint-causal outputs/checkpoints/toy_causal_predictor.pt \
  --a-norm outputs/results/toy_A_norm.npy \
  --threshold 0.5 \
  --output outputs/results/toy_eval_explain.json \
  --save-explanations \
  --num-explanations 2
```

## Output Files

Generated artifacts stay under `outputs/`:

- `outputs/checkpoints/`: SmartGuard backbone checkpoints, causal predictor checkpoints, fusion-stage checkpoints.
- `outputs/results/`: evaluation JSON, ablation CSV, context-shift summaries, normal causal graphs, centrality arrays.
- `outputs/results/explanations/`: case-study-ready explanation JSON files with scores, abnormal tokens, abnormal causal edges, context graph id, detector metadata, and warnings.
- `outputs/figures/`: score distributions and causal graph difference figures when `matplotlib` is available.
- `outputs/synthetic/`: offline SmartGen synthetic data and Causal-TOF kept/rejected/report files.
- `outputs/processed/`: normalized `BehaviorSequence` JSONL files.

## Safety Constraints

- Original projects are read-only:
  - `/home/heyang/projects/SmartGuard`
  - `/home/heyang/projects/SmartGen`
  - `/home/heyang/projects/GCAD`
- All modified or copied code must live under `/home/heyang/projects/CausalGenGuard`.
- No script should download datasets by default.
- No script should call an online LLM unless an API client is explicitly added and configured by the user.
- Smoke tests should run before full training or long experiments.

## Repository Layout

```text
configs/                  Experiment configuration files.
scripts/                  Data preparation and run entry points.
src/causal_gen_guard/     CausalGenGuard Python package.
tests/                    Unit and smoke tests.
outputs/                  Local outputs for checkpoints, logs, results, figures, synthetic data.
PROJECT_AUDIT.md          Initial source-project audit.
```

## TODO

- Replace the simplified SSC similarity compressor with exact SmartGen SPPC logic if needed.
- Add a real LLM client only when an API key is explicitly configured.
- Add more robust device-action mapping for ARGUS and SmartSense.
- Add stronger SmartSense-to-JSONL command-line preparation once real local source layouts are confirmed.
- Add full low-data target adaptation configs after real target splits are fixed.
