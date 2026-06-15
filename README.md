# CausalGenGuard

**CausalGenGuard** is a research prototype for context-shift-aware smart-home behavior anomaly detection.

The project is built around a focused question:

> When a SmartGuard-style source-only detector is trained on one smart-home context, how badly does it false-alarm on normal behavior from a shifted target context, and can target-context synthetic normal data reduce that false positive rate?

The current repository focuses on **FR** and **SP** smart-home datasets:

- **FR**: France smart-home dataset.
- **SP**: Spain smart-home dataset.

The current experimental claim is deliberately narrow:

> Target-context synthetic normal adaptation reduces target-context normal false positive rate under seasonal context shift.

The project **does not** claim that the causal branch is a universal performance winner, nor that the system is a new overall SOTA anomaly detector.

---

## 1. Project status

Current cleaned repository status:

- Core code path is runnable.
- `outputs/` is intentionally excluded from version control, except `.gitkeep`.
- Heavy generated artifacts such as checkpoints, processed JSONL files, mappings, logs, and experiment CSV files should not be committed.
- Tests pass with the expected result:
  - With sibling `../SmartGuard` available: `33 passed`.
  - Without sibling `../SmartGuard`: `30 passed, 3 skipped`.

Run tests with:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src \
/home/heyang/miniconda3/envs/smartguard_env/bin/python \
-m pytest -q tests -p no:cacheprovider
```

---

## 2. Main idea

SmartGuard-style anomaly detection works well in static benchmark settings, but real smart homes often experience **context shift**:

- seasonal changes;
- changed routines;
- different target-context normal behavior.

A source-only anomaly detector may treat unseen but normal target-context behavior as anomalous, causing high false positive rate.

CausalGenGuard therefore evaluates whether target-context synthetic normal data can reduce this false alarm problem.

The current pipeline is:

```text
SmartGuard numeric logs
    -> SmartGuard dictionary parser
    -> canonical semantic BehaviorSequence JSONL
    -> dictionary-based semantic anomaly injection
    -> source-only / synthetic-adapted / TOF / Causal-TOF / oracle experiments
    -> FR/SP seasonal context-shift evaluation
```

---

## 3. What is implemented

### 3.1 SmartGuard dictionary parsing

SmartGuard source data uses numeric IDs. This project parses SmartGuard `dictionary.py` files and converts numeric IDs into semantic device/action names.

Example controls:

```text
Light:switch on
Light:switch off
Camera:switch on
Camera:switch off
SmartLock:lock lock
SmartLock:lock unlock
WaterValve:valve open
WaterValve:valve close
```

Relevant files:

```text
src/causal_gen_guard/data/smartguard_dictionary.py
scripts/build_smartguard_mapping.py
```

### 3.2 Unified behavior schema

The project converts raw logs into unified behavior sequences.

Relevant files:

```text
src/causal_gen_guard/data/schemas.py
src/causal_gen_guard/data/smartguard_adapter.py
docs/DATA_FORMAT.md
```

### 3.3 Semantic anomaly injection

Instead of injecting arbitrary numeric IDs, the project injects semantic SmartGuard-style anomalies using canonical controls.

Examples:

```text
SD_light_flickering
SD_camera_flickering
MD_camera_off_while_lock
DM_watervalve_open_midnight
DD_microwave_long_time
```

Relevant files:

```text
src/causal_gen_guard/data/attack_injector.py
scripts/build_labeled_anomaly_dataset.py
docs/reports/ROUTE_B_INJECTION_STATUS.md
```

### 3.4 Context-shift experiments

The main experiment compares five methods:

```text
source_only
source_plus_raw_synthetic
source_plus_tof_synthetic
source_plus_causal_tof_synthetic
oracle_target
```

Relevant files:

```text
scripts/run_context_shift_final.py
scripts/run_context_shift_multidataset.py
src/causal_gen_guard/evaluation/metrics.py
docs/reports/FINAL_EXPERIMENT_RESULTS.md
```

### 3.5 TOF and Causal-TOF filtering

The project includes synthetic-data filtering variants:

- **Raw synthetic**: directly use target-context synthetic normal data.
- **TOF synthetic**: reconstruction-score-based filtering.
- **Causal-TOF synthetic**: filtering with causal deviation diagnostics.

The current results show that filtering is dataset-dependent. Causal-TOF should be interpreted as a filtering/diagnostic ablation, not as a consistently best method.

Relevant files:

```text
src/causal_gen_guard/generation/causal_tof.py
src/causal_gen_guard/models/causal_graph.py
```

---

## 4. Repository structure

```text
CausalGenGuard/
  README.md
  requirements.txt
  requirements-dev.txt
  configs/
    context_shift_fr.yaml
    archive/
  docs/
    DATA_FORMAT.md
    LIMITATIONS.md
    reports/
      FINAL_EXPERIMENT_RESULTS.md
      ROUTE_B_INJECTION_STATUS.md
      ...
  scripts/
    build_smartguard_mapping.py
    prepare_smartguard_data.py
    build_labeled_anomaly_dataset.py
    run_context_shift_final.py
    run_context_shift_multidataset.py
    diagnose_context_shift_anomaly.py
    clean_repository_outputs.sh
  src/
    causal_gen_guard/
      data/
      generation/
      models/
      training/
      evaluation/
  tests/
  outputs/
    .gitkeep
```

---

## 5. Installation

Create or activate the environment used for this project.

Example:

```bash
conda activate smartguard_env
```

Install dependencies:

```bash
pip install -r requirements.txt
pip install -r requirements-dev.txt
```

Core dependencies include:

```text
torch
numpy
pandas
scikit-learn
pyyaml
tqdm
matplotlib
pytest
```

---

## 6. External project assumptions

The repository assumes the following sibling project layout for full local reproduction:

```text
~/projects/
  CausalGenGuard/
  SmartGuard/
  SmartGen/
```

Expected SmartGuard dictionary paths:

```text
../SmartGuard/data/data/fr/dictionary.py
../SmartGuard/data/data/sp/dictionary.py
../SmartGuard/data/data/an/dictionary.py
```

FR and SP are the current main datasets. US is not part of the current final experimental claim because its mapping path is less stable in this project version.

---

## 7. Reproduction workflow

### 7.1 Build SmartGuard semantic mappings

```bash
cd ~/projects/CausalGenGuard

PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src \
/home/heyang/miniconda3/envs/smartguard_env/bin/python \
scripts/build_smartguard_mapping.py \
  --smartguard-root ../SmartGuard \
  --dataset fr \
  --output-dir outputs/mappings/smartguard/fr
```

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src \
/home/heyang/miniconda3/envs/smartguard_env/bin/python \
scripts/build_smartguard_mapping.py \
  --smartguard-root ../SmartGuard \
  --dataset sp \
  --output-dir outputs/mappings/smartguard/sp
```

Check:

```bash
ls outputs/mappings/smartguard/fr/control_to_id.json
ls outputs/mappings/smartguard/sp/control_to_id.json
```

### 7.2 Prepare canonical SmartGuard data

Example for FR:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src \
/home/heyang/miniconda3/envs/smartguard_env/bin/python \
scripts/prepare_smartguard_data.py \
  --smartguard-root ../SmartGuard \
  --dataset fr \
  --output outputs/processed/fr_sequences_canonical.jsonl \
  --smartguard-dictionary ../SmartGuard/data/data/fr/dictionary.py \
  --emit-canonical-control \
  --mapping-output-dir outputs/mappings/smartguard/fr
```

Example for SP:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src \
/home/heyang/miniconda3/envs/smartguard_env/bin/python \
scripts/prepare_smartguard_data.py \
  --smartguard-root ../SmartGuard \
  --dataset sp \
  --output outputs/processed/sp_sequences_canonical.jsonl \
  --smartguard-dictionary ../SmartGuard/data/data/sp/dictionary.py \
  --emit-canonical-control \
  --mapping-output-dir outputs/mappings/smartguard/sp
```

### 7.3 Build labeled anomaly datasets

Example for FR:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src \
/home/heyang/miniconda3/envs/smartguard_env/bin/python \
scripts/build_labeled_anomaly_dataset.py \
  --normal-jsonl outputs/processed/fr_sequences_canonical.jsonl \
  --output-jsonl outputs/labels/fr_smartguard_style_labeled.jsonl \
  --report outputs/labels/fr_smartguard_style_labeled_report.json \
  --ratio 1.0 \
  --seed 42 \
  --anomaly-family smartguard
```

Example for SP:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src \
/home/heyang/miniconda3/envs/smartguard_env/bin/python \
scripts/build_labeled_anomaly_dataset.py \
  --normal-jsonl outputs/processed/sp_sequences_canonical.jsonl \
  --output-jsonl outputs/labels/sp_smartguard_style_labeled.jsonl \
  --report outputs/labels/sp_smartguard_style_labeled_report.json \
  --ratio 1.0 \
  --seed 42 \
  --anomaly-family smartguard
```

### 7.4 Run FR/SP context-shift experiments

Run three seeds.

Seed 42:

```bash
PYTHONWARNINGS=ignore PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 \
PYTHONPATH=src /home/heyang/miniconda3/envs/smartguard_env/bin/python \
scripts/run_context_shift_multidataset.py \
  --datasets fr,sp \
  --transitions seasonal \
  --seed 42 \
  --output-dir outputs/results/context_shift_multidataset_seed42 \
  2>&1 | tee outputs/logs/CONTEXT_SHIFT_SEED42.log
```

Seed 43:

```bash
PYTHONWARNINGS=ignore PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 \
PYTHONPATH=src /home/heyang/miniconda3/envs/smartguard_env/bin/python \
scripts/run_context_shift_multidataset.py \
  --datasets fr,sp \
  --transitions seasonal \
  --seed 43 \
  --output-dir outputs/results/context_shift_multidataset_seed43 \
  2>&1 | tee outputs/logs/CONTEXT_SHIFT_SEED43.log
```

Seed 44:

```bash
PYTHONWARNINGS=ignore PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 \
PYTHONPATH=src /home/heyang/miniconda3/envs/smartguard_env/bin/python \
scripts/run_context_shift_multidataset.py \
  --datasets fr,sp \
  --transitions seasonal \
  --seed 44 \
  --output-dir outputs/results/context_shift_multidataset_seed44 \
  2>&1 | tee outputs/logs/CONTEXT_SHIFT_SEED44.log
```

Successful runs should report:

```text
successful_runs=2 missing_runs=0
```

---

## 8. Final experiment results

The final FR/SP seasonal context-shift results are summarized over three seeds: 42, 43, and 44.

Values are reported as mean ± sample standard deviation.

| Dataset | Method | Precision | Recall | F1 | Target-normal FPR | FNR | AUROC | AUPRC |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| FR | Source-only | 0.3838 ± 0.0632 | 0.2053 ± 0.0681 | 0.2664 ± 0.0722 | 0.6362 ± 0.0424 | 0.7947 ± 0.0681 | 0.2178 ± 0.0077 | 0.5026 ± 0.0028 |
| FR | Raw synthetic | 0.3535 ± 0.1350 | 0.0547 ± 0.0147 | 0.0945 ± 0.0271 | 0.2116 ± 0.0814 | 0.9453 ± 0.0147 | 0.2727 ± 0.0102 | 0.5330 ± 0.0100 |
| FR | TOF synthetic | 0.4224 ± 0.0423 | 0.0567 ± 0.0050 | 0.0999 ± 0.0086 | 0.1548 ± 0.0241 | 0.9433 ± 0.0050 | 0.2908 ± 0.0040 | 0.5431 ± 0.0022 |
| FR | Causal-TOF synthetic | 0.3701 ± 0.0578 | 0.0600 ± 0.0243 | 0.1005 ± 0.0313 | 0.2235 ± 0.1536 | 0.9400 ± 0.0243 | 0.2692 ± 0.0133 | 0.5310 ± 0.0113 |
| FR | Oracle target | 0.6260 ± 0.0673 | 0.0673 ± 0.0202 | 0.1213 ± 0.0345 | 0.0767 ± 0.0061 | 0.9327 ± 0.0202 | 0.6171 ± 0.0073 | 0.7071 ± 0.0073 |
| SP | Source-only | 0.0220 ± 0.0012 | 0.2417 ± 0.0144 | 0.0403 ± 0.0022 | 0.8607 ± 0.0042 | 0.7583 ± 0.0144 | 0.1150 ± 0.0710 | 0.0500 ± 0.0148 |
| SP | Raw synthetic | 0.0814 ± 0.0288 | 0.2583 ± 0.0144 | 0.1226 ± 0.0341 | 0.2493 ± 0.0699 | 0.7417 ± 0.0144 | 0.4169 ± 0.0674 | 0.0717 ± 0.0136 |
| SP | TOF synthetic | 0.0719 ± 0.0482 | 0.2667 ± 0.0144 | 0.1081 ± 0.0579 | 0.3687 ± 0.2266 | 0.7333 ± 0.0144 | 0.3758 ± 0.0545 | 0.0713 ± 0.0168 |
| SP | Causal-TOF synthetic | 0.0474 ± 0.0140 | 0.2500 ± 0.0000 | 0.0791 ± 0.0203 | 0.4333 ± 0.1602 | 0.7500 ± 0.0000 | 0.3140 ± 0.0169 | 0.0596 ± 0.0073 |
| SP | Oracle target | 0.1222 ± 0.1072 | 0.0167 ± 0.0144 | 0.0293 ± 0.0254 | 0.0107 ± 0.0031 | 0.9833 ± 0.0144 | 0.8022 ± 0.0134 | 0.2162 ± 0.0262 |

Main observations:

- Source-only has high target-context false positive rate on both FR and SP.
- Raw synthetic adaptation substantially reduces target-normal FPR on both datasets.
- TOF synthetic gives the lowest FR FPR among non-oracle methods.
- Raw synthetic gives the lowest SP FPR among non-oracle methods.
- Causal-TOF is not consistently the best strategy and should be presented as a filtering/diagnostic ablation.
- Recall is low and FNR is high for many methods; therefore, the safest claim is reduced false positive rate under context shift, not overall SOTA anomaly detection performance.

---

## 9. Metrics

The main reported metrics are:

```text
Precision
Recall
F1
Target-normal FPR
FNR
AUROC
AUPRC
```

Definitions:

- **Precision**: fraction of predicted anomalies that are true anomalies.
- **Recall**: fraction of true anomalies detected.
- **F1**: harmonic mean of precision and recall.
- **Target-normal FPR**: false positive rate on target-context normal behavior.
- **FNR**: false negative rate on anomaly samples.
- **AUROC**: threshold-independent ranking quality.
- **AUPRC**: precision-recall curve area.

In this project, **target-normal FPR** is the core metric because the main problem is false alarms under context shift.

---

## 10. What can and cannot be claimed

Supported claims:

- SmartGuard-style source-only detectors suffer high false positives under target context shift.
- Target-context synthetic normal adaptation reduces target-normal FPR on FR and SP.
- Oracle target normal data produces very low FPR, showing that target-context adaptation is important.
- Filtering strategies are dataset-dependent.

Unsupported or unsafe claims:

- Causal-TOF is universally better than raw synthetic or TOF.
- The causal branch is the main source of overall performance gain.
- The method achieves SOTA anomaly detection performance.
- The method solves US, ARGUS, or cross-home generalization in the current version.

---

## 11. Cleaning generated files

To clean generated artifacts before committing:

```bash
bash scripts/clean_repository_outputs.sh
```

This should remove generated files under `outputs/`, caches, logs, and local artifacts while keeping `outputs/.gitkeep`.

Do not commit:

```text
outputs/checkpoints/
outputs/processed/
outputs/labels/
outputs/mappings/
outputs/results/
outputs/logs/
outputs/synthetic/
*.pt
*.pth
*.npy
*.npz
*.pkl
*.pickle
__pycache__/
.pytest_cache/
```

---

## 12. Recommended final documentation layout

Recommended final docs:

```text
docs/
  DATA_FORMAT.md
  LIMITATIONS.md
  reports/
    FINAL_EXPERIMENT_RESULTS.md
    ROUTE_B_INJECTION_STATUS.md
```

Older Route-B intermediate reports can be removed after the final report has been committed.

---

## 13. Development notes

This repository is a research prototype for a small-paper experiment. The main value is a clean, reproducible experimental path for studying context-shift false positives in smart-home anomaly detection.

The most defensible paper framing is:

> CausalGenGuard reduces false alarms caused by seasonal context shift through target-context synthetic normal adaptation, while causal filtering is treated as an ablation and diagnostic component.