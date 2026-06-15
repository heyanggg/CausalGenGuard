# CausalGenGuard

CausalGenGuard is a research prototype for **context-shift-aware IoT behavior anomaly detection**. It builds on three ideas:

1. **SmartGuard-style sequence modeling** for IoT behavior reconstruction.
2. **SmartGen-style target-context synthetic adaptation** for reducing false positives under context shift.
3. **GCAD-style causal dependency analysis** for causal filtering and diagnostic ablations.

The current focus is not to simply improve accuracy on a saturated in-distribution SmartGuard benchmark. Instead, this project studies a harder setting: **IoT anomaly detection under context shift**, where a model trained in one context, such as a season, home, or region, is evaluated in another context.

---

## Project Status

The current implementation supports:

* SmartGuard dictionary parsing for FR/SP/AN datasets.
* Canonical semantic control conversion from SmartGuard numeric IDs.
* Dictionary-based SmartGuard-style semantic anomaly injection.
* Context-shift experiments over FR/SP seasonal transitions.
* Target-context synthetic adaptation baselines.
* TOF and Causal-TOF filtering ablations.
* Seeded multi-run evaluation for stability analysis.

Recent validated status:

* FR/SP canonical conversion works.
* FR/SP dictionary-based semantic anomaly injection supports 9 SmartGuard-style anomaly types.
* `fallback_numeric_injection_count = 0`.
* Route B context-shift smoke and main-scale runs complete successfully.
* Multi-seed FR/SP seasonal experiments show that target-context synthetic adaptation reduces target-normal false positive rate under context shift.

Important interpretation:

> Current results support the claim that target-context synthetic adaptation reduces false positives under seasonal context shift. TOF and Causal-TOF should currently be interpreted as filtering ablations rather than universally dominant methods.

---

## Repository Structure

```text
CausalGenGuard/
├── configs/                         # Experiment configs
├── docs/
│   └── reports/                     # Curated experiment reports
├── scripts/                         # Data preparation and experiment entrypoints
├── src/
│   └── causal_gen_guard/
│       ├── data/                    # SmartGuard dictionary and anomaly injection
│       ├── models/                  # Model components
│       └── ...
├── tests/                           # Unit tests
├── outputs/                         # Local generated outputs, ignored by git
├── requirements.txt
├── requirements-dev.txt
└── README.md
```

`outputs/` is a local generated directory. Large results, generated JSONL files, checkpoints, mappings, and logs should not be committed. Curated reports should be copied to `docs/reports/`.

---

## External Project Layout

This repository expects the original data/projects to be available locally as sibling directories:

```text
~/projects/
├── CausalGenGuard/
├── SmartGuard/
├── SmartGen/
└── GCAD/
```

Typical paths used by the scripts:

```text
../SmartGuard
../SmartGen
../GCAD
```

This repository does not download datasets automatically.

---

## Environment

The development environment used for the current experiments is:

```text
conda environment: smartguard_env
python: 3.8.x
```

Activate the environment:

```bash
cd /home/heyang/projects/CausalGenGuard

source /home/heyang/miniconda3/etc/profile.d/conda.sh
conda activate smartguard_env

export PYTHONPATH=src
```

Install dependencies if needed:

```bash
pip install -r requirements.txt
pip install -r requirements-dev.txt
```

Run tests:

```bash
PYTHONPATH=src python -m pytest -q tests
```

Current validated test status:

```text
35 passed
```

---

## Safety and Reproducibility Constraints

This project is intended to run fully offline on local files.

The experimental workflow should follow these constraints:

* Do not download datasets automatically.
* Do not call online LLM APIs during experiments.
* Do not commit generated large outputs, checkpoints, or temporary logs.
* Keep generated experiment files under `outputs/`.
* Move only curated Markdown reports to `docs/reports/`.

---

## SmartGuard Semantic Mapping

SmartGuard original behavior sequences use numeric device/control IDs. CausalGenGuard converts them into canonical semantic controls such as:

```text
Light:switch on
Camera:switch off
SmartLock:lock lock
WaterValve:valve open
```

Build FR mapping:

```bash
PYTHONPATH=src python scripts/build_smartguard_mapping.py \
  --smartguard-root ../SmartGuard \
  --dataset fr \
  --output-dir outputs/mappings/smartguard/fr
```

Build SP mapping:

```bash
PYTHONPATH=src python scripts/build_smartguard_mapping.py \
  --smartguard-root ../SmartGuard \
  --dataset sp \
  --output-dir outputs/mappings/smartguard/sp
```

The mapping outputs include:

```text
device_to_id.json
id_to_device.json
control_to_id.json
id_to_control.json
mapping_report.json
```

For FR/SP, the mapping should report that key SmartGuard-style semantic controls are available. AN can be parsed, but it does not contain the same FR/SP SmartGuard-style key controls and is not currently used for the main named anomaly set.

---

## Canonical SmartGuard Data Preparation

Prepare FR canonical sequences:

```bash
PYTHONPATH=src python scripts/prepare_smartguard_data.py \
  --smartguard-root ../SmartGuard \
  --dataset fr \
  --output outputs/processed/fr_sequences_canonical.jsonl \
  --smartguard-dictionary ../SmartGuard/data/data/fr/dictionary.py \
  --emit-canonical-control \
  --mapping-output-dir outputs/mappings/smartguard/fr
```

Prepare SP canonical sequences:

```bash
PYTHONPATH=src python scripts/prepare_smartguard_data.py \
  --smartguard-root ../SmartGuard \
  --dataset sp \
  --output outputs/processed/sp_sequences_canonical.jsonl \
  --smartguard-dictionary ../SmartGuard/data/data/sp/dictionary.py \
  --emit-canonical-control \
  --mapping-output-dir outputs/mappings/smartguard/sp
```

Each event contains semantic fields such as:

```text
device
canonical_control
action
raw_device_id
raw_control_id
day_name
hour_name
```

---

## Route B: Dictionary-based Semantic Anomaly Injection

Route B upgrades anomaly injection from source-sequence-only semantic injection to dictionary-based semantic control-pool injection.

The injection policy is:

1. If the target control exists in the current normal sequence, reuse the sequence event template.
2. If the target control does not exist in the current sequence but exists in the dictionary/mapping control pool, construct a valid canonical event from the dictionary.
3. If the target control does not exist in the dictionary/mapping, skip the anomaly type.
4. Never fall back to arbitrary numeric ID injection.

This means:

```text
fallback_numeric_injection_count must be 0
```

Build FR labeled anomalies:

```bash
PYTHONPATH=src python scripts/build_labeled_anomaly_dataset.py \
  --input-jsonl outputs/processed/fr_sequences_canonical.jsonl \
  --output-jsonl outputs/labels/fr_smartguard_style_labeled.jsonl \
  --report outputs/labels/fr_smartguard_style_labeled_report.json \
  --per-anomaly-type 50 \
  --seed 42 \
  --mapping-dir outputs/mappings/smartguard/fr
```

Build SP labeled anomalies:

```bash
PYTHONPATH=src python scripts/build_labeled_anomaly_dataset.py \
  --input-jsonl outputs/processed/sp_sequences_canonical.jsonl \
  --output-jsonl outputs/labels/sp_smartguard_style_labeled.jsonl \
  --report outputs/labels/sp_smartguard_style_labeled_report.json \
  --per-anomaly-type 50 \
  --seed 42 \
  --mapping-dir outputs/mappings/smartguard/sp
```

Expected Route B injection status:

```text
FR:
normal_count = 3196
anomaly_count = 450
named_injection_success_count = 450
source_sequence_template_injection_count = 44
dictionary_control_pool_injection_count = 406
fallback_numeric_injection_count = 0
supported anomaly types = 9/10

SP:
normal_count = 11062
anomaly_count = 450
named_injection_success_count = 450
source_sequence_template_injection_count = 44
dictionary_control_pool_injection_count = 406
fallback_numeric_injection_count = 0
supported anomaly types = 9/10
```

The unsupported anomaly type is:

```text
DD_shower_long_time
```

because the FR/SP dictionaries do not contain a Shower control.

Supported SmartGuard-style semantic anomaly types:

```text
DD_microwave_long_time
DM_ac_cool_in_winter
DM_watervalve_open_midnight
DM_window_open_midnight
MD_camera_off_while_lock
MD_window_open_while_lock
SD_camera_flickering
SD_light_flickering
SD_tv_flickering
```

---

## Context-shift Experiments

The main context-shift entrypoint is:

```bash
scripts/run_context_shift_multidataset.py
```

It supports:

```text
--datasets
--transitions
--max-normal
--max-anomaly
--max-synthetic
--epochs
--seed
--output-dir
```

The current SmartGuard mapping type used in new context-shift outputs is:

```text
smartguard_semantic
```

Run a small FR/SP smoke experiment:

```bash
PYTHONWARNINGS=ignore PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
PYTHONPATH=src python scripts/run_context_shift_multidataset.py \
  --datasets fr,sp \
  --transitions seasonal \
  --max-normal 300 \
  --max-anomaly 100 \
  --max-synthetic 100 \
  --epochs 1 \
  --seed 42 \
  --output-dir outputs/results/route_b_context_shift_frsp_smoke
```

Run a main-scale single run:

```bash
PYTHONWARNINGS=ignore PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
PYTHONPATH=src python scripts/run_context_shift_multidataset.py \
  --datasets fr,sp \
  --transitions seasonal \
  --max-normal 1000 \
  --max-anomaly 300 \
  --max-synthetic 300 \
  --epochs 3 \
  --seed 42 \
  --output-dir outputs/results/route_b_context_shift_frsp_main
```

Run a three-seed medium-scale stability experiment:

```bash
for seed in 42 43 44; do
  PYTHONWARNINGS=ignore PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  PYTHONPATH=src python scripts/run_context_shift_multidataset.py \
    --datasets fr,sp \
    --transitions seasonal \
    --max-normal 600 \
    --max-anomaly 200 \
    --max-synthetic 200 \
    --epochs 2 \
    --seed $seed \
    --output-dir outputs/results/route_b_context_shift_frsp_seed_${seed} \
    2>&1 | tee outputs/logs/ROUTE_B_CONTEXT_SHIFT_FRSP_SEED_${seed}.log
done
```

---

## Summarizing Multi-seed Results

Use the following helper snippet to summarize multiple seed outputs:

```bash
python - <<'PY'
import csv
import statistics as st
from pathlib import Path
from collections import defaultdict

paths = sorted(Path("outputs/results").glob("route_b_context_shift_frsp_seed_*/summary.csv"))
print("found summaries:", len(paths))
for p in paths:
    print(" ", p)

groups = defaultdict(list)

for p in paths:
    seed = p.parent.name.split("_")[-1]
    with open(p, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r["status"] != "success":
                print("NON-SUCCESS", p, r)
                continue
            key = (r["dataset"], r["method"])
            groups[key].append({
                "seed": seed,
                "fpr": float(r["target_normal_fpr"]),
                "f1": float(r["f1"]),
                "auroc": float(r["auroc"]),
                "auprc": float(r["auprc"]),
                "gain": None if r["adaptation_gain"] == "" else float(r["adaptation_gain"]),
            })

print("\n=== mean/std ===")
for (dataset, method), vals in sorted(groups.items()):
    def mean_std(name):
        xs = [v[name] for v in vals if v[name] is not None]
        if not xs:
            return "-"
        mean = st.mean(xs)
        std = st.stdev(xs) if len(xs) >= 2 else 0.0
        return f"{mean:.4f} ± {std:.4f}"

    print(
        dataset,
        method,
        "n=", len(vals),
        "FPR=", mean_std("fpr"),
        "F1=", mean_std("f1"),
        "AUROC=", mean_std("auroc"),
        "AUPRC=", mean_std("auprc"),
        "gain=", mean_std("gain"),
    )
PY
```

Current three-seed medium-scale result summary:

```text
FR:
source_only FPR = 0.4193 ± 0.0893
raw synthetic FPR = 0.3505 ± 0.0502
TOF synthetic FPR = 0.3175 ± 0.0525
Causal-TOF synthetic FPR = 0.3214 ± 0.0757
oracle target FPR = 0.0529 ± 0.0165

SP:
source_only FPR = 0.8507 ± 0.0431
raw synthetic FPR = 0.4993 ± 0.1212
TOF synthetic FPR = 0.5687 ± 0.2470
Causal-TOF synthetic FPR = 0.6033 ± 0.0514
oracle target FPR = 0.0093 ± 0.0050
```

---

## Current Experimental Interpretation

The stable conclusion is:

1. Source-only models suffer from high target-context false positive rates under seasonal context shift.
2. Target-context synthetic adaptation substantially reduces FPR on both FR and SP.
3. Oracle target training gives the expected lower-bound FPR reference.
4. TOF and Causal-TOF are useful filtering ablations, but their effect is dataset-dependent.
5. Current evidence does not support claiming that Causal-TOF is uniformly best.

This project should therefore be framed as a semantic context-shift evaluation and adaptation framework, with causal filtering as an ablation and diagnostic component.

---

## Important Reports

Curated reports should be stored in:

```text
docs/reports/
```

Recommended report files:

```text
ROUTE_B_INJECTION_STATUS.md
ROUTE_B_CONTEXT_SHIFT_FRSP_SMOKE_STATUS.md
ROUTE_B_CONTEXT_SHIFT_FRSP_MAIN_STATUS.md
ROUTE_B_CONTEXT_SHIFT_FRSP_SEED_STABILITY_STATUS.md
ROUTE_B_FILTER_DIAGNOSTIC_STATUS.md
```

Generated raw outputs remain under:

```text
outputs/
```

and should not be committed.

---

## Git Hygiene

Clean Python caches:

```bash
find . -type d -name "__pycache__" -prune -exec rm -rf {} +
find . -type d -name ".pytest_cache" -prune -exec rm -rf {} +
find . -type f -name "*.pyc" -delete
```

Remove already tracked caches from Git if needed:

```bash
git rm -r --cached tests/__pycache__ 2>/dev/null || true
git rm -r --cached src/**/__pycache__ 2>/dev/null || true
git rm -r --cached scripts/__pycache__ 2>/dev/null || true
```

Do not commit generated experiment outputs:

```text
outputs/processed/
outputs/labels/
outputs/mappings/
outputs/results/
outputs/checkpoints/
outputs/logs/*.log
```

Commit only:

```text
source code
tests
configs
README
curated docs/reports/*.md
```

---

## Test Suite

Run all tests:

```bash
PYTHONPATH=src python -m pytest -q tests
```

The test suite covers:

* SmartGuard dictionary parsing.
* Named semantic attack injection.
* Dictionary-based control-pool injection.
* Prevention of numeric fallback injection.
* Context-shift mapping type behavior.
* Seed argument behavior for multidataset context-shift runs.

---

## Citation / Research Use

This repository is currently a research prototype for a small-paper project. The current recommended paper framing is:

> Semantic context-shift evaluation and target-context synthetic adaptation for IoT behavior anomaly detection.

Causal filtering is included as an ablation and diagnostic component, not yet as a universally superior method.
