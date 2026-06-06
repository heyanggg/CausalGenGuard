# CausalGenGuard Project Audit

Audit date: 2026-06-05
Remote root: `/home/heyang/projects`

## Directory Presence

All four expected project directories exist under `/home/heyang/projects`:

- `/home/heyang/projects/SmartGuard`
- `/home/heyang/projects/SmartGen`
- `/home/heyang/projects/GCAD`
- `/home/heyang/projects/CausalGenGuard`

## Source Project Trees (max depth 3)

### SmartGuard

Key depth-3 structure observed:

```text
SmartGuard/
  SmartGuard.py
  train.py
  other_baseline.py
  evaluate_smartguard.py
  README.md
  data/
    an_data/
      an_test_instance_10.pkl
      an_trn_instance_10.pkl
      an_vld_instance_10.pkl
      attack/
    fr_data/
      fr_add_trn.pkl
      fr_test_instance_10.pkl
      fr_trn_instance_10.pkl
      fr_vld_instance_10.pkl
      labeled_fr_test.pkl
      attack/
    sp_data/
      sp_add_trn.pkl
      sp_test_instance_10.pkl
      sp_trn_instance_10.pkl
      sp_vld_instance_10.pkl
      attack/
    data/
      an/
      fr/
      sp/
  experiments/
    case/
    figure/
    results/
    Figure1_plot.py
    Figure6_plot.py
    RQ3_parameter_study.py
    RQ4_case_study.py
  figures/
    SmartGuard.png
  logs/
  results/
  saved_model/
```

Required SmartGuard items:

- `SmartGuard.py`: found
- `train.py`: found
- `other_baseline.py`: found
- `data/`: found

### SmartGen

Key depth-3 structure observed:

```text
SmartGen/
  SmartGen/
    split.py
    sppc.py
    text_translation_matrix.py
    security_check.py
  anomaly_detection_pipeline/
    Anomaly_Detection_pipeline_model.py
    main.py
    models1.py
    attack/
    check_model/
    results/
    synthetic_data/
    test/
  anomaly_detection_baseline/
    Anomaly_Detection_baseline_autoencoder.py
    Anomaly_Detection_baseline_models.py
    main.py
    baseline_data/
    results/
    saved_model/
  behavior_prediciton_pipeline/
    SASRec/
  behavior_prediciton_baseline/
    SASRec/
    CARnn/
    Caser/
    FMC/
    HMM/
    LSTM/
    SITAR/
  ablation_study/
    main.py
    model.py
    models1.py
    SASRec.py
    security_check.py
    transsas.py
    attack/
    data/
    results/
    study_data/
    test/
```

Required SmartGen items:

- `SmartGen/split.py`: found
- `SmartGen/sppc.py`: found
- `SmartGen/text_translation_matrix.py`: found
- `SmartGen/security_check.py`: found
- `anomaly_detection_pipeline/`: found
- Similar related directory `anomaly_detection_baseline/`: found

### GCAD

Key depth-3 structure observed:

```text
GCAD/
  main.py
  test.py
  requirements.txt
  README.md
  LICENSE
  run.sh
  datasets/
  img/
    model.png
  models/
    common.py
    tsmixer.py
  result/
  utils/
    dataloader.py
    general.py
```

Required GCAD items:

- `main.py`: found
- `test.py`: found
- `models/`: found
- `utils/`: found
- `requirements.txt`: found

## Target Project Status

`/home/heyang/projects/CausalGenGuard` was empty before this audit file was created. The read-only listing returned only:

```text
.
```

This audit writes only:

- `/home/heyang/projects/CausalGenGuard/PROJECT_AUDIT.md`

No training, dependency installation, or data download was run.

## Suggested Files To Copy Later

Recommended source files for a future integration pass, subject to design decisions:

- From SmartGuard:
  - `SmartGuard.py`
  - `train.py`
  - `other_baseline.py`
  - `evaluate_smartguard.py`
  - `README.md`
  - selected `data/` metadata or small sample references only, if needed
- From SmartGen:
  - `SmartGen/split.py`
  - `SmartGen/sppc.py`
  - `SmartGen/text_translation_matrix.py`
  - `SmartGen/security_check.py`
  - `anomaly_detection_pipeline/main.py`
  - `anomaly_detection_pipeline/Anomaly_Detection_pipeline_model.py`
  - `anomaly_detection_pipeline/models1.py`
  - possibly `anomaly_detection_baseline/` for baseline comparisons
- From GCAD:
  - `main.py`
  - `test.py`
  - `requirements.txt`
  - `models/`
  - `utils/`
  - `README.md`

## Next-Step Recommendations

1. Decide the intended `CausalGenGuard` package layout before copying code.
2. Copy code-only modules first; keep datasets, logs, checkpoints, and generated results out of the initial import unless explicitly needed.
3. Compare dependency requirements across the three source projects before installing anything.
4. Add a small smoke-test entry point after copying code, then run it without training or downloading data.
5. Preserve source provenance in comments or a migration log so imported modules remain traceable to SmartGuard, SmartGen, or GCAD.
