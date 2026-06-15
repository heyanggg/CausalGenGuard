#!/usr/bin/env bash
set -euo pipefail

cd /home/heyang/projects/CausalGenGuard

cat <<'EOF'
Example causal anomaly workflow:

1. Prepare normal SmartGuard or SmartGen sequences:
   python3 scripts/prepare_smartguard_data.py --smartguard-root ../SmartGuard --dataset fr --output outputs/processed/fr_sequences_canonical.jsonl
   python3 scripts/prepare_smartgen_data.py --smartgen-root ../SmartGen --dataset fr --transition ST --output outputs/synthetic/fr_ST.jsonl --apply-causal-tof

2. Inject causal anomalies from a small Python driver or notebook using:
   from causal_gen_guard.data.attack_injector import generate_anomaly_dataset
   mixed, report = generate_anomaly_dataset(normal_sequences, [{'family': 'causal', 'type': 'causal_edge_injection'}], ratio=0.2, seed=42)

3. Evaluate with the fusion detector:
   PYTHONPATH=src /home/heyang/miniconda3/envs/smartguard_env/bin/python -m causal_gen_guard.evaluation.evaluate --sequences outputs/processed/fr_causal_anomaly.jsonl --checkpoint-backbone outputs/checkpoints/smartguard_backbone_fr.pt --checkpoint-causal outputs/checkpoints/causal_predictor_fr.pt --a-norm outputs/results/A_norm_fr.npy --threshold 0.0 --output outputs/results/fr_causal_anomaly_eval.json

This script prints commands only. It does not inject data, train models, or run evaluation automatically.
EOF
