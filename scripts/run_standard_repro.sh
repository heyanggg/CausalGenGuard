#!/usr/bin/env bash
set -euo pipefail

cd /home/heyang/projects/CausalGenGuard

cat <<'EOF'
Example standard reconstruction workflow:

1. Prepare SmartGuard FR data:
   python3 scripts/prepare_smartguard_data.py --smartguard-root ../SmartGuard --dataset fr --output outputs/processed/fr_sequences.jsonl

2. Train a short SmartGuard backbone run:
   PYTHONPATH=src /home/heyang/miniconda3/envs/smartguard_env/bin/python -m causal_gen_guard.training.train_backbone --input-jsonl outputs/processed/fr_sequences.jsonl --epochs 1 --batch-size 32 --hidden-dim 64 --nhead 4 --num-layers 1 --checkpoint-path outputs/checkpoints/smartguard_backbone_fr.pt --loss-vector-output outputs/results/loss_vector_fr.npy

This script prints commands only. Run the commands manually when ready.
EOF
