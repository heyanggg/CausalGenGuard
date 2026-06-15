#!/usr/bin/env bash
# Clean generated artifacts from the Git working tree.
# Run from the CausalGenGuard repository root:
#   bash scripts/clean_repository_outputs.sh
set -euo pipefail

# Python caches are machine-generated and should never be committed.
find . -type d -name "__pycache__" -prune -exec rm -rf {} +
find . -type f -name "*.pyc" -delete
find . -type f -name "*.pyo" -delete

# Heavy or reproducible experiment outputs.  Keep scripts/configs/docs instead.
rm -rf outputs/checkpoints
rm -rf outputs/diagnostics
rm -rf outputs/feasibility
rm -rf outputs/figures/fr_acceptance
rm -rf outputs/processed
rm -rf outputs/labels
rm -rf outputs/synthetic
rm -rf outputs/tmp_argus_fake
rm -f outputs/tmp_context_shift_smoke.yaml
rm -f outputs/tmp_patch_summarize.py

# Remove binary arrays and model files that may remain under outputs.
find outputs -type f \( -name "*.npy" -o -name "*.npz" -o -name "*.pt" -o -name "*.pth" -o -name "*.ckpt" -o -name "*.pkl" -o -name "*.pickle" \) -delete 2>/dev/null || true

# Archive older diagnostic logs if they exist; these are useful historically but
# should not be mixed with final experiment reports.
mkdir -p docs/archive outputs
for file in \
  outputs/logs/CONTROL_MAPPING_AUDIT.md \
  outputs/logs/FEASIBILITY_REPORT.md \
  outputs/logs/CAUSAL_BRANCH_DIAGNOSTIC.md \
  outputs/logs/SMOKE_TEST_REPORT.md; do
  if [ -f "$file" ]; then
    mv "$file" docs/archive/
  fi
done

touch outputs/.gitkeep
printf 'Cleaned generated outputs. Keep source code, configs, docs, and reproducible scripts in Git.\n'
