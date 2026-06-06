#!/usr/bin/env bash
set -euo pipefail

# Context-shift experiment entry point.
# Default mode is smoke, so running this script without arguments validates the
# pipeline on a tiny subset instead of launching a full experiment.

DEFAULT_PYTHON=python3
if [[ -x /home/heyang/miniconda3/envs/smartguard_env/bin/python ]]; then
  DEFAULT_PYTHON=/home/heyang/miniconda3/envs/smartguard_env/bin/python
fi

PYTHON_BIN=${PYTHON_BIN:-${DEFAULT_PYTHON}}
CONFIG=${1:-configs/context_shift_fr.yaml}
MODE=${2:-smoke}

export PYTHONPATH=src${PYTHONPATH:+:${PYTHONPATH}}

case ${MODE} in
  smoke)
    echo Running context-shift smoke test for ${CONFIG}
    ${PYTHON_BIN} -m causal_gen_guard.training.train_fusion --config ${CONFIG} --smoke-test
    ${PYTHON_BIN} -m causal_gen_guard.evaluation.ablation --config ${CONFIG} --smoke-test
    ;;
  train)
    echo Running one configured context-shift training pipeline for ${CONFIG}
    ${PYTHON_BIN} -m causal_gen_guard.training.train_fusion --config ${CONFIG}
    ;;
  ablation)
    echo Running configured context-shift ablations for ${CONFIG}
    ${PYTHON_BIN} -m causal_gen_guard.evaluation.ablation --config ${CONFIG}
    ;;
  print)
    echo Smoke train: PYTHONPATH=src ${PYTHON_BIN} -m causal_gen_guard.training.train_fusion --config ${CONFIG} --smoke-test
    echo Smoke ablate: PYTHONPATH=src ${PYTHON_BIN} -m causal_gen_guard.evaluation.ablation --config ${CONFIG} --smoke-test
    echo Full train: PYTHONPATH=src ${PYTHON_BIN} -m causal_gen_guard.training.train_fusion --config ${CONFIG}
    echo Full ablate: PYTHONPATH=src ${PYTHON_BIN} -m causal_gen_guard.evaluation.ablation --config ${CONFIG}
    ;;
  *)
    echo Usage: scripts/run_context_shift.sh [config] [smoke|train|ablation|print] >&2
    exit 2
    ;;
esac
