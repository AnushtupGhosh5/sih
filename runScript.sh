#!/usr/bin/env bash
set -euo pipefail

cd /workspace

prepare() {
  python3 scripts/audit_synchronized_dataset.py
  python3 scripts/verify_synchronization.py
  python3 scripts/prepare_ecu_dataset.py
  python3 scripts/validate_ecu_dataset.py
}

usage() {
  printf '%s\n' \
    'Usage:' \
    '  ./run.sh prepare          Audit, prepare, and validate data (default)' \
    '  ./run.sh train [args...]  Train ECU-supervised CatBoost Δv model' \
    '  ./run.sh all [args...]    Prepare, validate, then train' \
    '  ./run.sh smoke            Run a small end-to-end training smoke test' \
    '  ./run.sh calibrate        Evaluate causal pre-blackout phone-axis calibration' \
    '  ./run.sh train-cnn [args] Train gated causal 1D CNN delta-v model' \
    '  ./run.sh smoke-cnn        Run a small causal CNN end-to-end smoke test' \
    '  ./run.sh recover-align    Recover inertial lags and build the corrected dataset' \
    '  ./run.sh train-ins [args] Train context-calibrated recurrent unrolled INS' \
    '  ./run.sh smoke-ins        CUDA/CPU smoke test for the unrolled INS pipeline' \
    '  ./run.sh hybrid-ins       Select causal drift controls on Driver A; test Driver B' \
    '  ./run.sh <command...>     Run an arbitrary command in the container'
}

mode="${1:-prepare}"
if (( $# > 0 )); then shift; fi
case "$mode" in
  prepare) prepare ;;
  train) python3 scripts/train_ecu_delta_v.py "$@" ;;
  all) prepare; python3 scripts/train_ecu_delta_v.py "$@" ;;
  smoke) python3 scripts/train_ecu_delta_v.py --smoke "$@" ;;
  calibrate) python3 scripts/evaluate_causal_calibration.py "$@" ;;
  train-cnn) python3 scripts/train_causal_cnn.py "$@" ;;
  smoke-cnn) python3 scripts/train_causal_cnn.py --smoke "$@" ;;
  train-ins) python3 scripts/train_unrolled_ins.py "$@" ;;
  smoke-ins) python3 scripts/train_unrolled_ins.py --smoke "$@" ;;
  hybrid-ins) python3 scripts/evaluate_hybrid_ins.py "$@" ;;
  recover-align)
    python3 scripts/recover_inertial_alignment.py
    python3 scripts/prepare_ecu_dataset.py --manifest artifacts/inertial_alignment_recovery/manifest.csv --output-dir artifacts/ecu_training_dataset_inertial_aligned --history-rows 350 --driver-split E,A,B
    python3 scripts/validate_ecu_dataset.py --dataset-dir artifacts/ecu_training_dataset_inertial_aligned
    ;;
  help|-h|--help) usage ;;
  *) exec "$mode" "$@" ;;
esac
