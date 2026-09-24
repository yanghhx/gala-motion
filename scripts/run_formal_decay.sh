#!/usr/bin/env bash
# Phase 1 formal experiment with aux ratio decay.
# r=0.03 for first 10k steps, then linear decay to r=0.01 by step 40k, stay at 0.01.
# Goal: reduce FID cost while maintaining skating reduction.
set -euo pipefail
export PYTHONNOUSERSITE=1
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
export PYTHONPATH="$ROOT"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export MAMBA_ROOT_PREFIX="${MAMBA_ROOT_PREFIX:-/home/qinyang/.local/share/mamba}"
cd "$ROOT"
if [[ -x /home/qinyang/桌面/.local-tools/bin/micromamba ]]; then
  python_bin() { /home/qinyang/桌面/.local-tools/bin/micromamba run -n gala-motion python "$@"; }
else
  python_bin() { "${PYTHON_BIN:-python}" "$@"; }
fi
run_py() { python_bin "$@"; }

STEPS="${FORMAL_STEPS:-50000}"
CONFIG="$ROOT/configs/gala_humanml3d_flow_physical.yaml"
INIT="$ROOT/checkpoints/gala_humanml3d_flow_distinct/best.pt"
LOG="$ROOT/runs/formal_phase1/decay.log"
mkdir -p "$ROOT/runs/formal_phase1" "$ROOT/outputs/formal_phase1" "$ROOT/checkpoints/formal_phase1"

name="skate_decay_r003_001_s${STEPS}"
ckpt_dir="$ROOT/checkpoints/formal_phase1/$name"
eval_out="$ROOT/outputs/formal_phase1/${name}_test_n20.json"
mkdir -p "$ckpt_dir"

if [[ -f "$eval_out" ]]; then
  echo "===== $(date) skip $name (test eval exists) =====" | tee -a "$LOG"
  exit 0
fi
extra=(--init-from "$INIT")
if [[ -f "$ckpt_dir/last.pt" ]]; then
  extra=(--resume "$ckpt_dir/last.pt")
  echo "===== $(date) resume $name from last.pt =====" | tee -a "$LOG"
else
  echo "===== $(date) train $name steps=$STEPS decay 0.03->0.01 =====" | tee -a "$LOG"
fi

run_py "$ROOT/trainers/train.py" \
  --config "$CONFIG" \
  "${extra[@]}" \
  --stage flow --epochs 200 --max-steps "$STEPS" --eval-every-steps 0 \
  --lambda-kin-acceleration 0.0 --lambda-kin-skating 1.0 \
  --adaptive-aux-weight \
  --aux-ratio-decay --aux-ratio-start 0.03 --aux-ratio-end 0.01 --aux-decay-steps 40000 \
  --checkpoint-dir "$ckpt_dir" --log-dir "$ROOT/runs/formal_phase1/$name" 2>&1 | tee -a "$LOG"

echo "===== $(date) official test $name 20-rep NFE=20 CFG=2.5 =====" | tee -a "$LOG"
run_py "$ROOT/scripts/evaluate_t2m.py" \
  --config "$CONFIG" \
  --checkpoint "$ckpt_dir/last.pt" \
  --protocol official --split test \
  --replication-times 20 --batch-size 32 \
  --steps 20 --guidance 2.5 \
  --output "$eval_out" 2>&1 | tee -a "$LOG"

echo "DECAY_DONE $(date)" | tee -a "$LOG"
