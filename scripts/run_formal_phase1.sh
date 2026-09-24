#!/usr/bin/env bash
# Phase 1 formal experiment: A3 control vs A3+Skate, 50k steps each from Distinct.
# Same seed, same hparams, same NFE=20 / CFG=2.5, official 20-replication test.
# A3 = λ=0,0 control. A3+Skate = skate adaptive r=0.02 (sweet spot from screen v3).
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
LOG="$ROOT/runs/formal_phase1/formal.log"
mkdir -p "$ROOT/runs/formal_phase1" "$ROOT/outputs/formal_phase1" "$ROOT/checkpoints/formal_phase1"

train_and_eval() {
  local name="$1" acc="$2" skate="$3" adaptive="${4:-false}" ratio="${5:-0.0}"
  local ckpt_dir="$ROOT/checkpoints/formal_phase1/$name"
  local eval_out="$ROOT/outputs/formal_phase1/${name}_test_n20.json"
  mkdir -p "$ckpt_dir"
  if [[ -f "$eval_out" ]]; then
    echo "===== $(date) skip $name (test eval exists) =====" | tee -a "$LOG"
    return
  fi
  local extra=(--init-from "$INIT")
  if [[ -f "$ckpt_dir/last.pt" && ! -f "$eval_out" ]]; then
    extra=(--resume "$ckpt_dir/last.pt")
    echo "===== $(date) resume $name from last.pt =====" | tee -a "$LOG"
  else
    echo "===== $(date) train $name acc=$acc skate=$skate adaptive=$adaptive ratio=$ratio steps=$STEPS =====" | tee -a "$LOG"
  fi
  local adaptive_flag=""
  if [[ "$adaptive" == "true" ]]; then
    adaptive_flag="--adaptive-aux-weight --aux-target-ratio $ratio"
  fi
  run_py "$ROOT/trainers/train.py" \
    --config "$CONFIG" \
    "${extra[@]}" \
    --stage flow --epochs 200 --max-steps "$STEPS" --eval-every-steps 0 \
    --lambda-kin-acceleration "$acc" --lambda-kin-skating "$skate" \
    $adaptive_flag \
    --checkpoint-dir "$ckpt_dir" --log-dir "$ROOT/runs/formal_phase1/$name" 2>&1 | tee -a "$LOG"
  echo "===== $(date) official test $name 20-rep NFE=20 CFG=2.5 full test set =====" | tee -a "$LOG"
  run_py "$ROOT/scripts/evaluate_t2m.py" \
    --config "$CONFIG" \
    --checkpoint "$ckpt_dir/last.pt" \
    --protocol official --split test \
    --replication-times 20 --batch-size 32 \
    --steps 20 --guidance 2.5 \
    --output "$eval_out" 2>&1 | tee -a "$LOG"
}

# A3 control: λ=0,0, same 50k budget from Distinct.
train_and_eval "a3_s${STEPS}" 0.0 0.0 false 0.0

# A3+Skate: skate adaptive r=0.02 (sweet spot between r=0.01 and r=0.03).
train_and_eval "skate_r0.02_s${STEPS}" 0.0 1.0 true 0.02

echo "FORMAL_PHASE1_DONE $(date)" | tee -a "$LOG"
