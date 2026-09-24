#!/usr/bin/env bash
# Phase 1 physical losses — second screen with adaptive weighting.
# Adaptive: eff_lambda = eta * flow_loss / aux_loss, so aux/flow ≈ eta automatically.
# This fixes the scale mismatch from the first screen (where aux/flow was ~1e-5).
# A3 control is reused from the first screen (already trained 4000 steps + eval).
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

STEPS="${SCREEN_STEPS:-10000}"
CONFIG="$ROOT/configs/gala_humanml3d_flow_physical.yaml"
INIT="$ROOT/checkpoints/gala_humanml3d_flow_distinct/best.pt"
LOG="$ROOT/runs/physical_screen_v2/screen.log"
mkdir -p "$ROOT/runs/physical_screen_v2" "$ROOT/outputs/physical/screen_v2" "$ROOT/checkpoints/physical_screen_v2"

# Reuse A3 from the first screen (already trained + evaluated).
A3_EVAL="$ROOT/outputs/physical/screen/a3_s4000_val.json"
if [[ -f "$A3_EVAL" ]]; then
  cp "$A3_EVAL" "$ROOT/outputs/physical/screen_v2/a3_s4000_val.json"
  echo "===== $(date) reuse A3 eval from first screen =====" | tee -a "$LOG"
fi

train_one() {
  local name="$1" acc="$2" skate="$3" adaptive="${4:-true}" ratio="${5:-0.15}"
  local ckpt_dir="$ROOT/checkpoints/physical_screen_v2/$name"
  local out="$ROOT/outputs/physical/screen_v2/${name}_val.json"
  mkdir -p "$ckpt_dir"
  if [[ -f "$out" && -f "$ckpt_dir/last.pt" ]]; then
    echo "===== $(date) skip $name (eval exists) =====" | tee -a "$LOG"
    return
  fi
  local extra=(--init-from "$INIT")
  if [[ -f "$ckpt_dir/last.pt" && ! -f "$out" ]]; then
    extra=(--resume "$ckpt_dir/last.pt")
    echo "===== $(date) resume $name from last.pt acc=$acc skate=$skate adaptive=$adaptive ratio=$ratio =====" | tee -a "$LOG"
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
    --stage flow --epochs 80 --max-steps "$STEPS" --eval-every-steps 0 \
    --lambda-kin-acceleration "$acc" --lambda-kin-skating "$skate" \
    $adaptive_flag \
    --checkpoint-dir "$ckpt_dir" --log-dir "$ROOT/runs/physical_screen_v2/$name" 2>&1 | tee -a "$LOG"
  echo "===== $(date) eval $name fixed val n=320 NFE=20 CFG=2.5 =====" | tee -a "$LOG"
  run_py "$ROOT/scripts/evaluate_t2m.py" \
    --config "$CONFIG" \
    --checkpoint "$ckpt_dir/last.pt" \
    --protocol official --split val --fixed-eval \
    --max-samples 320 --replication-times 1 --batch-size 32 \
    --steps 20 --guidance 2.5 \
    --output "$out" 2>&1 | tee -a "$LOG"
}

# Adaptive aux with target ratio 0.15 (aux stays at ~15% of flow automatically).
# lambda > 0 just signals "enable this loss"; adaptive sets the effective weight.
train_one "acc_adaptive_r015_s${STEPS}" 1.0 0.0 true 0.15
train_one "skate_adaptive_r015_s${STEPS}" 0.0 1.0 true 0.15

# Also try a higher ratio (0.25) to see if stronger constraint helps.
train_one "acc_adaptive_r025_s${STEPS}" 1.0 0.0 true 0.25
train_one "skate_adaptive_r025_s${STEPS}" 0.0 1.0 true 0.25

echo "===== $(date) score screening v2 =====" | tee -a "$LOG"
run_py "$ROOT/scripts/score_physical_screen.py" \
  --dir "$ROOT/outputs/physical/screen_v2" \
  --output "$ROOT/outputs/physical/screen_v2/weight_choice.json" 2>&1 | tee -a "$LOG"
echo "PHYSICAL_SCREEN_V2_DONE $(date)" | tee -a "$LOG"
