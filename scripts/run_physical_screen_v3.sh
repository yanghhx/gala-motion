#!/usr/bin/env bash
# Phase 1 physical losses — third screen.
# Skate only (acc loss was ineffective in v2), low adaptive ratios, 30k steps.
# Target: find the sweet spot where skating drops meaningfully but FID stays acceptable.
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

STEPS="${SCREEN_STEPS:-30000}"
CONFIG="$ROOT/configs/gala_humanml3d_flow_physical.yaml"
INIT="$ROOT/checkpoints/gala_humanml3d_flow_distinct/best.pt"
LOG="$ROOT/runs/physical_screen_v3/screen.log"
mkdir -p "$ROOT/runs/physical_screen_v3" "$ROOT/outputs/physical/screen_v3" "$ROOT/checkpoints/physical_screen_v3"

# Reuse A3 from the first screen as baseline.
A3_EVAL="$ROOT/outputs/physical/screen/a3_s4000_val.json"
if [[ -f "$A3_EVAL" && ! -f "$ROOT/outputs/physical/screen_v3/a3_s4000_val.json" ]]; then
  cp "$A3_EVAL" "$ROOT/outputs/physical/screen_v3/a3_s4000_val.json"
  echo "===== $(date) reuse A3 eval from first screen =====" | tee -a "$LOG"
fi

train_one() {
  local name="$1" acc="$2" skate="$3" ratio="$4"
  local ckpt_dir="$ROOT/checkpoints/physical_screen_v3/$name"
  local out="$ROOT/outputs/physical/screen_v3/${name}_val.json"
  mkdir -p "$ckpt_dir"
  if [[ -f "$out" && -f "$ckpt_dir/last.pt" ]]; then
    echo "===== $(date) skip $name (eval exists) =====" | tee -a "$LOG"
    return
  fi
  local extra=(--init-from "$INIT")
  if [[ -f "$ckpt_dir/last.pt" && ! -f "$out" ]]; then
    extra=(--resume "$ckpt_dir/last.pt")
    echo "===== $(date) resume $name from last.pt skate=$skate ratio=$ratio =====" | tee -a "$LOG"
  else
    echo "===== $(date) train $name skate=$skate ratio=$ratio steps=$STEPS =====" | tee -a "$LOG"
  fi
  run_py "$ROOT/trainers/train.py" \
    --config "$CONFIG" \
    "${extra[@]}" \
    --stage flow --epochs 80 --max-steps "$STEPS" --eval-every-steps 0 \
    --lambda-kin-acceleration "$acc" --lambda-kin-skating "$skate" \
    --adaptive-aux-weight --aux-target-ratio "$ratio" \
    --checkpoint-dir "$ckpt_dir" --log-dir "$ROOT/runs/physical_screen_v3/$name" 2>&1 | tee -a "$LOG"
  echo "===== $(date) eval $name fixed val n=320 NFE=20 CFG=2.5 =====" | tee -a "$LOG"
  run_py "$ROOT/scripts/evaluate_t2m.py" \
    --config "$CONFIG" \
    --checkpoint "$ckpt_dir/last.pt" \
    --protocol official --split val --fixed-eval \
    --max-samples 320 --replication-times 1 --batch-size 32 \
    --steps 20 --guidance 2.5 \
    --output "$out" 2>&1 | tee -a "$LOG"
}

# Skate only, low ratios. acc=0 (acc loss was ineffective in v2).
for r in 0.01 0.03 0.05; do
  train_one "skate_adaptive_r${r}_s${STEPS}" 0.0 1.0 "$r"
done

echo "===== $(date) score screening v3 =====" | tee -a "$LOG"
run_py "$ROOT/scripts/score_physical_screen.py" \
  --dir "$ROOT/outputs/physical/screen_v3" \
  --output "$ROOT/outputs/physical/screen_v3/weight_choice.json" 2>&1 | tee -a "$LOG"
echo "PHYSICAL_SCREEN_V3_DONE $(date)" | tee -a "$LOG"
