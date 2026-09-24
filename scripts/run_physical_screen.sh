#!/usr/bin/env bash
# Short-run weight screen for Phase 1 physical losses.
# A3 control and one-at-a-time Acc / Skate. Do not combine Acc+Skate here.
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

STEPS="${SCREEN_STEPS:-4000}"
INIT="$ROOT/checkpoints/gala_humanml3d_flow_distinct/best.pt"
CONFIG="$ROOT/configs/gala_humanml3d_flow_physical.yaml"
LOG="$ROOT/runs/physical_screen/screen.log"
mkdir -p "$ROOT/runs/physical_screen" "$ROOT/outputs/physical/screen" "$ROOT/checkpoints/physical_screen"

if [[ ! -f "$ROOT/outputs/physical/val_manifest.json" ]]; then
  echo "===== $(date) physical contact diagnosis =====" | tee -a "$LOG"
  run_py "$ROOT/scripts/make_physical_val_manifest.py" \
    --config "$CONFIG" --max-samples 320 \
    --output "$ROOT/outputs/physical/val_manifest.json" 2>&1 | tee -a "$LOG"
else
  echo "===== $(date) skip val manifest (exists) =====" | tee -a "$LOG"
fi
if [[ ! -f "$ROOT/outputs/physical/contact_diagnosis_val.json" ]]; then
  run_py "$ROOT/scripts/diagnose_contact.py" \
    --config "$CONFIG" --split val --max-samples 0 \
    --output "$ROOT/outputs/physical/contact_diagnosis_val.json" 2>&1 | tee -a "$LOG"
else
  echo "===== $(date) skip contact diagnosis (exists) =====" | tee -a "$LOG"
fi

train_one() {
  local name="$1" acc="$2" skate="$3"
  local ckpt_dir="$ROOT/checkpoints/physical_screen/$name"
  local out="$ROOT/outputs/physical/screen/${name}_val.json"
  mkdir -p "$ckpt_dir"
  if [[ -f "$out" && -f "$ckpt_dir/last.pt" ]]; then
    echo "===== $(date) skip $name (eval exists) =====" | tee -a "$LOG"
    return
  fi
  local extra=(--init-from "$INIT")
  if [[ -f "$ckpt_dir/last.pt" && ! -f "$out" ]]; then
    extra=(--resume "$ckpt_dir/last.pt")
    echo "===== $(date) resume $name from last.pt acc=$acc skate=$skate =====" | tee -a "$LOG"
  else
    echo "===== $(date) train $name acc=$acc skate=$skate steps=$STEPS =====" | tee -a "$LOG"
  fi
  run_py "$ROOT/trainers/train.py" \
    --config "$CONFIG" \
    "${extra[@]}" \
    --stage flow \
    --epochs 80 \
    --max-steps "$STEPS" \
    --eval-every-steps 0 \
    --lambda-kin-acceleration "$acc" \
    --lambda-kin-skating "$skate" \
    --checkpoint-dir "$ckpt_dir" \
    --log-dir "$ROOT/runs/physical_screen/$name" 2>&1 | tee -a "$LOG"
  echo "===== $(date) eval $name fixed val n=320 NFE=20 CFG=2.5 =====" | tee -a "$LOG"
  run_py "$ROOT/scripts/evaluate_t2m.py" \
    --config "$CONFIG" \
    --checkpoint "$ckpt_dir/last.pt" \
    --protocol official --split val --fixed-eval \
    --max-samples 320 --replication-times 1 --batch-size 32 \
    --steps 20 --guidance 2.5 \
    --output "$out" 2>&1 | tee -a "$LOG"
}

train_one "a3_s${STEPS}" 0.0 0.0
for w in 0.01 0.05 0.1; do
  train_one "acc_${w}_s${STEPS}" "$w" 0.0
done
for w in 0.01 0.05 0.1; do
  train_one "skate_${w}_s${STEPS}" 0.0 "$w"
done

echo "===== $(date) score screening =====" | tee -a "$LOG"
run_py "$ROOT/scripts/score_physical_screen.py" \
  --dir "$ROOT/outputs/physical/screen" \
  --output "$ROOT/outputs/physical/screen/weight_choice.json" 2>&1 | tee -a "$LOG"
echo "PHYSICAL_SCREEN_DONE $(date)" | tee -a "$LOG"
