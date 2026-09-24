#!/usr/bin/env bash
# Revision v2 experiments:
#   1. HumanML3D Anatomy re-run with anchor_gate init = -3 (sigmoid≈0.05)
#   2. KIT-ML Base (lambda=0, no anchor) from Distinct
#   3. KIT-ML +Anatomy (gate=-3) from Distinct
#   4. KIT-ML +Skate (decay 0.03->0.01) from Distinct
# Each: 50k steps from Distinct + 20-rep official test.
# Runs 2 concurrent to fit in 8GB GPU (~2.7GB each).
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
HUMANML_INIT="$ROOT/checkpoints/gala_humanml3d_flow_distinct/best.pt"
KITML_INIT="$ROOT/checkpoints/gala_kitml_flow_v2/best.pt"
OUTDIR="$ROOT/outputs/revision_v2"
LOGDIR="$ROOT/runs/revision_v2"
mkdir -p "$LOGDIR" "$OUTDIR" "$ROOT/checkpoints/revision_v2"

# train_and_eval <name> <config> <init_ckpt> <extra_args>
train_and_eval() {
  local name="$1" config="$2" init="$3" extra_args="$4"
  local ckpt_dir="$ROOT/checkpoints/revision_v2/$name"
  local eval_out="$OUTDIR/${name}_test_n20.json"
  local log="$LOGDIR/${name}.log"
  mkdir -p "$ckpt_dir"
  if [[ -f "$eval_out" ]]; then
    echo "[$(date)] skip $name (eval exists)" | tee -a "$log"
    return 0
  fi
  local init_args=(--init-from "$init")
  if [[ -f "$ckpt_dir/last.pt" ]]; then
    init_args=(--resume "$ckpt_dir/last.pt")
    echo "[$(date)] resume $name" | tee -a "$log"
  else
    echo "[$(date)] train $name steps=$STEPS from $(basename $init)" | tee -a "$log"
  fi
  run_py "$ROOT/trainers/train.py" \
    --config "$config" \
    "${init_args[@]}" \
    --stage flow --epochs 200 --max-steps "$STEPS" --eval-every-steps 0 \
    $extra_args \
    --checkpoint-dir "$ckpt_dir" --log-dir "$ROOT/runs/revision_v2/$name" 2>&1 | tee -a "$log"
  echo "[$(date)] official test $name 20-rep" | tee -a "$log"
  run_py "$ROOT/scripts/evaluate_t2m.py" \
    --config "$config" \
    --checkpoint "$ckpt_dir/last.pt" \
    --protocol official --split test \
    --replication-times 20 --batch-size 32 \
    --steps 20 --guidance 2.5 \
    --output "$eval_out" 2>&1 | tee -a "$log"
}

# --- Pair 1: HumanML3D Anatomy (gate=-3) + KIT-ML Base ---
train_and_eval "humanml_anatomy_gateNeg3" \
  "$ROOT/configs/gala_humanml3d_flow_anatomy.yaml" \
  "$HUMANML_INIT" "" &
PID1=$!

train_and_eval "kitml_base" \
  "$ROOT/configs/gala_kitml_flow_physical.yaml" \
  "$KITML_INIT" "" &
PID2=$!

wait $PID1
wait $PID2
echo "[$(date)] Pair 1 done" | tee -a "$LOGDIR/queue.log"

# --- Pair 2: KIT-ML Anatomy (gate=-3) + KIT-ML Skate ---
train_and_eval "kitml_anatomy" \
  "$ROOT/configs/gala_kitml_flow_anatomy.yaml" \
  "$KITML_INIT" "" &
PID3=$!

train_and_eval "kitml_skate" \
  "$ROOT/configs/gala_kitml_flow_skate.yaml" \
  "$KITML_INIT" "" &
PID4=$!

wait $PID3
wait $PID4
echo "[$(date)] Pair 2 done" | tee -a "$LOGDIR/queue.log"

echo "REVISION_V2_DONE $(date)" | tee -a "$LOGDIR/queue.log"
