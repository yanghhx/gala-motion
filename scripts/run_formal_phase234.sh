#!/usr/bin/env bash
# Phase 2/3/4 formal experiments: Anatomy Anchor, ST-CTR, Combined TMM.
# Each: 50k steps from Distinct + 20-rep official test.
# Runs sequentially. Skate decay is already running separately.
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
INIT="$ROOT/checkpoints/gala_humanml3d_flow_distinct/best.pt"
LOG="$ROOT/runs/formal_phase234/formal.log"
mkdir -p "$ROOT/runs/formal_phase234" "$ROOT/outputs/formal_phase234" "$ROOT/checkpoints/formal_phase234"

train_and_eval() {
  local name="$1" config="$2" extra_args="$3"
  local ckpt_dir="$ROOT/checkpoints/formal_phase234/$name"
  local eval_out="$ROOT/outputs/formal_phase234/${name}_test_n20.json"
  mkdir -p "$ckpt_dir"
  if [[ -f "$eval_out" ]]; then
    echo "===== $(date) skip $name (eval exists) =====" | tee -a "$LOG"
    return
  fi
  local init_args=(--init-from "$INIT")
  if [[ -f "$ckpt_dir/last.pt" ]]; then
    init_args=(--resume "$ckpt_dir/last.pt")
    echo "===== $(date) resume $name =====" | tee -a "$LOG"
  else
    echo "===== $(date) train $name steps=$STEPS =====" | tee -a "$LOG"
  fi
  run_py "$ROOT/trainers/train.py" \
    --config "$config" \
    "${init_args[@]}" \
    --stage flow --epochs 200 --max-steps "$STEPS" --eval-every-steps 0 \
    $extra_args \
    --checkpoint-dir "$ckpt_dir" --log-dir "$ROOT/runs/formal_phase234/$name" 2>&1 | tee -a "$LOG"
  echo "===== $(date) official test $name 20-rep =====" | tee -a "$LOG"
  run_py "$ROOT/scripts/evaluate_t2m.py" \
    --config "$config" \
    --checkpoint "$ckpt_dir/last.pt" \
    --protocol official --split test \
    --replication-times 20 --batch-size 32 \
    --steps 20 --guidance 2.5 \
    --output "$eval_out" 2>&1 | tee -a "$LOG"
}

# Phase 2: Anatomy Anchor (A4 in ablation matrix)
train_and_eval "anatomy_s${STEPS}" \
  "$ROOT/configs/gala_humanml3d_flow_anatomy.yaml" ""

# Phase 3: ST-CTR Graph (A5 in ablation matrix, without anchor)
train_and_eval "stctr_s${STEPS}" \
  "$ROOT/configs/gala_humanml3d_flow_stctr.yaml" ""

# Phase 4: Combined TMM = ST-CTR + Anatomy Anchor + Skate decay (A7 in ablation matrix)
train_and_eval "tmm_s${STEPS}" \
  "$ROOT/configs/gala_humanml3d_flow_tmm.yaml" ""

echo "PHASE234_DONE $(date)" | tee -a "$LOG"
