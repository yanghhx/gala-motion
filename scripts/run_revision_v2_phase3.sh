#!/usr/bin/env bash
# Run after pair 2: HumanML3D Anchor+Skate (final model) + semantic eval + viz
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
OUTDIR="$ROOT/outputs/revision_v2"
LOGDIR="$ROOT/runs/revision_v2"
mkdir -p "$LOGDIR" "$OUTDIR"

# 1. HumanML3D Anchor+Skate (final model = +Kinematic row in ablation)
name="humanml_anchor_skate"
config="$ROOT/configs/gala_humanml3d_flow_anchor_skate.yaml"
ckpt_dir="$ROOT/checkpoints/revision_v2/$name"
eval_out="$OUTDIR/${name}_test_n20.json"
log="$LOGDIR/${name}.log"
mkdir -p "$ckpt_dir"
if [[ ! -f "$eval_out" ]]; then
  init_args=(--init-from "$INIT")
  if [[ -f "$ckpt_dir/last.pt" ]]; then
    init_args=(--resume "$ckpt_dir/last.pt")
  fi
  run_py "$ROOT/trainers/train.py" \
    --config "$config" "${init_args[@]}" \
    --stage flow --epochs 200 --max-steps "$STEPS" --eval-every-steps 0 \
    --checkpoint-dir "$ckpt_dir" --log-dir "$ROOT/runs/revision_v2/$name" 2>&1 | tee -a "$log"
  run_py "$ROOT/scripts/evaluate_t2m.py" \
    --config "$config" --checkpoint "$ckpt_dir/last.pt" \
    --protocol official --split test --replication-times 20 --batch-size 32 \
    --steps 20 --guidance 2.5 --output "$eval_out" 2>&1 | tee -a "$log"
fi

# 2. Semantic evaluation on HumanML3D Anatomy (gate=-3)
anatomy_ckpt="$ROOT/checkpoints/revision_v2/humanml_anatomy_gateNeg3/last.pt"
sem_out="$OUTDIR/semantic_eval_anatomy.json"
if [[ -f "$anatomy_ckpt" && ! -f "$sem_out" ]]; then
  run_py "$ROOT/scripts/evaluate_part_semantic.py" \
    --config "$ROOT/configs/gala_humanml3d_flow_anatomy.yaml" \
    --checkpoint "$anatomy_ckpt" \
    --output "$sem_out" --steps 20 --guidance 2.5 --repeats 4 2>&1 | tee -a "$LOGDIR/semantic_eval.log"
fi

# Also run semantic eval on baseline (no anchor) for comparison
base_ckpt="$ROOT/checkpoints/gala_humanml3d_flow_distinct/best.pt"
sem_base_out="$OUTDIR/semantic_eval_baseline.json"
if [[ ! -f "$sem_base_out" ]]; then
  run_py "$ROOT/scripts/evaluate_part_semantic.py" \
    --config "$ROOT/configs/gala_humanml3d_flow_physical.yaml" \
    --checkpoint "$base_ckpt" \
    --output "$sem_base_out" --steps 20 --guidance 2.5 --repeats 4 2>&1 | tee -a "$LOGDIR/semantic_eval_baseline.log"
fi

# 3. Query-anchor similarity visualization
viz_out="$ROOT/outputs/anatomy_anchor_similarity.png"
if [[ -f "$anatomy_ckpt" && ! -f "$viz_out" ]]; then
  run_py "$ROOT/scripts/visualize_anchor_similarity.py" \
    --config "$ROOT/configs/gala_humanml3d_flow_anatomy.yaml" \
    --checkpoint "$anatomy_ckpt" \
    --output "$viz_out" 2>&1 | tee -a "$LOGDIR/anchor_viz.log"
fi

echo "REVISION_V2_PHASE3_DONE $(date)" | tee -a "$LOGDIR/queue.log"
