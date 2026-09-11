#!/usr/bin/env bash
# GPU queue after ST-GCN: tokenizer eval → +Part (from public +Global ckpt).
# Then GALA-v2 from +Part. Base RF / +Graph / KIT wait until those finish.
set -euo pipefail
export PYTHONNOUSERSITE=1
export PYTHONPATH=/home/qinyang/桌面/project
export HF_ENDPOINT=https://hf-mirror.com
cd /home/qinyang/桌面/project
PY=/home/qinyang/.local/share/mamba/envs/gala-motion/bin/python
LOG=/home/qinyang/桌面/project/runs/gala_round1/table3.log
mkdir -p /home/qinyang/桌面/project/runs/gala_round1

echo "===== $(date) table3 queue start =====" | tee -a "$LOG"

while pgrep -f "python trainers/train.py --config configs/gala_humanml3d_vae_stgcn.yaml" >/dev/null; do
  echo "$(date) waiting for ST-GCN VAE to finish" | tee -a "$LOG"
  sleep 30
done

if [[ ! -f checkpoints/gala_humanml3d_vae_stgcn/tokenizer_val.json ]]; then
  echo "===== $(date) eval ST-GCN tokenizer =====" | tee -a "$LOG"
  "$PY" scripts/eval_tokenizer.py \
    --config configs/gala_humanml3d_vae_stgcn.yaml \
    --checkpoint checkpoints/gala_humanml3d_vae_stgcn/best.pt \
    --split val \
    --output checkpoints/gala_humanml3d_vae_stgcn/tokenizer_val.json \
    2>&1 | tee -a "$LOG"
fi

echo "===== $(date) +Part from +Global flow =====" | tee -a "$LOG"
"$PY" trainers/train.py \
  --config configs/gala_humanml3d_flow_part.yaml \
  --init-from checkpoints/gala_humanml3d_flow/best.pt \
  2>&1 | tee -a "$LOG"

echo "===== $(date) GALA-v2 from +Part =====" | tee -a "$LOG"
"$PY" trainers/train.py \
  --config configs/gala_humanml3d_flow_v2.yaml \
  --init-from checkpoints/gala_humanml3d_flow_part/best.pt \
  2>&1 | tee -a "$LOG"

echo "===== $(date) Base RF =====" | tee -a "$LOG"
"$PY" trainers/train.py \
  --config configs/gala_humanml3d_flow_base.yaml \
  --init-from checkpoints/gala_humanml3d_vae_conv/best.pt \
  2>&1 | tee -a "$LOG"

echo "===== $(date) +Graph (no align) =====" | tee -a "$LOG"
"$PY" trainers/train.py \
  --config configs/gala_humanml3d_flow_noalign.yaml \
  --init-from checkpoints/gala_humanml3d_vae/best.pt \
  2>&1 | tee -a "$LOG"

echo "===== $(date) KIT-ML VAE =====" | tee -a "$LOG"
"$PY" trainers/train.py --config configs/gala_kitml.yaml \
  2>&1 | tee -a "$LOG"

echo "===== $(date) KIT-ML GALA-v2 =====" | tee -a "$LOG"
"$PY" trainers/train.py \
  --config configs/gala_kitml_flow_v2.yaml \
  --init-from checkpoints/gala_kitml_vae/best.pt \
  2>&1 | tee -a "$LOG"

echo "===== $(date) table3 queue done =====" | tee -a "$LOG"
