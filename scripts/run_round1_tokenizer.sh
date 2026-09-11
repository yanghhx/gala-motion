#!/bin/bash
set -euo pipefail
export PYTHONNOUSERSITE=1
export PYTHONPATH=/home/qinyang/桌面/project
export HF_ENDPOINT=https://hf-mirror.com
cd /home/qinyang/桌面/project
PY=/home/qinyang/.local/share/mamba/envs/gala-motion/bin/python

echo "=== CONV VAE train $(date) ==="
"$PY" trainers/train.py --config configs/gala_humanml3d_vae_conv.yaml
echo "=== CONV VAE eval $(date) ==="
"$PY" scripts/eval_tokenizer.py \
  --config configs/gala_humanml3d_vae_conv.yaml \
  --checkpoint checkpoints/gala_humanml3d_vae_conv/best.pt \
  --split val \
  --output checkpoints/gala_humanml3d_vae_conv/tokenizer_val.json

echo "=== STGCN VAE train $(date) ==="
"$PY" trainers/train.py --config configs/gala_humanml3d_vae_stgcn.yaml
echo "=== STGCN VAE eval $(date) ==="
"$PY" scripts/eval_tokenizer.py \
  --config configs/gala_humanml3d_vae_stgcn.yaml \
  --checkpoint checkpoints/gala_humanml3d_vae_stgcn/best.pt \
  --split val \
  --output checkpoints/gala_humanml3d_vae_stgcn/tokenizer_val.json

echo "ROUND1_TOKENIZER_DONE $(date)"
