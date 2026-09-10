#!/usr/bin/env bash
set -euo pipefail
ROOT=/home/qinyang/桌面/project
export MAMBA_ROOT_PREFIX=/home/qinyang/.local/share/mamba
export PYTHONPATH="$ROOT"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export PYTHONUNBUFFERED=1
cd "$ROOT"
printf '\n===== official eval steps=50 cfg=2.0 %s =====\n' "$(date '+%F %T')" >> "$ROOT/runs/gala_humanml3d_flow/eval.log"
exec "$ROOT/scripts/train_gala_humanml3d.sh" eval \
  --steps 50 --guidance 2.0 \
  --output "$ROOT/checkpoints/gala_humanml3d_flow/official_eval_s50_g2.json"
