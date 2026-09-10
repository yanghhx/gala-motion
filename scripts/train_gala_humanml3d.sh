#!/usr/bin/env bash
set -euo pipefail
ROOT=/home/qinyang/桌面/project
MAMBA=/home/qinyang/桌面/.local-tools/bin/micromamba
export MAMBA_ROOT_PREFIX=/home/qinyang/.local/share/mamba
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
STAGE="${1:-vae}"
shift || true
if [[ "$STAGE" == "flow" ]]; then
  exec "$MAMBA" run -n gala-motion python "$ROOT/trainers/train.py" \
    --config "$ROOT/configs/gala_humanml3d_flow.yaml" \
    --init-from "$ROOT/checkpoints/gala_humanml3d_vae/best.pt" \
    "$@"
fi
if [[ "$STAGE" == "eval" ]]; then
  exec "$MAMBA" run -n gala-motion python "$ROOT/scripts/evaluate_t2m.py" \
    --config "$ROOT/configs/gala_humanml3d_flow.yaml" \
    --checkpoint "$ROOT/checkpoints/gala_humanml3d_flow/best.pt" \
    --protocol official \
    --replication-times 20 \
    --batch-size 32 \
    --output "$ROOT/checkpoints/gala_humanml3d_flow/official_eval.json" \
    "$@"
fi
exec "$MAMBA" run -n gala-motion python "$ROOT/trainers/train.py" \
  --config "$ROOT/configs/gala_humanml3d.yaml" \
  "$@"
