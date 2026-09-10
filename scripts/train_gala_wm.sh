#!/usr/bin/env bash
set -euo pipefail
ROOT=/home/qinyang/桌面/project
MAMBA=/home/qinyang/桌面/.local-tools/bin/micromamba
export MAMBA_ROOT_PREFIX=/home/qinyang/.local/share/mamba
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
VARIANT="${1:-root4}"
shift || true
if [[ "$VARIANT" == "idm" ]]; then
  CONFIG="$ROOT/configs/gala_humanml3d_wm_idm.yaml"
elif [[ "$VARIANT" == "v2" ]]; then
  CONFIG="$ROOT/configs/gala_humanml3d_wm_v2.yaml"
elif [[ "$VARIANT" == "eval" ]]; then
  exec "$MAMBA" run -n gala-motion python "$ROOT/scripts/eval_wm.py" \
    --config "$ROOT/configs/gala_humanml3d_wm.yaml" \
    --checkpoint "$ROOT/checkpoints/gala_humanml3d_wm/best.pt" \
    --output "$ROOT/checkpoints/gala_humanml3d_wm/eval.json" \
    "$@"
elif [[ "$VARIANT" == "analyze" ]]; then
  exec "$MAMBA" run -n gala-motion python "$ROOT/scripts/analyze_wm.py" "$@"
else
  CONFIG="$ROOT/configs/gala_humanml3d_wm.yaml"
fi
exec "$MAMBA" run -n gala-motion python "$ROOT/trainers/train_wm.py" \
  --config "$CONFIG" \
  "$@"
