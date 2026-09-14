#!/usr/bin/env bash
# GPU queue after ST-GCN: tokenizer eval → +Part (from public +Global ckpt).
# Then GALA-v2 from +Part. Base RF / +Graph / KIT wait until those finish.
set -euo pipefail
export PYTHONNOUSERSITE=1
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
export PYTHONPATH="$ROOT"
export HF_ENDPOINT=https://hf-mirror.com
cd "$ROOT"
PY=${PYTHON_BIN:-python}
LOG="$ROOT/runs/gala_round1/table3.log"
mkdir -p "$ROOT/runs/gala_round1"

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

PART_LAST=checkpoints/gala_humanml3d_flow_part/last.pt
if [[ -f "$PART_LAST" ]]; then
  echo "===== $(date) +Part resume from $PART_LAST =====" | tee -a "$LOG"
  "$PY" trainers/train.py \
    --config configs/gala_humanml3d_flow_part.yaml \
    --resume "$PART_LAST" \
    2>&1 | tee -a "$LOG"
else
  echo "===== $(date) +Part from +Global flow =====" | tee -a "$LOG"
  "$PY" trainers/train.py \
    --config configs/gala_humanml3d_flow_part.yaml \
    --init-from checkpoints/gala_humanml3d_flow/best.pt \
    2>&1 | tee -a "$LOG"
fi

PART_BEST=checkpoints/gala_humanml3d_flow_part/best.pt
GLOBAL_BEST=checkpoints/gala_humanml3d_flow/best.pt
V2_INIT="$PART_BEST"
if [[ -f "$PART_BEST" ]]; then
  PART_FID="$("$PY" - <<'PY'
import torch
ck = torch.load("checkpoints/gala_humanml3d_flow_part/best.pt", map_location="cpu", weights_only=False)
print(float(ck.get("metric", 1e9)))
PY
)"
  echo "===== $(date) +Part best val FID=${PART_FID} =====" | tee -a "$LOG"
  # +Global val FID is 0.335 (n=1504, 20 steps, CFG 2.5). +Part uses n=320.
  awk_ok="$(awk -v fid="$PART_FID" 'BEGIN { print (fid+0 <= 0.38) ? "yes" : "no" }')"
  if [[ "$awk_ok" != "yes" && -f "$GLOBAL_BEST" ]]; then
    echo "===== $(date) +Part did not beat +Global; GALA-v2 inits from $GLOBAL_BEST =====" | tee -a "$LOG"
    V2_INIT="$GLOBAL_BEST"
  fi
fi

echo "===== $(date) GALA-v2 from ${V2_INIT} =====" | tee -a "$LOG"
"$PY" trainers/train.py \
  --config configs/gala_humanml3d_flow_v2.yaml \
  --init-from "$V2_INIT" \
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
