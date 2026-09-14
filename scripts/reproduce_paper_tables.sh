#!/usr/bin/env bash
# Reproduce Tables 1--5 of the GALA ICASSP 2027 paper from saved checkpoints.
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$ROOT"
export PYTHONNOUSERSITE=1
export PYTHONPATH="$ROOT"
export HF_ENDPOINT=${HF_ENDPOINT:-https://hf-mirror.com}
PY=${PYTHON_BIN:-python}
OUT=${OUTPUT_DIR:-outputs/reproduction}
REPS=${REPLICATION_TIMES:-20}
BATCH=${BATCH_SIZE:-32}
mkdir -p "$OUT"/{table1,table2,table3,table4,table5}

required=(
  checkpoints/gala_humanml3d_flow_distinct/best.pt
  checkpoints/gala_kitml_flow_v2/best.pt
  checkpoints/gala_humanml3d_flow_base/best.pt
  checkpoints/gala_humanml3d_flow_noalign/best.pt
  checkpoints/gala_humanml3d_flow/best.pt
  checkpoints/gala_humanml3d_flow_part_distinct/best.pt
  checkpoints/gala_humanml3d_vae_conv/best.pt
  checkpoints/gala_humanml3d_vae_stgcn/best.pt
  checkpoints/gala_humanml3d_vae/best.pt
)
for path in "${required[@]}"; do
  [[ -f "$path" ]] || { echo "missing required checkpoint: $path" >&2; exit 2; }
done

eval_t2m() {
  local config=$1 checkpoint=$2 output=$3 steps=$4 guidance=$5
  "$PY" scripts/evaluate_t2m.py \
    --config "$config" --checkpoint "$checkpoint" \
    --protocol official --split test --replication-times "$REPS" \
    --batch-size "$BATCH" --steps "$steps" --guidance "$guidance" \
    --output "$output"
}

echo "[Table 1] HumanML3D official test"
eval_t2m configs/gala_humanml3d_flow_distinct.yaml \
  checkpoints/gala_humanml3d_flow_distinct/best.pt \
  "$OUT/table1/gala_n20_cfg2.5.json" 20 2.5
eval_t2m configs/gala_humanml3d_flow_distinct.yaml \
  checkpoints/gala_humanml3d_flow_distinct/best.pt \
  "$OUT/table1/gala_n50_cfg2.0.json" 50 2.0

echo "[Table 2] KIT-ML official test"
eval_t2m configs/gala_kitml_flow_v2.yaml \
  checkpoints/gala_kitml_flow_v2/best.pt \
  "$OUT/table2/gala_n20_cfg2.5.json" 20 2.5

echo "[Table 3] HumanML3D component ablation"
eval_t2m configs/gala_humanml3d_flow_base.yaml \
  checkpoints/gala_humanml3d_flow_base/best.pt \
  "$OUT/table3/base_rf.json" 20 2.5
eval_t2m configs/gala_humanml3d_flow_noalign.yaml \
  checkpoints/gala_humanml3d_flow_noalign/best.pt \
  "$OUT/table3/graph_rf.json" 20 2.5
eval_t2m configs/gala_humanml3d_flow.yaml \
  checkpoints/gala_humanml3d_flow/best.pt \
  "$OUT/table3/graph_global.json" 20 2.5
eval_t2m configs/gala_humanml3d_flow_part_distinct.yaml \
  checkpoints/gala_humanml3d_flow_part_distinct/best.pt \
  "$OUT/table3/graph_global_part.json" 20 2.5
cp "$OUT/table1/gala_n20_cfg2.5.json" "$OUT/table3/gala.json"

echo "[Table 4] HumanML3D tokenizer reconstruction"
for spec in \
  "conv|configs/gala_humanml3d_vae_conv.yaml|checkpoints/gala_humanml3d_vae_conv/best.pt" \
  "stgcn|configs/gala_humanml3d_vae_stgcn.yaml|checkpoints/gala_humanml3d_vae_stgcn/best.pt" \
  "ctr_graph|configs/gala_humanml3d.yaml|checkpoints/gala_humanml3d_vae/best.pt"; do
  IFS='|' read -r name config checkpoint <<< "$spec"
  "$PY" scripts/eval_tokenizer.py --config "$config" --checkpoint "$checkpoint" \
    --split val --batch-size "$BATCH" --output "$OUT/table4/${name}.json"
done

echo "[Table 5] Validation CFG/NFE scan"
"$PY" scripts/sweep_cfg_steps.py \
  --config configs/gala_humanml3d_flow.yaml \
  --checkpoint checkpoints/gala_humanml3d_flow/best.pt \
  --split val --batch-size "$BATCH" --output "$OUT/table5/cfg_steps_sweep.json"

echo "[Table 5 supplement] RTX latency (hardware-dependent)"
"$PY" scripts/bench_efficiency.py \
  --config configs/gala_humanml3d_flow.yaml \
  --checkpoint checkpoints/gala_humanml3d_flow/best.pt \
  --batch-size 1 --frames 196 --steps 10 20 50 --guidance 2.5 \
  --warmup 3 --repeats 10 --output "$OUT/table5/efficiency.json"

echo "Done. Results: $ROOT/$OUT"
