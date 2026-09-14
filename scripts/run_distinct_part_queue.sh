#!/usr/bin/env bash
# Repair run for anatomically distinct queries. Old checkpoints/results remain untouched.
set -euo pipefail
export PYTHONNOUSERSITE=1
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
export PYTHONPATH="$ROOT"
export HF_ENDPOINT=https://hf-mirror.com
cd "$ROOT"
PY=${PYTHON_BIN:-python}
LOG=runs/gala_round1/distinct_part.log
mkdir -p runs/gala_round1 outputs/attention outputs/figures outputs/humanml3d outputs/ablation

echo "===== $(date) distinct +Part from Graph+Global =====" | tee -a "$LOG"
PART_LAST=checkpoints/gala_humanml3d_flow_part_distinct/last.pt
if [[ -f "$PART_LAST" ]]; then
  "$PY" trainers/train.py \
    --config configs/gala_humanml3d_flow_part_distinct.yaml \
    --resume "$PART_LAST" 2>&1 | tee -a "$LOG"
else
  "$PY" trainers/train.py \
    --config configs/gala_humanml3d_flow_part_distinct.yaml \
    --init-from checkpoints/gala_humanml3d_flow/best.pt 2>&1 | tee -a "$LOG"
fi

echo "===== $(date) attention-collapse audit =====" | tee -a "$LOG"
"$PY" scripts/export_attention.py \
  --config configs/gala_humanml3d_flow_part_distinct.yaml \
  --checkpoint checkpoints/gala_humanml3d_flow_part_distinct/best.pt \
  --output-json outputs/attention/part_attention_distinct.json \
  --output-figure outputs/figures/fig_part_attention_distinct.pdf 2>&1 | tee -a "$LOG"

"$PY" - <<'PY' | tee -a "$LOG"
import json
import numpy as np

records = json.load(open("outputs/attention/part_attention_distinct.json", encoding="utf-8"))
scores = []
for record in records:
    attention = np.asarray(record["attention"], dtype=np.float64)
    attention /= np.linalg.norm(attention, axis=1, keepdims=True).clip(1e-12)
    similarity = attention @ attention.T
    scores.extend(similarity[np.triu_indices(5, 1)].tolist())
mean_cosine = float(np.mean(scores))
print(f"ATTENTION_OFFDIAG_COSINE={mean_cosine:.6f}")
if mean_cosine >= 0.90:
    raise SystemExit("distinct-part audit failed: attention queries remain collapsed")
PY

echo "===== $(date) distinct full GALA from repaired +Part =====" | tee -a "$LOG"
"$PY" trainers/train.py \
  --config configs/gala_humanml3d_flow_distinct.yaml \
  --init-from checkpoints/gala_humanml3d_flow_part_distinct/best.pt 2>&1 | tee -a "$LOG"

eval_official() {
  local config="$1" checkpoint="$2" output="$3" steps="$4" guidance="$5"
  "$PY" scripts/evaluate_t2m.py --config "$config" --checkpoint "$checkpoint" \
    --protocol official --split test --replication-times 20 --batch-size 32 \
    --steps "$steps" --guidance "$guidance" --output "$output" 2>&1 | tee -a "$LOG"
}

eval_official configs/gala_humanml3d_flow_part_distinct.yaml \
  checkpoints/gala_humanml3d_flow_part_distinct/best.pt \
  outputs/ablation/graph_global_part_distinct_test_n20.json 20 2.5
eval_official configs/gala_humanml3d_flow_distinct.yaml \
  checkpoints/gala_humanml3d_flow_distinct/best.pt \
  outputs/humanml3d/gala_distinct_test_n20_cfg2.5.json 20 2.5
eval_official configs/gala_humanml3d_flow_distinct.yaml \
  checkpoints/gala_humanml3d_flow_distinct/best.pt \
  outputs/humanml3d/gala_distinct_test_n50_cfg2.0.json 50 2.0

echo "DISTINCT_PART_QUEUE_DONE $(date)" | tee -a "$LOG"
