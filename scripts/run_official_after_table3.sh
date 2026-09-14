#!/usr/bin/env bash
# Follow-on queue: run only comparable official test evaluations after all
# Table-3/KIT training in run_table3_queue.sh has exited successfully.
set -euo pipefail
export PYTHONNOUSERSITE=1
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
export PYTHONPATH="$ROOT"
export HF_ENDPOINT=https://hf-mirror.com
cd "$ROOT"
PY=${PYTHON_BIN:-python}
LOG=runs/gala_round1/official_after_table3.log
mkdir -p outputs/humanml3d outputs/kitml outputs/ablation outputs/latex

while pgrep -f "^bash ${ROOT}/scripts/run_table3_queue.sh$" >/dev/null; do
  sleep 30
done

run_eval() {
  local config="$1" checkpoint="$2" output="$3" steps="$4" guidance="$5"
  if [[ ! -f "$output" ]]; then
    "$PY" scripts/evaluate_t2m.py --config "$config" --checkpoint "$checkpoint" \
      --protocol official --split test --replication-times 20 --batch-size 32 \
      --steps "$steps" --guidance "$guidance" --output "$output" 2>&1 | tee -a "$LOG"
  fi
}

run_eval configs/gala_humanml3d_flow_base.yaml checkpoints/gala_humanml3d_flow_base/best.pt outputs/ablation/base_rf_test_n20.json 20 2.5
run_eval configs/gala_humanml3d_flow_noalign.yaml checkpoints/gala_humanml3d_flow_noalign/best.pt outputs/ablation/graph_rf_test_n20.json 20 2.5
run_eval configs/gala_humanml3d_flow.yaml checkpoints/gala_humanml3d_flow/best.pt outputs/ablation/graph_global_test_n20.json 20 2.5
run_eval configs/gala_humanml3d_flow_part.yaml checkpoints/gala_humanml3d_flow_part/best.pt outputs/ablation/graph_global_part_test_n20.json 20 2.5
run_eval configs/gala_humanml3d_flow_v2.yaml checkpoints/gala_humanml3d_flow_v2/best.pt outputs/humanml3d/gala_test_n20_cfg2.5.json 20 2.5
run_eval configs/gala_humanml3d_flow_v2.yaml checkpoints/gala_humanml3d_flow_v2/best.pt outputs/humanml3d/gala_test_n50_cfg2.0.json 50 2.0
run_eval configs/gala_kitml_flow_v2.yaml checkpoints/gala_kitml_flow_v2/best.pt outputs/kitml/gala_test_n20_cfg2.5.json 20 2.5

for spec in \
  "outputs/ablation/base_rf_test_n20.json|Base RF|humanml3d|20|2.5" \
  "outputs/ablation/graph_rf_test_n20.json|+Graph|humanml3d|20|2.5" \
  "outputs/ablation/graph_global_test_n20.json|+Global|humanml3d|20|2.5" \
  "outputs/ablation/graph_global_part_test_n20.json|+Part|humanml3d|20|2.5" \
  "outputs/humanml3d/gala_test_n20_cfg2.5.json|GALA|humanml3d|20|2.5" \
  "outputs/humanml3d/gala_test_n50_cfg2.0.json|GALA|humanml3d|50|2.0" \
  "outputs/kitml/gala_test_n20_cfg2.5.json|GALA|kitml|20|2.5"; do
  IFS='|' read -r input name dataset nfe cfg <<< "$spec"
  "$PY" scripts/aggregate_replications.py "$input" --name "$name" \
    --dataset "$dataset" --nfe "$nfe" --cfg "$cfg"
done

echo "OFFICIAL_AFTER_TABLE3_DONE $(date)" | tee -a "$LOG"
