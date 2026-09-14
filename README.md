# GALA: Graph-Aware Latent Alignment

Code for **GALA** (text-to-motion with a graph VAE + rectified flow) and **GALA-WM** (action-conditioned latent world model).

Paper drafts live under `paper/`. Official HumanML3D numbers use the Guo / MDM evaluator.

- **Code:** https://github.com/yanghhx/gala-motion
- **T2M Overleaf pack:** `paper/gala_t2m_overleaf/gala_t2m_overleaf/`
- **World-model Overleaf pack:** `paper/overleaf/`

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
pip install pytest pyyaml tensorboard
```

HumanML3D, KIT-ML, and the Guo evaluator assets are **not** in this repository. Place them under `data/` (or edit the `dataset.root` fields in `configs/`).

## Text-to-motion (GALA)

```bash
# stage 1: tokenizer, then stage 2: flow (see config train_stage)
python trainers/train.py --config configs/gala_humanml3d_flow.yaml

# official Guo protocol
python scripts/evaluate_t2m.py \
  --config configs/gala_humanml3d_flow.yaml \
  --checkpoint checkpoints/gala_humanml3d_flow/best.pt \
  --protocol official --replication-times 20 --batch-size 32 \
  --steps 20 --guidance 2.5
```

Reported operating points on HumanML3D test (\(n=4544\), 20 replications):

| Setting | NFE | Latency (4060, \(T=196\)) | R@3 | FID | MM Dist | Diversity |
| --- | --- | --- | --- | --- | --- | --- |
| 20 steps, CFG 2.5 | 20 | 140 ms | 0.806 | 0.272 | 3.061 | 9.692 |
| 50 steps, CFG 2.0 | 50 | 340 ms | 0.793 | 0.224 | 3.107 | 9.643 |

Tokenizer reconstruction on val (1504 clips): Conv-VAE FID 0.012 / MPJPE 0.105 / R@3 0.755; ST-GCN-VAE 0.003 / 0.088 / 0.748; CTR-Graph-VAE 0.003 / 0.080 / 0.750.

The final GALA rows use the non-collapsed part-query checkpoints and official 20-replication evaluation. Mean off-diagonal attention-profile cosine falls from 0.9988 to 0.3474 after query-diversity repair. Relevant configs are `configs/gala_humanml3d_flow_part_distinct.yaml` and `configs/gala_humanml3d_flow_distinct.yaml`; evaluation artifacts are under `outputs/`.

Tokenizer / efficiency scripts: `scripts/eval_tokenizer.py`, `scripts/bench_efficiency.py`. GALA-10 latency is 70 ms (14.4 clips/s) on the same RTX 4060 protocol.

To reproduce every GALA-owned result in Tables 1--5 from the saved checkpoints (official test protocol uses 20 replications):

```bash
PYTHON_BIN=python bash scripts/reproduce_paper_tables.sh
```

The command writes a fresh, table-indexed copy of all metrics to `outputs/reproduction/`. Dataset and evaluator assets are intentionally not bundled; see the setup section above.

## World model (GALA-WM)

```bash
python trainers/train_wm.py --config configs/gala_humanml3d_wm_v2.yaml
python scripts/eval_wm.py --config configs/gala_humanml3d_wm_v2.yaml
```

## Layout

```
models/          graph tokenizer, DiT flow, GALA-WM dynamics
trainers/        T2M and WM training loops
diffusion/       rectified-flow matching
evaluation/      Guo T2M metrics and WM MPJPE / CEM
configs/         HumanML3D / KIT-ML YAML
scripts/         eval, sweeps, paper figures
paper/           LaTeX drafts and figures
```
