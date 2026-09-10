# GALA: Graph-Aware Latent Alignment

Code for **GALA** (text-to-motion with a graph VAE + rectified flow) and **GALA-WM** (action-conditioned latent world model).

Paper drafts live under `paper/`. Official HumanML3D numbers use the Guo / MDM evaluator.

- **Code:** https://github.com/yanghhx/gala-motion
- **T2M Overleaf pack:** `paper/gala_t2m_overleaf/`
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

| Setting | R@3 | FID | MM Dist | Diversity |
| --- | --- | --- | --- | --- |
| 20 steps, CFG 2.5 | 0.768 | 0.330 | 3.248 | 9.529 |
| 50 steps, CFG 2.0 | 0.749 | 0.312 | 3.342 | 9.365 |

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
