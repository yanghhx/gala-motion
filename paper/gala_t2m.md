# GALA: Anatomy-Aware Language–Motion Alignment for Efficient Text-to-Motion Rectified Flow

Draft · HumanML3D test, \(n=4544\), 20 replications · Guo / MDM protocol

**Code:** https://github.com/yanghhx/gala-motion

## Story

Skeleton graph → part-aware motion latents ↔ language tokens → kinematic rectified flow.

1. Anatomy-aware continuous tokenizer (CTR graph + stride-4 VAE).
2. Token-to-body-part alignment (global InfoNCE + part queries that also condition the DiT).
3. Kinematic rectified flow (latent velocity matching + decoded velocity / bone / foot losses, 20–50 NFE).

GENMO is a generalist SMPL baseline, not the comparison target.

## HumanML3D (filled)

Official test, 4544 clips, batch 32, 20 replications. GALA-20/50 are the public graph + global-align + RF checkpoint.

| Method | NFE | R@3 ↑ | FID ↓ | MM Dist ↓ | Diversity → |
| --- | --- | --- | --- | --- | --- |
| Real | — | 0.797 | 0.002 | 2.974 | 9.503 |
| MDM | 1000 | 0.611 | 0.544 | 5.566 | 9.559 |
| MLD | 50 | 0.772 | 0.473 | 3.196 | 9.724 |
| M2DM | — | 0.763 | 0.352 | 3.134 | 9.926 |
| T2M-GPT | — | 0.775 | 0.141 | 3.121 | 9.761 |
| EMDM | 10 | 0.786 | 0.112 | 3.110 | 9.551 |
| MoMask | — | 0.807 | 0.045 | 2.958 | — |
| GENMO | — | 0.632 | 0.216 | 3.466 | 11.342 |
| **GALA 20 / CFG 2.5** | 20 | **0.768 ± 0.002** | 0.330 ± 0.011 | **3.248 ± 0.008** | **9.529 ± 0.085** |
| **GALA 50 / CFG 2.0** | 50 | 0.749 ± 0.003 | **0.312 ± 0.009** | 3.342 ± 0.009 | 9.365 ± 0.115 |

R@1 / R@2: GALA-20 = 0.447 / 0.654; GALA-50 = 0.428 / 0.632.

VAE recon FID = 0.003 (CTR) vs 0.012 (Conv). Params = 59.9M.

Efficiency on RTX 4060, batch 1, \(T=196\): GALA-20 = 140 ms (7.1 clips/s); GALA-50 = 340 ms.

## Tokenizer ablation (HumanML3D val, 1504 clips)

| Tokenizer | Recon FID ↓ | MPJPE ↓ | Bone ↓ | Vel. ↓ |
| --- | --- | --- | --- | --- |
| Conv-VAE | 0.012 | 0.105 | 0.061 | 0.039 |
| ST-GCN-VAE | — | — | — | — |
| CTR-Graph-VAE | **0.003** | 0.080 | 0.051 | 0.038 |

Graph is not a no-op: FID falls from 0.012 to 0.003.

## Must-run before submission

Do not invent the empty cells. Order:

1. Conv-VAE vs ST-GCN-VAE vs CTR-VAE reconstruction (`scripts/eval_tokenizer.py`). Conv vs CTR is filled; ST-GCN is still training.
2. Component ablation: Base RF / +Graph / +Global (filled) / +Part / GALA-v2.
3. `scripts/bench_efficiency.py` latency / FPS on the 4060. **Done** (140 ms / 340 ms).
4. Part alignment, then kinematic flow, as separate runs.
5. KIT-ML official 20-rep Guo numbers.

Configs live under `configs/gala_humanml3d_flow_*.yaml` and `configs/gala_kitml_flow_v2.yaml`. Commands are in `paper/gala_t2m_overleaf/gala_t2m_overleaf/README.txt`.

## Paper files

Overleaf: `paper/gala_t2m_overleaf/gala_t2m_overleaf/`
