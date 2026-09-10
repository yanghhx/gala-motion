# GALA: Graph-Aware Latent Alignment for Text-to-Motion with Rectified Flow

Draft · HumanML3D test, \(n=4544\), 20 replications · Guo / MDM protocol

**Code:** https://github.com/yanghhx/gala-motion

## Abstract

Synthesizing 3D human motion from natural language remains difficult: a generated clip must obey skeletal kinematics yet stay faithful to a prompt whose statistics are far from pose space. Pose-level diffusion is expressive but expensive, because it denoises redundant raw trajectories; discrete tokenizers recover fidelity only after learning a codebook; generalist SMPL models, when scored on HumanML3D, further pay a conversion tax between body representations. We present GALA, a graph-aware latent alignment model for text-to-motion. Our key insight is that a topology-aware variational tokenizer already yields a compact, near-lossless motion latent, so generation reduces to learning a straight generative path in that space, explicitly aligned with language. GALA encodes joint trajectories with a channel-wise topology graph, compresses them with a stride-4 VAE, aligns CLIP text and motion latents by contrastive learning, and synthesizes latent clips with a rectified-flow transformer under classifier-free guidance. Extensive experiments on HumanML3D under the standard Guo protocol show that GALA substantially outperforms the Motion Diffusion Model and the recent generalist baseline GENMO in text–motion R-Precision (Top-3 0.768 vs. 0.611 / 0.632) and multimodal distance, while matching the diversity of real motions.

## Comparison with GENMO Table 4

Official test, 4544 clips, batch 32, 20 replications. Literature rows copied from GENMO ICCV 2025 Table 4.

| Method | Rep. | R@3 ↑ | FID ↓ | MM Dist ↓ | Diversity → |
| --- | --- | --- | --- | --- | --- |
| Real | HML3D | 0.797 | 0.002 | 2.974 | 9.503 |
| T2M | HML3D | 0.740 | 1.067 | 3.340 | 9.188 |
| MDM | HML3D | 0.611 | 0.544 | 5.566 | 9.559 |
| M2DM | HML3D | 0.763 | 0.352 | 3.134 | 9.926 |
| EMDM | HML3D | 0.786 | 0.112 | 3.110 | 9.551 |
| GENMO | SMPL | 0.632 | 0.216 | 3.466 | 11.342 |
| **GALA 20 steps / CFG 2.5** | HML3D | **0.768 ± 0.002** | 0.330 ± 0.011 | **3.248 ± 0.008** | **9.529 ± 0.085** |
| **GALA 50 steps / CFG 2.0** | HML3D | 0.749 ± 0.003 | **0.312 ± 0.009** | 3.342 ± 0.009 | 9.365 ± 0.115 |

R@1 / R@2: GALA-20 = 0.447 / 0.654; GALA-50 = 0.428 / 0.632.

**Beat GENMO:** R@3, MM Dist, Diversity (closer to Real). **Do not beat GENMO:** FID. **Not SOTA vs EMDM.**

## Sampling scan (val, 1504 clips, 1 seed)

VAE recon FID = 0.003. Best generated FID = 50 steps / CFG 2.0 (0.290). CFG = 1.0 hurts both FID and R@3.

## Paper files

Overleaf: `paper/gala_t2m_overleaf/`
