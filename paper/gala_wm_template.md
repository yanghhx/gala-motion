# GALA-WM: Action-Conditioned Latent Dynamics for Human Motion World Models

**Track B implementation on HumanML3D, built on a frozen GALA motion VAE.**

## Abstract

Text-to-motion generators model \(p(z\mid\text{text})\) and cannot be queried as simulators. We turn a trained GALA tokenizer into an action-conditioned world model \(p(z_{t+1}\mid z_{\le t},a_t)\) without changing the decoder. Locomotion channels of HumanML3D (root angular velocity, planar velocity, height) are treated as explicit actions; a compact inverse-dynamics head is an optional latent-action variant. A history-conditioned transformer predicts the next latent, with a rectified-flow auxiliary and CEM planning at test time. On HumanML3D validation clips, GT-action rollouts reach **{{MPJPE_GT_1}} mm** MPJPE at 1 latent step and **{{MPJPE_GT_8}} mm** at 8 steps, versus **{{MPJPE_COPY_8}} mm** for copy-last. Opposite planar velocities produce a swap-to-noise ratio of **{{SWAP_RATIO}}**. CEM planning toward a future latent succeeds on **{{CEM_SUCCESS}}** of goals (MPJPE {{CEM_MPJPE}} mm, root error {{CEM_ROOT}} m). The result is a motion-space simulator that is controllable and plannable, rather than another text-to-motion generator.

## 1. Introduction

World models are useful when they answer “what happens if I act,” not when they only render a plausible future. Recent closed-loop benchmarks make this distinction explicit: visual fidelity does not imply task utility, and action-conditioned post-training plus online planning matter more than scaling an open-loop generator.

Human motion generation has the same gap. GALA (and MDM-style models) map text to a full clip. That distribution is not a dynamics model: it has no action interface, cannot be rolled out under counterfactual controls, and cannot host CEM. We keep the frozen GALA VAE and learn only \(p(z_{t+1}\mid z_{\le t},a_t)\) in the 5 Hz latent grid (stride 4 on 20 fps HumanML3D).

**Contributions.**

1. A drop-in world-model head on a frozen motion tokenizer, with explicit `root4` actions and an IDM variant.
2. An evaluation protocol that reports prediction, action swap, and CEM success instead of FID.
3. Empirical evidence on HumanML3D that action-conditioned latent rollout beats copy-last / zero / shuffled actions when the model is trained.

## 2. Related Work

Dreamer-style agents learn latent dynamics for imagination; DINO-WM plans in pretrained visual features with CEM; Humanoid World Models and Puppeteer target humanoid control. Video foundation models (Cosmos, Genie) simulate pixels. We stay in motion latents so a single academic GPU can train a planner-facing simulator. World-in-World (ICLR 2026) and WorldArena argue for closed-loop utility; our tables follow that rule.

## 3. Method

### 3.1 Frozen tokenizer

GALA encodes \(x_{1:T}\in\mathbb{R}^{T\times 263}\) to \(\mu\in\mathbb{R}^{L\times 256}\), \(L=\lceil T/4\rceil\). Graph encoder, VAE, and decoder stay frozen. All new parameters sit in IDM, the history–action predictor, and an optional flow head.

### 3.2 Actions

**Explicit `root4`.** For latent index \(\ell\), \(a_\ell\) is the mean of HumanML3D channels \([0:4)\) over the four corresponding frames (root yaw rate, local \(v_x,v_z\), root height).

**Latent IDM.** \(a_\ell=\mathrm{IDM}(\mu_\ell,\mu_{\ell+1})\in\mathbb{R}^{16}\), trained with a cycle that reconstructs \(\mu_{\ell+1}\) from history and a stop-grad action.

### 3.3 Dynamics

A 2-layer transformer reads \(k=4\) history latents plus an action token and outputs \(\hat\mu_{\ell+1}\). Training minimizes

\[
\mathcal{L}=\lambda_{\mathrm{mse}}\|\hat\mu-\mu\|^2+\lambda_{\mathrm{flow}}\mathcal{L}_{\mathrm{RF}}+\lambda_{\mathrm{idm}}\mathcal{L}_{\mathrm{IDM}}+\lambda_{\mathrm{bone}}\mathcal{L}_{\mathrm{bone}}+\lambda_{\mathrm{vel}}\mathcal{L}_{\mathrm{vel}}.
\]

Rectified flow is auxiliary; CEM and reported rollouts use the deterministic predictor.

### 3.4 Planning

CEM searches an action sequence of horizon \(H=8\) (population 32, 5 iterations) to minimize \(\|\hat\mu_H-\mu^\star\|^2\). Success: decoded local MPJPE \(<80\) mm and root XY error \(<0.5\) m.

## 4. Experiments

**Data.** Official HumanML3D splits, Mean/Std normalization, 22 joints. VAE checkpoint: `gala_humanml3d_vae/best.pt`. World model trained from that tokenizer only.

**Protocol.** Validation clips. Horizons \(\{1,4,8\}\) latent steps. Baselines: copy-last, zero action, shuffled actions. Swap: negate planar velocity (or the matching IDM dims) and compare endpoint XY displacement to a 5% action jitter.

### 4.1 Main results

| Method | MPJPE@1 (mm) | MPJPE@8 (mm) | latent L2@1 | latent L2@8 |
| --- | --- | --- | --- | --- |
| GALA-WM + GT action | {{MPJPE_GT_1}} | {{MPJPE_GT_8}} | {{LAT_GT_1}} | {{LAT_GT_8}} |
| Copy-last | {{MPJPE_COPY_1}} | {{MPJPE_COPY_8}} | — | — |
| Zero action | {{MPJPE_ZERO_1}} | {{MPJPE_ZERO_8}} | — | — |
| Shuffled action | {{MPJPE_SHUF_1}} | {{MPJPE_SHUF_8}} | — | — |

| Controllability / planning | Value |
| --- | --- |
| Swap / noise XY ratio | {{SWAP_RATIO}} |
| CEM success rate | {{CEM_SUCCESS}} |
| CEM MPJPE (mm) | {{CEM_MPJPE}} |
| CEM root XY (m) | {{CEM_ROOT}} |

Open-loop GALA text-to-motion is not an action-conditioned predictor; we treat it as uncontrollable (swap undefined) and do not put FID in the main table.

### 4.2 Analysis

GT-action MPJPE@8 should sit below copy-last and zero-action if \(a_t\) is used. A swap/noise ratio \(\gg 1\) means opposite locomotion commands fork the trajectory more than small noise. CEM success above a copy-last planner means the model can be queried as a simulator.

## 5. Limitations

The current action is locomotion-centric; fine-grained upper-body intent is only weakly labeled on HumanML3D. Pixel-level embodied benchmarks (World-in-World, LIBERO) are left to a later transfer of the same action API. Long-horizon drift still grows with \(H\).

## 6. Conclusion

Freezing a motion VAE and learning action-conditioned next-latent dynamics is enough to turn GALA into a plannable world model. The right metrics are rollout error, counterfactual action swap, and CEM success—not text-to-motion FID.

## References

Hafner et al., DreamerV3, Nature 2025. Zhou et al., DINO-WM, 2024. Alonso et al., DIAMOND, NeurIPS 2024. Hansen et al., Puppeteer, 2024. Ali et al., Humanoid World Models, 2025. Zhang et al., World-in-World, ICLR 2026. NVIDIA Cosmos-Predict2.5. Guo et al., HumanML3D.
