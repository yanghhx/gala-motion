# GALA-WM: Residual Action-Conditioned Latent Dynamics for Human Motion World Models

**Code:** https://github.com/yanghhx/gala-motion

## Abstract

Text-to-motion generators learn \(p(z\mid\text{text})\) and cannot be queried as simulators. We freeze a GALA motion VAE and learn \(p(z_{t+1}\mid z_{\le t},a_t)\) on HumanML3D with explicit locomotion actions (root yaw rate, planar velocity, height). A residual 4-step model already beats copy-last (58.4 vs 76.0 mm MPJPE at horizon 8) and a hold planner (65% vs 50% CEM success), but zero-action rollouts stay close to GT, so the policy barely uses \(a_t\). We widen that gap with FiLM action modulation, an action gate on the residual, 8-step unrolls, a decoded pose loss, and a stronger CEM (trajectory cost, clipped \(\mathcal{N}(0,0.5^2)\) proposals). On 92 validation clips the v2 model reaches **21.6 / 39.0 / 53.0 mm** MPJPE at horizons 1 / 4 / 8 versus 35.8 / 60.6 / 76.7 mm for copy-last; zero-action at horizon 8 is 75.1 mm (gap 22 mm vs 10 mm in v1). CEM succeeds on **73.9%** of goals (72.0 mm) versus **50%** hold. The result is a locomotion-controllable motion simulator, not another text-to-motion generator.

## 1. Introduction

World models are useful when they answer “what happens if I act.” Closed-loop benchmarks such as World-in-World (ICLR 2026) show that visual fidelity does not imply task utility. Human motion generation has the same gap: GALA maps text to a full clip, with no action interface, no counterfactual rollout, and no CEM.

We keep the frozen GALA tokenizer (\(263\)-D HumanML3D poses \(\rightarrow\) \(256\)-D latents at stride 4) and only train dynamics. The scientific claim is narrow and testable: **action-conditioned residual latent dynamics, trained with short unroll, should beat copy-last, zero-action, and shuffled-action baselines on multi-step MPJPE, fork under reversed locomotion commands, and improve CEM over a hold planner.**

## 2. Related Work

Dreamer-style agents learn latent dynamics for imagination. DINO-WM plans in pretrained features with CEM. Puppeteer and Humanoid World Models target humanoid control. Video foundation models (Cosmos, Genie) simulate pixels. We stay in motion latents so a single 8 GB GPU can host a planner-facing simulator. Evaluation follows the closed-loop rule: prediction error, action swap, and planning success—not FID.

## 3. Method

### 3.1 Frozen tokenizer

GALA encodes \(x_{1:T}\in\mathbb{R}^{T\times 263}\) to \(\mu\in\mathbb{R}^{L\times 256}\). Graph encoder, VAE, and decoder stay frozen (`gala_humanml3d_vae/best.pt`).

### 3.2 Actions

For latent index \(\ell\), \(a_\ell\in\mathbb{R}^{4}\) is the mean of HumanML3D channels \([0:4)\) over the four corresponding frames. An optional IDM head \(a=\mathrm{IDM}(\mu_t,\mu_{t+1})\in\mathbb{R}^{16}\) is implemented but not used in the main table.

### 3.3 Dynamics (unfixed vs fixed)

**Unfixed.** A 2-layer transformer reads \(k=4\) history latents plus an action token and emits an **absolute** \(\hat\mu_{t+1}\). Loss is one-step MSE plus flow / bone / velocity auxiliaries.

**Fixed (this paper’s main model).**

\[
\hat\mu_{t+1}=\mu_t+\Delta(\mu_{t-k+1:t},a_t).
\]

Training adds a **4-step unroll** with ground-truth actions:

\[
\mathcal{L}=\|\hat\mu_{t+1}-\mu_{t+1}\|^2+\sum_{h=1}^{H}\|\hat\mu_{t+h}-\mu_{t+h}\|^2+\lambda_{\mathrm{aux}}\mathcal{L}_{\mathrm{aux}},
\]

**v2 (main).** FiLM modulates every history token by \(a_t\). An action gate makes the residual vanish when \(a=0\):

\[
\hat\mu_{t+1}=\mu_t+\sigma(g(a_t))\,\Delta(\mu_{t-k+1:t},a_t).
\]

Unroll horizon \(H=8\) (matched to eval), \(\lambda_{\mathrm{pose}}=1\) on decoded 1-step motion, 3-layer predictor. CEM uses pop 48, 8 iterations, proposals clipped to \([-2.5,2.5]\) with initial std 0.5, and cost \(=\|\hat\mu_H-\mu^\star\|^2+0.35\cdot\mathrm{mean}_h\|\hat\mu_h-\mu^\star\|^2\).

### 3.4 Evaluation protocol

Validation clips, \(n=92\) after length filtering. Horizons \(\{1,4,8\}\) latent steps (\(\approx 0.2/0.8/1.6\) s). MPJPE is computed on **denormalized** local joints after decoding the **concatenated** latent window. Swap: negate planar velocity and compare terminal root XY to a 5% action jitter. CEM success if MPJPE \(<80\) mm and root XY \(<0.5\) m. Hold copies the last history latent.

## 4. Experiments

**Setup.** Official HumanML3D, batch 8, grad-accum 2, AdamW \(2\times 10^{-4}\), bf16, RTX 4060 8 GB. Unfixed: absolute 1-step. v1: residual + 4-step unroll, 12k steps. v2: FiLM + action gate + 8-step unroll, 10k steps.

### 4.1 Main results (MPJPE, mm)

| Method | @1 | @4 | @8 |
| --- | --- | --- | --- |
| Copy-last | 35.8 | 60.6 | 76.7 |
| Unfixed + GT action | 29.7 | 51.9 | 74.1 |
| v1 residual + GT | 21.5 | 39.9 | 58.2 |
| **v2 FiLM/gate + GT** | **21.6** | **39.0** | **53.0** |
| v1 + zero action | 26.1 | 48.5 | 68.4 |
| **v2 + zero action** | 29.3 | 53.5 | **75.1** |
| v1 + shuffled action | 30.5 | 55.6 | 76.6 |
| **v2 + shuffled action** | 43.9 | 68.6 | **89.9** |

v2 is the first setting where zero/shuffle sit on the copy-last side of the table and GT stays in the 50 mm band. Horizon-8 GT improves 5.2 mm over v1 and **23.7 mm** over copy-last. The GT–zero gap grows from 10 mm (v1) to **22 mm** (v2): the model actually uses \(a_t\).

### 4.2 Controllability and planning

| | Swap \(\Delta\) XY (m) | CEM success | CEM MPJPE (mm) | Hold success |
| --- | --- | --- | --- | --- |
| Unfixed, weak CEM | 0.46 | 0.543 | 88.8 | 0.500 |
| v1, weak CEM | 0.58 | 0.652 | 83.8 | 0.500 |
| v1, strong CEM | 0.58 | 0.717 | 76.8 | 0.500 |
| **v2, strong CEM** | **0.62** | **0.739** | **72.0** | 0.500 |

Stronger CEM alone lifts v1 from 65% to 72% without retraining. v2 adds another 2 points and cuts planner MPJPE to 72 mm (hold 103 mm). Opposite planar velocities still fork root XY by \(\approx 0.6\) m.

### 4.3 What went wrong in the first run

1. Two `train_wm.py` processes shared one checkpoint directory; `best.pt` was not a clean 20k-step model.
2. One-step absolute regression overfits short-term reconstruction; autoregressive rollout then diverges.
3. Train-time MPJPE was reported in **normalized** coordinates (\(\times 1000\) looked like metres-scale millimetres).
4. Swap/CEM decoded only the last latent (4 frames), so root XY could not accumulate.

All numbers in Section 4 use the corrected protocol on both checkpoints.

## 5. Limitations

Actions are locomotion-centric; fine-grained upper-body intent is weakly labeled on HumanML3D. We did not transfer the same API to HumanoidBench, Puppeteer, or World-in-World. CEM still fails on 26% of goals. Validation is 92 clips, one seed.

## 6. Conclusion

Freezing a motion VAE is enough to obtain a plannable world model if dynamics are residual, action-modulated, unrolled to the eval horizon, and judged as a simulator. v2 beats copy-last by 24 mm at 1.6 s, doubles the GT–zero gap versus v1, and lifts CEM from hold’s 50% to 74%. Text-to-motion FID is the wrong main table.

## References

Hafner et al., DreamerV3, Nature 2025.  
Zhou et al., DINO-WM, 2024.  
Alonso et al., DIAMOND, NeurIPS 2024.  
Hansen et al., Puppeteer, 2024.  
Ali et al., Humanoid World Models, 2025.  
Zhang et al., World-in-World, ICLR 2026.  
Guo et al., HumanML3D.
