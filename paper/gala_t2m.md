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

## Tokenizer ablation (HumanML3D val, 1504 clips)

| Tokenizer | R@3 ↑ | Recon FID ↓ | MPJPE ↓ | Bone ↓ | Vel. ↓ |
| --- | --- | --- | --- | --- | --- |
| Conv-VAE | 0.755 | 0.012 | 0.105 | 0.061 | 0.039 |
| ST-GCN-VAE | 0.748 | 0.003 | 0.088 | 0.050 | 0.038 |
| CTR-Graph-VAE | 0.750 | **0.003** | **0.080** | 0.051 | 0.038 |

Graph is not a no-op: FID falls from 0.012 to 0.003. Retrieval barely moves. Static ST-GCN already matches CTR recon FID; CTR further cuts MPJPE (0.088 → 0.080).

Efficiency on RTX 4060, batch 1, \(T=196\): GALA-10 = 70 ms (14.4 clips/s); GALA-20 = 140 ms (7.1 clips/s); GALA-50 = 340 ms.

## +Part (done, val only)

100k steps on top of the public +Global checkpoint. Best ckpt = last step.

| Split | n | R@3 ↑ | FID ↓ |
| --- | --- | --- | --- |
| +Global val (sweep, 20 / 2.5) | 1504 | 0.781 | 0.335 |
| +Part val (train eval, 20 / 2.5) | 320 | 0.800 | 0.370 |
| +Global test (official 20-rep) | 4544 | 0.768 | 0.330 |
| +Part test | 4544 | — | — |

R@3 looks better on the small val; FID does not beat +Global. Not entered as a 20-rep test number. GALA-v2 started from this ckpt at 17:05.

## Must-run before submission

Do not invent the empty cells. Order:

1. Conv-VAE vs ST-GCN-VAE vs CTR-VAE reconstruction (`scripts/eval_tokenizer.py`). **Done.**
2. Component ablation: Base RF / +Graph / +Global (filled) / +Part (val n=320 done; official 20-rep pending) / GALA-v2 (running on 4060 from +Part).
3. `scripts/bench_efficiency.py` latency / FPS on the 4060. **Done** (70 / 140 / 340 ms).
4. Official 20-rep for +Part and GALA-v2 after v2 finishes.
5. KIT-ML official 20-rep Guo numbers.

Configs live under `configs/gala_humanml3d_flow_*.yaml` and `configs/gala_kitml_flow_v2.yaml`. Commands are in `paper/gala_t2m_overleaf/gala_t2m_overleaf/README.txt`.

## Paper files

Overleaf: `paper/gala_t2m_overleaf/gala_t2m_overleaf/`

---

## TMM Extension: Contact-aware Physical Constraints (Phase 1)

### Method

Contact-aware foot-skating loss on the kinematic flow endpoint. The skating
loss operates in world-frame joint positions (recovered from Guo 263-dim
features via `recover_from_ric`) and penalises foot displacement during
GT-derived contact frames:

\[
L_{\text{skate}} = \frac{\sum_{t,f} c_{t,f} \|\hat p_{t+1,f} - \hat p_{t,f}\|_2^2}{\sum_{t,f} c_{t,f} + \epsilon}
\]

where \(c_{t,f}\) is the GT-contact-conditioned contact mask (last-4
HumanML3D binary channels).  The loss is applied to the kinematic flow
endpoint \(\hat z_1 = z_t + (1-t) v_\theta\) decoded as \(\hat X =
D(\hat z_1)\).

**Adaptive weighting.**  The skating loss raw value is ~\(1.7 \times
10^{-4}\), while the flow loss is ~60.  A fixed \(\lambda\) would need to
be ~\(3 \times 10^5\) to reach 10–20% of flow.  Instead we use adaptive
weighting:

\[
\lambda_{\text{eff}} = \eta \cdot \frac{L_{\text{RF}}}{L_{\text{skate}}}
\quad\text{so}\quad
\frac{\lambda_{\text{eff}} L_{\text{skate}}}{L_{\text{RF}}} \approx \eta
\]

With \(\eta = 0.02\) (chosen by screening), the skating loss stays at
~2% of flow throughout training.  A decay schedule \(\eta: 0.03 \to
0.01\) over the first 40k steps further reduces FID cost.

### Screening (fixed val, n=320, NFE=20, CFG=2.5, 1 replication)

Three rounds of weight screening from the same Distinct checkpoint:

| Round | Strategy | skate/flow | Skating ↓ | FID | R@3 |
| --- | --- | --- | --- | --- | --- |
| v1 (4k) | Fixed λ=0.1 | 0.004% | -0% | 0.481 | 0.750 |
| v2 (10k) | Adaptive r=0.15 | 15% | -70% | 1.656 | 0.691 |
| v2 (10k) | Adaptive r=0.25 | 25% | -82% | 2.679 | 0.669 |
| v3 (30k) | Adaptive r=0.01 | 1% | -9% | 0.495 | 0.766 |
| v3 (30k) | Adaptive r=0.03 | 3% | -29% | 0.639 | 0.753 |
| v3 (30k) | Adaptive r=0.05 | 5% | -44% | 0.688 | 0.716 |

Low ratio + long training is optimal. r=0.01 preserves R@3 exactly while
reducing skating 9%. r=0.03 gives 29% skating reduction with acceptable
quality cost.

### Formal experiment (HumanML3D test, 20 replications, NFE=20, CFG=2.5)

Both trained 50k steps from the same Distinct checkpoint, same seed and
hparams.

| Method | FID ↓ | R@3 ↑ | MM Dist ↓ | Diversity → | Paired Acc ↓ | Skating ↓ | Acc. Mag. ↓ |
| --- | --- | --- | --- | --- | --- | --- | --- |
| **A3** (λ=0, control) | **0.354 ± 0.010** | **0.807 ± 0.002** | **3.064** | 9.744 | 0.004426 ± 0.000005 | 0.001550 ± 0.000012 | 0.005523 |
| **A3+Skate** (r=0.02) | 0.446 ± 0.008 | 0.794 ± 0.002 | 3.148 | 9.723 | 0.004271 ± 0.000003 | **0.001168 ± 0.000008** | **0.005052** |

**Skating reduced 24.6%** (0.00155 → 0.00117, CI non-overlapping, p <
0.001).  Acceleration reduced 3.5% (paired) / 8.5% (magnitude).  Quality
cost: FID +0.092, R@3 −0.013.

The skating loss is a **GT-contact-conditioned generated foot
displacement** metric: contact comes from GT, displacement from generated
motion.  The main table reports paired acceleration error (not
acceleration magnitude, which is diagnostic only).

### Aux ratio decay (done)

r=0.03 for first 10k steps, linear decay to r=0.01 by step 40k.  Final
skate/flow = 0.010 at step 50k.

| Method | FID ↓ | R@3 ↑ | MM Dist ↓ | Div → | Paired Acc ↓ | Skating ↓ | Acc. Mag. ↓ |
| --- | --- | --- | --- | --- | --- | --- | --- |
| **A3** (control) | **0.354 ± 0.010** | **0.807 ± 0.002** | **3.064** | 9.744 | 0.004426 ± 0.000005 | 0.001550 ± 0.000012 | 0.005523 |
| **A3+Skate** (fixed r=0.02) | 0.446 ± 0.008 | 0.794 ± 0.002 | 3.148 | 9.723 | 0.004271 ± 0.000003 | **0.001168 ± 0.000008** | **0.005052** |
| **A3+Skate decay** (0.03→0.01) | 0.403 ± 0.010 | 0.799 ± 0.002 | 3.112 | 9.740 | 0.004336 ± 0.000007 | 0.001314 ± 0.000010 | 0.005251 |

The decay schedule recovers most of the FID cost (0.446 → 0.403) and R@3
(0.794 → 0.799) while retaining 15% skating reduction.  This is the
recommended configuration: **skating −15%, FID +0.049, R@3 −0.008** vs
control.

---

## TMM Extension: Anatomically Anchored Part-Semantic Alignment (Phase 2)

### Method

Learnable part queries are regularised toward precomputed CLIP embeddings
of anatomical prompts ("torso, body and head movement", etc.):

\[
q_p^{\text{anchor}} = q_p^{\text{learn}} + \sigma(g_p) \, W_a \, a_p
\]
\[
L_{\text{anchor}} = \frac{1}{P} \sum_p \left[1 - \cos(\bar q_p, \bar a_p)\right]
\]

where \(a_p = E_{\text{CLIP}}(s_p)\) is the cached CLIP embedding of the
anatomical prompt, \(g_p\) is a learnable per-part gate (init 0, so
\(\sigma(0) = 0.5\)), and \(W_a\) is a projection layer.  The full
part-semantic loss:

\[
L_{\text{PSA}} = L_{\text{part-NCE}} + \lambda_d L_{\text{div}} + \lambda_a L_{\text{anchor}}
\]

**Results** (50k steps from Distinct + 20-rep official test):

| Method | FID ↓ | R@3 ↑ | MM Dist ↓ | Div → | Paired Acc ↓ | Skating ↓ | Acc. Mag. ↓ |
| --- | --- | --- | --- | --- | --- | --- | --- |
| A3 (control) | **0.354 ± 0.010** | 0.807 ± 0.002 | **3.064** | 9.744 | 0.004426 | 0.001550 | 0.005523 |
| **A4 (+Anatomy)** | 0.369 ± 0.008 | **0.805 ± 0.002** | 3.062 | **9.743** | 0.004417 | 0.001551 | 0.005485 |

Anatomy anchor adds negligible quality cost (FID +0.015, R@3 −0.002,
both within CI).  The anchor grounds the five learnable part queries in
CLIP anatomical semantics without hurting generation quality.

### Method

CTR spatial graph + temporal self-attention per joint:

\[
H^{l+1} = H^l + \alpha_s G^{\text{spa}}_{\text{CTR}}(H^l, A) + \alpha_t G^{\text{temp}}(H^l)
\]

The temporal branch does per-joint self-attention across frames with
learnable relative-position bias \(B_{\Delta t}\) and frame-mask awareness.
Gates start at \(\alpha_s = 1, \alpha_t = 0\) so the model begins from the
original spatial-only CTR behaviour.

**Results** (50k steps from Distinct + 20-rep official test):

| Method | FID ↓ | R@3 ↑ | MM Dist ↓ | Div → | Paired Acc ↓ | Skating ↓ | Acc. Mag. ↓ |
| --- | --- | --- | --- | --- | --- | --- | --- |
| A3 (control) | **0.354 ± 0.010** | **0.807 ± 0.002** | **3.064** | **9.744** | 0.004426 | 0.001550 | 0.005523 |
| **A5 (+ST-CTR)** | 0.458 ± 0.010 | 0.792 ± 0.002 | 3.125 | 9.379 | 0.004389 | 0.001468 | 0.005348 |

ST-CTR adds quality cost (FID +0.104, R@3 −0.015, Diversity −0.37).
The temporal gate starts at 0 and may need longer training or a warmer
gate schedule to recover quality.  Physical metrics improve slightly
(skating −5.3%).

---

## TMM Extension: Combined Model (Phase 4)

ST-CTR + Anatomy Anchor + Skate decay.  Ablation matrix entry A7.

**Results** (50k steps from Distinct + 20-rep official test):

| Method | FID ↓ | R@3 ↑ | MM Dist ↓ | Div → | Paired Acc ↓ | Skating ↓ | Acc. Mag. ↓ |
| --- | --- | --- | --- | --- | --- | --- | --- |
| A3 (control) | **0.354 ± 0.010** | **0.807 ± 0.002** | **3.064** | **9.744** | 0.004426 | 0.001550 | 0.005523 |
| **A7 (TMM combined)** | 0.502 ± 0.010 | 0.787 ± 0.002 | 3.166 | 9.449 | 0.004322 ± 0.000005 | 0.001292 ± 0.000009 | 0.005154 |

The full combination accumulates quality costs from all three modules
(FID +0.148, R@3 −0.020) rather than showing synergy.  Skating reduction
(−16.6%) is similar to skate-decay alone (−15.3%), confirming that the
physical benefit comes from the skating loss, not from ST-CTR or Anatomy.
The modules are **not complementary** in the current 50k-step budget.

---

## Ablation Matrix

| ID | CTR | Temporal | Part Align | Distinct | Anatomy | Acc | Skate | FID ↓ | R@3 ↑ | Skate ↓ | Status |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| A3 | ✓ | | ✓ | ✓ | | | | **0.354** | **0.807** | 0.00155 | **Done** |
| A3+Skate | ✓ | | ✓ | ✓ | | | ✓ | 0.446 | 0.794 | 0.00117 | **Done** |
| A3+Skate decay | ✓ | | ✓ | ✓ | | | ✓ | 0.403 | 0.799 | 0.00131 | **Done** |
| A4 | ✓ | | ✓ | ✓ | ✓ | | | 0.369 | 0.805 | 0.00155 | **Done** |
| A5 | ✓ | ✓ | ✓ | ✓ | | | | 0.458 | 0.792 | 0.00147 | **Done** |
| A7 | ✓ | ✓ | ✓ | ✓ | ✓ | | ✓ | 0.502 | 0.787 | 0.00129 | **Done** |

All 20-rep official test, HumanML3D test set, NFE=20, CFG=2.5, 50k steps
from same Distinct checkpoint.
