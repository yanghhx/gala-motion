# GALA: Anatomy-Aware Language–Motion Alignment for Efficient Text-to-Motion Rectified Flow

Draft · HumanML3D test, 20 replications · Guo / MDM protocol

**Code:** github.com/yanghhx/gala-motion

## Story — Three Contributions

1. **CTR Graph Motion Tokenizer** — topology-aware graph encoder preserves skeleton structure.
2. **Anatomically Grounded Part-Semantic Alignment** — learnable part queries anchored to CLIP anatomical prompt embeddings (gate init -3, sigmoid≈0.05).
3. **Kinematic Rectified Flow** — latent velocity matching + decoded velocity/bone/foot/acceleration/skating losses.

Solves: (1) skeleton structure preservation, (2) fine-grained language-body correspondence, (3) physical plausibility.

## HumanML3D Main Table

Official test, 4544 clips, batch 32, 20 replications.

| Method | NFE | R@3 ↑ | FID ↓ | MM Dist ↓ | Diversity → |
| --- | --- | --- | --- | --- | --- |
| Real | — | 0.797 | 0.002 | 2.974 | 9.503 |
| MDM | 1000 | 0.611 | 0.544 | 5.566 | 9.559 |
| MLD | 50 | 0.772 | 0.473 | 3.196 | 9.724 |
| MoMask | — | 0.807 | 0.045 | 2.958 | — |
| **GALA 20** | 20 | **0.768±0.002** | 0.330±0.011 | **3.248±0.008** | **9.529±0.085** |
| **GALA 50** | 50 | 0.749±0.003 | **0.312±0.009** | 3.342±0.009 | 9.365±0.115 |

VAE recon FID = 0.003 (CTR) vs 0.012 (Conv). Params = 59.9M.

## Tokenizer Ablation (HumanML3D val, 1504 clips)

| Tokenizer | R@3 ↑ | Recon FID ↓ | MPJPE ↓ | Bone ↓ | Vel. ↓ |
| --- | --- | --- | --- | --- | --- |
| Conv-VAE | 0.755 | 0.012 | 0.105 | 0.061 | 0.039 |
| CTR-Graph-VAE | 0.750 | **0.003** | **0.080** | 0.051 | 0.038 |

Efficiency: GALA-10 = 70ms; GALA-20 = 140ms; GALA-50 = 340ms on RTX 4060.

---

## Contribution 2: Anatomically Grounded Part-Semantic Alignment

### Method

Anchored part query: q_p^anchor = q_p^learn + sigmoid(g_p) * W_a * a_p

Anchor loss: L_anchor = mean_p [1 - cos(q_bar_p, a_bar_p)]

Gate init g_p = -3, so sigmoid(-3) ≈ 0.05 — avoids strong perturbation early in training.

Full loss: L_PSA = L_part-NCE + λ_d L_div + λ_a L_anchor, with λ_a = 0.05.

### Formal Results (HumanML3D, 20-rep, gate_init=-3)

*Pending — training in progress (revision_v2/humanml_anatomy_gateNeg3).*

| Method | FID ↓ | R@3 ↑ | MM Dist ↓ | Skating ↓ |
| --- | --- | --- | --- | --- |
| A3 (control) | **0.354±0.010** | **0.807±0.002** | 3.064 | 0.00155 |
| +Anatomy (gate=-3) | *pending* | *pending* | *pending* | *pending* |

Previous result with gate_init=0: FID 0.369, R@3 0.805 (near-zero cost). Gate=-3 should be even closer to baseline.

### Fine-grained Body-Part Semantic Evaluation (Experiment 2)

Test prompts: "raises left arm", "kicks with right leg", etc. Measure cos(z_part, c_part) per part. Target part should have highest cosine.

*Pending — script `scripts/evaluate_part_semantic.py` ready, will run after training.*

### Query-Anchor Similarity Matrix

S_ij = cos(q_i, a_j), expected diagonal > off-diagonal. *Pending.*

---

## Contribution 3: Kinematic Rectified Flow with Physical Constraints

### Method

Contact-aware skating loss on kinematic flow endpoint, world-frame positions via recover_from_ric:

L_skate = sum c_{t,f} ||p_hat_{t+1,f} - p_hat_{t,f}||^2 / (sum c_{t,f} + eps)

Adaptive weighting: λ_eff = η * L_RF / L_skate, so λ_eff L_skate / L_RF ≈ η.

Decay schedule η: 0.03 → 0.01 over 40k steps.

### Formal Results (HumanML3D, 20-rep)

| Method | FID ↓ | R@3 ↑ | MM Dist ↓ | Div → | Paired Acc ↓ | Skating ↓ |
| --- | --- | --- | --- | --- | --- | --- |
| A3 (control) | **0.354±0.010** | **0.807±0.002** | **3.064** | 9.744 | 0.004426 | 0.001550 |
| A3+Skate (r=0.02) | 0.446±0.008 | 0.794±0.002 | 3.148 | 9.723 | 0.004271 | **0.001168** |
| A3+Skate decay | 0.403±0.010 | 0.799±0.002 | 3.112 | 9.740 | 0.004336 | 0.001314 |

Skating reduced 15% (decay config) with FID +0.049, R@3 -0.008. Claim: improves physical plausibility with limited quality degradation.

---

## KIT-ML Experiments (Experiment 1)

*Pending — training in progress (revision_v2/kitml_{base,anatomy,skate}).*

| Method | FID ↓ | R@3 ↑ | MM Dist ↓ | Diversity → | Skating ↓ |
| --- | --- | --- | --- | --- | --- |
| Base GALA | *pending* | *pending* | *pending* | *pending* | *pending* |
| +Anatomy | *pending* | *pending* | *pending* | *pending* | *pending* |
| +Skate | *pending* | *pending* | *pending* | *pending* | *pending* |

---

## Final Ablation Table (Experiment 3)

HumanML3D test, 20-rep, NFE=20, CFG=2.5, 50k steps from Distinct.

| Model | CTR | Part | Anchor | Kinematic | FID ↓ | R@3 ↑ | Skating ↓ |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Baseline | ✓ | | | | 0.354±0.010 | 0.807±0.002 | 0.00155 |
| +Part | ✓ | ✓ | | | 0.354±0.010 | 0.807±0.002 | 0.00155 |
| +Anchor | ✓ | ✓ | ✓ | | *pending* | *pending* | *pending* |
| +Kinematic | ✓ | ✓ | ✓ | ✓ | *pending* | *pending* | *pending* |

Note: "+Part" = Baseline because Distinct checkpoint already includes part alignment. Ablation isolates anchor and kinematic contributions. ST-CTR is NOT in this table.

---

## Analysis: Temporal Dependency Modeling (ST-CTR) — NOT a Contribution

ST-CTR (spatial CTR + temporal per-joint self-attention) hurts generation quality and is excluded from the final model. Included as analysis only.

| Method | FID ↓ | R@3 ↑ | MM Dist ↓ | Div → | Skating ↓ |
| --- | --- | --- | --- | --- | --- |
| A3 (control) | **0.354±0.010** | **0.807±0.002** | **3.064** | **9.744** | 0.00155 |
| A5 (+ST-CTR) | 0.458±0.010 | 0.792±0.002 | 3.125 | 9.379 | 0.00147 |

ST-CTR adds quality cost (FID +0.104, R@3 -0.015, Diversity -0.37). Temporal modeling improves motion dynamics but introduces optimization difficulty in latent generation. The final model keeps the original CTR graph tokenizer for better quality-efficiency tradeoff.

### Combined Model Analysis (A7)

ST-CTR + Anatomy + Skate decay combined:

| Method | FID ↓ | R@3 ↑ | Skating ↓ |
| --- | --- | --- | --- |
| A3 (control) | **0.354±0.010** | **0.807±0.002** | 0.00155 |
| A7 (combined) | 0.502±0.010 | 0.787±0.002 | 0.00129 |

The full combination accumulates quality costs (FID +0.148) rather than showing synergy. Skating reduction (-17%) is similar to skate-decay alone (-15%), confirming physical benefit comes from the skating loss, not ST-CTR or Anatomy. Modules are not complementary in the 50k-step budget.
