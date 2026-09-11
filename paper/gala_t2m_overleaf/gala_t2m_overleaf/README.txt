GALA T2M Overleaf pack (ICASSP / CCF-B revision)
================================================

Upload this folder as a new Overleaf project.

Compiler: pdfLaTeX
Main file: main.tex
Recompile twice so citations resolve.

Needs: tikz, booktabs, natbib (Overleaf has them).

Figures (regenerate from the project root):

  python paper/gala_t2m_overleaf/gala_t2m_overleaf/plot_architecture.py
  python paper/gala_t2m_overleaf/gala_t2m_overleaf/plot_figures.py

  figures/fig_architecture.pdf  three-block GALA-v2 pipeline
  figures/fig_compare.pdf       R@3 / FID vs native-HML3D methods
  figures/fig_sweep.pdf         CFG × ODE-step val scan
  figures/fig_train_curve.pdf   Flow training
  figures/fig_recon_gap.pdf     VAE recon vs generated FID

What is filled vs. what must be run
-----------------------------------
Filled (real numbers):

  Table 1 GALA-20/50 (official 20-rep test)
  Val sampling scan
  CTR-Graph-VAE recon FID 0.003 / MPJPE 0.080 / bone 0.051 / vel 0.038
  Conv-VAE recon FID 0.012 / MPJPE 0.105 / bone 0.061 / vel 0.039
  ST-GCN-VAE recon FID 0.003 / MPJPE 0.088 / bone 0.050 / vel 0.038
  GALA-20 latency 140 ms, GALA-50 340 ms (RTX 4060, T=196)

Empty cells (do not invent numbers):

  Table KIT-ML GALA row
  Table component ablation except "+ Global" (that row is GALA-20)

If Conv-VAE reconstruction FID had also been ~0.003, we would demote the
graph tokenizer from contribution (1). It is 0.012, so the graph stays.

Training order (do not enable every flag in one jump)
-----------------------------------------------------
Round 1 — justify Graph and InfoNCE, measure speed

  # tokenizer controls, 35k steps each
  python trainers/train.py --config configs/gala_humanml3d_vae_conv.yaml
  python trainers/train.py --config configs/gala_humanml3d_vae_stgcn.yaml

  python scripts/eval_tokenizer.py --config configs/gala_humanml3d.yaml \
    --checkpoint checkpoints/gala_humanml3d_vae/best.pt --split val
  python scripts/eval_tokenizer.py --config configs/gala_humanml3d_vae_conv.yaml \
    --checkpoint checkpoints/gala_humanml3d_vae_conv/best.pt --split val
  python scripts/eval_tokenizer.py --config configs/gala_humanml3d_vae_stgcn.yaml \
    --checkpoint checkpoints/gala_humanml3d_vae_stgcn/best.pt --split val

  # flow ablations (init from the matching frozen VAE, except +Part)
  python trainers/train.py --config configs/gala_humanml3d_flow_base.yaml \
    --init-from checkpoints/gala_humanml3d_vae_conv/best.pt
  python trainers/train.py --config configs/gala_humanml3d_flow_noalign.yaml \
    --init-from checkpoints/gala_humanml3d_vae/best.pt

  python scripts/bench_efficiency.py --config configs/gala_humanml3d_flow.yaml \
    --checkpoint checkpoints/gala_humanml3d_flow/best.pt --output checkpoints/gala_humanml3d_flow/efficiency.json

Round 2 — part-level alignment only (expect ΔR@3 / ΔMM Dist)
  Additive on the public +Global checkpoint, not a from-scratch DiT:

  python trainers/train.py --config configs/gala_humanml3d_flow_part.yaml \
    --init-from checkpoints/gala_humanml3d_flow/best.pt

Round 3 — kinematic flow (expect ΔFID / bone / foot)
  Continue from +Part if that row helps; otherwise from +Global:

  python trainers/train.py --config configs/gala_humanml3d_flow_v2.yaml \
    --init-from checkpoints/gala_humanml3d_flow_part/best.pt

Round 4 — KIT-ML + official 20-rep tests

  python trainers/train.py --config configs/gala_kitml.yaml --stage vae
  python trainers/train.py --config configs/gala_kitml_flow_v2.yaml \
    --init-from checkpoints/gala_kitml_vae/best.pt

  python scripts/evaluate_t2m.py --config configs/gala_humanml3d_flow_v2.yaml \
    --checkpoint checkpoints/gala_humanml3d_flow_v2/best.pt \
    --protocol official --replication-times 20 --batch-size 32 --steps 20 --guidance 2.5

If part alignment does not move R@3, drop it rather than keep a decorative module.
