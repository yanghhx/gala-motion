GALA T2M Overleaf pack
======================

Upload this folder (or the zip) as a new Overleaf project.

Compiler: pdfLaTeX
Main file: main.tex
Recompile twice so citations resolve.

Needs: tikz, booktabs, natbib (Overleaf has them).
If TikZ errors, comment out \input{fig_architecture} and
\includegraphics{fig1_architecture.pdf} is not provided; keep the TikZ.

Figures:
  figures/fig_architecture.pdf  three-block GALA pipeline (print figure)
  fig_architecture.tex          TikZ twin, not included by main.tex
  figures/fig_compare.pdf       R@3 / FID vs GENMO Table 4
  figures/fig_sweep.pdf         CFG × ODE-step val scan
  figures/fig_train_curve.pdf   Flow training
  figures/fig_recon_gap.pdf     VAE recon vs generated FID
