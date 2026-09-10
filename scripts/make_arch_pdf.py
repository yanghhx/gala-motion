#!/usr/bin/env python3
"""Print-size architecture figure for GALA-WM.

Drawn at column-pair width so fonts are not shrunk by includegraphics.
Layout follows DINO-WM Fig. 2: frozen encoder | dynamics | latent CEM,
with MLD/MoMask-style panel gutters and short labels.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, FancyBboxPatch as _
from matplotlib.patches import Rectangle
import numpy as np

ROOT = Path("/home/qinyang/桌面/project")
OUTS = [ROOT / "paper" / "figures", ROOT / "paper" / "overleaf" / "figures"]

INK = "#1F2933"
MUTE = "#5B6770"
LINE = "#3D4A54"
BLUE = "#2F5D8A"
GREEN = "#2C6A4A"
ORANGE = "#A15C28"
PURPLE = "#5A4580"
B_FILL = "#F3F6FA"
G_FILL = "#F2F7F4"
P_FILL = "#F5F3F8"
CHIP_B = "#D9E4F0"
CHIP_G = "#D4E8DC"
CHIP_O = "#F3E0D0"
CHIP_P = "#E4DDF0"
WHITE = "#FFFFFF"


def rbox(ax, x, y, w, h, fc, ec, lw=0.9, rad=0.05, z=2):
    p = FancyBboxPatch(
        (x, y), w, h,
        boxstyle=f"round,pad=0.012,rounding_size={rad}",
        facecolor=fc, edgecolor=ec, linewidth=lw, zorder=z, clip_on=False,
    )
    ax.add_patch(p)
    return p


def arr(ax, x0, y0, x1, y1, color=LINE, lw=0.95):
    ax.add_patch(FancyArrowPatch(
        (x0, y0), (x1, y1),
        arrowstyle="-|>", mutation_scale=8, lw=lw, color=color,
        shrinkA=0, shrinkB=0.4, zorder=6,
    ))


def stick(ax, x, y, s=0.22, color=INK, lw=0.95):
    ax.plot([x, x], [y, y + 0.55 * s], color=color, lw=lw, solid_capstyle="round", zorder=5)
    ax.plot([x - 0.32 * s, x, x + 0.32 * s], [y + 0.22 * s, y + 0.55 * s, y + 0.22 * s],
            color=color, lw=lw, solid_capstyle="round", zorder=5)
    ax.plot([x - 0.24 * s, x, x + 0.24 * s], [y - 0.38 * s, y, y - 0.38 * s],
            color=color, lw=lw, solid_capstyle="round", zorder=5)
    ax.add_patch(Circle((x, y + 0.72 * s), 0.11 * s, facecolor=color, edgecolor="none", zorder=5))


def panel(ax, x, y, w, h, fill, title, tag):
    rbox(ax, x, y, w, h, fill, "#D5DCE3", lw=0.85, rad=0.07, z=0)
    ax.text(x + 0.10, y + h - 0.16, tag, ha="left", va="center",
            fontsize=8.2, fontweight="bold", color=INK, zorder=3)
    ax.text(x + 0.32, y + h - 0.16, title, ha="left", va="center",
            fontsize=8.2, color=INK, zorder=3)


def chips(ax, x, y, n, w, h, gap, fc, ec):
    for i in range(n):
        rbox(ax, x + i * (w + gap), y, w, h, fc, ec, lw=0.7, rad=0.03, z=4)


def main():
    # Near final print width (two-column textwidth ≈ 7.06 in)
    W, H = 7.36, 4.05
    fig, ax = plt.subplots(figsize=(W, H), dpi=300)
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.axis("off")
    fig.patch.set_facecolor(WHITE)
    ax.set_facecolor(WHITE)

    # ----- panels -----
    pa, pb, pc = 0.06, 2.48, 5.02
    pw_a, pw_b, pw_c = 2.32, 2.44, 2.28
    py, ph = 0.10, 3.86
    panel(ax, pa, py, pw_a, ph, B_FILL, "Frozen observation model", "(a)")
    panel(ax, pb, py, pw_b, ph, G_FILL, "Residual dynamics  $p_\\theta$", "(b)")
    panel(ax, pc, py, pw_c, ph, P_FILL, "Latent CEM planning", "(c)")

    # ===== (a) observations =====
    ax.text(pa + 0.14, 3.48, "pose history", ha="left", va="center", fontsize=7.0, color=MUTE)
    frames = [r"$x_{t-2}$", r"$x_{t-1}$", r"$x_t$"]
    for i, lab in enumerate(frames):
        fx = pa + 0.22 + i * 0.70
        rbox(ax, fx, 2.72, 0.58, 0.70, WHITE, "#C5CDD6", lw=0.75, rad=0.04)
        stick(ax, fx + 0.29, 2.98, s=0.28, color=BLUE)
        ax.text(fx + 0.29, 2.62, lab, ha="center", va="top", fontsize=7.4, color=MUTE)

    arr(ax, pa + 1.16, 2.70, pa + 1.16, 2.52)
    rbox(ax, pa + 0.18, 2.02, 1.96, 0.46, WHITE, BLUE, lw=1.05, rad=0.05)
    ax.text(pa + 1.16, 2.32, "Frozen GALA VAE", ha="center", va="center",
            fontsize=8.0, fontweight="bold", color=BLUE)
    ax.text(pa + 1.16, 2.13, r"$263$-D  $\rightarrow$  $z\in\mathbb{R}^{256}$",
            ha="center", va="center", fontsize=7.2, color=MUTE)

    arr(ax, pa + 1.16, 2.02, pa + 1.16, 1.84)
    ax.text(pa + 0.18, 1.72, r"history tokens $z_{t-k:t}$", ha="left", va="center",
            fontsize=7.0, color=MUTE)
    chips(ax, pa + 0.22, 1.38, 5, 0.30, 0.26, 0.08, CHIP_B, BLUE)

    ax.text(pa + 0.18, 1.14, "explicit locomotion action", ha="left", va="center",
            fontsize=7.0, color=MUTE)
    rbox(ax, pa + 0.18, 0.28, 1.96, 0.72, WHITE, ORANGE, lw=1.05, rad=0.05)
    ax.text(pa + 1.16, 0.78, r"$a_t=\mathrm{mean}(\mathrm{root4})$",
            ha="center", va="center", fontsize=7.6, color=ORANGE)
    ax.text(pa + 1.16, 0.52, r"$(\dot\psi,\,v_x,\,v_z,\,h)\in\mathbb{R}^{4}$",
            ha="center", va="center", fontsize=7.2, color=MUTE)

    # ===== (b) dynamics =====
    rbox(ax, pb + 0.14, 3.12, 2.16, 0.42, WHITE, GREEN, lw=0.95, rad=0.05)
    ax.text(pb + 1.22, 3.33, r"$\hat{z}_{t+1}=z_t+\sigma(g(a_t))\,\Delta$",
            ha="center", va="center", fontsize=8.0, color=INK)

    ax.text(pb + 0.16, 2.96, "history  (FiLM by $a_t$)", ha="left", va="center",
            fontsize=7.0, color=MUTE)
    chips(ax, pb + 0.18, 2.58, 4, 0.32, 0.26, 0.08, CHIP_G, GREEN)
    rbox(ax, pb + 1.78, 2.58, 0.44, 0.26, CHIP_O, ORANGE, lw=0.75, rad=0.03)
    ax.text(pb + 2.00, 2.71, r"$a_t$", ha="center", va="center", fontsize=7.2, color=ORANGE)

    arr(ax, pb + 1.22, 2.58, pb + 1.22, 2.40)
    rbox(ax, pb + 0.18, 1.88, 2.16, 0.50, WHITE, GREEN, lw=1.05, rad=0.05)
    ax.text(pb + 1.22, 2.22, "3-layer Transformer", ha="center", va="center",
            fontsize=8.0, fontweight="bold", color=GREEN)
    ax.text(pb + 1.22, 2.02, "self-attn on $z$, action as token",
            ha="center", va="center", fontsize=7.0, color=MUTE)

    arr(ax, pb + 1.22, 1.88, pb + 1.22, 1.70)
    rbox(ax, pb + 0.18, 1.18, 2.16, 0.48, WHITE, ORANGE, lw=1.0, rad=0.05)
    ax.text(pb + 1.22, 1.50, r"action gate  $\sigma(g(a_t))$", ha="center", va="center",
            fontsize=7.8, fontweight="bold", color=ORANGE)
    ax.text(pb + 1.22, 1.30, r"$a{=}0$  $\to$  copy-last",
            ha="center", va="center", fontsize=7.2, color=MUTE)

    arr(ax, pb + 1.22, 1.18, pb + 1.22, 1.00)
    rbox(ax, pb + 0.18, 0.28, 2.16, 0.68, WHITE, GREEN, lw=1.05, rad=0.05)
    ax.text(pb + 1.22, 0.78, r"residual unroll  $H{=}8$", ha="center", va="center",
            fontsize=7.8, fontweight="bold", color=GREEN)
    ax.text(pb + 1.22, 0.52, r"GT actions  $+$  decoded pose loss",
            ha="center", va="center", fontsize=7.0, color=MUTE)

    # stage arrows — short horizontals in the gutter, not diagonals
    arr(ax, pa + pw_a - 0.01, 1.51, pb + 0.05, 1.51)
    arr(ax, pb + pw_b - 0.01, 1.10, pc + 0.05, 1.10)

    # ===== (c) CEM =====
    rbox(ax, pc + 0.16, 3.08, 1.96, 0.46, WHITE, PURPLE, lw=1.0, rad=0.05)
    ax.text(pc + 1.14, 3.38, r"goal  $x^{\star}$", ha="center", va="center",
            fontsize=7.8, fontweight="bold", color=PURPLE)
    ax.text(pc + 1.14, 3.18, r"frozen encoder  $\to z^{\star}$",
            ha="center", va="center", fontsize=7.0, color=MUTE)

    arr(ax, pc + 1.14, 3.08, pc + 1.14, 2.90)
    rbox(ax, pc + 0.16, 1.55, 1.96, 1.32, WHITE, PURPLE, lw=1.1, rad=0.06)
    ax.text(pc + 1.14, 2.68, "CEM in $z$", ha="center", va="center",
            fontsize=8.2, fontweight="bold", color=PURPLE)
    ax.text(pc + 1.14, 2.46, r"sample $a\sim\mathcal{N}(0,0.5^2)$",
            ha="center", va="center", fontsize=7.0, color=INK)
    ax.text(pc + 1.14, 2.26, r"rollout $\hat z_{1:H}$ with $p_\theta$",
            ha="center", va="center", fontsize=7.0, color=INK)
    ax.text(pc + 1.14, 2.06, r"cost  $\|\hat{z}_H-z^{\star}\|^2$",
            ha="center", va="center", fontsize=7.0, color=INK)
    ax.text(pc + 1.14, 1.82, "pop 48  ·  8 iters  ·  clip",
            ha="center", va="center", fontsize=6.8, color=MUTE)
    ax.text(pc + 1.14, 1.64, "decoder not in the loop",
            ha="center", va="center", fontsize=6.8, color=MUTE)

    arr(ax, pc + 0.70, 1.55, pc + 0.70, 1.36)
    arr(ax, pc + 1.58, 1.55, pc + 1.58, 1.36)
    rbox(ax, pc + 0.16, 0.28, 0.90, 0.92, WHITE, "#C5CDD6", lw=0.9, rad=0.05)
    ax.text(pc + 0.61, 0.92, "Hold", ha="center", va="center",
            fontsize=7.6, fontweight="bold", color=MUTE)
    ax.text(pc + 0.61, 0.68, r"repeat $z_t$", ha="center", va="center", fontsize=6.8, color=MUTE)
    ax.text(pc + 0.61, 0.46, "open-loop", ha="center", va="center", fontsize=6.6, color=MUTE)

    rbox(ax, pc + 1.18, 0.28, 0.94, 0.92, WHITE, BLUE, lw=1.0, rad=0.05)
    ax.text(pc + 1.65, 0.92, "Decode", ha="center", va="center",
            fontsize=7.6, fontweight="bold", color=BLUE)
    ax.text(pc + 1.65, 0.68, r"$\hat x$ for viz", ha="center", va="center", fontsize=6.8, color=MUTE)
    ax.text(pc + 1.65, 0.46, "/ MPJPE", ha="center", va="center", fontsize=6.8, color=MUTE)

    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    for out in OUTS:
        out.mkdir(parents=True, exist_ok=True)
        fig.savefig(out / "fig1_architecture.pdf", bbox_inches="tight", pad_inches=0.03)
        fig.savefig(out / "fig1_architecture.png", dpi=340, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)
    print("wrote print-size fig1_architecture")


if __name__ == "__main__":
    main()
