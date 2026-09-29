#!/usr/bin/env python3
"""GALA-WM architecture in the DINO-WM Fig. 2 / MLD Fig. 2 convention.

Three connected stages, left-to-right:
  (a) frozen observation model
  (b) residual transition p_theta  (actions enter HERE, not in the encoder)
  (c) latent CEM planning
All forward computation lives in z; the decoder is visualization-only.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch

ROOT = Path("/home/qinyang/桌面/project")
OUT = ROOT / "paper" / "overleaf" / "figures"

INK = "#1F2933"
MUTE = "#5B6770"
LINE = "#4A5560"
BLUE = "#2F5D8A"
TEAL = "#2C6A4A"
ORANGE = "#A15C28"
PURPLE = "#5A4580"
B_FILL = "#F4F7FB"
G_FILL = "#F3F8F4"
P_FILL = "#F6F4F9"
CHIP_B = "#D7E3F0"
CHIP_T = "#D3E7DB"
CHIP_O = "#F1DDCC"
CHIP_P = "#E4DDF0"
WHITE = "#FFFFFF"
PANEL_E = "#C9D2DB"


def rbox(ax, x, y, w, h, fc, ec, lw=0.95, rad=0.055, z=2, ls="-"):
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h,
        boxstyle=f"round,pad=0.004,rounding_size={rad}",
        facecolor=fc, edgecolor=ec, linewidth=lw, linestyle=ls,
        zorder=z, clip_on=False,
    ))


def arr(ax, x0, y0, x1, y1, color=LINE, lw=1.05, ms=7.5, ls="-"):
    ax.add_patch(FancyArrowPatch(
        (x0, y0), (x1, y1),
        arrowstyle="-|>", mutation_scale=ms, lw=lw, color=color,
        linestyle=ls, shrinkA=0, shrinkB=0, zorder=7, capstyle="round",
    ))


def varr(ax, x, y0, y1, color=LINE, lw=0.95):
    arr(ax, x, y0, x, y1, color=color, lw=lw, ms=7.0)


def gutter_arr(ax, x_left, x_right, y, color=LINE):
    gap = x_right - x_left
    shaft = min(0.16, 0.58 * gap)
    mid = 0.5 * (x_left + x_right)
    arr(ax, mid - 0.5 * shaft, y, mid + 0.5 * shaft, y, color=color, lw=1.15, ms=8.0)


def stick(ax, x, y, s=0.22, color=BLUE, lw=0.95):
    ax.plot([x, x], [y, y + 0.55 * s], color=color, lw=lw, solid_capstyle="round", zorder=5)
    ax.plot([x - 0.32 * s, x, x + 0.32 * s], [y + 0.22 * s, y + 0.55 * s, y + 0.22 * s],
            color=color, lw=lw, solid_capstyle="round", zorder=5)
    ax.plot([x - 0.24 * s, x, x + 0.24 * s], [y - 0.38 * s, y, y - 0.38 * s],
            color=color, lw=lw, solid_capstyle="round", zorder=5)
    ax.add_patch(Circle((x, y + 0.72 * s), 0.11 * s, facecolor=color, edgecolor="none", zorder=5))


def panel(ax, x, y, w, h, fill, tag, title):
    rbox(ax, x, y, w, h, fill, PANEL_E, lw=0.80, rad=0.07, z=0)
    ax.text(x + 0.10, y + h - 0.155, tag, ha="left", va="center",
            fontsize=8.0, fontweight="bold", color=INK, zorder=3)
    ax.text(x + 0.32, y + h - 0.155, title, ha="left", va="center",
            fontsize=8.0, color=INK, zorder=3)
    ax.plot([x + 0.10, x + w - 0.10], [y + h - 0.29, y + h - 0.29],
            color=PANEL_E, lw=0.7, zorder=1, solid_capstyle="round")


def chips(ax, x, y, n, w, h, gap, fc, ec):
    for i in range(n):
        rbox(ax, x + i * (w + gap), y, w, h, fc, ec, lw=0.65, rad=0.03, z=4)


def module(ax, x, y, w, h, title, sub, ec, tc):
    rbox(ax, x, y, w, h, WHITE, ec, lw=1.05, rad=0.05)
    if sub:
        ax.text(x + w / 2, y + h * 0.64, title, ha="center", va="center",
                fontsize=7.8, fontweight="bold", color=tc, zorder=3)
        ax.text(x + w / 2, y + h * 0.30, sub, ha="center", va="center",
                fontsize=6.6, color=MUTE, zorder=3)
    else:
        ax.text(x + w / 2, y + h / 2, title, ha="center", va="center",
                fontsize=7.8, fontweight="bold", color=tc, zorder=3)


def main():
    # DINO-WM Fig. 2: one connected pipeline, not three isolated posters.
    W, H = 7.16, 2.58
    fig, ax = plt.subplots(figsize=(W, H), dpi=300)
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.axis("off")
    fig.patch.set_facecolor(WHITE)

    # ---- band (a)+(b): world model ----
    rbox(ax, 0.04, 1.08, 3.38, 1.44, B_FILL, PANEL_E, lw=0.75, rad=0.07, z=0)
    rbox(ax, 3.46, 1.08, 3.66, 1.44, G_FILL, PANEL_E, lw=0.75, rad=0.07, z=0)
    ax.text(0.14, 2.38, "(a)", ha="left", va="center", fontsize=8.0, fontweight="bold", color=INK)
    ax.text(0.36, 2.38, "Frozen observation", ha="left", va="center", fontsize=8.0, color=INK)
    ax.text(3.56, 2.38, "(b)", ha="left", va="center", fontsize=8.0, fontweight="bold", color=INK)
    ax.text(3.78, 2.38, r"Residual transition  $p_\theta$", ha="left", va="center",
            fontsize=8.0, color=INK)
    ax.plot([0.14, 3.30], [2.26, 2.26], color=PANEL_E, lw=0.7, zorder=1)
    ax.plot([3.56, 7.00], [2.26, 2.26], color=PANEL_E, lw=0.7, zorder=1)

    # poses
    fw, fh = 0.46, 0.50
    f0, fy = 0.16, 1.52
    for i, lab in enumerate((r"$x_{t-2}$", r"$x_{t-1}$", r"$x_t$")):
        fx = f0 + i * 0.52
        rbox(ax, fx, fy, fw, fh, WHITE, "#C5CDD6", lw=0.7, rad=0.04)
        stick(ax, fx + fw / 2, fy + 0.16, s=0.20)
        ax.text(fx + fw / 2, fy - 0.08, lab, ha="center", va="center", fontsize=6.4, color=MUTE)

    # VAE
    vae_x, vae_w, vae_y, vae_h = 1.82, 1.55, 1.48, 0.58
    arr(ax, f0 + 2 * 0.52 + fw, fy + fh / 2, vae_x, vae_y + vae_h / 2, color=BLUE)
    module(ax, vae_x, vae_y, vae_w, vae_h,
           "Frozen GALA VAE", r"stride 4, $z\in\mathbb{R}^{256}$", BLUE, BLUE)

    # z chips
    cw, ch, cg = 0.22, 0.28, 0.05
    chips_x, chips_y = 3.50, 1.62
    arr(ax, vae_x + vae_w, vae_y + vae_h / 2, chips_x, chips_y + ch / 2, color=BLUE)
    chips(ax, chips_x, chips_y, 4, cw, ch, cg, CHIP_B, BLUE)
    ax.text(chips_x + 0.50, chips_y + ch + 0.08, r"$z_{t-k:t}$",
            ha="center", va="center", fontsize=6.5, color=MUTE)

    # a_t
    ax_x, aw, ah = 4.62, 0.42, 0.28
    rbox(ax, ax_x, chips_y, aw, ah, CHIP_O, ORANGE, lw=0.75, rad=0.03)
    ax.text(ax_x + aw / 2, chips_y + ah / 2, r"$a_t$", ha="center", va="center",
            fontsize=7.6, fontweight="bold", color=ORANGE)

    # residual transformer
    tr_x, tr_w, tr_y, tr_h = 5.18, 1.82, 1.22, 1.10
    arr(ax, ax_x + aw, chips_y + ah / 2, tr_x, tr_y + 0.62, color=TEAL)
    rbox(ax, tr_x, tr_y, tr_w, tr_h, WHITE, TEAL, lw=1.12, rad=0.05)
    ax.text(tr_x + tr_w / 2, tr_y + tr_h - 0.14, r"$p_\theta$  Residual Transformer",
            ha="center", va="center", fontsize=7.4, fontweight="bold", color=TEAL)
    ax.text(tr_x + tr_w / 2, tr_y + tr_h - 0.32,
            r"FiLM + gate  ·  3 layers  ·  $H{=}8$",
            ha="center", va="center", fontsize=6.2, color=MUTE)
    rbox(ax, tr_x + 0.08, tr_y + 0.10, tr_w - 0.16, 0.52, CHIP_T, TEAL, lw=0.6, rad=0.03)
    ax.text(tr_x + tr_w / 2, tr_y + 0.42,
            r"$\hat z_{t+1}=z_t+\sigma(g(a_t))\,\Delta$",
            ha="center", va="center", fontsize=7.4, fontweight="bold", color=TEAL)
    ax.text(tr_x + tr_w / 2, tr_y + 0.22,
            r"$a{=}0$  $\to$  copy-last",
            ha="center", va="center", fontsize=6.1, color=MUTE)

    # ---- band (c): planning ----
    rbox(ax, 0.04, 0.06, 7.08, 0.94, P_FILL, PANEL_E, lw=0.75, rad=0.07, z=0)
    ax.text(0.14, 0.86, "(c)", ha="left", va="center", fontsize=8.0, fontweight="bold", color=INK)
    ax.text(0.36, 0.86, r"Latent CEM  —  all forward computation in $z$",
            ha="left", va="center", fontsize=8.0, color=INK)
    ax.plot([0.14, 7.00], [0.74, 0.74], color=PANEL_E, lw=0.7, zorder=1)

    ax.text(0.18, 0.52, r"goal $x^\star$", ha="left", va="center", fontsize=6.5, color=MUTE)
    chips(ax, 0.18, 0.18, 4, 0.24, 0.22, 0.05, CHIP_P, PURPLE)
    ax.text(0.72, 0.08, r"frozen enc $\to z^\star$", ha="center", va="center",
            fontsize=6.1, color=MUTE)

    arr(ax, 1.30, 0.29, 1.55, 0.29, color=PURPLE)
    module(ax, 1.58, 0.14, 2.05, 0.52,
           "CEM in $z$",
           r"pop 48 · 8 iters · $\|\hat z_H-z^\star\|^2$",
           PURPLE, PURPLE)

    arr(ax, 3.66, 0.40, 3.92, 0.40, color=LINE)
    module(ax, 3.96, 0.14, 1.28, 0.52, "Hold", r"repeat $z_t$", ORANGE, ORANGE)
    rbox(ax, 5.36, 0.14, 1.60, 0.52, WHITE, BLUE, lw=1.0, rad=0.05, ls="--")
    ax.text(6.16, 0.48, "Decode", ha="center", va="center",
            fontsize=7.6, fontweight="bold", color=BLUE, zorder=3)
    ax.text(6.16, 0.28, r"viz / MPJPE only", ha="center", va="center",
            fontsize=6.3, color=MUTE, zorder=3)

    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / "fig1_architecture.pdf", bbox_inches="tight", pad_inches=0.02)
    fig.savefig(OUT / "fig1_architecture.png", dpi=360, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print("wrote fig1_architecture")


if __name__ == "__main__":
    main()
