#!/usr/bin/env python3
"""GALA T2M architecture, print width.

Three-column pipeline in the spirit of MLD Fig. 2 / MoMask Fig. 2:
wide gutters, one vertical stack per column, short horizontal arrows
that live in the gutter (never diagonals), no jammed 'x N' titles.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch

ROOT = Path("/home/qinyang/桌面/project")
OUT = ROOT / "paper" / "gala_t2m_overleaf" / "figures"

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
WHITE = "#FFFFFF"
PANEL_E = "#C9D2DB"


def rbox(ax, x, y, w, h, fc, ec, lw=0.95, rad=0.055, z=2):
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h,
        boxstyle=f"round,pad=0.004,rounding_size={rad}",
        facecolor=fc, edgecolor=ec, linewidth=lw, zorder=z, clip_on=False,
    ))


def arr(ax, x0, y0, x1, y1, color=LINE, lw=1.05, ms=7.5):
    ax.add_patch(FancyArrowPatch(
        (x0, y0), (x1, y1),
        arrowstyle="-|>", mutation_scale=ms, lw=lw, color=color,
        shrinkA=0, shrinkB=0, zorder=7, capstyle="round",
    ))


def varr(ax, x, y0, y1, color=LINE):
    arr(ax, x, y0, x, y1, color=color, lw=0.95, ms=7.0)


def gutter_arr(ax, x_left, x_right, y, color=LINE):
    """Short horizontal arrow centered in the column gutter."""
    gap = x_right - x_left
    shaft = min(0.15, 0.55 * gap)
    mid = 0.5 * (x_left + x_right)
    arr(ax, mid - 0.5 * shaft, y, mid + 0.5 * shaft, y, color=color, lw=1.15, ms=8.0)


def stick(ax, x, y, s=0.24, color=BLUE, lw=0.95):
    ax.plot([x, x], [y, y + 0.55 * s], color=color, lw=lw, solid_capstyle="round", zorder=5)
    ax.plot([x - 0.32 * s, x, x + 0.32 * s], [y + 0.22 * s, y + 0.55 * s, y + 0.22 * s],
            color=color, lw=lw, solid_capstyle="round", zorder=5)
    ax.plot([x - 0.24 * s, x, x + 0.24 * s], [y - 0.38 * s, y, y - 0.38 * s],
            color=color, lw=lw, solid_capstyle="round", zorder=5)
    ax.add_patch(Circle((x, y + 0.72 * s), 0.11 * s, facecolor=color, edgecolor="none", zorder=5))


def panel(ax, x, y, w, h, fill, tag, title):
    rbox(ax, x, y, w, h, fill, PANEL_E, lw=0.80, rad=0.08, z=0)
    ax.text(x + 0.11, y + h - 0.17, tag, ha="left", va="center",
            fontsize=8.3, fontweight="bold", color=INK, zorder=3)
    ax.text(x + 0.34, y + h - 0.17, title, ha="left", va="center",
            fontsize=8.3, color=INK, zorder=3)
    ax.plot([x + 0.12, x + w - 0.12], [y + h - 0.32, y + h - 0.32],
            color=PANEL_E, lw=0.7, zorder=1, solid_capstyle="round")


def chips(ax, x, y, n, w, h, gap, fc, ec):
    for i in range(n):
        rbox(ax, x + i * (w + gap), y, w, h, fc, ec, lw=0.65, rad=0.03, z=4)


def module(ax, x, y, w, h, title, sub, ec, tc):
    rbox(ax, x, y, w, h, WHITE, ec, lw=1.05, rad=0.055)
    if sub:
        ax.text(x + w / 2, y + h * 0.64, title, ha="center", va="center",
                fontsize=8.0, fontweight="bold", color=tc, zorder=3)
        ax.text(x + w / 2, y + h * 0.30, sub, ha="center", va="center",
                fontsize=6.9, color=MUTE, zorder=3)
    else:
        ax.text(x + w / 2, y + h / 2, title, ha="center", va="center",
                fontsize=8.0, fontweight="bold", color=tc, zorder=3)


def rbrace(ax, x, y0, y1, label, color=TEAL):
    """Right-side curly brace (ViT / DiT stacked-block convention)."""
    ym = 0.5 * (y0 + y1)
    x1 = x + 0.045
    x2 = x + 0.09
    kw = dict(color=color, lw=0.85, solid_capstyle="round", solid_joinstyle="round", zorder=5)
    ax.plot([x, x1, x1], [y1, y1, ym + 0.07], **kw)
    ax.plot([x1, x2], [ym + 0.07, ym], **kw)
    ax.plot([x2, x1], [ym, ym - 0.07], **kw)
    ax.plot([x1, x1, x], [ym - 0.07, y0, y0], **kw)
    ax.text(x + 0.12, ym, label, ha="left", va="center",
            fontsize=6.5, color=color, zorder=5, linespacing=0.95)


def main():
    W, H = 7.16, 3.55
    fig, ax = plt.subplots(figsize=(W, H), dpi=300)
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.axis("off")
    fig.patch.set_facecolor(WHITE)

    pa, pwa = 0.05, 2.10
    gap = 0.30
    pb, pwb = pa + pwa + gap, 2.42
    pc, pwc = pb + pwb + gap, 1.99
    py, ph = 0.08, 3.40
    panel(ax, pa, py, pwa, ph, B_FILL, "(a)", "Graph tokenizer")
    panel(ax, pb, py, pwb, ph, G_FILL, "(b)", "Text-conditioned flow")
    panel(ax, pc, py, pwc, ph, P_FILL, "(c)", "Sample and decode")
    content_top = py + ph - 0.40  # below the header rule

    # ---------- (a) ----------
    fw, fh = 0.70, 0.56
    f0 = pa + 0.20
    f1 = pa + pwa - 0.20 - fw
    fy = content_top - fh - 0.04
    rbox(ax, f0, fy, fw, fh, WHITE, "#C5CDD6", lw=0.7, rad=0.045)
    rbox(ax, f1, fy, fw, fh, WHITE, "#C5CDD6", lw=0.7, rad=0.045)
    stick(ax, f0 + fw / 2, fy + 0.20, s=0.24)
    stick(ax, f1 + fw / 2, fy + 0.20, s=0.24)
    ax.text((f0 + fw + f1) / 2, fy + fh / 2, r"$\cdots$", ha="center", va="center",
            fontsize=9.0, color=MUTE)
    ax.text(f0 + fw / 2, fy - 0.09, r"$x_1$", ha="center", va="center", fontsize=7.2, color=MUTE)
    ax.text(f1 + fw / 2, fy - 0.09, r"$x_T$", ha="center", va="center", fontsize=7.2, color=MUTE)

    mx = pa + pwa / 2
    enc_y, enc_h = 1.70, 0.42
    varr(ax, mx, fy, enc_y + enc_h)
    module(ax, pa + 0.14, enc_y, pwa - 0.28, enc_h,
           "Graph encoder", "CTR topology  +  joint velocity", BLUE, BLUE)
    vae_y, vae_h = 1.14, 0.42
    varr(ax, mx, enc_y, vae_y + vae_h)
    module(ax, pa + 0.14, vae_y, pwa - 0.28, vae_h,
           "Topology VAE", r"stride 4,   $z\in\mathbb{R}^{L\times 256}$", BLUE, BLUE)

    cw, ch, cg = 0.28, 0.26, 0.07
    nchip = 5
    chips_w = nchip * cw + (nchip - 1) * cg
    chips_x = pa + (pwa - chips_w) / 2
    chips_y = 0.50
    varr(ax, mx, vae_y, chips_y + ch + 0.16)
    ax.text(pa + 0.14, chips_y + ch + 0.10, "latent tokens  (frozen, 5 Hz)",
            ha="left", va="center", fontsize=6.7, color=MUTE)
    chips(ax, chips_x, chips_y, nchip, cw, ch, cg, CHIP_B, BLUE)
    y_ab = chips_y + ch / 2

    # ---------- (b) ----------
    clip_h = 0.44
    clip_y = content_top - clip_h
    module(ax, pb + 0.14, clip_y, pwb - 0.28, clip_h,
           "Frozen CLIP", r"ViT-B/32   $c_{1:N}\in\mathbb{R}^{N\times 512}$", ORANGE, ORANGE)
    ali_y, ali_h = 2.08, 0.44
    varr(ax, pb + pwb / 2, clip_y, ali_y + ali_h)
    module(ax, pb + 0.14, ali_y, pwb - 0.28, ali_h,
           "Language–motion alignment", r"InfoNCE on pooled $c$ and $z$", TEAL, TEAL)

    dit_x, dit_w = pb + 0.14, pwb - 0.52
    dit_y, dit_h = 0.20, 1.74
    varr(ax, pb + pwb / 2, ali_y, dit_y + dit_h)
    rbox(ax, dit_x, dit_y, dit_w, dit_h, WHITE, TEAL, lw=1.15, rad=0.06)
    ax.text(dit_x + dit_w / 2, dit_y + dit_h - 0.15, "Rectified-flow DiT",
            ha="center", va="center", fontsize=8.1, fontweight="bold", color=TEAL)
    ax.text(dit_x + dit_w / 2, dit_y + dit_h - 0.36,
            r"$z_t=(1-t)\varepsilon+t\,z$  ·  $v=z-\varepsilon$",
            ha="center", va="center", fontsize=6.7, color=MUTE)

    layers = [
        ("AdaLN", "time + pooled text"),
        ("Self-attention", r"on $z_t$"),
        ("Cross-attention", "to CLIP tokens"),
        ("FFN", r"CFG drop  $0.1$"),
    ]
    lh, lg = 0.25, 0.045
    layer_top = dit_y + dit_h - 0.50
    for i, (name, extra) in enumerate(layers):
        yy = layer_top - (i + 1) * lh - i * lg
        rbox(ax, dit_x + 0.10, yy, dit_w - 0.20, lh,
             CHIP_T if i % 2 == 0 else WHITE, TEAL, lw=0.6, rad=0.03)
        ax.text(dit_x + dit_w / 2, yy + lh * 0.64, name,
                ha="center", va="center", fontsize=6.9, fontweight="bold", color=INK)
        ax.text(dit_x + dit_w / 2, yy + lh * 0.28, extra,
                ha="center", va="center", fontsize=6.1, color=MUTE)
    layer_bot = layer_top - 4 * lh - 3 * lg
    rbrace(ax, dit_x + dit_w + 0.02, layer_bot, layer_top - lg,
           "8\nblocks", color=TEAL)

    # ---------- (c) ----------
    ax.text(pc + 0.14, content_top - 0.02, r"noise  $\varepsilon$",
            ha="left", va="center", fontsize=6.8, color=MUTE)
    ocw, och, ocg = 0.30, 0.24, 0.08
    on = 4
    ow = on * ocw + (on - 1) * ocg
    ox = pc + (pwc - ow) / 2
    ochip_y = clip_y + 0.02
    chips(ax, ox, ochip_y, on, ocw, och, ocg, CHIP_O, ORANGE)

    eul_h = 0.70
    eul_y = 1.78
    varr(ax, pc + pwc / 2, ochip_y, eul_y + eul_h)
    module(ax, pc + 0.14, eul_y, pwc - 0.28, eul_h,
           "Euler ODE", r"$N{=}20$ or $50$   ·   $v=v_u+\omega(v_c-v_u)$",
           PURPLE, PURPLE)
    y_bc = eul_y + eul_h / 2

    zlab_y = 1.22
    ax.text(pc + 0.14, zlab_y + 0.12, r"clean $z$", ha="left", va="center",
            fontsize=6.7, color=MUTE)
    chips(ax, ox, zlab_y - 0.18, on, ocw, och, ocg, CHIP_B, BLUE)
    varr(ax, pc + pwc / 2, eul_y, zlab_y + 0.12)

    dec_y, dec_h = 0.22, 0.52
    varr(ax, pc + pwc / 2, zlab_y - 0.18, dec_y + dec_h)
    module(ax, pc + 0.14, dec_y, pwc - 0.28, dec_h,
           "VAE decoder", r"$\hat x_{1:T}\in\mathbb{R}^{T\times 263}$", BLUE, BLUE)

    gutter_arr(ax, pa + pwa, pb, y_ab, color=BLUE)
    gutter_arr(ax, pb + pwb, pc, y_bc, color=TEAL)

    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / "fig_architecture.pdf", bbox_inches="tight", pad_inches=0.02)
    fig.savefig(OUT / "fig_architecture.png", dpi=360, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print("wrote T2M fig_architecture")


if __name__ == "__main__":
    main()
