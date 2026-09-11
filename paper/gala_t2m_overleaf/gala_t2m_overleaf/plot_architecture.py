#!/usr/bin/env python3
"""GALA T2M architecture, print width.

(a) five anatomical part latents
(b) CLIP tokens → anatomical queries → c_p ↔ z_p
(c) Euler sampling + a training-only kinematic branch
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch

ROOT = Path("/home/qinyang/桌面/project")
OUT = ROOT / "paper" / "gala_t2m_overleaf" / "gala_t2m_overleaf" / "figures"

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
DASH = "#8A6A3A"

# Torso, L-arm, R-arm, L-leg, R-leg
PARTS = [
    ("Torso", "#2F5D8A", "#D7E3F0"),
    ("L-arm", "#2C6A4A", "#D3E7DB"),
    ("R-arm", "#A15C28", "#F1DDCC"),
    ("L-leg", "#5A4580", "#E6E0EF"),
    ("R-leg", "#8A3B4A", "#F3D6DC"),
]


def rbox(ax, x, y, w, h, fc, ec, lw=0.95, rad=0.055, z=2, ls="-"):
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h,
        boxstyle=f"round,pad=0.004,rounding_size={rad}",
        facecolor=fc, edgecolor=ec, linewidth=lw, zorder=z, clip_on=False,
        linestyle=ls,
    ))


def arr(ax, x0, y0, x1, y1, color=LINE, lw=1.05, ms=7.5, ls="-"):
    ax.add_patch(FancyArrowPatch(
        (x0, y0), (x1, y1),
        arrowstyle="-|>", mutation_scale=ms, lw=lw, color=color,
        shrinkA=0, shrinkB=0, zorder=7, capstyle="round",
        linestyle=ls,
    ))


def varr(ax, x, y0, y1, color=LINE, lw=0.95):
    arr(ax, x, y0, x, y1, color=color, lw=lw, ms=7.0)


def darr(ax, x, y0, y1, color=TEAL):
    """Short vertical double arrow (alignment)."""
    ax.annotate(
        "", xy=(x, y1), xytext=(x, y0),
        arrowprops=dict(arrowstyle="<->", color=color, lw=0.95,
                         mutation_scale=6.5, shrinkA=0, shrinkB=0),
        zorder=7,
    )


def gutter_arr(ax, x_left, x_right, y, color=LINE):
    gap = x_right - x_left
    shaft = min(0.15, 0.55 * gap)
    mid = 0.5 * (x_left + x_right)
    arr(ax, mid - 0.5 * shaft, y, mid + 0.5 * shaft, y, color=color, lw=1.15, ms=8.0)


def anatomy(ax, x, y, s=0.42):
    """Frontal stick figure with five body-part colors."""
    torso, la, ra, ll, rl = [c for _, c, _ in PARTS]
    head = (x, y + 0.78 * s)
    neck = (x, y + 0.58 * s)
    hip = (x, y + 0.08 * s)
    sh_l = (x - 0.28 * s, y + 0.52 * s)
    sh_r = (x + 0.28 * s, y + 0.52 * s)
    el_l = (x - 0.48 * s, y + 0.28 * s)
    el_r = (x + 0.48 * s, y + 0.28 * s)
    wr_l = (x - 0.58 * s, y + 0.04 * s)
    wr_r = (x + 0.58 * s, y + 0.04 * s)
    kn_l = (x - 0.18 * s, y - 0.28 * s)
    kn_r = (x + 0.18 * s, y - 0.28 * s)
    ft_l = (x - 0.22 * s, y - 0.55 * s)
    ft_r = (x + 0.22 * s, y - 0.55 * s)
    kw = dict(solid_capstyle="round", zorder=5)
    ax.plot([neck[0], hip[0]], [neck[1], hip[1]], color=torso, lw=1.55, **kw)
    ax.plot([sh_l[0], sh_r[0]], [sh_l[1], sh_r[1]], color=torso, lw=1.35, **kw)
    ax.plot([sh_l[0], el_l[0], wr_l[0]], [sh_l[1], el_l[1], wr_l[1]], color=la, lw=1.45, **kw)
    ax.plot([sh_r[0], el_r[0], wr_r[0]], [sh_r[1], el_r[1], wr_r[1]], color=ra, lw=1.45, **kw)
    ax.plot([hip[0], kn_l[0], ft_l[0]], [hip[1], kn_l[1], ft_l[1]], color=ll, lw=1.45, **kw)
    ax.plot([hip[0], kn_r[0], ft_r[0]], [hip[1], kn_r[1], ft_r[1]], color=rl, lw=1.45, **kw)
    ax.add_patch(Circle(head, 0.12 * s, facecolor=torso, edgecolor="none", zorder=5))


def panel(ax, x, y, w, h, fill, tag, title):
    rbox(ax, x, y, w, h, fill, PANEL_E, lw=0.80, rad=0.08, z=0)
    ax.text(x + 0.11, y + h - 0.16, tag, ha="left", va="center",
            fontsize=8.3, fontweight="bold", color=INK, zorder=3)
    ax.text(x + 0.34, y + h - 0.16, title, ha="left", va="center",
            fontsize=8.2, color=INK, zorder=3)
    ax.plot([x + 0.12, x + w - 0.12], [y + h - 0.30, y + h - 0.30],
            color=PANEL_E, lw=0.7, zorder=1, solid_capstyle="round")


def module(ax, x, y, w, h, title, sub, ec, tc):
    rbox(ax, x, y, w, h, WHITE, ec, lw=1.05, rad=0.055)
    if sub:
        ax.text(x + w / 2, y + h * 0.64, title, ha="center", va="center",
                fontsize=7.7, fontweight="bold", color=tc, zorder=3)
        ax.text(x + w / 2, y + h * 0.30, sub, ha="center", va="center",
                fontsize=6.5, color=MUTE, zorder=3)
    else:
        ax.text(x + w / 2, y + h / 2, title, ha="center", va="center",
                fontsize=7.7, fontweight="bold", color=tc, zorder=3)


def rbrace(ax, x, y0, y1, label, color=TEAL):
    ym = 0.5 * (y0 + y1)
    x1 = x + 0.045
    x2 = x + 0.09
    kw = dict(color=color, lw=0.85, solid_capstyle="round", solid_joinstyle="round", zorder=5)
    ax.plot([x, x1, x1], [y1, y1, ym + 0.07], **kw)
    ax.plot([x1, x2], [ym + 0.07, ym], **kw)
    ax.plot([x2, x1], [ym, ym - 0.07], **kw)
    ax.plot([x1, x1, x], [ym - 0.07, y0, y0], **kw)
    ax.text(x + 0.12, ym, label, ha="left", va="center",
            fontsize=6.4, color=color, zorder=5, linespacing=0.95)


def part_row(ax, x, y, w, h, gap, labels=True, prefix=""):
    n = 5
    chips_w = n * w + (n - 1) * gap
    xs = []
    for i, (name, ec, fc) in enumerate(PARTS):
        xx = x + i * (w + gap)
        rbox(ax, xx, y, w, h, fc, ec, lw=0.7, rad=0.03, z=4)
        text = f"${prefix}_{{{['T','LA','RA','LL','RL'][i]}}}$" if prefix else name
        ax.text(xx + w / 2, y + h / 2, text if prefix else name,
                ha="center", va="center", fontsize=5.9 if not prefix else 6.4,
                color=ec, fontweight="bold", zorder=5)
        xs.append(xx + w / 2)
    return chips_w, xs


def main():
    W, H = 7.16, 3.92
    fig, ax = plt.subplots(figsize=(W, H), dpi=300)
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.axis("off")
    fig.patch.set_facecolor(WHITE)

    pa, pwa = 0.05, 2.08
    gap = 0.26
    pb, pwb = pa + pwa + gap, 2.52
    pc, pwc = pb + pwb + gap, 2.00
    py, ph = 0.06, 3.80
    panel(ax, pa, py, pwa, ph, B_FILL, "(a)", "Part-aware tokenizer")
    panel(ax, pb, py, pwb, ph, G_FILL, "(b)", "Language–part alignment")
    panel(ax, pc, py, pwc, ph, P_FILL, "(c)", "Kinematic rectified flow")
    content_top = py + ph - 0.38

    # ---------- (a) ----------
    mx = pa + pwa / 2
    sk_y = content_top - 0.72
    anatomy(ax, mx, sk_y + 0.22, s=0.50)
    ax.text(mx, sk_y - 0.10, r"$x_{1:T}$  ·  22 joints",
            ha="center", va="center", fontsize=6.5, color=MUTE)

    enc_y, enc_h = 2.18, 0.40
    varr(ax, mx, sk_y - 0.02, enc_y + enc_h)
    module(ax, pa + 0.12, enc_y, pwa - 0.24, enc_h,
           "Part-aware graph", "CTR  +  pooled parts", BLUE, BLUE)
    vae_y, vae_h = 1.64, 0.40
    varr(ax, mx, enc_y, vae_y + vae_h)
    module(ax, pa + 0.12, vae_y, pwa - 0.24, vae_h,
           "Topology VAE", r"stride 4,   $z,\,z_p$", BLUE, BLUE)

    pw, phh, pg = 0.34, 0.28, 0.04
    row_w = 5 * pw + 4 * pg
    row_x = pa + (pwa - row_w) / 2
    chips_y = 0.86
    varr(ax, mx, vae_y, chips_y + phh + 0.12)
    ax.text(pa + 0.12, chips_y + phh + 0.10, "part latents  (frozen)",
            ha="left", va="center", fontsize=6.4, color=MUTE)
    part_row(ax, row_x, chips_y, pw, phh, pg, prefix="")

    zcw, zch, zcg = 0.28, 0.22, 0.06
    zn = 5
    zw = zn * zcw + (zn - 1) * zcg
    zx = pa + (pwa - zw) / 2
    zy = 0.22
    for i in range(zn):
        rbox(ax, zx + i * (zcw + zcg), zy, zcw, zch, CHIP_B, BLUE, lw=0.6, rad=0.03)
    ax.text(pa + 0.12, zy + zch + 0.10, r"latent tokens $z$  (5 Hz)",
            ha="left", va="center", fontsize=6.4, color=MUTE)
    varr(ax, mx, chips_y, zy + zch + 0.18)
    y_ab = chips_y + phh / 2

    # ---------- (b) ----------
    clip_h = 0.40
    clip_y = content_top - clip_h
    module(ax, pb + 0.12, clip_y, pwb - 0.24, clip_h,
           "Frozen CLIP", r"$c_{1:N}$  ViT-B/32", ORANGE, ORANGE)

    ccw, cch, ccg = 0.28, 0.18, 0.05
    cn = 6
    cw = cn * ccw + (cn - 1) * ccg
    cx = pb + (pwb - cw) / 2
    cy = clip_y - 0.28
    varr(ax, pb + pwb / 2, clip_y, cy + cch)
    for i in range(cn):
        rbox(ax, cx + i * (ccw + ccg), cy, ccw, cch, CHIP_O, ORANGE, lw=0.55, rad=0.03)
    ax.text(cx + cw + 0.04, cy + cch / 2, r"$\cdots$", ha="left", va="center",
            fontsize=7.0, color=MUTE)

    q_y, q_h = 2.18, 0.36
    varr(ax, pb + pwb / 2, cy, q_y + q_h)
    module(ax, pb + 0.12, q_y, pwb - 0.24, q_h,
           "Anatomical queries", r"$\tilde c_p=\mathrm{Attn}(q_p,\,c_{1:N})$", TEAL, TEAL)

    pw2, ph2, pg2 = 0.40, 0.26, 0.06
    row_w2 = 5 * pw2 + 4 * pg2
    row_x2 = pb + (pwb - row_w2) / 2
    c_y = 1.72
    _, cxs = part_row(ax, row_x2, c_y, pw2, ph2, pg2, prefix="c")
    z_y = 1.26
    _, zxs = part_row(ax, row_x2, z_y, pw2, ph2, pg2, prefix="z")
    for xc, xz in zip(cxs, zxs):
        darr(ax, xc, z_y + ph2 + 0.015, c_y - 0.015)
    ax.text(pb + pwb - 0.14, 0.5 * (c_y + z_y) + 0.05, "align",
            ha="right", va="center", fontsize=6.2, color=TEAL, fontstyle="italic")

    dit_x, dit_w = pb + 0.12, pwb - 0.48
    dit_y, dit_h = 0.16, 0.96
    varr(ax, pb + pwb / 2, z_y, dit_y + dit_h)
    rbox(ax, dit_x, dit_y, dit_w, dit_h, WHITE, TEAL, lw=1.15, rad=0.06)
    ax.text(dit_x + dit_w / 2, dit_y + dit_h - 0.13, "Rectified-flow DiT",
            ha="center", va="center", fontsize=7.6, fontweight="bold", color=TEAL)
    layers = [
        ("AdaLN", "time + pooled text"),
        ("Self-attn / Cross-attn", r"$z_t$  /  CLIP$+\,\tilde c_p$"),
        ("FFN", r"CFG drop  $0.1$"),
    ]
    lh, lg = 0.20, 0.035
    layer_top = dit_y + dit_h - 0.24
    for i, (name, extra) in enumerate(layers):
        yy = layer_top - (i + 1) * lh - i * lg
        rbox(ax, dit_x + 0.08, yy, dit_w - 0.16, lh,
             CHIP_T if i % 2 == 0 else WHITE, TEAL, lw=0.6, rad=0.03)
        ax.text(dit_x + dit_w / 2, yy + lh * 0.64, name,
                ha="center", va="center", fontsize=6.5, fontweight="bold", color=INK)
        ax.text(dit_x + dit_w / 2, yy + lh * 0.28, extra,
                ha="center", va="center", fontsize=5.8, color=MUTE)
    layer_bot = layer_top - 3 * lh - 2 * lg
    rbrace(ax, dit_x + dit_w + 0.02, layer_bot, layer_top - lg,
           "8\nblocks", color=TEAL)
    y_bc = dit_y + dit_h * 0.55

    # ---------- (c) ----------
    ax.text(pc + 0.12, content_top - 0.02, r"noise  $\varepsilon$",
            ha="left", va="center", fontsize=6.5, color=MUTE)
    ocw, och, ocg = 0.28, 0.20, 0.07
    on = 4
    ow = on * ocw + (on - 1) * ocg
    ox = pc + (pwc - ow) / 2
    ochip_y = content_top - 0.48
    for i in range(on):
        rbox(ax, ox + i * (ocw + ocg), ochip_y, ocw, och, CHIP_O, ORANGE, lw=0.55, rad=0.03)

    eul_h = 0.52
    eul_y = 2.28
    varr(ax, pc + pwc / 2, ochip_y, eul_y + eul_h)
    module(ax, pc + 0.12, eul_y, pwc - 0.24, eul_h,
           "Euler ODE", r"$N{=}20$ or $50$   ·   CFG  $\omega$",
           PURPLE, PURPLE)

    zlab_y = 1.88
    ax.text(pc + 0.12, zlab_y + 0.08, r"clean $z$", ha="left", va="center",
            fontsize=6.4, color=MUTE)
    for i in range(on):
        rbox(ax, ox + i * (ocw + ocg), zlab_y - 0.16, ocw, och, CHIP_B, BLUE, lw=0.55, rad=0.03)
    varr(ax, pc + pwc / 2, eul_y, zlab_y + 0.10)

    dec_y, dec_h = 1.14, 0.42
    varr(ax, pc + pwc / 2, zlab_y - 0.16, dec_y + dec_h)
    module(ax, pc + 0.12, dec_y, pwc - 0.24, dec_h,
           "VAE decoder", r"$\hat x_{1:T}$", BLUE, BLUE)

    kin_y, kin_h = 0.14, 0.88
    rbox(ax, pc + 0.10, kin_y, pwc - 0.20, kin_h, "#FBF6EF", DASH, lw=0.95, rad=0.05, ls="--")
    ax.text(pc + pwc / 2, kin_y + kin_h - 0.12, "train only",
            ha="center", va="center", fontsize=6.3, fontweight="bold", color=DASH)
    ax.text(pc + pwc / 2, kin_y + kin_h - 0.30, r"$\hat z_1=z_t+(1-t)v_\theta$",
            ha="center", va="center", fontsize=6.2, color=INK)
    varr(ax, pc + pwc / 2, kin_y + kin_h - 0.38, kin_y + 0.48, color=DASH)
    ax.text(pc + pwc / 2, kin_y + 0.40, "Frozen decoder",
            ha="center", va="center", fontsize=6.3, fontweight="bold", color=DASH)
    kw, kh, kg = 0.48, 0.20, 0.06
    krow = 3 * kw + 2 * kg
    kx = pc + (pwc - krow) / 2
    ky = kin_y + 0.10
    for i, (lab, ec, fc) in enumerate([
        ("Vel.", ORANGE, CHIP_O),
        ("Bone", BLUE, CHIP_B),
        ("Foot", PURPLE, "#E6E0EF"),
    ]):
        rbox(ax, kx + i * (kw + kg), ky, kw, kh, fc, ec, lw=0.6, rad=0.03)
        ax.text(kx + i * (kw + kg) + kw / 2, ky + kh / 2, lab,
                ha="center", va="center", fontsize=6.2, color=ec, fontweight="bold")

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
