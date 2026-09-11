#!/usr/bin/env python3
"""Quantitative figures for the GALA T2M paper."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path("/home/qinyang/桌面/project")
OUT = Path("/home/qinyang/桌面/project/paper/gala_t2m_overleaf/gala_t2m_overleaf/figures")
OUT.mkdir(parents=True, exist_ok=True)

INK = "#1F2937"
MUTE = "#6B7280"
TEAL = "#2A9D8F"
BLUE = "#4C72B0"
ORANGE = "#E07A3D"
RED = "#C44E52"
GRAY = "#9CA3AF"


def style():
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "axes.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "legend.frameon": False,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.06,
        "pdf.fonttype": 42,
    })


def save(fig, name):
    fig.savefig(OUT / f"{name}.pdf")
    fig.savefig(OUT / f"{name}.png", dpi=220)
    plt.close(fig)
    print("wrote", name)


def fig_train():
    steps = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100]
    fid = [1.978, 2.737, 1.088, 1.512, 1.104, 0.717, 0.412, 0.836, 0.555, 0.459,
           0.696, 0.705, 0.584, 0.900, 0.918, 0.838, 0.792, 0.663, 1.054, 0.858]
    r3 = [0.672, 0.622, 0.666, 0.688, 0.706, 0.725, 0.753, 0.722, 0.700, 0.728,
          0.741, 0.744, 0.759, 0.734, 0.766, 0.769, 0.759, 0.784, 0.741, 0.763]
    x = np.array(steps) * 1000

    fig, ax1 = plt.subplots(figsize=(4.55, 2.55))
    ax2 = ax1.twinx()
    ax2.spines["top"].set_visible(False)
    l1, = ax1.plot(x, fid, color=TEAL, lw=1.6, marker="o", ms=3.2, label="Val FID")
    l2, = ax2.plot(x, r3, color=ORANGE, lw=1.6, marker="s", ms=3.2, label="Val R@3")
    ax1.axvline(35000, color=MUTE, ls="--", lw=0.9)
    ax1.annotate("best.pt", xy=(35000, 0.412), xytext=(48000, 1.55),
                 color=MUTE, fontsize=8,
                 arrowprops=dict(arrowstyle="->", color=MUTE, lw=0.8))
    ax1.set_xlabel("Optimizer step")
    ax1.set_ylabel("FID (320-clip val, 10 steps)")
    ax2.set_ylabel("R@3")
    ax1.set_ylim(0, 3.0)
    ax2.set_ylim(0.58, 0.82)
    ax1.legend([l1, l2], ["Val FID", "Val R@3"], loc="upper right")
    save(fig, "fig_train_curve")


def fig_sweep():
    sweep = json.loads((ROOT / "checkpoints/gala_humanml3d_flow/cfg_steps_sweep.json").read_text())
    steps = [10, 20, 50]
    cfgs = [1.0, 1.5, 2.0, 2.5]
    fid = np.zeros((len(steps), len(cfgs)))
    r3 = np.zeros_like(fid)
    for item in sweep["grid"]:
        i = steps.index(item["steps"])
        j = cfgs.index(item["guidance"])
        fid[i, j] = item["FID"]
        r3[i, j] = item["R@3"]

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.55))
    for ax, mat, title, cmap, fmt in (
        (axes[0], fid, "Val FID (lower better)", "YlGn_r", "{:.2f}"),
        (axes[1], r3, "Val R@3 (higher better)", "YlOrBr", "{:.2f}"),
    ):
        im = ax.imshow(mat, cmap=cmap, origin="upper")
        ax.set_xticks(range(len(cfgs)), [str(c) for c in cfgs])
        ax.set_yticks(range(len(steps)), [str(s) for s in steps])
        ax.set_xlabel("CFG")
        ax.set_ylabel("ODE steps")
        ax.set_title(title, fontsize=10)
        ax.spines["top"].set_visible(True)
        ax.spines["right"].set_visible(True)
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                ax.text(j, i, fmt.format(mat[i, j]), ha="center", va="center", fontsize=8, color=INK)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    axes[0].add_patch(plt.Rectangle((-0.5 + 2, 1.5), 1, 1, fill=False, lw=1.6, edgecolor=TEAL))
    save(fig, "fig_sweep")


def fig_compare():
    names = ["MDM", "MLD", "M2DM", "GALA\n20", "GALA\n50", "T2M-GPT", "EMDM", "MoMask"]
    r3 = [0.611, 0.772, 0.763, 0.768, 0.749, 0.775, 0.786, 0.807]
    fid = [0.544, 0.473, 0.352, 0.330, 0.312, 0.141, 0.112, 0.045]
    colors = [GRAY, GRAY, GRAY, TEAL, TEAL, GRAY, GRAY, GRAY]

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.7))
    x = np.arange(len(names))
    for ax, vals, title, ylabel, invert in (
        (axes[0], r3, "Text alignment", "R@3 (higher better)", False),
        (axes[1], fid, "Motion fidelity", "FID (lower better)", True),
    ):
        bars = ax.bar(x, vals, color=colors, edgecolor=INK, lw=0.5)
        ax.set_xticks(x, names, fontsize=7.0)
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        if invert:
            ax.set_ylim(0, 0.65)
        else:
            ax.set_ylim(0, 0.95)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + (0.015 if not invert else 0.012),
                    f"{v:.3f}", ha="center", va="bottom", fontsize=6.4)
    save(fig, "fig_compare")


def fig_recon_gap():
    labels = ["VAE recon\n(val)", "GALA 50\n(test)", "EMDM\n(test)", "Real"]
    fid = [0.003, 0.312, 0.112, 0.002]
    colors = [BLUE, TEAL, GRAY, GRAY]
    fig, ax = plt.subplots(figsize=(4.4, 2.55))
    bars = ax.bar(labels, fid, color=colors, edgecolor=INK, lw=0.5)
    ax.set_ylabel("FID")
    ax.set_title("Tokenizer is not the FID bottleneck")
    ax.set_ylim(0, 0.40)
    for bar, v in zip(bars, fid):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.008,
                f"{v:.3f}", ha="center", va="bottom", fontsize=8)
    save(fig, "fig_recon_gap")


if __name__ == "__main__":
    style()
    fig_train()
    fig_sweep()
    fig_compare()
    fig_recon_gap()
