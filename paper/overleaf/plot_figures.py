#!/usr/bin/env python3
"""Compact full-width quantitative + qualitative panels (conference figure*)."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec
from matplotlib.image import imread

ROOT = Path("/home/qinyang/桌面/project")
OUT = ROOT / "paper" / "overleaf" / "figures"
CKPT = ROOT / "checkpoints"

INK = "#1F2933"
MUTE = "#5B6770"
COPY = "#6B7280"
UNFIXED = "#C44E52"
V1 = "#4C72B0"
V2 = "#2C6A4A"
ZERO = "#A15C28"
SHUFFLE = "#5A4580"
HOLD = "#9A7B4F"


def style():
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 7.6,
        "axes.titlesize": 8.0,
        "axes.labelsize": 7.4,
        "xtick.labelsize": 6.8,
        "ytick.labelsize": 6.8,
        "legend.fontsize": 6.2,
        "axes.linewidth": 0.65,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "legend.frameon": False,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "pdf.fonttype": 42,
        "mathtext.default": "regular",
    })


def load_json(name):
    return json.loads((CKPT / name / "eval.json").read_text(encoding="utf-8"))


def mpjpe(report, method, horizon):
    return report["rollout"][method][str(horizon)]["mpjpe_mm"]["mean"]


def save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight", pad_inches=0.02)
    fig.savefig(OUT / f"{name}.png", dpi=280, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print("wrote", name)


def panel_tag(ax, tag):
    ax.set_title(tag, loc="left", fontsize=8.0, fontweight="bold", color=INK, pad=2)


def fig_quant():
    v2 = load_json("gala_humanml3d_wm_v2")
    v1 = load_json("gala_humanml3d_wm")
    unf = load_json("gala_humanml3d_wm_unfixed")
    xs = np.array([1, 4, 8])
    sec = xs * 0.2

    # One compact 2x2 spanning print width — not four scattered column floats.
    fig = plt.figure(figsize=(7.16, 2.62))
    gs = GridSpec(2, 4, figure=fig, wspace=0.34, hspace=0.48,
                  left=0.048, right=0.995, top=0.90, bottom=0.13)

    ax = fig.add_subplot(gs[0, :2])
    series = [
        ("Copy-last", [mpjpe(v2, "copy_last", h) for h in xs], COPY, "--", 1.3),
        ("Zero $a$", [mpjpe(v2, "zero", h) for h in xs], ZERO, ":", 1.3),
        ("Shuffle $a$", [mpjpe(v2, "shuffle", h) for h in xs], SHUFFLE, ":", 1.3),
        ("Unfixed", [mpjpe(unf, "gt", h) for h in xs], UNFIXED, "-.", 1.3),
        ("v1 residual", [mpjpe(v1, "gt", h) for h in xs], V1, "-", 1.3),
        ("v2 (ours)", [mpjpe(v2, "gt", h) for h in xs], V2, "-", 1.9),
    ]
    for name, ys, color, ls, lw in series:
        ax.plot(sec, ys, color=color, ls=ls, marker="o", lw=lw, ms=3.2, label=name)
    ax.axhline(80, color="#D1D5DB", ls="--", lw=0.65)
    ax.set_xticks(sec)
    ax.set_xticklabels(["0.2 s", "0.8 s", "1.6 s"])
    ax.set_ylabel("MPJPE (mm)")
    ax.set_ylim(15, 98)
    ax.legend(ncol=2, loc="upper left", handlelength=1.45, columnspacing=0.7,
              labelspacing=0.12, borderaxespad=0.15)
    panel_tag(ax, "(a)  Horizon MPJPE")

    ax = fig.add_subplot(gs[0, 2:])
    labels = ["Copy-last", "Zero $a$", "Shuffle $a$", "GT $a$"]
    v1_y = [mpjpe(v1, m, 8) for m in ("copy_last", "zero", "shuffle", "gt")]
    v2_y = [mpjpe(v2, m, 8) for m in ("copy_last", "zero", "shuffle", "gt")]
    x = np.arange(len(labels))
    w = 0.36
    ax.bar(x - w / 2, v1_y, w, label="v1", color=V1, zorder=2)
    ax.bar(x + w / 2, v2_y, w, label="v2 (ours)", color=V2, zorder=2)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("MPJPE @ 1.6 s")
    ax.set_ylim(0, 108)
    ax.legend(loc="upper left", handlelength=0.95)
    panel_tag(ax, r"(b)  Action ablation  ($H{=}8$)")

    ax = fig.add_subplot(gs[1, :2])
    rows = [
        ("Hold", 50.0, HOLD),
        ("Unfixed", 54.3, UNFIXED),
        ("v1 weak", 65.2, V1),
        ("v1 strong", 71.7, "#5B8FA8"),
        ("v2 (ours)", 73.9, V2),
    ]
    xb = np.arange(len(rows))
    ax.bar(xb, [r[1] for r in rows], color=[r[2] for r in rows], zorder=2, width=0.68)
    ax.axhline(50, color="#D1D5DB", ls="--", lw=0.65)
    ax.set_xticks(xb)
    ax.set_xticklabels([r[0] for r in rows], fontsize=6.6)
    ax.set_ylabel("Success (%)")
    ax.set_ylim(0, 100)
    for i, r in enumerate(rows):
        ax.text(i, r[1] + 1.6, f"{r[1]:.0f}", ha="center", fontsize=6.4, color=INK)
    panel_tag(ax, "(c)  CEM vs hold")

    ax = fig.add_subplot(gs[1, 2:])
    ax.plot([2000, 4000, 6000, 8000, 10000, 12000, 14000],
            [0.5195, 0.5112, 0.6053, 0.5133, 0.6963, 0.6200, 1.5885],
            color=UNFIXED, ls="-.", marker="o", ms=2.8, lw=1.2, label="Unfixed")
    ax.plot([2000, 4000, 6000, 8000, 10000, 12000],
            [0.5226, 0.4612, 0.4494, 0.4257, 0.4108, 0.4179],
            color=V1, marker="o", ms=2.8, lw=1.2, label="v1")
    ax.plot([2000, 4000, 6000, 8000, 10000],
            [0.4056, 0.3744, 0.3744, 0.3600, 0.3358],
            color=V2, marker="o", ms=3.0, lw=1.8, label="v2 (ours)")
    ax.set_xlabel("Training step")
    ax.set_ylabel(r"Val latent $\ell_2$ @ $H{=}8$")
    ax.set_ylim(0.25, 1.05)
    ax.legend(loc="upper left", handlelength=1.3)
    panel_tag(ax, r"(d)  Train  ($H{=}8$)")

    save(fig, "fig_quant")


def _imshow(ax, path, tag):
    img = imread(path)
    ax.imshow(img)
    ax.set_axis_off()
    ax.set_title(tag, loc="left", fontsize=8.0, fontweight="bold", color=INK, pad=1.5)


def fig_qual():
    # Single compact figure*: rollout | planning / root — no dataset teaser.
    fig = plt.figure(figsize=(7.16, 2.72))
    gs = GridSpec(2, 2, figure=fig, height_ratios=[1.05, 1.0], width_ratios=[1.02, 1.10],
                  wspace=0.06, hspace=0.08, left=0.008, right=0.995, top=0.935, bottom=0.02)
    ax0 = fig.add_subplot(gs[:, 0])
    ax1 = fig.add_subplot(gs[0, 1])
    ax2 = fig.add_subplot(gs[1, 1])
    _imshow(ax0, OUT / "fig_rollout_grid.png", "(a) Open-loop rollout")
    _imshow(ax1, OUT / "fig_planning_vis.png", "(b) CEM vs hold")
    _imshow(ax2, OUT / "fig_planning_xy.png", r"(c) Root $Z(t)\,/\,X(t)$")
    save(fig, "fig_qual")


if __name__ == "__main__":
    style()
    fig_quant()
    fig_qual()
