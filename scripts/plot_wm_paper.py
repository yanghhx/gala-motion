#!/usr/bin/env python3
"""Paper figures for GALA-WM. Quantitative panels from eval JSON; architecture is drawn."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

ROOT = Path("/home/qinyang/桌面/project")
OUT = ROOT / "paper" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

C = {
    "copy": "#6B7280",
    "unfixed": "#C44E52",
    "v1": "#4C72B0",
    "v2": "#2A9D8F",
    "zero": "#E07A3D",
    "shuffle": "#7B68A6",
    "hold": "#9A7B4F",
    "ink": "#1F2937",
    "mute": "#6B7280",
    "box": "#F4F1EA",
    "box2": "#E8F4F1",
    "box3": "#EEF2FF",
}


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
        "savefig.pad_inches": 0.08,
        "pdf.fonttype": 42,
    })


def load_json(rel):
    return json.loads((ROOT / rel).read_text(encoding="utf-8"))


def mpjpe(report, method, horizon):
    return report["rollout"][method][str(horizon)]["mpjpe_mm"]["mean"]


def save(fig, name):
    png = OUT / f"{name}.png"
    pdf = OUT / f"{name}.pdf"
    fig.savefig(png, dpi=220)
    fig.savefig(pdf)
    plt.close(fig)
    print("wrote", png)


def fig_architecture():
    """Layout mirrors DINO-WM Fig.1: frozen encoder → action-conditioned predictor → CEM."""
    fig, ax = plt.subplots(figsize=(12.4, 4.6))
    ax.set_xlim(0, 12.4)
    ax.set_ylim(0, 4.6)
    ax.axis("off")

    def box(x, y, w, h, text, fc, title=None, fs=8.5):
        ax.add_patch(FancyBboxPatch(
            (x, y), w, h, boxstyle="round,pad=0.04,rounding_size=0.08",
            facecolor=fc, edgecolor="#374151", linewidth=1.0,
        ))
        if title:
            ax.text(x + w / 2, y + h - 0.22, title, ha="center", va="top",
                    fontsize=7.5, color=C["mute"], fontstyle="italic")
            ax.text(x + w / 2, y + h / 2 - 0.08, text, ha="center", va="center",
                    fontsize=fs, color=C["ink"], wrap=True)
        else:
            ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
                    fontsize=fs, color=C["ink"])

    def arrow(x1, y1, x2, y2, text=None):
        ax.add_patch(FancyArrowPatch(
            (x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=10,
            color="#374151", lw=1.1, shrinkA=0, shrinkB=0,
        ))
        if text:
            ax.text((x1 + x2) / 2, (y1 + y2) / 2 + 0.14, text,
                    ha="center", va="bottom", fontsize=7, color=C["mute"])

    # Column titles matching DINO-WM / Dreamer
    ax.text(1.55, 4.38, "Frozen tokenizer  (GALA VAE)", ha="center", fontsize=9, color=C["mute"])
    ax.text(6.2, 4.38, "Action-conditioned residual dynamics", ha="center", fontsize=9, color=C["mute"])
    ax.text(10.55, 4.38, "Latent CEM  (DINO-WM protocol)", ha="center", fontsize=9, color=C["mute"])

    box(0.25, 3.15, 2.6, 0.85, r"HumanML3D pose  $x_{1:T}$" + "\n263-D RIC, 22 joints", C["box"], fs=8)
    arrow(1.55, 3.15, 1.55, 2.55)
    box(0.25, 1.55, 2.6, 1.0, "Graph encoder + VAE\nstride 4  →  $z\\in\\mathbb{R}^{256}$", C["box2"],
        title="frozen  ·  not trained as WM", fs=8)
    box(0.25, 0.35, 2.6, 0.85, r"root4 action  $a_t$" + "\n" + r"$(\dot\psi, v_x, v_z, h)$", C["box"], fs=8)
    arrow(2.85, 2.05, 3.55, 2.05)
    arrow(2.85, 0.78, 4.55, 1.55, "explicit $a$")

    box(3.55, 1.55, 5.3, 2.15, "", C["box2"])
    ax.text(6.2, 3.5, r"$\hat z_{t+1}=z_t+\sigma(g(a_t))\,\Delta(z_{t-k:t},a_t)$",
            ha="center", va="center", fontsize=10, color=C["ink"])
    ax.text(6.2, 2.95, "FiLM on every history token   ·   action gate on residual",
            ha="center", fontsize=8, color=C["mute"])
    ax.text(6.2, 2.45, "3-layer transformer predictor   ·   unroll H = 8 at train time",
            ha="center", fontsize=8, color=C["ink"])
    ax.text(6.2, 1.9, "v2 vs DINO-WM: motion latent, not DINO pixels; gate forces $a{=}0\\approx$ copy-last",
            ha="center", fontsize=7.4, color=C["mute"])

    box(3.55, 0.35, 2.4, 0.85, "Decoded pose loss\n1-step $\\hat x$", C["box"], fs=8)
    box(6.15, 0.35, 2.7, 0.85, "Optional flow / IDM\n(aux, not main table)", C["box"], fs=8)

    arrow(8.85, 2.6, 9.35, 2.6)
    box(9.35, 2.85, 2.8, 1.15, "Imagine $\\hat z_{1:H}(a)$\ncost: terminal + path", C["box3"],
        title="CEM  ·  pop 48 / 8 iters", fs=8)
    box(9.35, 1.45, 2.8, 1.05, "Hold planner\nrepeat last $z_t$", C["box"],
        title="open-loop baseline", fs=8)
    box(9.35, 0.25, 2.8, 0.9, "Frozen decoder\nMPJPE / root XY", C["box2"], fs=8)
    arrow(10.75, 2.85, 10.75, 2.5)
    ax.annotate("", xy=(10.75, 1.45), xytext=(10.75, 1.05),
                arrowprops=dict(arrowstyle="-|>", color="#374151", lw=1.1))

    ax.text(6.2, 0.08, "Fig. 1  ·  Same three blocks as DINO-WM (frozen encoder / dynamics / CEM), instantiated on a motion VAE.",
            ha="center", fontsize=7.5, color=C["mute"])
    save(fig, "fig1_architecture")


def fig_horizon():
    v2 = load_json("checkpoints/gala_humanml3d_wm_v2/eval.json")
    v1 = load_json("checkpoints/gala_humanml3d_wm/eval.json")
    unf = load_json("checkpoints/gala_humanml3d_wm_unfixed/eval.json")
    xs = np.array([1, 4, 8])
    sec = xs * 0.2
    series = [
        ("Copy-last  (repeat $z_t$; DINO-WM / video-pred.)", [mpjpe(v2, "copy_last", h) for h in xs], C["copy"], "--"),
        ("Zero action  (null-$a$; Visual Foresight / Genie)", [mpjpe(v2, "zero", h) for h in xs], C["zero"], ":"),
        ("Shuffle action  (wrong $a$; controllability)", [mpjpe(v2, "shuffle", h) for h in xs], C["shuffle"], ":"),
        ("Unfixed  + GT $a$  (absolute 1-step)", [mpjpe(unf, "gt", h) for h in xs], C["unfixed"], "-."),
        ("v1 residual  + GT $a$", [mpjpe(v1, "gt", h) for h in xs], C["v1"], "-"),
        ("v2 FiLM+gate  + GT $a$  (ours)", [mpjpe(v2, "gt", h) for h in xs], C["v2"], "-"),
    ]
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    for name, ys, color, ls in series:
        ax.plot(sec, ys, color=color, ls=ls, marker="o", lw=2.0 if "ours" in name else 1.5,
                markersize=5.5, label=name)
    ax.set_xlabel("Rollout horizon (seconds)")
    ax.set_ylabel("Denormalized MPJPE (mm)")
    ax.set_xticks(sec)
    ax.set_xticklabels(["0.2 s  (H=1)", "0.8 s  (H=4)", "1.6 s  (H=8)"])
    ax.set_ylim(15, 100)
    ax.legend(fontsize=7.4, loc="upper left")
    ax.set_title("Open-loop rollout vs horizon  ·  HumanML3D val, n=92")
    ax.axhline(80, color="#D1D5DB", ls="--", lw=0.8)
    ax.text(1.62, 81.5, "CEM success threshold 80 mm", ha="right", fontsize=7, color=C["mute"])
    save(fig, "fig2_horizon_mpjpe")


def fig_ablation():
    v2 = load_json("checkpoints/gala_humanml3d_wm_v2/eval.json")
    v1 = load_json("checkpoints/gala_humanml3d_wm/eval.json")
    labels = ["Copy-last", "Zero $a$", "Shuffle $a$", "GT $a$"]
    v1_y = [mpjpe(v1, m, 8) for m in ("copy_last", "zero", "shuffle", "gt")]
    v2_y = [mpjpe(v2, m, 8) for m in ("copy_last", "zero", "shuffle", "gt")]
    x = np.arange(len(labels))
    w = 0.36
    fig, ax = plt.subplots(figsize=(6.6, 4.2))
    b1 = ax.bar(x - w / 2, v1_y, w, label="v1 residual", color=C["v1"], zorder=2)
    b2 = ax.bar(x + w / 2, v2_y, w, label="v2 FiLM+gate (ours)", color=C["v2"], zorder=2)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("MPJPE @ 1.6 s (mm)")
    ax.set_ylim(0, 110)
    ax.legend(fontsize=8)
    ax.set_title("Action ablation at horizon 8  ·  GT should sit far below copy-last")
    for bars in (b1, b2):
        for rect in bars:
            ax.text(rect.get_x() + rect.get_width() / 2, rect.get_height() + 1.5,
                    f"{rect.get_height():.0f}", ha="center", va="bottom", fontsize=7.5)
    ax.annotate("GT–zero gap\n10 mm → 22 mm",
                xy=(3 + w / 2, v2_y[3]), xytext=(2.15, 42),
                fontsize=8, color=C["v2"],
                arrowprops=dict(arrowstyle="->", color=C["v2"]))
    save(fig, "fig3_action_ablation")


def fig_cem():
    rows = [
        ("Hold\n(open-loop)", 50.0, 103.3, C["hold"]),
        ("Unfixed\nweak CEM", 54.3, 88.8, C["unfixed"]),
        ("v1\nweak CEM", 65.2, 83.8, C["v1"]),
        ("v1\nstrong CEM", 71.7, 76.8, "#5B8FA8"),
        ("v2\nstrong CEM", 73.9, 72.0, C["v2"]),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.9))
    names = [r[0] for r in rows]
    x = np.arange(len(rows))
    colors = [r[3] for r in rows]
    axes[0].bar(x, [r[1] for r in rows], color=colors, zorder=2)
    axes[0].set_ylabel("Goal success (%)")
    axes[0].set_ylim(0, 100)
    axes[0].axhline(50, color="#D1D5DB", ls="--", lw=0.9)
    axes[0].set_title("CEM success  (MPJPE<80 mm, XY<0.5 m)")
    axes[1].bar(x, [r[2] for r in rows], color=colors, zorder=2)
    axes[1].set_ylabel("Planner MPJPE (mm)")
    axes[1].set_ylim(0, 130)
    axes[1].set_title("CEM terminal pose error")
    for ax in axes:
        ax.set_xticks(x)
        ax.set_xticklabels(names, fontsize=8)
    for i, r in enumerate(rows):
        axes[0].text(i, r[1] + 1.5, f"{r[1]:.0f}%", ha="center", fontsize=8)
        axes[1].text(i, r[2] + 2, f"{r[2]:.0f}", ha="center", fontsize=8)
    fig.suptitle("Latent CEM vs hold  ·  protocol cloned from DINO-WM (Zhou et al., 2024)", fontsize=10, y=1.02)
    save(fig, "fig4_cem")


def fig_train():
    # Residual v1 (second training pass) and v2 from train.log; unfixed latent@8 from first pass.
    v1_steps = [2000, 4000, 6000, 8000, 10000, 12000]
    v1_lat = [0.5226, 0.4612, 0.4494, 0.4257, 0.4108, 0.4179]
    v2_steps = [2000, 4000, 6000, 8000, 10000]
    v2_lat = [0.4056, 0.3744, 0.3744, 0.3600, 0.3358]
    unf_steps = [2000, 4000, 6000, 8000, 10000, 12000, 14000]
    unf_lat = [0.5195, 0.5112, 0.6053, 0.5133, 0.6963, 0.6200, 1.5885]
    fig, ax = plt.subplots(figsize=(6.8, 3.8))
    ax.plot(unf_steps, unf_lat, color=C["unfixed"], ls="-.", marker="o", label="Unfixed (absolute 1-step)")
    ax.plot(v1_steps, v1_lat, color=C["v1"], marker="o", label="v1 residual + unroll 4")
    ax.plot(v2_steps, v2_lat, color=C["v2"], marker="o", lw=2, label="v2 FiLM+gate + unroll 8")
    ax.set_xlabel("Training step")
    ax.set_ylabel("Val latent L2 @ horizon 8")
    ax.set_title("Online validation  ·  lower is better; unfixed diverges")
    ax.legend(fontsize=8)
    save(fig, "fig5_train_latent")


def fig_baseline_map():
    fig, ax = plt.subplots(figsize=(10.6, 3.6))
    ax.axis("off")
    ax.set_xlim(0, 10.6)
    ax.set_ylim(0, 3.6)
    headers = ["What we report", "Paper we copy the protocol from", "Not a reimplementation of"]
    rows = [
        ["Copy-last / repeat $z_t$", "DINO-WM Table 1 (repeat); video pred. copy-frame", "MDM / T2M FID tables"],
        ["Zero / shuffle $a$", "Visual Foresight; Genie; World-in-World ablations", "Puppeteer physics WM"],
        ["Hold planner", "DINO-WM / PlaNet open-loop control", "DreamerV3 actor-critic"],
        ["Latent CEM", "DINO-WM (Zhou et al. 2024); PlaNet CEM", "pixel CEM in DIAMOND"],
    ]
    col_x = [0.15, 3.55, 7.35]
    widths = [3.25, 3.65, 3.1]
    ax.text(5.3, 3.35, "Baseline provenance  ·  diagnostic protocols, not competing codebases",
            ha="center", fontsize=10, color=C["ink"])
    for i, h in enumerate(headers):
        ax.add_patch(FancyBboxPatch(
            (col_x[i], 2.55), widths[i], 0.55, boxstyle="square,pad=0",
            facecolor="#111827", edgecolor="none",
        ))
        ax.text(col_x[i] + widths[i] / 2, 2.82, h, ha="center", va="center",
                fontsize=8, color="white")
    for r, row in enumerate(rows):
        y = 1.95 - r * 0.48
        bg = "#F8FAFC" if r % 2 == 0 else "#EEF2FF"
        for i, cell in enumerate(row):
            ax.add_patch(FancyBboxPatch(
                (col_x[i], y), widths[i], 0.46, boxstyle="square,pad=0",
                facecolor=bg, edgecolor="#E5E7EB", linewidth=0.6,
            ))
            ax.text(col_x[i] + 0.12, y + 0.23, cell, ha="left", va="center", fontsize=7.6, color=C["ink"])
    save(fig, "fig0_baseline_map")


def main():
    style()
    fig_baseline_map()
    fig_architecture()
    fig_horizon()
    fig_ablation()
    fig_cem()
    fig_train()
    print("figures in", OUT)


if __name__ == "__main__":
    main()
