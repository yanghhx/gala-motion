#!/usr/bin/env python3
"""DINO-WM Fig.3/4/5 analogues: compact publication-quality skeleton figures."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml

from evaluation.wm_metrics import integrate_root_xy, local_joints
from models.skeleton_graph import HML22_EDGES
from trainers.train import _build_loader

_spec = importlib.util.spec_from_file_location("eval_wm", Path("/home/qinyang/桌面/project/scripts/eval_wm.py"))
_eval_wm = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_eval_wm)
decode_latents = _eval_wm.decode_latents
gt_actions = _eval_wm.gt_actions
history_batch = _eval_wm.history_batch
load_wm = _eval_wm.load_wm
cem_plan = _eval_wm.cem_plan

ROOT = Path("/home/qinyang/桌面/project")
OUT = ROOT / "paper" / "figures"
OVER = ROOT / "paper" / "overleaf" / "figures"
OUT.mkdir(parents=True, exist_ok=True)
OVER.mkdir(parents=True, exist_ok=True)

LEFT_EDGES = {(0, 1), (1, 2), (2, 3), (8, 11), (11, 12), (12, 13), (13, 14), (14, 15)}
RIGHT_EDGES = {(0, 4), (4, 5), (5, 6), (8, 16), (16, 17), (17, 18), (18, 19), (19, 20), (20, 21)}
SPINE_EDGES = {(0, 7), (7, 8), (8, 9), (9, 10)}

COL = {
    "gt": "#1A202C",
    "left": "#2B6CB0",
    "right": "#DD6B20",
    "spine": "#2D3748",
    "pred": "#276749",
    "copy": "#718096",
    "zero": "#C05621",
    "swap": "#C53030",
    "cem": "#2B6CB0",
    "hold": "#805AD5",
}


def world_joints(motion: torch.Tensor, num_joints: int = 22) -> np.ndarray:
    local = local_joints(motion, num_joints)
    xy = integrate_root_xy(motion)
    height = motion[..., 3]
    world = local.clone()
    world[..., 0] = world[..., 0] + xy[..., 0].unsqueeze(-1)
    world[..., 2] = world[..., 2] + xy[..., 1].unsqueeze(-1)
    world[..., 1] = world[..., 1] + height.unsqueeze(-1)
    return world.detach().cpu().numpy()


def draw_skel(ax, joints, color=None, lr=False, lw=2.2, alpha=1.0, offset_z=0.0):
    u = joints[:, 2] + offset_z
    v = joints[:, 1]
    for i, j in HML22_EDGES:
        if lr:
            if (i, j) in LEFT_EDGES or (j, i) in LEFT_EDGES:
                c = COL["left"]
            elif (i, j) in RIGHT_EDGES or (j, i) in RIGHT_EDGES:
                c = COL["right"]
            else:
                c = COL["spine"]
        else:
            c = color or COL["pred"]
        ax.plot([u[i], u[j]], [v[i], v[j]], color=c, lw=lw, alpha=alpha,
                solid_capstyle="round", solid_joinstyle="round", zorder=2)
    ax.scatter(u, v, s=14, color=COL["spine"] if lr else (color or COL["pred"]),
               alpha=min(1.0, alpha + 0.05), zorder=3, linewidths=0, edgecolors="none")


def set_cell(ax, joints_list, pad=0.20, offsets=None):
    pts = []
    for k, j in enumerate(joints_list):
        off = 0.0 if offsets is None else offsets[k]
        pts.append(np.stack([j[:, 2] + off, j[:, 1]], axis=1))
    pts = np.concatenate(pts, axis=0)
    cx, cy = pts.mean(axis=0)
    span = max(float(np.max(pts.max(axis=0) - pts.min(axis=0))), 0.55)
    ax.set_xlim(cx - span / 2 - pad, cx + span / 2 + pad)
    ax.set_ylim(cy - span / 2 - pad, cy + span / 2 + pad)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("#CBD5E0")
        spine.set_linewidth(0.8)
    ax.set_facecolor("#FAFBFC")


def pick_samples(model, loader, device, horizon, k=4):
    cands = []
    seen = 0
    for batch in loader:
        motion = batch["motion"].to(device)
        lengths = batch["lengths"].to(device)
        captions = batch.get("captions") or [""] * motion.shape[0]
        mu, latent_mask, _ = model.encode_latents(motion, lengths)
        history = model.wm.history_latents
        keep = latent_mask.sum(dim=1).long() >= history + horizon
        if not keep.any():
            continue
        idx = torch.where(keep)[0]
        motion_k, mu_k = motion[idx], mu[idx]
        start = torch.full((mu_k.shape[0],), history, device=device, dtype=torch.long)
        z_hist = history_batch(mu_k, history, start)
        act = gt_actions(model, mu_k, motion_k, start, horizon)
        speed = act[..., 1:3].norm(dim=-1).mean(dim=1)
        for i in range(mu_k.shape[0]):
            cap = captions[int(idx[i])] if isinstance(captions, list) else ""
            cands.append({
                "speed": float(speed[i]),
                "caption": cap.replace("\n", " ").strip(),
                "z_hist": z_hist[i:i + 1].cpu(),
                "act": act[i:i + 1].cpu(),
                "goal": mu_k[i, start[i] + horizon - 1].cpu(),
                "mu_future": mu_k[i, start[i]: start[i] + horizon].cpu(),
            })
            seen += 1
        if seen >= 64:
            break
    cands.sort(key=lambda x: x["speed"], reverse=True)
    # Prefer diverse captions
    picked, used = [], set()
    for c in cands:
        key = " ".join(c["caption"].lower().split()[:4])
        if key in used:
            continue
        used.add(key)
        picked.append(c)
        if len(picked) >= k:
            break
    return picked or cands[:k]


def save_both(fig, name):
    for folder in (OUT, OVER):
        fig.savefig(folder / f"{name}.pdf", bbox_inches="tight", pad_inches=0.08)
        fig.savefig(folder / f"{name}.png", dpi=280, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)
    print("wrote", name)


@torch.no_grad()
def rollout_bundle(model, item, device, scale, shift, horizon):
    z_hist = item["z_hist"].to(device)
    act = item["act"].to(device)
    swapped = act.clone()
    swapped[..., 1:3] = -swapped[..., 1:3]
    zero = torch.zeros_like(act)
    fut = item["mu_future"].unsqueeze(0).to(device)
    pred = model.rollout(z_hist, act)
    pred_swap = model.rollout(z_hist, swapped)
    pred_zero = model.rollout(z_hist, zero)
    copy = z_hist[:, -1:].expand(-1, horizon, -1).contiguous()
    planned = cem_plan(model, z_hist, item["goal"].to(device), horizon)
    pred_cem = model.rollout(z_hist, planned.unsqueeze(0))
    hold = z_hist[:, -1:].expand(-1, horizon, -1).contiguous()
    keys = {
        "gt": fut, "pred": pred, "copy": copy, "zero": pred_zero,
        "swap": pred_swap, "cem": pred_cem, "hold": hold,
    }
    skel = {k: world_joints(decode_latents(model, z, scale, shift)[0].unsqueeze(0))[0] for k, z in keys.items()}
    paths = {k: integrate_root_xy(decode_latents(model, z, scale, shift))[0].cpu().numpy() for k, z in keys.items()}
    return skel, paths


def short_cap(cap, n=34):
    cap = cap.strip()
    return cap if len(cap) <= n else cap[: n - 1] + "…"


def fig_dataset_teaser(bundles, times):
    """Conference-style domain grid: no in-figure title (caption carries it)."""
    n = len(bundles)
    fig, axes = plt.subplots(n, 4, figsize=(6.4, 1.45 * n))

    if n == 1:
        axes = axes[None, :]
    t_idx = [times[0], times[1], times[2], times[-1]]
    headers = ["$t{=}0$", "0.4s", "0.8s", "1.6s"]
    for r, (skel, _, cap) in enumerate(bundles):
        all_j = [skel["gt"][t] for t in t_idx]
        for c, t in enumerate(t_idx):
            ax = axes[r, c]
            draw_skel(ax, skel["gt"][t], lr=True, lw=2.0)
            set_cell(ax, all_j, pad=0.18)
            if r == 0:
                ax.set_title(headers[c], fontsize=8, pad=2)
            if c == 0:
                ax.text(-0.04, 0.5, short_cap(cap, 22), transform=ax.transAxes,
                        ha="right", va="center", fontsize=7, color="#1A1A1A", clip_on=False)
    fig.tight_layout(h_pad=0.18, w_pad=0.04, rect=(0.12, 0.0, 1.0, 1.0))
    save_both(fig, "fig_dataset_teaser")


def fig_rollout_grid(bundles, times, titles):
    """Conference-style open-loop grid (DINO-WM Fig.4)."""
    rows = [
        ("pred", "Ours", False, COL["pred"]),
        ("copy", "Copy-last", False, COL["copy"]),
        ("zero", "Zero act.", False, COL["zero"]),
        ("swap", "Swap $v$", False, COL["swap"]),
        ("gt", "GT", True, COL["gt"]),
    ]
    skel, _, _cap = bundles[0]
    n_t, n_r = len(times), len(rows)
    fig, axes = plt.subplots(n_r, n_t, figsize=(5.8, 3.6))

    all_j = [skel[k][t] for k in ("gt", "pred", "copy", "zero", "swap") for t in times]
    for r, (key, label, lr, color) in enumerate(rows):
        for col, t in enumerate(times):
            ax = axes[r, col]
            draw_skel(ax, skel[key][t], color=color, lr=lr, lw=2.0)
            set_cell(ax, all_j, pad=0.16)
            if r == 0:
                ax.set_title(titles[col], fontsize=8, pad=2)
            if col == 0:
                ax.text(-0.04, 0.5, label, transform=ax.transAxes, ha="right", va="center",
                        fontsize=8, clip_on=False)
    fig.tight_layout(h_pad=0.06, w_pad=0.03, rect=(0.10, 0.0, 1.0, 1.0))
    save_both(fig, "fig_rollout_grid")


def fig_planning_vis(bundles, times, titles):
    """Conference-style planning grid (DINO-WM Fig.5)."""
    n_clip = min(2, len(bundles))
    later = times[1:]
    headers = ["Start"] + [f"GT {h}" for h in titles[1:]] + [f"CEM {h}" for h in titles[1:]] + ["Hold", "Goal"]
    n_col = len(headers)
    fig, axes = plt.subplots(n_clip, n_col, figsize=(7.2, 1.25 * n_clip + 0.1))

    if n_clip == 1:
        axes = axes[None, :]
    for c in range(n_clip):
        skel, _, cap = bundles[c]
        t0, tH = times[0], times[-1]
        cells = [skel["gt"][t0]]
        cells += [skel["gt"][t] for t in later]
        cells += [skel["cem"][t] for t in later]
        cells += [skel["hold"][tH], skel["gt"][tH]]
        styles = (
            [("gt", True, COL["gt"])] * (1 + len(later))
            + [("cem", False, COL["cem"])] * len(later)
            + [("hold", False, COL["hold"]), ("gt", True, COL["gt"])]
        )
        for col, (joints, (_k, lr, color)) in enumerate(zip(cells, styles)):
            ax = axes[c, col]
            draw_skel(ax, joints, color=color, lr=lr, lw=1.85)
            set_cell(ax, cells, pad=0.16)
            if c == 0:
                ax.set_title(headers[col], fontsize=7, pad=2)
        axes[c, 0].text(-0.04, 0.5, short_cap(cap, 20), transform=axes[c, 0].transAxes,
                        ha="right", va="center", fontsize=7, clip_on=False)
    fig.tight_layout(h_pad=0.12, w_pad=0.02, rect=(0.10, 0.0, 1.0, 1.0))
    save_both(fig, "fig_planning_vis")


def fig_planning_xy(bundles):
    """Compact Z(t)/X(t) panels without a verbose suptitle."""
    n = min(2, len(bundles))
    fig, axes = plt.subplots(2, n, figsize=(2.9 * n, 3.4), sharex="col")

    if n == 1:
        axes = np.array(axes)[:, None]
    series_style = (
        ("gt", COL["gt"], "-", 2.0, "GT"),
        ("pred", COL["pred"], "-", 1.9, "Ours"),
        ("cem", COL["cem"], "--", 1.8, "CEM"),
        ("hold", COL["hold"], ":", 1.7, "Hold"),
        ("swap", COL["swap"], "-.", 1.8, "Swap"),
    )
    for c in range(n):
        _, paths, cap = bundles[c]
        T = paths["gt"].shape[0]
        t = np.arange(T) / 20.0
        ax_z, ax_x = axes[0, c], axes[1, c]
        for key, color, ls, lw, name in series_style:
            p = paths[key]
            ax_z.plot(t, p[:, 1], color=color, ls=ls, lw=lw, label=name)
            ax_x.plot(t, p[:, 0], color=color, ls=ls, lw=lw)
        ax_z.set_title(short_cap(cap, 36), fontsize=7.5)
        if c == 0:
            ax_z.set_ylabel("Z (m)")
            ax_x.set_ylabel("X (m)")
        ax_x.set_xlabel("time (s)")
        for ax in (ax_z, ax_x):
            ax.grid(True, alpha=0.22, lw=0.5)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
    axes[0, 0].legend(fontsize=6, loc="best", frameon=False, ncol=1)
    fig.tight_layout(h_pad=0.35, w_pad=0.35)
    save_both(fig, "fig_planning_xy")


@torch.no_grad()
def main():
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 9,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "axes.titlesize": 9,
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
    })
    config = yaml.safe_load(open(ROOT / "configs/gala_humanml3d_wm_v2.yaml", encoding="utf-8"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_wm(config, str(ROOT / "checkpoints/gala_humanml3d_wm_v2/best.pt"), device, residual=True)
    train_cfg = {**config["training"], "batch_size_per_gpu": 8, "num_workers": 0}
    dataset, loader, _ = _build_loader(
        config["dataset"], "val", train_cfg, distributed=False, seed=config["seed"],
        shuffle=False, mode="eval", hash_text=False, max_samples=96,
    )
    scale = torch.as_tensor(dataset.std, device=device, dtype=torch.float32)
    shift = torch.as_tensor(dataset.mean, device=device, dtype=torch.float32)
    horizon = 8
    samples = pick_samples(model, loader, device, horizon, k=4)
    bundles = []
    for item in samples:
        skel, paths = rollout_bundle(model, item, device, scale, shift, horizon)
        bundles.append((skel, paths, item["caption"] or "clip"))
        print("clip", item["caption"], "speed", f"{item['speed']:.2f}")
    T = bundles[0][0]["gt"].shape[0]
    times = [min(int(round(x)), T - 1) for x in (0, 8, 16, 24, T - 1)]
    titles = ["$t{=}0$", "0.4 s", "0.8 s", "1.2 s", "1.6 s"]

    def forward_score(b):
        p = b[1]["gt"]
        return float(p[-1, 1] - p[0, 1])

    def is_loco(cap: str, score: float) -> bool:
        c = cap.lower()
        bad = ("turn", "spin", "around", "rotate", "pivot", "wave", "stand", "sit",
               "jump", "backwards", "backward", "shove", "stagger")
        if any(k in c for k in bad):
            return False
        good = ("walk", "run", "lift", "carry", "crawl")
        return score > 0.6 and any(k in c for k in good)

    scored = [(forward_score(b), b) for b in bundles]
    ranked = [b for s, b in sorted(scored, reverse=True) if is_loco(b[2], s)]
    if len(ranked) < 2:
        ranked = [b for s, b in sorted(scored, reverse=True) if "walk" in b[2].lower()]
    if len(ranked) < 2:
        ranked = [b for s, b in sorted(scored, reverse=True)]
    teaser = ranked[:2]
    main_clip = ranked[:2]
    print("teaser:", [b[2] for b in teaser])

    fig_dataset_teaser(teaser, times)
    fig_rollout_grid(main_clip, times, titles)
    fig_planning_vis(main_clip, times, titles)
    fig_planning_xy(main_clip)


if __name__ == "__main__":
    main()
