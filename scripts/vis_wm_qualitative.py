#!/usr/bin/env python3
"""Qualitative paper figures: swap XY paths and skeleton snapshots from GALA-WM v2."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

from evaluation.wm_metrics import integrate_root_xy, local_joints
from models.skeleton_graph import HML22_EDGES
from trainers.train import _build_loader

import importlib.util

_spec = importlib.util.spec_from_file_location("eval_wm", Path("/home/qinyang/桌面/project/scripts/eval_wm.py"))
_eval_wm = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_eval_wm)
decode_latents = _eval_wm.decode_latents
gt_actions = _eval_wm.gt_actions
history_batch = _eval_wm.history_batch
load_wm = _eval_wm.load_wm

ROOT = Path("/home/qinyang/桌面/project")
OUT = ROOT / "paper" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

C = {
    "gt": "#1F2937",
    "pred": "#2A9D8F",
    "copy": "#6B7280",
    "swap": "#C44E52",
    "zero": "#E07A3D",
}


def world_joints(motion: torch.Tensor, num_joints: int = 22) -> np.ndarray:
    local = local_joints(motion, num_joints)
    xy = integrate_root_xy(motion)
    height = motion[..., 3]
    world = local.clone()
    world[..., 0] = world[..., 0] + xy[..., 0].unsqueeze(-1)
    world[..., 2] = world[..., 2] + xy[..., 1].unsqueeze(-1)
    world[..., 1] = world[..., 1] + height.unsqueeze(-1)
    return world.cpu().numpy()


def draw_skel(ax, joints, color, lw=1.6, alpha=1.0, view="xz"):
    """joints: [J, 3] world XYZ with Y up."""
    if view == "xz":
        u, v = joints[:, 0], joints[:, 2]
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Z (m)")
    else:
        u, v = joints[:, 0], joints[:, 1]
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
    for i, j in HML22_EDGES:
        ax.plot([u[i], u[j]], [v[i], v[j]], color=color, lw=lw, alpha=alpha, solid_capstyle="round")
    ax.scatter(u, v, s=10, color=color, alpha=alpha, zorder=3, linewidths=0)


def pick_samples(model, loader, device, mean, std, horizon, k=3):
    """Prefer clips with large planar speed so swap forks are visible."""
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
        tgt = mu_k[torch.arange(mu_k.shape[0], device=device), start + horizon - 1]
        for i in range(mu_k.shape[0]):
            cap = captions[int(idx[i])] if isinstance(captions, list) else ""
            cands.append({
                "speed": float(speed[i]),
                "caption": cap[:80],
                "z_hist": z_hist[i:i + 1].cpu(),
                "act": act[i:i + 1].cpu(),
                "goal": tgt[i:i + 1].cpu(),
                "mu_future": mu_k[i, start[i]: start[i] + horizon].cpu(),
            })
            seen += 1
        if seen >= 48:
            break
    cands.sort(key=lambda x: x["speed"], reverse=True)
    return cands[:k]


@torch.no_grad()
def main():
    config = yaml.safe_load(open(ROOT / "configs/gala_humanml3d_wm_v2.yaml", encoding="utf-8"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_wm(config, str(ROOT / "checkpoints/gala_humanml3d_wm_v2/best.pt"), device, residual=True)
    train_cfg = {**config["training"], "batch_size_per_gpu": 8, "num_workers": 0}
    dataset, loader, _ = _build_loader(
        config["dataset"], "val", train_cfg, distributed=False, seed=config["seed"],
        shuffle=False, mode="eval", hash_text=False, max_samples=96,
    )
    mean, std = dataset.mean, dataset.std
    horizon = 8
    samples = pick_samples(model, loader, device, mean, std, horizon, k=3)
    scale = torch.as_tensor(std, device=device, dtype=torch.float32)
    shift = torch.as_tensor(mean, device=device, dtype=torch.float32)

    fig_xy, axes_xy = plt.subplots(1, 3, figsize=(10.8, 3.6), sharex=False, sharey=False)
    fig_sk, axes_sk = plt.subplots(3, 4, figsize=(11.2, 8.2))

    for s, item in enumerate(samples):
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

        motions = {
            "gt": decode_latents(model, fut, scale, shift)[0],
            "pred": decode_latents(model, pred, scale, shift)[0],
            "swap": decode_latents(model, pred_swap, scale, shift)[0],
            "copy": decode_latents(model, copy, scale, shift)[0],
            "zero": decode_latents(model, pred_zero, scale, shift)[0],
        }
        paths = {k: integrate_root_xy(v.unsqueeze(0))[0].cpu().numpy() for k, v in motions.items()}
        ax = axes_xy[s]
        for key, color, ls, lw in (
            ("gt", C["gt"], "-", 1.8),
            ("pred", C["pred"], "-", 2.2),
            ("swap", C["swap"], "--", 1.8),
            ("copy", C["copy"], ":", 1.4),
            ("zero", C["zero"], "-.", 1.2),
        ):
            p = paths[key]
            ax.plot(p[:, 0], p[:, 1], color=color, ls=ls, lw=lw, label=key)
            ax.scatter(p[-1, 0], p[-1, 1], color=color, s=18, zorder=4)
        ax.scatter(paths["gt"][0, 0], paths["gt"][0, 1], color=C["gt"], s=28, marker="s", zorder=5)
        ax.set_aspect("equal", adjustable="datalim")
        ax.set_title(item["caption"] or f"clip {s + 1}", fontsize=8)
        ax.set_xlabel("X (m)")
        if s == 0:
            ax.set_ylabel("Z (m)")
            ax.legend(fontsize=7, loc="best")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        skel = {k: world_joints(v.unsqueeze(0))[0] for k, v in motions.items() if k in ("gt", "pred", "copy", "swap")}
        frames = [0, 15, 31]  # ~0 / 0.75 / 1.55 s at 20 fps after stride-4 decode of 8 latents
        frames = [min(t, skel["gt"].shape[0] - 1) for t in frames]
        # overlay GT vs pred at last frame, plus copy and swap
        row = axes_sk[s]
        titles = [
            "t=0  start",
            "H=8  GT vs pred",
            "H=8  copy-last",
            "H=8  swapped $v_x,v_z$",
        ]
        specs = [
            [("gt", C["gt"], 1.0, frames[0])],
            [("gt", C["gt"], 0.35, frames[-1]), ("pred", C["pred"], 1.0, frames[-1])],
            [("gt", C["gt"], 0.35, frames[-1]), ("copy", C["copy"], 1.0, frames[-1])],
            [("pred", C["pred"], 0.45, frames[-1]), ("swap", C["swap"], 1.0, frames[-1])],
        ]
        for col, (title, spec) in enumerate(zip(titles, specs)):
            ax = row[col]
            for name, color, alpha, t in spec:
                draw_skel(ax, skel[name][t], color, lw=1.7, alpha=alpha, view="xy")
            ax.set_title(title if s == 0 else "", fontsize=9)
            ax.set_aspect("equal")
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            if col == 0:
                ax.set_ylabel(f"clip {s + 1}", fontsize=8)

    fig_xy.suptitle("Action swap on root XY  ·  opposite $(v_x,v_z)$ should fork the path (DINO-WM / Visual Foresight counterfactual)", fontsize=10)
    fig_xy.tight_layout()
    fig_xy.savefig(OUT / "fig6_swap_xy.png", dpi=220, bbox_inches="tight")
    fig_xy.savefig(OUT / "fig6_swap_xy.pdf", bbox_inches="tight")
    plt.close(fig_xy)

    fig_sk.suptitle("Decoded skeletons at 1.6 s  ·  green = v2 GT-action rollout, grey = copy-last, red = swapped planar velocity", fontsize=10)
    fig_sk.tight_layout()
    fig_sk.savefig(OUT / "fig7_skeletons.png", dpi=220, bbox_inches="tight")
    fig_sk.savefig(OUT / "fig7_skeletons.pdf", bbox_inches="tight")
    plt.close(fig_sk)
    print("wrote", OUT / "fig6_swap_xy.png")
    print("wrote", OUT / "fig7_skeletons.png")
    for i, item in enumerate(samples):
        print(f"sample {i}: speed={item['speed']:.3f}  {item['caption']}")


if __name__ == "__main__":
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 9,
        "axes.titlesize": 10,
        "pdf.fonttype": 42,
    })
    main()
