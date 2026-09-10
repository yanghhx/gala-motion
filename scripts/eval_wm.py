#!/usr/bin/env python3
"""Closed-loop world-model evaluation: rollout, swap-action, CEM planning."""
from __future__ import annotations

import argparse
import json
from dataclasses import fields
from pathlib import Path

import numpy as np
import torch
import yaml

from evaluation.wm_metrics import integrate_root_xy, latent_l2, local_joints, mpjpe_mm, summarize
from models.gala_motion import GALAMotionConfig
from models.gala_wm import GALAWorldConfig, build_world_model
from trainers.train import _build_loader


def load_wm(config, checkpoint, device, residual=None):
    data_cfg, model_cfg = config["dataset"], config["model"]
    allowed = {item.name for item in fields(GALAMotionConfig)}
    gala = GALAMotionConfig(
        motion_dim=data_cfg["motion_dim"], num_joints=data_cfg["num_joints"],
        max_frames=data_cfg["max_frames"],
        **{key: value for key, value in model_cfg.items() if key in allowed},
    )
    world_kwargs = {
        key: value for key, value in config.get("world", {}).items()
        if key in {item.name for item in fields(GALAWorldConfig)}
    }
    state = torch.load(checkpoint, map_location=device, weights_only=False)
    ckpt_world = {}
    if isinstance(state, dict):
        packed = state.get("config") or {}
        ckpt_world = packed.get("world") or {}
        if isinstance(ckpt_world, dict):
            world_kwargs.update({k: v for k, v in ckpt_world.items() if k in {item.name for item in fields(GALAWorldConfig)}})
    if residual is not None:
        world_kwargs["residual"] = residual
    world = GALAWorldConfig(**world_kwargs)
    init_from = config["training"].get("init_from")
    model = build_world_model(gala, world, init_from, device)
    raw = state["model"] if isinstance(state, dict) and "model" in state else state
    model.load_state_dict(raw, strict=False)
    model.eval()
    return model


def history_batch(mu, history, start):
    batch = torch.arange(mu.shape[0], device=mu.device)
    offsets = torch.arange(history, device=mu.device)
    return mu[batch[:, None], start[:, None] - history + offsets]


def gt_actions(model, mu, motion, start, horizon):
    explicit = model.root4_actions(motion, mu.shape[1])
    if model.wm.action_type == "idm":
        inferred = model.idm(mu[:, :-1], mu[:, 1:])
        source = inferred
    else:
        source = explicit
    steps = start[:, None] + torch.arange(horizon, device=mu.device)[None] - 1
    steps = steps.clamp(0, source.shape[1] - 1)
    batch = torch.arange(mu.shape[0], device=mu.device)
    return source[batch[:, None], steps]


def decode_latents(model, z, scale, shift):
    """z: [B, H, D] -> denormalized motion [B, H*stride, 263]."""
    batch, steps, dim = z.shape
    flat = z.reshape(batch * steps, 1, dim)
    motion = model.backbone.decode(flat, model.cfg.stride)
    motion = motion.reshape(batch, steps * model.cfg.stride, motion.shape[-1])
    return motion * scale + shift


def per_sample_l2(pred, target):
    return (pred - target).square().mean(dim=-1)


def per_sample_mpjpe(pred_m, true_m, num_joints):
    dist = (local_joints(pred_m, num_joints) - local_joints(true_m, num_joints)).norm(dim=-1)
    return dist.mean(dim=(1, 2)) * 1000


@torch.no_grad()
def rollout_metrics(model, batch, device, horizons, mean, std):
    motion = batch["motion"].to(device)
    lengths = batch["lengths"].to(device)
    mu, latent_mask, _ = model.encode_latents(motion, lengths)
    history = model.wm.history_latents
    keep = latent_mask.sum(dim=1).long() >= history + max(horizons)
    if not keep.any():
        return []
    motion, mu = motion[keep], mu[keep]
    start = torch.full((mu.shape[0],), history, device=device, dtype=torch.long)
    z_hist = history_batch(mu, history, start)
    scale = torch.as_tensor(std, device=device, dtype=motion.dtype)
    shift = torch.as_tensor(mean, device=device, dtype=motion.dtype)
    max_h = max(horizons)
    idx = torch.arange(mu.shape[0], device=device)
    methods = {
        "gt": gt_actions(model, mu, motion, start, max_h),
        "zero": None,
        "shuffle": None,
        "copy_last": None,
    }
    methods["zero"] = torch.zeros_like(methods["gt"])
    methods["shuffle"] = methods["gt"][torch.randperm(mu.shape[0], device=device)]
    records = []
    for name, act in methods.items():
        if name == "copy_last":
            pred_z = z_hist[:, -1:].expand(-1, max_h, -1).contiguous()
        else:
            pred_z = model.rollout(z_hist, act)
        for horizon in horizons:
            target = mu[idx, start + horizon - 1]
            pred = pred_z[:, :horizon]
            tgt_seq = mu[idx[:, None], start[:, None] + torch.arange(horizon, device=device)]
            pred_m = decode_latents(model, pred, scale, shift)
            true_m = decode_latents(model, tgt_seq, scale, shift)
            l2 = per_sample_l2(pred[:, -1], target)
            mpjpe = per_sample_mpjpe(pred_m, true_m, model.cfg.num_joints)
            for i in range(mu.shape[0]):
                records.append({
                    "method": name,
                    "horizon": horizon,
                    "latent_l2": float(l2[i]),
                    "mpjpe_mm": float(mpjpe[i]),
                })
    return records


@torch.no_grad()
def swap_metrics(model, batch, device, horizon, mean, std):
    motion = batch["motion"].to(device)
    lengths = batch["lengths"].to(device)
    mu, latent_mask, _ = model.encode_latents(motion, lengths)
    history = model.wm.history_latents
    keep = latent_mask.sum(dim=1).long() >= history + horizon
    if not keep.any():
        return []
    motion, mu = motion[keep], mu[keep]
    start = torch.full((mu.shape[0],), history, device=device, dtype=torch.long)
    z_hist = history_batch(mu, history, start)
    act = gt_actions(model, mu, motion, start, horizon)
    swapped = act.clone()
    if swapped.shape[-1] >= 3:
        swapped[..., 1:3] = -swapped[..., 1:3]
    pred_a = model.rollout(z_hist, act)
    pred_b = model.rollout(z_hist, swapped)
    pred_n = model.rollout(z_hist, act + 0.05 * torch.randn_like(act))
    scale = torch.as_tensor(std, device=device, dtype=motion.dtype)
    shift = torch.as_tensor(mean, device=device, dtype=motion.dtype)
    xy_a = integrate_root_xy(decode_latents(model, pred_a, scale, shift))[:, -1]
    xy_b = integrate_root_xy(decode_latents(model, pred_b, scale, shift))[:, -1]
    xy_n = integrate_root_xy(decode_latents(model, pred_n, scale, shift))[:, -1]
    swap_delta = (xy_a - xy_b).norm(dim=-1)
    noise_delta = (xy_a - xy_n).norm(dim=-1)
    return [{
        "swap_delta": float(swap_delta[i]),
        "noise_delta": float(noise_delta[i]),
        "ratio": float(swap_delta[i] / noise_delta[i].clamp_min(1e-4)),
    } for i in range(mu.shape[0])]


@torch.no_grad()
def cem_plan(model, z_hist, goal, horizon, pop=48, elite=12, iters=8, act_std=0.5, act_clip=2.5):
    action_dim = model.action_dim
    mean = torch.zeros(horizon, action_dim, device=z_hist.device, dtype=z_hist.dtype)
    std = torch.full((horizon, action_dim), act_std, device=z_hist.device, dtype=z_hist.dtype)
    best = mean
    for _ in range(iters):
        acts = (mean + std * torch.randn(pop, horizon, action_dim, device=z_hist.device, dtype=z_hist.dtype)).clamp(-act_clip, act_clip)
        hist = z_hist.expand(pop, -1, -1)
        pred = model.rollout(hist, acts)
        terminal = (pred[:, -1] - goal).square().mean(dim=-1)
        path = (pred - goal).square().mean(dim=(1, 2))
        cost = terminal + 0.35 * path
        idx = cost.topk(elite, largest=False).indices
        chosen = acts[idx]
        mean, std = chosen.mean(0), chosen.std(0).clamp_min(0.05)
        best = chosen[0]
    return best


@torch.no_grad()
def cem_metrics(model, batch, device, horizon, mean, std, pop, elite, iters):
    motion = batch["motion"].to(device)
    lengths = batch["lengths"].to(device)
    mu, latent_mask, _ = model.encode_latents(motion, lengths)
    history = model.wm.history_latents
    keep = latent_mask.sum(dim=1).long() >= history + horizon
    if not keep.any():
        return []
    mu = mu[keep]
    start = torch.full((mu.shape[0],), history, device=device, dtype=torch.long)
    scale = torch.as_tensor(std, device=device, dtype=mu.dtype)
    shift = torch.as_tensor(mean, device=device, dtype=mu.dtype)
    records = []
    for i in range(mu.shape[0]):
        z_hist = history_batch(mu[i:i + 1], history, start[i:i + 1])
        goal = mu[i, start[i] + horizon - 1]
        planned = cem_plan(model, z_hist, goal, horizon, pop, elite, iters)
        pred = model.rollout(z_hist, planned.unsqueeze(0))
        hold = z_hist[:, -1:]
        pred_m = decode_latents(model, pred[:, -1:], scale, shift)
        hold_m = decode_latents(model, hold, scale, shift)
        true_m = decode_latents(model, goal.view(1, 1, -1), scale, shift)
        mpjpe = float(per_sample_mpjpe(pred_m, true_m, model.cfg.num_joints)[0])
        hold_mpjpe = float(per_sample_mpjpe(hold_m, true_m, model.cfg.num_joints)[0])
        root = float((integrate_root_xy(pred_m)[:, -1] - integrate_root_xy(true_m)[:, -1]).norm())
        records.append({
            "mpjpe_mm": mpjpe,
            "hold_mpjpe_mm": hold_mpjpe,
            "root_xy_m": root,
            "success": int(mpjpe < 80.0 and root < 0.5),
            "hold_success": int(hold_mpjpe < 80.0),
            "latent_l2": float(latent_l2(pred[:, -1], goal)),
        })
    return records


def aggregate(rollout_rows, swap_rows, cem_rows, horizons):
    report = {"rollout": {}, "swap": summarize([row["ratio"] for row in swap_rows]), "cem": {}}
    methods = sorted({row["method"] for row in rollout_rows})
    for method in methods:
        report["rollout"][method] = {}
        for horizon in horizons:
            subset = [row for row in rollout_rows if row["method"] == method and row["horizon"] == horizon]
            report["rollout"][method][str(horizon)] = {
                "latent_l2": summarize([row["latent_l2"] for row in subset]),
                "mpjpe_mm": summarize([row["mpjpe_mm"] for row in subset]),
            }
    if cem_rows:
        report["cem"] = {
            "mpjpe_mm": summarize([row["mpjpe_mm"] for row in cem_rows]),
            "hold_mpjpe_mm": summarize([row.get("hold_mpjpe_mm", row["mpjpe_mm"]) for row in cem_rows]),
            "root_xy_m": summarize([row["root_xy_m"] for row in cem_rows]),
            "success_rate": float(np.mean([row["success"] for row in cem_rows])),
            "hold_success_rate": float(np.mean([row.get("hold_success", 0) for row in cem_rows])),
            "latent_l2": summarize([row["latent_l2"] for row in cem_rows]),
            "n": len(cem_rows),
        }
    if swap_rows:
        report["swap"] = {
            "ratio": summarize([row["ratio"] for row in swap_rows]),
            "swap_delta": summarize([row["swap_delta"] for row in swap_rows]),
            "noise_delta": summarize([row["noise_delta"] for row in swap_rows]),
        }
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", default="val")
    parser.add_argument("--max-samples", type=int, default=128)
    parser.add_argument("--horizons", default="1,4,8")
    parser.add_argument("--cem-horizon", type=int, default=8)
    parser.add_argument("--cem-pop", type=int, default=48)
    parser.add_argument("--cem-iters", type=int, default=8)
    parser.add_argument("--output", required=True)
    parser.add_argument("--residual", dest="residual", action="store_true")
    parser.add_argument("--no-residual", dest="residual", action="store_false")
    parser.set_defaults(residual=None)
    args = parser.parse_args()
    config = yaml.safe_load(open(args.config, encoding="utf-8"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_wm(config, args.checkpoint, device, residual=args.residual)
    train_cfg = {**config["training"], "batch_size_per_gpu": 8, "num_workers": 0}
    dataset, loader, _ = _build_loader(
        config["dataset"], args.split, train_cfg, distributed=False, seed=config["seed"],
        shuffle=False, mode="eval", hash_text=False, max_samples=args.max_samples,
    )
    horizons = tuple(int(x) for x in args.horizons.split(","))
    rollout_rows, swap_rows, cem_rows = [], [], []
    for batch in loader:
        rollout_rows.extend(rollout_metrics(model, batch, device, horizons, dataset.mean, dataset.std))
        swap_rows.extend(swap_metrics(model, batch, device, max(horizons), dataset.mean, dataset.std))
        cem_rows.extend(cem_metrics(
            model, batch, device, args.cem_horizon, dataset.mean, dataset.std,
            args.cem_pop, max(args.cem_pop // 4, 8), args.cem_iters,
        ))
    report = aggregate(rollout_rows, swap_rows, cem_rows, horizons)
    report["meta"] = {
        "checkpoint": args.checkpoint,
        "split": args.split,
        "samples": args.max_samples,
        "action_type": model.wm.action_type,
        "residual": model.wm.residual,
        "device": str(device),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
