#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import random
from dataclasses import asdict, fields

import numpy as np
import torch
import yaml

from models.gala_motion import GALAMotionConfig
from models.gala_wm import GALAWorldConfig, build_world_model
from trainers.checkpoint import CheckpointManager
from trainers.train import _build_loader, distributed_setup

try:
    from torch.utils.tensorboard import SummaryWriter
except Exception:
    SummaryWriter = None


def parse_configs(raw):
    data_cfg, model_cfg, train_cfg = raw["dataset"], raw["model"], raw["training"]
    allowed = {item.name for item in fields(GALAMotionConfig)}
    gala = GALAMotionConfig(
        motion_dim=data_cfg["motion_dim"], num_joints=data_cfg["num_joints"],
        max_frames=data_cfg["max_frames"],
        **{key: value for key, value in model_cfg.items() if key in allowed},
    )
    world = GALAWorldConfig(**{key: value for key, value in raw.get("world", {}).items() if key in {f.name for f in fields(GALAWorldConfig)}})
    return data_cfg, train_cfg, gala, world


@torch.no_grad()
def quick_val(model, loader, device, horizons=(1, 8)):
    from evaluation.wm_metrics import latent_l2, local_joints, mpjpe_mm

    raw = model.module if hasattr(model, "module") else model
    raw.eval()
    stats = {f"latent_h{h}": [] for h in horizons}
    stats.update({f"mpjpe_h{h}": [] for h in horizons})
    for batch in loader:
        motion = batch["motion"].to(device)
        lengths = batch["lengths"].to(device)
        mu, latent_mask, _ = raw.encode_latents(motion, lengths)
        actions = raw.root4_actions(motion, mu.shape[1])
        history = raw.wm.history_latents
        latent_len = latent_mask.sum(dim=1).long()
        keep = latent_len >= history + max(horizons)
        if not keep.any():
            continue
        mu, actions, motion = mu[keep], actions[keep], motion[keep]
        start = torch.full((mu.shape[0],), history, device=device, dtype=torch.long)
        batch_idx = torch.arange(mu.shape[0], device=device)
        offsets = torch.arange(history, device=device)
        z_hist = mu[batch_idx[:, None], start[:, None] - history + offsets]
        if raw.wm.action_type == "idm":
            act_seq = raw.idm(mu[:, :-1], mu[:, 1:])
        else:
            act_seq = actions
        preds = raw.rollout(z_hist, act_seq[batch_idx[:, None], start[:, None] + torch.arange(max(horizons), device=device) - 1])
        mean = torch.as_tensor(getattr(loader.dataset, "mean", 0.0), device=device, dtype=mu.dtype)
        std = torch.as_tensor(getattr(loader.dataset, "std", 1.0), device=device, dtype=mu.dtype)
        for horizon in horizons:
            target = mu[batch_idx, start + horizon - 1]
            stats[f"latent_h{horizon}"].append(float(latent_l2(preds[:, horizon - 1], target)))
            pred_m = raw.backbone.decode(preds[:, horizon - 1].unsqueeze(1), raw.cfg.stride) * std + mean
            true_m = raw.backbone.decode(target.unsqueeze(1), raw.cfg.stride) * std + mean
            stats[f"mpjpe_h{horizon}"].append(float(mpjpe_mm(local_joints(pred_m, raw.cfg.num_joints), local_joints(true_m, raw.cfg.num_joints))))
    raw.train()
    return {key: float(np.mean(vals)) if vals else float("nan") for key, vals in stats.items()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--resume")
    parser.add_argument("--init-from")
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--max-samples", type=int)
    args = parser.parse_args()
    config = yaml.safe_load(open(args.config, encoding="utf-8"))
    if args.max_steps:
        config["training"]["max_steps"] = args.max_steps
    data_cfg, train_cfg, gala_cfg, world_cfg = parse_configs(config)
    distributed, local_rank, _ = distributed_setup()
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda", local_rank) if use_cuda else torch.device("cpu")
    seed = config["seed"] + local_rank
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    _, loader, sampler = _build_loader(
        data_cfg, data_cfg.get("split", "train"), train_cfg, distributed, config["seed"],
        shuffle=True, mode="train", hash_text=False, max_samples=args.max_samples or 0,
    )
    init_from = args.init_from or train_cfg.get("init_from")
    model = build_world_model(gala_cfg, world_cfg, init_from, device)
    if distributed:
        from torch.nn.parallel import DistributedDataParallel

        model = DistributedDataParallel(model, device_ids=[local_rank], find_unused_parameters=True)
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=train_cfg["learning_rate"], weight_decay=train_cfg["weight_decay"])
    max_steps = int(train_cfg.get("max_steps") or 0)
    grad_accum = max(int(train_cfg.get("grad_accum", 1)), 1)
    steps_per_epoch = max((len(loader) + grad_accum - 1) // grad_accum, 1)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(max_steps, int(train_cfg["epochs"]) * steps_per_epoch),
    )
    manager = CheckpointManager(train_cfg["checkpoint_dir"], mode="min")
    writer = SummaryWriter(train_cfg["log_dir"]) if local_rank == 0 and SummaryWriter is not None else None
    start_epoch = global_step = optimizer_step = 0
    if args.resume:
        state = manager.resume(model, optimizer, args.resume, scheduler=scheduler, map_location=device)
        start_epoch, global_step = state["epoch"] + 1, state["step"]
        optimizer_step = int(state.get("optimizer_step", global_step))

    val_loader = None
    if local_rank == 0 and int(train_cfg.get("eval_every_steps") or 0) > 0:
        _, val_loader, _ = _build_loader(
            data_cfg, "val", {**train_cfg, "batch_size_per_gpu": 16, "num_workers": 0},
            distributed=False, seed=config["seed"], shuffle=False, mode="eval",
            hash_text=False, max_samples=int(train_cfg.get("eval_max_samples", 128)),
        )

    use_amp = use_cuda and train_cfg.get("precision") in {"bf16", "fp16"}
    amp_dtype = torch.bfloat16 if train_cfg.get("precision") == "bf16" else torch.float16
    eval_every = int(train_cfg.get("eval_every_steps") or 0)
    stop = False
    for epoch in range(start_epoch, int(train_cfg["epochs"])):
        if sampler:
            sampler.set_epoch(epoch)
        model.train()
        sums, samples = {}, 0
        optimizer.zero_grad(set_to_none=True)
        for step, batch in enumerate(loader, start=1):
            motion = batch["motion"].to(device, non_blocking=True)
            lengths = batch["lengths"].to(device, non_blocking=True)
            with torch.autocast("cuda", dtype=amp_dtype, enabled=use_amp):
                losses = model(motion, lengths)
                scaled = losses["total"] / grad_accum
            scaled.backward()
            if step % grad_accum == 0 or step == len(loader):
                torch.nn.utils.clip_grad_norm_(model.parameters(), train_cfg["gradient_clip"])
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()
                optimizer_step += 1
            count = motion.shape[0]
            samples += count
            for name, value in losses.items():
                sums[name] = sums.get(name, 0.0) + value.detach().float().item() * count
                if writer:
                    writer.add_scalar(f"train_step/{name}", value.detach().float().item(), global_step)
            global_step += 1
            if local_rank == 0 and val_loader is not None and eval_every and optimizer_step > 0:
                if optimizer_step % eval_every == 0 and (step % grad_accum == 0 or step == len(loader)):
                    metrics = quick_val(model, val_loader, device)
                    print(
                        f"val step={optimizer_step} latent@1={metrics['latent_h1']:.4f} "
                        f"latent@8={metrics['latent_h8']:.4f} mpjpe@8={metrics['mpjpe_h8']:.2f}",
                        flush=True,
                    )
                    if writer:
                        for key, value in metrics.items():
                            writer.add_scalar(f"val/{key}", value, optimizer_step)
                    manager.save(
                        model, optimizer, epoch, global_step, metrics.get("mpjpe_h8", losses["total"].item()),
                        {"gala": asdict(gala_cfg), "world": asdict(world_cfg), **config, "optimizer_step": optimizer_step},
                        scheduler=scheduler,
                    )
            if max_steps and optimizer_step >= max_steps:
                stop = True
                break
        epoch_loss = sums["total"] / max(samples, 1)
        if local_rank == 0:
            manager.save(
                model, optimizer, epoch, global_step, epoch_loss,
                {"gala": asdict(gala_cfg), "world": asdict(world_cfg), **config, "optimizer_step": optimizer_step},
                scheduler=scheduler, update_best=val_loader is None,
            )
            print(f"epoch={epoch:03d} loss={epoch_loss:.4f} steps={global_step} opt_steps={optimizer_step}", flush=True)
        if stop:
            break
    if writer:
        writer.close()
    if distributed:
        import torch.distributed as dist

        dist.destroy_process_group()


if __name__ == "__main__":
    main()
