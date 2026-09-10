import argparse
import os
import random
from dataclasses import asdict, fields

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

import numpy as np
import torch
import torch.distributed as dist
import yaml
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler

if not hasattr(np, "bool8"):
    np.bool8 = np.bool_

try:
    from torch.utils.tensorboard import SummaryWriter
except Exception:
    SummaryWriter = None

from datasets.humanml import HumanMLDataset, collate_motion_text
from models.gala_motion import GALAMotion, GALAMotionConfig
from trainers.checkpoint import CheckpointManager
from trainers.ema import ModelEMA


def encode_batch_text(batch, device, clip_encoder):
    if clip_encoder is not None:
        return clip_encoder.encode(batch["captions"], device)
    return batch["text"].to(device, non_blocking=True), batch["text_mask"].to(device, non_blocking=True)


def distributed_setup():
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    distributed = world_size > 1
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if distributed:
        dist.init_process_group("nccl")
        torch.cuda.set_device(local_rank)
    return distributed, local_rank, world_size


def _build_loader(data_cfg, split, train_cfg, distributed, seed, shuffle, mode, hash_text, max_samples=0, drop_last=False):
    dataset = HumanMLDataset(
        data_cfg["root"], split, max_frames=data_cfg["max_frames"], hash_text=hash_text, mode=mode,
    )
    if max_samples:
        dataset.items = dataset.items[:max_samples]
        dataset.ids = dataset.ids[:max_samples]
    sampler = DistributedSampler(dataset, shuffle=shuffle, seed=seed) if distributed and shuffle else None
    loader = DataLoader(
        dataset,
        batch_size=train_cfg["batch_size_per_gpu"],
        shuffle=sampler is None and shuffle,
        sampler=sampler,
        num_workers=train_cfg["num_workers"] if torch.cuda.is_available() else 0,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=torch.cuda.is_available() and train_cfg["num_workers"] > 0,
        collate_fn=collate_motion_text,
        drop_last=drop_last,
    )
    return dataset, loader, sampler


@torch.no_grad()
def official_val_fid(model, loader, device, clip_encoder, guo, dataset, steps=20, guidance=2.5):
    from evaluation.guo_evaluator import official_replication_metrics

    raw = model.module if hasattr(model, "module") else model
    raw.eval()
    reals, gens, texts = [], [], []
    for batch in loader:
        motion = batch["motion"].to(device)
        lengths = batch["lengths"].to(device)
        text, text_mask = encode_batch_text(batch, device, clip_encoder)
        sampled = raw.sample(text, text_mask, lengths, steps=steps, guidance_scale=guidance)
        real_eval, real_len = guo.prepare_motion(motion, lengths, dataset.mean, dataset.std)
        gen_eval, gen_len = guo.prepare_motion(sampled, lengths, dataset.mean, dataset.std)
        text_emb, gen_emb = guo.embed_batch(gen_eval, gen_len, batch["tokens"])
        reals.append(guo.embed_motion(real_eval, real_len))
        gens.append(gen_emb)
        texts.append(text_emb)
    raw.train()
    if not gens:
        return None
    metrics = official_replication_metrics(np.concatenate(reals), np.concatenate(gens), np.concatenate(texts))
    return metrics["FID"], metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--resume")
    parser.add_argument("--init-from")
    parser.add_argument("--stage", choices=["vae", "flow", "joint"])
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--max-samples", type=int)
    args = parser.parse_args()
    config = yaml.safe_load(open(args.config, encoding="utf-8"))
    if args.epochs:
        config["training"]["epochs"] = args.epochs
    if args.max_steps:
        config["training"]["max_steps"] = args.max_steps
    if args.stage:
        config["model"]["train_stage"] = args.stage
    distributed, local_rank, world_size = distributed_setup()
    use_cuda = torch.cuda.is_available()
    if distributed and not use_cuda:
        raise RuntimeError("Distributed training requires CUDA")
    device = torch.device("cuda", local_rank) if use_cuda else torch.device("cpu")
    seed = config["seed"] + local_rank
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)

    data_cfg, model_cfg, train_cfg = config["dataset"], config["model"], config["training"]
    allowed = {item.name for item in fields(GALAMotionConfig)}
    model_kwargs = {key: value for key, value in model_cfg.items() if key in allowed}
    cfg = GALAMotionConfig(
        motion_dim=data_cfg["motion_dim"], num_joints=data_cfg["num_joints"],
        max_frames=data_cfg["max_frames"], **model_kwargs,
    )
    use_clip = cfg.text_encoder_type == "clip"
    hash_text = not use_clip
    dataset, loader, sampler = _build_loader(
        data_cfg, data_cfg.get("split", "train"), train_cfg, distributed, config["seed"],
        shuffle=True, mode="train", hash_text=hash_text, max_samples=args.max_samples or 0,
    )
    model = GALAMotion(cfg).to(device)
    clip_encoder = None
    if use_clip and cfg.train_stage != "vae":
        from models.clip_text import FrozenCLIPTextEncoder

        clip_encoder = FrozenCLIPTextEncoder().to(device)
    if distributed:
        model = DistributedDataParallel(model, device_ids=[local_rank], find_unused_parameters=True)
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=train_cfg["learning_rate"], weight_decay=train_cfg["weight_decay"])
    max_steps = int(train_cfg.get("max_steps") or 0)
    epoch_ceiling = int(train_cfg["epochs"])
    grad_accum = max(int(train_cfg.get("grad_accum", 1)), 1)
    steps_per_epoch = max((len(loader) + grad_accum - 1) // grad_accum, 1)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(max_steps, epoch_ceiling * steps_per_epoch),
    )
    manager = CheckpointManager(train_cfg["checkpoint_dir"], mode="min")
    writer = SummaryWriter(train_cfg["log_dir"]) if local_rank == 0 and SummaryWriter is not None else None
    start_epoch = global_step = optimizer_step = 0
    raw_model = model.module if hasattr(model, "module") else model
    if args.init_from:
        state = torch.load(args.init_from, map_location=device, weights_only=False)
        raw = state["model"] if isinstance(state, dict) and "model" in state else state
        raw_model.load_state_dict(raw, strict=False)
        raw_model.cfg.train_stage = cfg.train_stage
        raw_model.apply_train_stage()
        if local_rank == 0:
            print(f"initialized weights from {args.init_from} stage={cfg.train_stage}", flush=True)
    ema_decay = float(train_cfg.get("ema_decay") or 0)
    ema = ModelEMA(raw_model, ema_decay) if ema_decay > 0 and cfg.train_stage != "vae" else None
    if args.resume:
        state = manager.resume(
            model, optimizer, args.resume, scheduler=scheduler, ema=ema, map_location=device,
        )
        start_epoch, global_step = state["epoch"] + 1, state["step"]
        optimizer_step = int(state.get("optimizer_step", global_step))

    use_amp = use_cuda and train_cfg.get("precision") in {"bf16", "fp16"}
    amp_dtype = torch.bfloat16 if train_cfg.get("precision") == "bf16" else torch.float16
    eval_every = int(train_cfg.get("eval_every_steps", 0) or 0)
    patience = int(train_cfg.get("early_stop_fid_patience", 0) or 0)
    eval_max_samples = int(train_cfg.get("eval_max_samples", 256))
    eval_steps = int(train_cfg.get("eval_steps", 20) or 20)
    eval_guidance = float(train_cfg.get("eval_guidance", 2.5))
    guo = val_loader = val_dataset = None
    fid_history = []
    if local_rank == 0 and cfg.train_stage == "flow" and eval_every > 0:
        from evaluation.guo_evaluator import GuoEvaluator, evaluator_ready, install_mdm_eval_stats

        if evaluator_ready():
            install_mdm_eval_stats()
            dataset_name = "humanml" if data_cfg["motion_dim"] == 263 else "kit"
            guo = GuoEvaluator(dataset_name, device)
            val_dataset, val_loader, _ = _build_loader(
                data_cfg, "val", {**train_cfg, "batch_size_per_gpu": 32, "num_workers": 0},
                distributed=False, seed=config["seed"], shuffle=True, mode="eval",
                hash_text=False, max_samples=eval_max_samples, drop_last=True,
            )
            print(f"official val FID every {eval_every} optimizer steps, n={len(val_dataset)}", flush=True)

    stop = False
    for epoch in range(start_epoch, epoch_ceiling):
        if sampler:
            sampler.set_epoch(epoch)
        model.train()
        sums, samples = {}, 0
        optimizer.zero_grad(set_to_none=True)
        for step, batch in enumerate(loader, start=1):
            motion = batch["motion"].to(device, non_blocking=True)
            lengths = batch["lengths"].to(device, non_blocking=True)
            text, text_mask = encode_batch_text(batch, device, clip_encoder)
            with torch.autocast("cuda", dtype=amp_dtype, enabled=use_amp):
                losses = model(motion, lengths, text, text_mask)
                scaled = losses["total"] / grad_accum
            scaled.backward()
            if step % grad_accum == 0 or step == len(loader):
                torch.nn.utils.clip_grad_norm_(model.parameters(), train_cfg["gradient_clip"])
                optimizer.step()
                if ema is not None:
                    ema.update(raw_model)
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()
                optimizer_step += 1
            count = batch["motion"].shape[0]
            samples += count
            for name, value in losses.items():
                sums[name] = sums.get(name, 0.0) + value.detach().float().item() * count
                if writer:
                    writer.add_scalar(f"train_step/{name}", value.detach().float().item(), global_step)
            global_step += 1
            if local_rank == 0 and guo is not None and optimizer_step > 0 and optimizer_step % eval_every == 0 and (
                step % grad_accum == 0 or step == len(loader)
            ):
                if ema is not None:
                    ema.apply(raw_model)
                fid_pack = official_val_fid(
                    model, val_loader, device, clip_encoder, guo, val_dataset,
                    steps=eval_steps, guidance=eval_guidance,
                )
                if ema is not None:
                    ema.restore(raw_model)
                if fid_pack is not None:
                    fid, metrics = fid_pack
                    fid_history.append(fid)
                    if writer:
                        writer.add_scalar("val/official_fid", fid, optimizer_step)
                        writer.add_scalar("val/official_r3", metrics["R@3"], optimizer_step)
                    print(
                        f"val_fid step={optimizer_step} FID={fid:.4f} R@3={metrics['R@3']:.4f} "
                        f"Diversity={metrics['Diversity']:.4f} steps={eval_steps} cfg={eval_guidance}",
                        flush=True,
                    )
                    manager.save(
                        model, optimizer, epoch, global_step, fid,
                        {"model": asdict(cfg), **config, "optimizer_step": optimizer_step},
                        scheduler=scheduler, ema=ema,
                    )
                    if patience > 0 and len(fid_history) > patience:
                        recent = fid_history[-(patience + 1):]
                        if all(recent[i] < recent[i + 1] for i in range(len(recent) - 1)):
                            print(f"early stop: official FID rose for {patience} evals {recent}", flush=True)
                            stop = True
                            break
            if max_steps and optimizer_step >= max_steps:
                stop = True
                break
        epoch_loss = sums["total"] / max(samples, 1)
        if local_rank == 0:
            if writer:
                for name, total in sums.items():
                    writer.add_scalar(f"train_epoch/{name}", total / max(samples, 1), epoch)
                writer.add_scalar("train_epoch/lr", scheduler.get_last_lr()[0], epoch)
            update_best = guo is None
            metric = epoch_loss if update_best else (fid_history[-1] if fid_history else epoch_loss)
            manager.save(
                model, optimizer, epoch, global_step, metric,
                {"model": asdict(cfg), **config, "optimizer_step": optimizer_step},
                scheduler=scheduler, ema=ema, update_best=update_best or bool(fid_history),
            )
            print(
                f"epoch={epoch:03d} loss={epoch_loss:.4f} steps={global_step} opt_steps={optimizer_step}",
                flush=True,
            )
        if stop:
            break
    if writer:
        writer.close()
    if distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
