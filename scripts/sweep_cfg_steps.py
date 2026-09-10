#!/usr/bin/env python3
"""Sweep CFG and ODE steps on val, plus VAE reconstruction FID."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from evaluation.guo_evaluator import (
    GuoEvaluator,
    evaluator_ready,
    install_mdm_eval_stats,
    official_replication_metrics,
)
from models.gala_motion import sequence_mask
from evaluate_t2m import collect_official, load_model, make_loader

os_environ_default = "https://hf-mirror.com"


def collect_recon(model, loader, device, guo, dataset):
    real_motion, gen_motion, texts = [], [], []
    mean, std = dataset.mean, dataset.std
    model.eval()
    amp = device.type == "cuda"
    for batch in loader:
        motion = batch["motion"].to(device)
        lengths = batch["lengths"].to(device)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
            _, mu, _, _, _, _, _ = model.encode(motion, lengths)
            recon = model.decode(mu.float(), motion.shape[1]) * sequence_mask(lengths, motion.shape[1]).unsqueeze(-1)
        real_eval, real_len = guo.prepare_motion(motion, lengths, mean, std)
        gen_eval, gen_len = guo.prepare_motion(recon.float(), lengths, mean, std)
        text_emb, gen_emb = guo.embed_batch(gen_eval, gen_len, batch["tokens"])
        real_motion.append(guo.embed_motion(real_eval, real_len))
        gen_motion.append(gen_emb)
        texts.append(text_emb)
        del motion, lengths, mu, recon, real_eval, gen_eval
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return np.concatenate(real_motion), np.concatenate(gen_motion), np.concatenate(texts)


def main():
    import os

    os.environ.setdefault("HF_ENDPOINT", os_environ_default)
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", default="val")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    config = yaml.safe_load(open(args.config, encoding="utf-8"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(config, args.checkpoint, device)
    from models.clip_text import FrozenCLIPTextEncoder

    clip_encoder = FrozenCLIPTextEncoder().to(device)
    if not evaluator_ready():
        raise FileNotFoundError("Guo evaluator is not ready")
    install_mdm_eval_stats()
    guo = GuoEvaluator("humanml" if config["dataset"]["motion_dim"] == 263 else "kit", device)
    dataset, loader = make_loader(
        config, args.split, args.batch_size, args.max_samples,
        hash_text=False, mode="eval", shuffle=True,
    )
    print(f"sweep split={args.split} n={len(dataset)} batches={len(loader)}", flush=True)

    recon_dataset, recon_loader = make_loader(
        config, args.split, 8, args.max_samples,
        hash_text=False, mode="eval", shuffle=False,
    )
    real, recon, texts = collect_recon(model, recon_loader, device, guo, recon_dataset)
    recon_metrics = official_replication_metrics(real, recon, texts, batch_size=args.batch_size)
    recon_metrics["kind"] = "vae_recon"
    print(json.dumps(recon_metrics), flush=True)
    del real, recon, texts, recon_loader
    if device.type == "cuda":
        torch.cuda.empty_cache()

    grid = []
    for steps in (10, 20, 50):
        for guidance in (1.0, 1.5, 2.0, 2.5):
            dataset, loader = make_loader(
                config, args.split, args.batch_size, args.max_samples,
                hash_text=False, mode="eval", shuffle=True,
            )
            real, generated, texts = collect_official(
                model, loader, device, clip_encoder, guo,
                generate=True, steps=steps, guidance=guidance, dataset=dataset,
            )
            metrics = official_replication_metrics(real, generated, texts, batch_size=args.batch_size)
            metrics.update(kind="sample", steps=steps, guidance=guidance)
            print(json.dumps(metrics), flush=True)
            grid.append(metrics)
            del real, generated, texts
            if device.type == "cuda":
                torch.cuda.empty_cache()

    ranked = sorted(grid, key=lambda item: (item["FID"], -item["R@3"]))
    summary = {
        "checkpoint": str(args.checkpoint),
        "split": args.split,
        "vae_recon": recon_metrics,
        "grid": grid,
        "best_by_fid": ranked[0] if ranked else None,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({"best_by_fid": ranked[0]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
