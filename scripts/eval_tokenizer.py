#!/usr/bin/env python3
"""Tokenizer ablation: reconstruction FID / R@3 plus kinematic errors."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import yaml

from evaluation.guo_evaluator import GuoEvaluator, evaluator_ready, install_mdm_eval_stats, official_replication_metrics
from evaluation.kinematics import reconstruction_errors
from models.gala_motion import sequence_mask
from scripts.evaluate_t2m import load_model, make_loader


@torch.no_grad()
def reconstruct_batch(model, motion, lengths):
    _, mu, _, _, _, _, _ = model.encode(motion, lengths)
    recon = model.decode(mu, motion.shape[1])
    return recon * sequence_mask(lengths, motion.shape[1]).unsqueeze(-1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", default="val")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    config = yaml.safe_load(open(args.config, encoding="utf-8"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(config, args.checkpoint, device)
    dataset, loader = make_loader(
        config, args.split, args.batch_size, args.max_samples,
        hash_text=False, mode="eval", shuffle=False,
    )
    kin_sums, n_batches = {}, 0
    if not evaluator_ready():
        raise FileNotFoundError("Guo evaluator files are missing under checkpoints/t2m_evaluators")
    install_mdm_eval_stats()
    name = str(config.get("experiment", "")).lower()
    guo = GuoEvaluator("humanml" if "humanml" in name or config["dataset"]["motion_dim"] == 263 else "kit", device)
    reals, gens, texts = [], [], []
    for batch in loader:
        motion = batch["motion"].to(device)
        lengths = batch["lengths"].to(device)
        recon = reconstruct_batch(model, motion, lengths)
        errors = reconstruction_errors(model, motion, lengths)
        for key, value in errors.items():
            kin_sums[key] = kin_sums.get(key, 0.0) + value
        n_batches += 1
        real_eval, real_len = guo.prepare_motion(motion, lengths, dataset.mean, dataset.std)
        gen_eval, gen_len = guo.prepare_motion(recon, lengths, dataset.mean, dataset.std)
        text_emb, gen_emb = guo.embed_batch(gen_eval, gen_len, batch["tokens"])
        reals.append(guo.embed_motion(real_eval, real_len))
        gens.append(gen_emb)
        texts.append(text_emb)
    metrics = official_replication_metrics(np.concatenate(reals), np.concatenate(gens), np.concatenate(texts))
    metrics.update({key: total / max(n_batches, 1) for key, total in kin_sums.items()})
    metrics["split"] = args.split
    metrics["num_batches"] = n_batches
    print(json.dumps(metrics, indent=2))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(metrics, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
