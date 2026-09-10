#!/usr/bin/env python3
"""Evaluate a GALA checkpoint with the Guo HumanML3D / KIT-ML protocol.

Official numbers are comparable to GENMO Table 4 / MDM eval_humanml.py:
R-Precision on batches of 32, FID, MM Dist, Diversity, repeated 20 times.
Internal GALA alignment is only a training monitor.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import fields
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from datasets.humanml import HumanMLDataset, collate_motion_text
from evaluation.guo_evaluator import (
    GuoEvaluator,
    evaluator_ready,
    install_mdm_eval_stats,
    official_replication_metrics,
    summarize_replications,
)
from evaluation.metrics import diversity, frechet_distance, r_precision
from models.gala_motion import GALAMotion, GALAMotionConfig
from trainers.train import encode_batch_text

GENMO_HUMANML3D = {
    "source": "GENMO ICCV 2025 Table 4",
    "R@3": 0.632,
    "FID": 0.216,
    "MM Dist": 3.466,
    "Diversity": 11.342,
}
MDM_KITML = {
    "source": "MDM / HumanML3D-protocol literature (KIT-ML)",
    "R@3": 0.396,
    "FID": 0.497,
    "MM Dist": 9.191,
    "Diversity": 10.85,
}
GALA_HUMANML3D_TARGET = {"R@3": 0.70, "FID": 0.20, "Diversity": (9.0, 12.0)}
GALA_KITML_FLOOR = {"R@3": 0.45, "FID": 0.45, "Diversity": 10.0}


def load_model(config, checkpoint, device):
    data_cfg, model_cfg = config["dataset"], dict(config["model"])
    model_cfg["train_stage"] = "joint"
    allowed = {item.name for item in fields(GALAMotionConfig)}
    model_kwargs = {key: value for key, value in model_cfg.items() if key in allowed}
    cfg = GALAMotionConfig(
        motion_dim=data_cfg["motion_dim"],
        num_joints=data_cfg["num_joints"],
        max_frames=data_cfg["max_frames"],
        **model_kwargs,
    )
    model = GALAMotion(cfg).to(device)
    state = torch.load(checkpoint, map_location=device, weights_only=False)
    raw = state["model"] if isinstance(state, dict) and "model" in state else state
    model.load_state_dict(raw, strict=False)
    ema_state = state.get("ema") if isinstance(state, dict) else None
    if ema_state and ema_state.get("shadow"):
        from trainers.ema import ModelEMA

        ema = ModelEMA(model, float(ema_state.get("decay") or 0.9999))
        ema.load_state_dict(ema_state)
        ema.apply(model)
        print("loaded EMA weights for evaluation", flush=True)
    model.eval()
    return model


def compare(metrics, baseline, higher_better=("R@1", "R@2", "R@3")):
    report = {}
    for key, base in baseline.items():
        if key == "source":
            continue
        value = metrics.get(key)
        if value is None:
            continue
        if key == "Diversity":
            better = abs(value - base) <= 2.0
        else:
            better = (value >= base) if key in higher_better else (value <= base)
        report[key] = {"ours": value, "baseline": base, "beats_or_matches": bool(better)}
    return report


def beats_target(metrics, experiment_name):
    if "humanml" in experiment_name:
        lo, hi = GALA_HUMANML3D_TARGET["Diversity"]
        return (
            metrics.get("R@3", 0) >= GALA_HUMANML3D_TARGET["R@3"]
            and metrics.get("FID", 1e9) <= GALA_HUMANML3D_TARGET["FID"]
            and lo <= metrics.get("Diversity", 0) <= hi
        )
    return (
        metrics.get("R@3", 0) >= GALA_KITML_FLOOR["R@3"]
        and metrics.get("FID", 1e9) <= GALA_KITML_FLOOR["FID"]
    )


@torch.no_grad()
def collect_official(model, loader, device, clip_encoder, guo, generate, steps, guidance, dataset):
    real_motion, gen_motion, texts = [], [], []
    mean, std = dataset.mean, dataset.std
    for batch in loader:
        motion = batch["motion"].to(device)
        lengths = batch["lengths"].to(device)
        text, text_mask = encode_batch_text(batch, device, clip_encoder)
        sampled = (
            model.sample(text, text_mask, lengths, steps=steps, guidance_scale=guidance)
            if generate
            else motion
        )
        real_eval, real_len = guo.prepare_motion(motion, lengths, mean, std)
        gen_eval, gen_len = guo.prepare_motion(sampled, lengths, mean, std)
        text_emb, gen_emb = guo.embed_batch(gen_eval, gen_len, batch["tokens"])
        real_emb = guo.embed_motion(real_eval, real_len)
        real_motion.append(real_emb)
        gen_motion.append(gen_emb)
        texts.append(text_emb)
    return np.concatenate(real_motion), np.concatenate(gen_motion), np.concatenate(texts)


@torch.no_grad()
def collect_internal(model, loader, device, clip_encoder, generate, steps, guidance):
    real_motion, gen_motion, texts, pairs = [], [], [], []
    for batch in loader:
        motion = batch["motion"].to(device)
        lengths = batch["lengths"].to(device)
        text, text_mask = encode_batch_text(batch, device, clip_encoder)
        real = model.encode_motion_embedding(motion, lengths)
        sampled = (
            model.sample(text, text_mask, lengths, steps=steps, guidance_scale=guidance)
            if generate
            else motion
        )
        generated = model.encode_motion_embedding(sampled, lengths)
        dummy_z = torch.zeros(text.shape[0], 1, model.cfg.latent_dim, device=device)
        dummy_mask = torch.ones(text.shape[0], 1, device=device, dtype=torch.bool)
        text_tokens = model.encode_text(text, text_mask)
        text_emb, _ = model.alignment.embeddings(text_tokens, text_mask, dummy_z, dummy_mask)
        real_motion.append(real.cpu().numpy())
        gen_motion.append(generated.cpu().numpy())
        texts.append(text_emb.cpu().numpy())
        pairs.append(np.linalg.norm(text_emb.cpu().numpy() - generated.cpu().numpy(), axis=-1))
    return (
        np.concatenate(real_motion),
        np.concatenate(gen_motion),
        np.concatenate(texts),
        np.concatenate(pairs),
    )


def make_loader(config, split, batch_size, max_samples, hash_text, mode, shuffle):
    dataset = HumanMLDataset(
        config["dataset"]["root"],
        split,
        max_frames=config["dataset"]["max_frames"],
        hash_text=hash_text,
        mode=mode,
    )
    if max_samples:
        dataset.items = dataset.items[:max_samples]
        dataset.ids = dataset.ids[:max_samples]
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=mode == "eval",
        collate_fn=collate_motion_text,
    )
    return dataset, loader


def run_official(model, config, args, device, clip_encoder, guo, generate):
    runs = []
    for replication in range(args.replication_times):
        dataset, loader = make_loader(
            config, args.split, args.batch_size, args.max_samples,
            hash_text=False, mode="eval", shuffle=True,
        )
        real, generated, texts = collect_official(
            model, loader, device, clip_encoder, guo,
            generate=generate, steps=args.steps, guidance=args.guidance, dataset=dataset,
        )
        metrics = official_replication_metrics(real, generated, texts, batch_size=args.batch_size)
        metrics["replication"] = replication
        print(json.dumps(metrics), flush=True)
        runs.append(metrics)
    summary = summarize_replications(runs)
    summary["replications"] = runs
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--guidance", type=float, default=2.5)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--replication-times", type=int, default=20)
    parser.add_argument("--no-generate", action="store_true")
    parser.add_argument("--protocol", choices=["auto", "official", "internal"], default="auto")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    config = yaml.safe_load(open(args.config, encoding="utf-8"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(config, args.checkpoint, device)
    use_clip = config["model"].get("text_encoder_type") == "clip"
    clip_encoder = None
    if use_clip:
        from models.clip_text import FrozenCLIPTextEncoder

        clip_encoder = FrozenCLIPTextEncoder().to(device)

    name = str(config.get("experiment", "")).lower()
    baseline = GENMO_HUMANML3D if "humanml" in name else MDM_KITML
    use_official = args.protocol == "official" or (args.protocol == "auto" and evaluator_ready())
    if args.protocol == "official" and not evaluator_ready():
        raise FileNotFoundError("Guo evaluator files are missing under checkpoints/t2m_evaluators")

    if use_official:
        if args.batch_size != 32:
            print("warning: official R-Precision expects batch_size=32", flush=True)
        install_mdm_eval_stats()
        dataset_name = "humanml" if "humanml" in name else "kit"
        guo = GuoEvaluator(dataset_name, device)
        metrics = run_official(model, config, args, device, clip_encoder, guo, generate=not args.no_generate)
    else:
        dataset, loader = make_loader(
            config, args.split, args.batch_size, args.max_samples,
            hash_text=not use_clip, mode="train", shuffle=False,
        )
        real, generated, texts, pair_dist = collect_internal(
            model, loader, device, clip_encoder,
            generate=not args.no_generate, steps=args.steps, guidance=args.guidance,
        )
        ranking = r_precision(texts, generated)
        metrics = {
            "R@1": ranking["top_1"],
            "R@2": ranking["top_2"],
            "R@3": ranking["top_3"],
            "FID": frechet_distance(real, generated),
            "MM Dist": float(pair_dist.mean()),
            "Diversity": diversity(generated, pairs=min(300, max(len(generated) // 2, 1))),
            "protocol": "internal_gala_alignment",
            "num_samples": int(len(generated)),
        }
    metrics["generate"] = not args.no_generate
    metrics["baseline"] = baseline
    metrics["comparison"] = compare(metrics, baseline)
    metrics["beats_gala_target"] = beats_target(metrics, name)
    print(json.dumps({key: value for key, value in metrics.items() if key != "replications"}, indent=2))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(metrics, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
