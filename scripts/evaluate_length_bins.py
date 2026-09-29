#!/usr/bin/env python
"""Length-wise robustness evaluation: split HumanML3D test by sequence length and evaluate FID/R@3/MM."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from scripts.evaluate_t2m import load_model, encode_batch_text
from evaluation.guo_evaluator import GuoEvaluator, official_replication_metrics
from datasets.humanml import HumanMLDataset, collate_motion_text


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/gala_humanml3d_flow_distinct.yaml")
    parser.add_argument("--checkpoint", default="checkpoints/gala_humanml3d_flow_distinct/best.pt")
    parser.add_argument("--output", type=Path, default=Path("outputs/length_analysis/length_bins.json"))
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--guidance", type=float, default=2.5)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = yaml.safe_load(open(args.config))

    # Load full test dataset to get lengths
    dataset = HumanMLDataset(
        config["dataset"]["root"], "test", max_frames=config["dataset"]["max_frames"], mode="eval_fixed"
    )
    print(f"Loaded {len(dataset)} test clips")

    # Get raw lengths from npy files
    motion_dir = Path(config["dataset"]["root"]) / "new_joint_vecs"
    lengths_map = {}
    for item in dataset.items:
        mid = item["id"]
        motion_path = motion_dir / f"{mid}.npy"
        if motion_path.exists():
            motion = np.load(motion_path)
            lengths_map[mid] = len(motion)

    # Bins: Short < 60, Medium 60-120, Long > 120
    bin_ids = {"short": [], "medium": [], "long": []}
    for mid, l in lengths_map.items():
        if l < 60:
            bin_ids["short"].append(mid)
        elif l <= 120:
            bin_ids["medium"].append(mid)
        else:
            bin_ids["long"].append(mid)

    print("Bin distribution:")
    for name, ids in bin_ids.items():
        print(f"  {name}: {len(ids)} clips")

    # Load model
    model = load_model(config, args.checkpoint, device)
    model.eval()
    model.set_motion_stats(dataset.mean, dataset.std)

    from models.clip_text import FrozenCLIPTextEncoder
    clip_encoder = FrozenCLIPTextEncoder().to(device)
    guo = GuoEvaluator("humanml", device)

    results = {}
    for bin_name, ids in bin_ids.items():
        n = len(ids)
        if n < 32:
            results[bin_name] = {"count": n, "note": "fewer than 32 samples, R-Precision not stable"}
            print(f"  {bin_name}: {n} clips (too few for R-Precision)")
            continue

        # Filter dataset items to this bin
        id_set = set(ids)
        bin_dataset = HumanMLDataset(
            config["dataset"]["root"], "test", max_frames=config["dataset"]["max_frames"], mode="eval_fixed"
        )
        bin_dataset.items = [item for item in bin_dataset.items if item["id"] in id_set]
        bin_dataset.ids = [item["id"] for item in bin_dataset.items]
        print(f"  {bin_name}: evaluating {len(bin_dataset)} clips...")

        loader = DataLoader(bin_dataset, batch_size=32, shuffle=False, drop_last=False, collate_fn=collate_motion_text)

        real_embs, gen_embs, text_embs = [], [], []
        for batch in loader:
            motion = batch["motion"].to(device)
            lengths = batch["lengths"].to(device)
            text, text_mask = encode_batch_text(batch, device, clip_encoder)
            with torch.no_grad():
                sampled = model.sample(text, text_mask, lengths, steps=args.steps, guidance_scale=args.guidance)
            real_eval, real_len = guo.prepare_motion(motion, lengths, dataset.mean, dataset.std)
            gen_eval, gen_len = guo.prepare_motion(sampled, lengths, dataset.mean, dataset.std)
            text_emb, gen_emb = guo.embed_batch(gen_eval, gen_len, batch["tokens"])
            real_emb = guo.embed_motion(real_eval, real_len)
            real_embs.append(real_emb)
            gen_embs.append(gen_emb)
            text_embs.append(text_emb)

        real_all = np.concatenate(real_embs)
        gen_all = np.concatenate(gen_embs)
        text_all = np.concatenate(text_embs)

        # Need at least 32 for R-Precision
        n_eval = (len(gen_all) // 32) * 32
        if n_eval < 32:
            results[bin_name] = {"count": n, "note": "fewer than 32 after embedding"}
            continue

        metrics = official_replication_metrics(
            real_all[:n_eval], gen_all[:n_eval], text_all[:n_eval], batch_size=32, diversity_times=min(300, n_eval)
        )
        results[bin_name] = {
            "count": n,
            "FID": float(metrics["FID"]),
            "R@3": float(metrics["R@3"]),
            "MM Dist": float(metrics["MM Dist"]),
            "Diversity": float(metrics["Diversity"]),
        }
        print(f"  {bin_name}: FID={metrics['FID']:.3f} R@3={metrics['R@3']:.3f} MM={metrics['MM Dist']:.3f}")

    results["total"] = len(lengths_map)
    results["bin_boundaries"] = {"short": "<60", "medium": "60-120", "long": ">120"}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2))
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
