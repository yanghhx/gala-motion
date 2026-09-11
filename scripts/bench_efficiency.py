#!/usr/bin/env python3
"""GALA sampling efficiency: params, NFE, latency, FPS on one prompt-length batch."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
import yaml

from scripts.evaluate_t2m import load_model


def count_params(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return int(total), int(trainable)


@torch.no_grad()
def bench(model, device, batch_size, frames, text_len, steps, guidance, warmup, repeats):
    model.eval()
    text = torch.randn(batch_size, text_len, model.cfg.text_dim, device=device)
    text_mask = torch.ones(batch_size, text_len, device=device, dtype=torch.bool)
    lengths = torch.full((batch_size,), frames, device=device, dtype=torch.long)
    for _ in range(warmup):
        model.sample(text, text_mask, lengths, steps=steps, guidance_scale=guidance)
        if device.type == "cuda":
            torch.cuda.synchronize()
    times = []
    for _ in range(repeats):
        if device.type == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()
        motion = model.sample(text, text_mask, lengths, steps=steps, guidance_scale=guidance)
        if device.type == "cuda":
            torch.cuda.synchronize()
        times.append(time.perf_counter() - start)
    latency = sum(times) / len(times)
    return {
        "latency_s": latency,
        "latency_ms": 1000.0 * latency,
        "fps": batch_size / latency,
        "ms_per_clip": 1000.0 * latency / batch_size,
        "shape": list(motion.shape),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--frames", type=int, default=196)
    parser.add_argument("--text-len", type=int, default=77)
    parser.add_argument("--steps", type=int, nargs="+", default=[10, 20, 50])
    parser.add_argument("--guidance", type=float, default=2.5)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    config = yaml.safe_load(open(args.config, encoding="utf-8"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(config, args.checkpoint, device)
    total, trainable = count_params(model)
    report = {
        "device": str(device),
        "params": total,
        "params_m": round(total / 1e6, 2),
        "trainable": trainable,
        "batch_size": args.batch_size,
        "frames": args.frames,
        "guidance": args.guidance,
        "settings": [],
    }
    for steps in args.steps:
        stats = bench(
            model, device, args.batch_size, args.frames, args.text_len,
            steps, args.guidance, args.warmup, args.repeats,
        )
        stats["nfe"] = steps
        report["settings"].append(stats)
        print(json.dumps({"nfe": steps, **stats, "params_m": report["params_m"]}), flush=True)
    print(json.dumps({k: v for k, v in report.items() if k != "settings"}, indent=2))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
