#!/usr/bin/env python3
"""Freeze the 320-clip validation manifest used by physical screening."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from datasets.humanml import HumanMLDataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/gala_humanml3d_flow_physical.yaml")
    parser.add_argument("--split", default="val")
    parser.add_argument("--max-samples", type=int, default=320)
    parser.add_argument("--output", type=Path, default=Path("outputs/physical/val_manifest.json"))
    args = parser.parse_args()
    config = yaml.safe_load(open(args.config, encoding="utf-8"))
    dataset = HumanMLDataset(
        config["dataset"]["root"], args.split,
        max_frames=config["dataset"]["max_frames"], mode="eval_fixed",
    )
    items = dataset.items[: args.max_samples]
    payload = {
        "split": args.split,
        "mode": "eval_fixed",
        "seed": config.get("seed", 3407),
        "max_samples": len(items),
        "nfe": 20,
        "guidance": 2.5,
        "items": [
            {
                "id": item["id"],
                "start": item["start"],
                "end": item["end"],
                "caption": item["captions"][0]["caption"],
            }
            for item in items
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"wrote {args.output} n={len(items)}", flush=True)


if __name__ == "__main__":
    main()
