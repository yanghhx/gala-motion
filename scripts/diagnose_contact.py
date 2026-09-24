#!/usr/bin/env python3
"""Diagnose Guo foot-contact channels before skating-loss training.

Contact is GT-contact-conditioned generated foot displacement, not a predicted
contact head. This script checks whether the last-4 HumanML3D channels are a
reliable stance mask, including:

- per-foot / heel-vs-toe contact ratio
- r_valid = Σ c 1(v < vmax) / Σ c
- contact-segment length distribution
- caption-keyword action bins
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import yaml

from datasets.humanml import HumanMLDataset
from models.gala_motion import sequence_mask
from models.motion_repr import (
    contact_from_features,
    contact_velocity_valid_ratio,
    recover_from_ric,
    skating_error,
)
from models.skeleton_graph import foot_names, foot_joints

ACTION_BINS = {
    "walk": ("walk", "walking", "stroll", "stride"),
    "run": ("run", "running", "jog", "jogging", "sprint"),
    "jump": ("jump", "jumping", "hop", "leap"),
    "dance": ("dance", "dancing", "waltz", "ballet"),
    "sit_stand": ("sit", "sitting", "stand", "standing", "stand up", "sit down"),
    "turn": ("turn", "turning", "spin", "rotate"),
    "kick": ("kick", "kicking"),
    "wave": ("wave", "waving"),
}


def classify_caption(caption: str) -> str:
    text = caption.lower()
    for name, keys in ACTION_BINS.items():
        if any(key in text for key in keys):
            return name
    return "other"


def run_lengths(flags: np.ndarray) -> list[int]:
    lengths = []
    current = 0
    for value in flags.tolist():
        if value:
            current += 1
        elif current:
            lengths.append(current)
            current = 0
    if current:
        lengths.append(current)
    return lengths


def summarize(values: list[float]) -> dict:
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return {"mean": None, "p50": None, "p90": None, "n": 0}
    return {
        "mean": float(arr.mean()),
        "p50": float(np.median(arr)),
        "p90": float(np.quantile(arr, 0.9)),
        "n": int(arr.size),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/gala_humanml3d_flow_physical.yaml")
    parser.add_argument("--split", default="val")
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--vmax", type=float, nargs="+", default=[0.01, 0.05])
    parser.add_argument("--output", type=Path, default=Path("outputs/physical/contact_diagnosis.json"))
    args = parser.parse_args()

    config = yaml.safe_load(open(args.config, encoding="utf-8"))
    data_cfg = config["dataset"]
    dataset = HumanMLDataset(data_cfg["root"], args.split, max_frames=data_cfg["max_frames"], mode="eval_fixed")
    if args.max_samples:
        dataset.items = dataset.items[: args.max_samples]
        dataset.ids = dataset.ids[: args.max_samples]
    num_joints = int(data_cfg["num_joints"])
    names = list(foot_names(num_joints))
    mean = torch.from_numpy(dataset.mean)
    std = torch.from_numpy(dataset.std)

    per_channel = {name: [] for name in names}
    r_valid = {str(v): [] for v in args.vmax}
    skate = []
    segments = {name: [] for name in names}
    actions = defaultdict(lambda: {"n": 0, "contact": [], "r_valid_0.01": []})
    n_clips = 0

    for index in range(len(dataset)):
        item = dataset[index]
        motion = item["motion"].unsqueeze(0)
        length = int(item["length"])
        mask = sequence_mask(torch.tensor([length]), motion.shape[1])
        denorm = motion * std + mean
        contact = contact_from_features(denorm)
        positions = recover_from_ric(denorm, num_joints)
        clip_contact = (contact[:, :-1] & mask[:, 1:, None] & mask[:, :-1, None]).float()
        for channel, name in enumerate(names):
            per_channel[name].append(float(clip_contact[0, :, channel].mean()))
            segments[name].extend(run_lengths(clip_contact[0, :, channel].bool().numpy()))
        for vmax in args.vmax:
            r_valid[str(vmax)].append(float(contact_velocity_valid_ratio(
                positions, contact, mask, num_joints, vmax,
            )))
        skate.append(float(skating_error(positions, contact, mask, num_joints)))
        bucket = classify_caption(item["caption"])
        actions[bucket]["n"] += 1
        actions[bucket]["contact"].append(float(clip_contact.mean()))
        actions[bucket]["r_valid_0.01"].append(r_valid["0.01"][-1] if "0.01" in r_valid else r_valid[str(args.vmax[0])][-1])
        n_clips += 1
        if (index + 1) % 200 == 0:
            print(f"scanned {index + 1}/{len(dataset)}", flush=True)

    left = [name for name in names if name.startswith("left_")]
    right = [name for name in names if name.startswith("right_")]
    heel = [name for name in names if "ankle" in name or "toe" in name]
    toe = [name for name in names if name.endswith("_foot")]
    report = {
        "split": args.split,
        "num_clips": n_clips,
        "foot_names": names,
        "foot_joints": list(foot_joints(num_joints)),
        "contact_ratio": {name: summarize(values) for name, values in per_channel.items()},
        "contact_ratio_left": summarize(sum((per_channel[name] for name in left), [])),
        "contact_ratio_right": summarize(sum((per_channel[name] for name in right), [])),
        "contact_ratio_heel_or_ball": summarize(sum((per_channel[name] for name in heel), [])),
        "contact_ratio_foot": summarize(sum((per_channel[name] for name in toe), [])),
        "r_valid": {vmax: summarize(values) for vmax, values in r_valid.items()},
        "gt_self_skating": summarize(skate),
        "contact_segment_frames": {name: summarize(values) for name, values in segments.items()},
        "action_bins": {
            name: {
                "n": stats["n"],
                "contact_ratio": summarize(stats["contact"]),
                "r_valid_0.01": summarize(stats["r_valid_0.01"]),
            }
            for name, stats in sorted(actions.items())
        },
        "paper_term": "GT-contact-conditioned generated foot displacement",
        "recommendation": (
            "Use gt_contact_feature for the main skating loss only if r_valid@0.01 "
            "is high. Otherwise prefer kinematic_threshold or raise vmax / filter."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({k: report[k] for k in (
        "num_clips", "contact_ratio", "r_valid", "gt_self_skating", "action_bins",
    )}, indent=2))
    print(f"wrote {args.output}", flush=True)


if __name__ == "__main__":
    main()
