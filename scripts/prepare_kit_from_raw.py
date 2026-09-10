#!/usr/bin/env python3
"""Build a HumanML-style KIT-ML folder from the downloaded KIT raw dump.

This writes texts, official-style splits, Mean/Std, and 251-d motion vectors.
The motion vectors pack root trajectory + joint angles into the KIT feature
width so GALA can train. They are not the official Guo 251-d RIC features,
so FID/R-Precision against published tables is only meaningful after replacing
new_joint_vecs with the official processed KIT-ML release.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np


KIT_DIM = 251
TARGET_FPS = 20


def parse_motion(xml_path: Path, max_frames: int = 196) -> np.ndarray:
    root = ET.parse(xml_path).getroot()
    frames = []
    times = []
    for node in root.findall(".//MotionFrame"):
        time = node.findtext("Timestep")
        pos = node.findtext("RootPosition")
        rot = node.findtext("RootRotation")
        joints = node.findtext("JointPosition")
        if pos is None or rot is None or joints is None:
            continue
        times.append(float(time or 0.0))
        values = [float(x) for x in (pos + " " + rot + " " + joints).split()]
        frames.append(values)
    if not frames:
        raise ValueError(f"No motion frames in {xml_path}")
    array = np.asarray(frames, dtype=np.float32)
    times = np.asarray(times, dtype=np.float32)
    duration = max(float(times[-1] - times[0]), 1.0 / TARGET_FPS)
    length = min(max_frames, max(2, int(round(duration * TARGET_FPS)) + 1))
    query = np.linspace(times[0], times[-1], length)
    resampled = np.empty((length, array.shape[1]), dtype=np.float32)
    for dim in range(array.shape[1]):
        resampled[:, dim] = np.interp(query, times, array[:, dim])
    packed = np.zeros((length, KIT_DIM), dtype=np.float32)
    packed[:, : min(KIT_DIM, resampled.shape[1])] = resampled[:, :KIT_DIM]
    return packed


def split_name(sample_id: str) -> str:
    bucket = int(hashlib.md5(sample_id.encode()).hexdigest(), 16) % 10
    if bucket == 0:
        return "test"
    if bucket == 1:
        return "val"
    return "train"


def write_text(path: Path, captions: list[str]):
    lines = [f"{caption}#{caption}#0.0#0.0" for caption in captions if caption.strip()]
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-root", type=Path, default=Path("/home/qinyang/桌面/project/data/KIT-ML-raw"))
    parser.add_argument("--output", type=Path, default=Path("/home/qinyang/桌面/project/data/KIT-ML"))
    parser.add_argument("--max-frames", type=int, default=196)
    args = parser.parse_args()

    motion_dir = args.output / "new_joint_vecs"
    text_dir = args.output / "texts"
    motion_dir.mkdir(parents=True, exist_ok=True)
    text_dir.mkdir(parents=True, exist_ok=True)

    splits = {"train": [], "val": [], "test": []}
    stats_sum = None
    stats_sq = None
    frames = 0
    kept = 0
    for annotation_path in sorted(args.raw_root.glob("*_annotations.json")):
        sample_id = annotation_path.name.split("_")[0]
        xml_path = annotation_path.with_name(f"{sample_id}_mmm.xml")
        if not xml_path.exists():
            continue
        captions = json.loads(annotation_path.read_text(encoding="utf-8"))
        if not captions:
            continue
        try:
            motion = parse_motion(xml_path, args.max_frames)
        except Exception:
            continue
        if not np.isfinite(motion).all():
            continue
        np.save(motion_dir / f"{sample_id}.npy", motion)
        write_text(text_dir / f"{sample_id}.txt", [str(item) for item in captions])
        splits[split_name(sample_id)].append(sample_id)
        if stats_sum is None:
            stats_sum = motion.sum(0)
            stats_sq = np.square(motion).sum(0)
        else:
            stats_sum += motion.sum(0)
            stats_sq += np.square(motion).sum(0)
        frames += len(motion)
        kept += 1

    for name, ids in splits.items():
        (args.output / f"{name}.txt").write_text("\n".join(ids) + "\n", encoding="utf-8")
    all_ids = splits["train"] + splits["val"] + splits["test"]
    (args.output / "all.txt").write_text("\n".join(all_ids) + "\n", encoding="utf-8")
    (args.output / "train_val.txt").write_text("\n".join(splits["train"] + splits["val"]) + "\n", encoding="utf-8")

    mean = stats_sum / max(frames, 1)
    std = np.sqrt(np.clip(stats_sq / max(frames, 1) - np.square(mean), 0, None))
    std = np.maximum(std, 1e-6)
    np.save(args.output / "Mean.npy", mean.astype(np.float32))
    np.save(args.output / "Std.npy", std.astype(np.float32))
    print(f"Prepared {kept} KIT-ML clips -> {args.output}")
    print({name: len(ids) for name, ids in splits.items()})


if __name__ == "__main__":
    main()
