#!/usr/bin/env python
"""Contact-ratio sanity check: verify HumanML3D vs KIT-ML contact definitions.

Computes ContactRatio = #valid_contact_indicators / #valid_foot_frame_indicators
for both datasets, reporting per-foot (left heel, right heel, left toe, right toe)
and overall.
"""
from __future__ import annotations

import json
from pathlib import Path
import numpy as np

# HumanML3D: 263-D, 22 joints. Foot contact channels are the last 4:
#   index 259 = left heel, 260 = right heel, 261 = left toe, 262 = right toe
HML3D_CONTACT_SLICES = {
    "left_heel": slice(259, 260),
    "right_heel": slice(260, 261),
    "left_toe": slice(261, 262),
    "right_toe": slice(262, 263),
}

# KIT-ML: 251-D, 21 joints. Foot contact channels are the last 4:
#   index 247 = left heel, 248 = right heel, 249 = left toe, 250 = right toe
KIT_CONTACT_SLICES = {
    "left_heel": slice(247, 248),
    "right_heel": slice(248, 249),
    "left_toe": slice(249, 250),
    "right_toe": slice(250, 251),
}


def compute_contact_ratio(motion_dir, split_file, contact_slices, max_files=None):
    """Compute contact ratio over a dataset split."""
    with open(split_file) as f:
        ids = [l.strip() for l in f if l.strip()]

    total_contact = {k: 0 for k in contact_slices}
    total_frames = 0
    total_foot_frames = 0  # frames where any foot indicator is valid (non-padded)
    n_processed = 0

    for mid in ids:
        motion_path = motion_dir / f"{mid}.npy"
        if not motion_path.exists():
            continue
        motion = np.load(motion_path).astype(np.float32)
        T = motion.shape[0]
        total_frames += T
        # For each frame, check if it's a valid (non-zero) frame
        # In HumanML3D/KIT, padding is zeros, so valid frames have non-zero root
        valid_mask = np.abs(motion[:, 0]) > 1e-8  # root motion non-zero
        n_valid = valid_mask.sum()
        total_foot_frames += n_valid

        for name, sl in contact_slices.items():
            contact_vals = motion[valid_mask, sl]
            # Contact is binary (0 or 1) in the original, but stored as float
            # Count frames where contact > 0.5
            total_contact[name] += int((contact_vals > 0.5).sum())

        n_processed += 1
        if max_files and n_processed >= max_files:
            break

    overall_contact = sum(total_contact.values())
    result = {
        "n_clips": n_processed,
        "total_frames": total_frames,
        "total_valid_frames": total_foot_frames,
        "per_foot": {k: v for k, v in total_contact.items()},
        "overall_contact_indicators": overall_contact,
        "contact_ratio_overall": overall_contact / max(total_foot_frames * 4, 1),  # 4 channels per frame
        "contact_ratio_per_foot": {k: v / max(total_foot_frames, 1) for k, v in total_contact.items()},
    }
    return result


def main():
    out_dir = Path("outputs/physical")
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=== HumanML3D contact ratio ===")
    hml3d = compute_contact_ratio(
        Path("data/HumanML3D/new_joint_vecs"),
        "data/HumanML3D/test.txt",
        HML3D_CONTACT_SLICES,
    )
    print(f"  Clips: {hml3d['n_clips']}")
    print(f"  Valid frames: {hml3d['total_valid_frames']}")
    print(f"  Overall contact ratio: {hml3d['contact_ratio_overall']:.4f}")
    for k, v in hml3d["contact_ratio_per_foot"].items():
        print(f"    {k}: {v:.4f}")

    print("\n=== KIT-ML contact ratio ===")
    kit = compute_contact_ratio(
        Path("data/KIT-ML-official/new_joint_vecs"),
        "data/KIT-ML-official/test.txt",
        KIT_CONTACT_SLICES,
    )
    print(f"  Clips: {kit['n_clips']}")
    print(f"  Valid frames: {kit['total_valid_frames']}")
    print(f"  Overall contact ratio: {kit['contact_ratio_overall']:.4f}")
    for k, v in kit["contact_ratio_per_foot"].items():
        print(f"    {k}: {v:.4f}")

    ratio = hml3d["contact_ratio_overall"] / max(kit["contact_ratio_overall"], 1e-8)
    print(f"\nHumanML3D / KIT-ML ratio: {ratio:.1f}x")

    result = {
        "humanml3d": {k: (int(v) if isinstance(v, np.integer) else v) for k, v in hml3d.items()},
        "kit_ml": {k: (int(v) if isinstance(v, np.integer) else v) for k, v in kit.items()},
        "hml3d_to_kit_ratio": float(ratio),
        "contact_channel_indices": {
            "humanml3d": {k: f"[{v.start}:{v.stop}]" for k, v in HML3D_CONTACT_SLICES.items()},
            "kit_ml": {k: f"[{v.start}:{v.stop}]" for k, v in KIT_CONTACT_SLICES.items()},
        },
    }
    out_path = out_dir / "contact_ratio_sanity_check.json"
    out_path.write_text(json.dumps(result, indent=2))
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
