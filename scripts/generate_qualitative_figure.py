#!/usr/bin/env python
"""Qualitative comparison figure: Text | GT | Global | GALA skeleton keyframes."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
import yaml
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from scripts.evaluate_t2m import load_model

# HumanML3D kinematic chain (from MDM paramUtil)
T2M_KINEMATIC_CHAIN = [[0, 2, 5, 8, 11], [0, 1, 4, 7, 10], [0, 3, 6, 9, 12, 15], [9, 14, 17, 19, 21], [9, 13, 16, 18, 20]]

QUALITATIVE_PROMPTS = [
    {"text": "a person raises their left arm up above the head", "label": "L-arm raise"},
    {"text": "a person raises their right arm up above the head", "label": "R-arm raise"},
    {"text": "a person kicks forward with their left leg", "label": "L-leg kick"},
    {"text": "a person kicks forward with their right leg", "label": "R-leg kick"},
    {"text": "a person bends the torso forward", "label": "Torso bend"},
    {"text": "a person walks forward", "label": "Walk"},
    {"text": "a person waves their left hand while stepping forward", "label": "Compound"},
    {"text": "a person raises the left arm while stepping forward", "label": "L-arm+step"},
]


def find_gt_motion(prompt_text, test_ids, texts_dir, max_search=800):
    prompt_words = set(prompt_text.lower().replace("a person ", "").replace("their ", "").split())
    best_id, best_score, best_text = None, -1, ""
    count = 0
    for mid in test_ids:
        if count >= max_search:
            break
        count += 1
        txt_path = os.path.join(texts_dir, f"{mid}.txt")
        if not os.path.exists(txt_path):
            continue
        with open(txt_path) as f:
            lines = f.readlines()
        for line in lines:
            caption = line.strip().lower()
            caption_words = set(caption.replace("a person ", "").replace("their ", "").split())
            overlap = len(prompt_words & caption_words)
            if overlap > best_score:
                best_score = overlap
                best_id = mid
                best_text = line.strip()
    return best_id, best_text, best_score


def plot_skeleton_3d(ax, joints, color="steelblue", title=""):
    ax.cla()
    root = joints[0]
    jc = joints - root
    for chain in T2M_KINEMATIC_CHAIN:
        xs = [jc[j, 0] for j in chain]
        ys = [jc[j, 1] for j in chain]
        zs = [jc[j, 2] for j in chain]
        ax.plot(xs, ys, zs, color=color, linewidth=1.2, alpha=0.8)
    ax.scatter(jc[:, 0], jc[:, 1], jc[:, 2], color=color, s=6, alpha=0.9)
    ax.set_xlim(-1.5, 1.5); ax.set_ylim(-1.5, 1.5); ax.set_zlim(0, 2.5)
    ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
    ax.view_init(elev=15, azim=-60)
    if title:
        ax.set_title(title, fontsize=5, pad=1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gala-config", default="configs/gala_humanml3d_flow_distinct.yaml")
    parser.add_argument("--gala-ckpt", default="checkpoints/gala_humanml3d_flow_distinct/best.pt")
    parser.add_argument("--global-config", default="configs/gala_humanml3d_flow.yaml")
    parser.add_argument("--global-ckpt", default="checkpoints/gala_humanml3d_flow/best.pt")
    parser.add_argument("--output", type=Path, default=Path("outputs/qualitative/fig_qualitative.pdf"))
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--guidance", type=float, default=2.5)
    parser.add_argument("--num_keyframes", type=int, default=4)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    mean = np.load("data/HumanML3D/Mean.npy")
    std = np.load("data/HumanML3D/Std.npy")
    with open("data/HumanML3D/test.txt") as f:
        test_ids = [l.strip() for l in f if l.strip()]
    texts_dir = "data/HumanML3D/texts"

    # Load GT motions directly from npy files
    motion_dir = Path("data/HumanML3D/new_joint_vecs")

    # Load models
    gala_config = yaml.safe_load(open(args.gala_config))
    gala_model = load_model(gala_config, args.gala_ckpt, device)
    gala_model.eval()
    global_config = yaml.safe_load(open(args.global_config))
    global_model = load_model(global_config, args.global_ckpt, device)
    global_model.eval()

    from models.clip_text import FrozenCLIPTextEncoder
    clip_encoder = FrozenCLIPTextEncoder().to(device)

    from models.motion_repr import recover_from_ric, denormalize_motion

    n_prompts = len(QUALITATIVE_PROMPTS)
    n_kf = args.num_keyframes
    # Columns: text(1) + GT(n_kf) + Global(n_kf) + GALA(n_kf)
    n_cols = 1 + 3 * n_kf

    fig = plt.figure(figsize=(12, 2.0 * n_prompts))
    gt_matches = []

    for row_idx, prompt in enumerate(QUALITATIVE_PROMPTS):
        text = prompt["text"]
        label = prompt["label"]

        gt_id, gt_text, score = find_gt_motion(text, test_ids, texts_dir)
        gt_matches.append({"prompt": text, "gt_id": gt_id, "gt_text": gt_text, "score": score})
        print(f"[{row_idx}] {label}: GT={gt_id} score={score} text='{gt_text[:50]}'")

        # Load GT motion (normalized in npy)
        gt_length = 60
        gt_motion = np.zeros((gt_length, 263), dtype=np.float32)
        gt_npy_path = Path(f"data/HumanML3D/{gt_id}.npy") if gt_id else None
        if gt_npy_path and gt_npy_path.exists():
            gt_motion_raw = np.load(gt_npy_path).astype(np.float32)
            gt_length = min(len(gt_motion_raw), 100)
            gt_motion = gt_motion_raw[:gt_length]
        else:
            # Try new_joint_vecs
            gt_npy_path = motion_dir / f"{gt_id}.npy" if gt_id else None
            if gt_npy_path and gt_npy_path.exists():
                gt_motion_raw = np.load(gt_npy_path).astype(np.float32)
                gt_length = min(len(gt_motion_raw), 100)
                gt_motion = gt_motion_raw[:gt_length]

        # Generate
        text_enc, text_mask = clip_encoder.encode([text], device)
        length = torch.tensor([gt_length], device=device, dtype=torch.long)

        with torch.no_grad():
            gala_motion = gala_model.sample(text_enc, text_mask, length, steps=args.steps, guidance_scale=args.guidance)
            gala_motion = gala_motion[0].cpu().numpy()
        with torch.no_grad():
            global_motion = global_model.sample(text_enc, text_mask, length, steps=args.steps, guidance_scale=args.guidance)
            global_motion = global_motion[0].cpu().numpy()

        # Recover 3D joints (motions are normalized)
        gt_joints = recover_from_ric(denormalize_motion(torch.from_numpy(gt_motion), mean, std), 22).numpy()
        gala_joints = recover_from_ric(denormalize_motion(torch.from_numpy(gala_motion), mean, std), 22).numpy()
        global_joints = recover_from_ric(denormalize_motion(torch.from_numpy(global_motion), mean, std), 22).numpy()

        T = min(gt_joints.shape[0], gala_joints.shape[0], global_joints.shape[0])
        kf_indices = np.linspace(int(T*0.2), int(T*0.8), n_kf, dtype=int)

        # Text
        ax_text = fig.add_subplot(n_prompts, n_cols, row_idx * n_cols + 1)
        ax_text.axis("off")
        ax_text.text(0.5, 0.5, f"{label}\n\"{text}\"", fontsize=4, ha="center", va="center", wrap=True)

        for kf_i, kf in enumerate(kf_indices):
            ax_gt = fig.add_subplot(n_prompts, n_cols, row_idx*n_cols + 2 + kf_i, projection="3d")
            plot_skeleton_3d(ax_gt, gt_joints[kf], "forestgreen", "GT" if kf_i == 0 else "")
            ax_g = fig.add_subplot(n_prompts, n_cols, row_idx*n_cols + 2 + n_kf + kf_i, projection="3d")
            plot_skeleton_3d(ax_g, global_joints[kf], "coral", "Global" if kf_i == 0 else "")
            ax_gala = fig.add_subplot(n_prompts, n_cols, row_idx*n_cols + 2 + 2*n_kf + kf_i, projection="3d")
            plot_skeleton_3d(ax_gala, gala_joints[kf], "steelblue", "GALA" if kf_i == 0 else "")

    fig.suptitle("Qualitative: GT (green) | Global (orange) | GALA (blue)", fontsize=7)
    plt.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=150, bbox_inches="tight")
    fig.savefig(args.output.with_suffix(".png"), dpi=150, bbox_inches="tight")
    print(f"Saved {args.output}")

    meta_path = args.output.parent / "qualitative_meta.json"
    meta_path.write_text(json.dumps({"prompts": QUALITATIVE_PROMPTS, "gt_matches": gt_matches,
                                      "steps": args.steps, "guidance": args.guidance}, indent=2))
    print(f"Saved {meta_path}")


if __name__ == "__main__":
    main()
