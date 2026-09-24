#!/usr/bin/env python
"""Fine-grained body-part semantic evaluation (Revision v2, Experiment 2).

For each test prompt we:
  1. Generate motion from the text prompt.
  2. Encode the motion → graph_joint → pool_body_parts → part_motion z_part.
  3. Extract part text tokens c_part from the CLIP text via the part alignment module.
  4. Compute cos(z_part[p], c_part[p]) for each anatomical part p.
  5. Check whether the target body part (e.g. left_arm) has the highest cosine.

Reports per-part mean cosine, per-group top-1 accuracy, and a confusion matrix.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import yaml

from models.skeleton_graph import PART_NAMES, pool_body_parts
from scripts.evaluate_t2m import load_model

# (prompt, target_part_index)
# PART_NAMES = ("torso", "left_arm", "right_arm", "left_leg", "right_leg")  → 0..4
SEMANTIC_PROMPTS: list[tuple[str, int]] = [
    # left arm
    ("a person raises their left arm up above the head", 1),
    ("a person waves their left hand in the air", 1),
    ("a person lifts their left arm sideways", 1),
    ("a person stretches their left arm forward", 1),
    ("a person puts their left hand on the hip", 1),
    # right arm
    ("a person raises their right arm up above the head", 2),
    ("a person waves their right hand in the air", 2),
    ("a person lifts their right arm sideways", 2),
    ("a person stretches their right arm forward", 2),
    ("a person puts their right hand on the hip", 2),
    # left leg
    ("a person kicks forward with their left leg", 3),
    ("a person lifts their left knee up", 3),
    ("a person steps forward with their left leg", 3),
    ("a person raises their left foot off the ground", 3),
    ("a person bends their left knee", 3),
    # right leg
    ("a person kicks forward with their right leg", 4),
    ("a person lifts their right knee up", 4),
    ("a person steps forward with their right leg", 4),
    ("a person raises their right foot off the ground", 4),
    ("a person bends their right knee", 4),
    # torso
    ("a person turns their body around to the left", 0),
    ("a person bends their torso down forward", 0),
    ("a person twists their upper body sideways", 0),
    ("a person leans their torso to the right", 0),
    ("a person rotates their body slowly", 0),
]

PART_LABELS = list(PART_NAMES)  # torso, left_arm, right_arm, left_leg, right_leg


@torch.no_grad()
def evaluate_part_semantic(model, clip_encoder, device, steps=20, guidance=2.5, repeats=4):
    model.eval()
    expanded = []
    for prompt, target in SEMANTIC_PROMPTS:
        for _ in range(repeats):
            expanded.append((prompt, target))

    # flat list: per_part_cos[p][i] = cosine for part p at expanded index i
    per_part_cos = {p: [] for p in range(len(PART_LABELS))}
    confusion = torch.zeros(len(PART_LABELS), len(PART_LABELS), dtype=torch.long)
    correct = 0
    total = 0

    batch_size = 8
    for start in range(0, len(expanded), batch_size):
        batch = expanded[start : start + batch_size]
        prompts = [b[0] for b in batch]
        targets = [b[1] for b in batch]
        lengths = torch.full((len(prompts),), 60, device=device, dtype=torch.long)

        text, text_mask = clip_encoder.encode(prompts, device)
        motion = model.sample(text, text_mask, lengths, steps=steps, guidance_scale=guidance)
        clean_motion, frame_mask, _, graph_joint = model._features(motion, lengths)
        part_motion = pool_body_parts(graph_joint, frame_mask, model.parts)
        part_tokens = model.part_alignment.part_text(text, text_mask)
        z_part = torch.nn.functional.normalize(model.part_alignment.motion_proj(part_motion), dim=-1)
        c_part = torch.nn.functional.normalize(model.part_alignment.lang_proj(part_tokens), dim=-1)
        cos = (z_part * c_part).sum(-1)  # (B, P)

        for b in range(len(prompts)):
            target = targets[b]
            cos_vals = cos[b].cpu()
            pred = int(cos_vals.argmax().item())
            confusion[target, pred] += 1
            if pred == target:
                correct += 1
            total += 1
            for p in range(len(PART_LABELS)):
                per_part_cos[p].append(float(cos_vals[p].item()))

    group_names = list(PART_LABELS)
    group_acc = {}
    for g in range(len(group_names)):
        row_sum = confusion[g].sum().item()
        group_acc[group_names[g]] = confusion[g, g].item() / max(row_sum, 1)

    part_mean_cos = {
        PART_LABELS[p]: sum(per_part_cos[p]) / max(len(per_part_cos[p]), 1)
        for p in range(len(PART_LABELS))
    }

    target_mean = {}
    nontarget_mean = {}
    for p in range(len(PART_LABELS)):
        t_vals, n_vals = [], []
        for i, (_, target) in enumerate(expanded):
            v = per_part_cos[p][i]
            (t_vals if target == p else n_vals).append(v)
        target_mean[PART_LABELS[p]] = sum(t_vals) / max(len(t_vals), 1)
        nontarget_mean[PART_LABELS[p]] = sum(n_vals) / max(len(n_vals), 1)

    results = {
        "overall_top1_accuracy": correct / max(total, 1),
        "group_accuracy": group_acc,
        "part_mean_cosine_all": part_mean_cos,
        "part_mean_cosine_when_target": target_mean,
        "part_mean_cosine_when_nontarget": nontarget_mean,
        "semantic_gap": {p: target_mean[p] - nontarget_mean[p] for p in PART_LABELS},
        "confusion_matrix": confusion.tolist(),
        "part_labels": PART_LABELS,
        "num_prompts": len(expanded),
        "num_unique_prompts": len(SEMANTIC_PROMPTS),
        "repeats": repeats,
        "steps": steps,
        "guidance": guidance,
    }
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--guidance", type=float, default=2.5)
    parser.add_argument("--repeats", type=int, default=4)
    args = parser.parse_args()

    config = yaml.safe_load(open(args.config, encoding="utf-8"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(config, args.checkpoint, device)

    use_clip = config["model"].get("text_encoder_type") == "clip"
    if not use_clip:
        raise ValueError("Part semantic evaluation requires CLIP text encoder")
    from models.clip_text import FrozenCLIPTextEncoder

    clip_encoder = FrozenCLIPTextEncoder().to(device)

    results = evaluate_part_semantic(
        model, clip_encoder, device, steps=args.steps, guidance=args.guidance, repeats=args.repeats
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
