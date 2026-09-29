#!/usr/bin/env python
"""Expanded fine-grained body-part semantic evaluation (v2).

Loads prompts from evaluation/prompts/part_semantic_v2.json, generates motion
for each unique prompt, computes per-part cosine, Top-1 accuracy, per-part
semantic gap, left/right contrast accuracy, and bootstrap 95% CI over unique
prompts.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import yaml

from models.skeleton_graph import PART_NAMES, pool_body_parts
from scripts.evaluate_t2m import load_model

PART_INDEX = {p: i for i, p in enumerate(PART_NAMES)}


@torch.no_grad()
def evaluate_part_semantic_v2(
    model, clip_encoder, device, prompts, steps=20, guidance=2.5, repeats=4
):
    model.eval()
    # expanded: each unique prompt repeated `repeats` times
    expanded = []
    for p in prompts:
        for _ in range(repeats):
            expanded.append(p)

    per_prompt_cos = []  # per unique prompt: list of (P,) arrays averaged over repeats
    per_prompt_pred = []  # per unique prompt: predicted part index (majority vote)
    per_prompt_target = []  # per unique prompt: target part index

    batch_size = 8
    # group expanded by prompt for aggregation
    prompt_cos = {p["id"]: [] for p in prompts}

    for start in range(0, len(expanded), batch_size):
        batch = expanded[start : start + batch_size]
        texts = [b["text"] for b in batch]
        targets = [PART_INDEX[b["target_part"]] for b in batch]
        ids = [b["id"] for b in batch]
        lengths = torch.full((len(texts),), 60, device=device, dtype=torch.long)

        text, text_mask = clip_encoder.encode(texts, device)
        motion = model.sample(text, text_mask, lengths, steps=steps, guidance_scale=guidance)
        clean_motion, frame_mask, _, graph_joint = model._features(motion, lengths)
        part_motion = pool_body_parts(graph_joint, frame_mask, model.parts)
        part_tokens = model.part_alignment.part_text(text, text_mask)
        z_part = torch.nn.functional.normalize(model.part_alignment.motion_proj(part_motion), dim=-1)
        c_part = torch.nn.functional.normalize(model.part_alignment.lang_proj(part_tokens), dim=-1)
        cos = (z_part * c_part).sum(-1)  # (B, P)

        for b in range(len(texts)):
            cos_vals = cos[b].cpu().numpy()
            prompt_cos[ids[b]].append(cos_vals)

    # Aggregate per unique prompt
    for p in prompts:
        cos_list = prompt_cos[p["id"]]  # list of (P,) arrays, length=repeats
        avg_cos = np.mean(cos_list, axis=0)  # (P,)
        pred = int(np.argmax(avg_cos))
        per_prompt_cos.append(avg_cos)
        per_prompt_pred.append(pred)
        per_prompt_target.append(PART_INDEX[p["target_part"]])

    per_prompt_cos = np.array(per_prompt_cos)  # (N, P)
    per_prompt_pred = np.array(per_prompt_pred)
    per_prompt_target = np.array(per_prompt_target)

    # Top-1 accuracy
    correct = (per_prompt_pred == per_prompt_target).sum()
    total = len(per_prompt_target)
    top1 = correct / total

    # Per-part semantic gap
    gaps = {}
    for p_idx, p_name in enumerate(PART_NAMES):
        target_mask = per_prompt_target == p_idx
        nontarget_mask = per_prompt_target != p_idx
        t_vals = per_prompt_cos[target_mask, p_idx]
        n_vals = per_prompt_cos[nontarget_mask, p_idx]
        gaps[p_name] = float(t_vals.mean() - n_vals.mean()) if len(t_vals) > 0 and len(n_vals) > 0 else 0.0

    # Left/right contrast accuracy
    # For arm pairs: target left_arm (1) vs right_arm (2)
    # For leg pairs: target left_leg (3) vs right_leg (4)
    lr_arm_correct = 0
    lr_arm_total = 0
    lr_leg_correct = 0
    lr_leg_total = 0
    for i, p in enumerate(prompts):
        tgt = per_prompt_target[i]
        pred = per_prompt_pred[i]
        if tgt == 1:  # left_arm, correct if pred == 1 (not 2)
            lr_arm_total += 1
            if pred == 1:
                lr_arm_correct += 1
        elif tgt == 2:  # right_arm
            lr_arm_total += 1
            if pred == 2:
                lr_arm_correct += 1
        elif tgt == 3:  # left_leg
            lr_leg_total += 1
            if pred == 3:
                lr_leg_correct += 1
        elif tgt == 4:  # right_leg
            lr_leg_total += 1
            if pred == 4:
                lr_leg_correct += 1

    # Bootstrap CI over unique prompts
    rng = np.random.default_rng(42)
    n_boot = 1000
    boot_top1 = []
    boot_gaps = {p: [] for p in PART_NAMES}
    for _ in range(n_boot):
        idx = rng.choice(total, size=total, replace=True)
        boot_top1.append((per_prompt_pred[idx] == per_prompt_target[idx]).mean())
        for p_idx, p_name in enumerate(PART_NAMES):
            t_mask = per_prompt_target[idx] == p_idx
            n_mask = per_prompt_target[idx] != p_idx
            if t_mask.sum() > 0 and n_mask.sum() > 0:
                t_v = per_prompt_cos[idx][t_mask, p_idx].mean()
                n_v = per_prompt_cos[idx][n_mask, p_idx].mean()
                boot_gaps[p_name].append(t_v - n_v)
            else:
                boot_gaps[p_name].append(0.0)

    def ci(arr):
        arr = np.array(arr)
        return float(np.percentile(arr, 2.5)), float(np.percentile(arr, 97.5))

    top1_lo, top1_hi = ci(boot_top1)
    gap_ci = {p: ci(boot_gaps[p]) for p in PART_NAMES}

    # Confusion matrix
    confusion = np.zeros((len(PART_NAMES), len(PART_NAMES)), dtype=int)
    for t, p in zip(per_prompt_target, per_prompt_pred):
        confusion[t, p] += 1

    # Per-group accuracy
    group_acc = {}
    for g in range(len(PART_NAMES)):
        row_sum = confusion[g].sum()
        group_acc[PART_NAMES[g]] = int(confusion[g, g]) / max(int(row_sum), 1)

    results = {
        "overall_top1_accuracy": float(top1),
        "top1_ci_95": [top1_lo, top1_hi],
        "chance_level": 0.20,
        "group_accuracy": group_acc,
        "semantic_gap": gaps,
        "semantic_gap_ci_95": {p: list(gap_ci[p]) for p in PART_NAMES},
        "lr_arm_accuracy": float(lr_arm_correct / max(lr_arm_total, 1)),
        "lr_leg_accuracy": float(lr_leg_correct / max(lr_leg_total, 1)),
        "lr_arm_count": lr_arm_total,
        "lr_leg_count": lr_leg_total,
        "confusion_matrix": confusion.tolist(),
        "part_labels": list(PART_NAMES),
        "num_unique_prompts": len(prompts),
        "repeats": repeats,
        "steps": steps,
        "guidance": guidance,
        "bootstrap_samples": n_boot,
        "per_prompt_cos": per_prompt_cos.tolist(),
        "per_prompt_pred": per_prompt_pred.tolist(),
        "per_prompt_target": per_prompt_target.tolist(),
        "prompt_ids": [p["id"] for p in prompts],
    }
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--prompts", default="evaluation/prompts/part_semantic_v2.json")
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

    prompts = json.load(open(args.prompts))
    print(f"Loaded {len(prompts)} unique prompts")

    results = evaluate_part_semantic_v2(
        model, clip_encoder, device, prompts,
        steps=args.steps, guidance=args.guidance, repeats=args.repeats,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2))
    print(json.dumps({k: v for k, v in results.items() if k not in ("per_prompt_cos",)}, indent=2))


if __name__ == "__main__":
    main()
