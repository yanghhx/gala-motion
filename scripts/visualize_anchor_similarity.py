#!/usr/bin/env python
"""Visualize query-anchor similarity matrix S_ij = cos(q_i, a_j).

Expected: diagonal > off-diagonal (each query matches its anchor).
Saves outputs/anatomy_anchor_similarity.png and .json.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import yaml

from scripts.evaluate_t2m import load_model

PART_LABELS = ["torso", "left_arm", "right_arm", "left_leg", "right_leg"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", type=Path, default=Path("outputs/anatomy_anchor_similarity.png"))
    args = parser.parse_args()

    config = yaml.safe_load(open(args.config, encoding="utf-8"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(config, args.checkpoint, device)

    pa = model.part_alignment
    if not pa.use_anatomical_anchor:
        raise ValueError("Model does not use anatomical anchor")

    with torch.no_grad():
        q = torch.nn.functional.normalize(pa.query_proj(pa.part_queries), dim=-1)
        a = torch.nn.functional.normalize(pa.anchor_to_embed(pa.anchor_embeddings), dim=-1)
        S = (q @ a.T).cpu()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(4, 3.5))
    im = ax.imshow(S.numpy(), cmap="RdYlGn", vmin=-1, vmax=1)
    ax.set_xticks(range(len(PART_LABELS)))
    ax.set_yticks(range(len(PART_LABELS)))
    ax.set_xticklabels(PART_LABELS, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(PART_LABELS, fontsize=8)
    ax.set_xlabel("Anchor $a_j$")
    ax.set_ylabel("Query $q_i$")
    ax.set_title("Query–Anchor Cosine Similarity")
    for i in range(len(PART_LABELS)):
        for j in range(len(PART_LABELS)):
            ax.text(j, i, f"{S[i,j]:.2f}", ha="center", va="center", fontsize=7,
                    color="white" if abs(S[i,j]) > 0.5 else "black")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=200, bbox_inches="tight")
    print(f"Saved {args.output}")

    diag = S.diag().mean().item()
    off = (S.sum() - S.diag().sum()) / (S.numel() - len(PART_LABELS))
    result = {
        "similarity_matrix": S.tolist(),
        "part_labels": PART_LABELS,
        "diagonal_mean": diag,
        "off_diagonal_mean": off,
        "gap": diag - off,
    }
    json_path = args.output.with_suffix(".json")
    json_path.write_text(json.dumps(result, indent=2))
    print(f"Diagonal mean={diag:.4f}, off-diagonal mean={off:.4f}, gap={diag-off:.4f}")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
