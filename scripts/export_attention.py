#!/usr/bin/env python3
"""Export learned anatomical-query attention and a publication-ready heatmap."""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import fields
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.clip_text import FrozenCLIPTextEncoder
from models.gala_motion import GALAMotion, GALAMotionConfig
from models.skeleton_graph import PART_NAMES

DEFAULT_CAPTIONS = (
    "a person raises the left arm and then kicks with the right leg",
    "a person waves with the right hand while standing still",
    "a person steps forward with the left leg and bends the torso",
)
DISPLAY_NAMES = ("Torso", "L-arm", "R-arm", "L-leg", "R-leg")


def load_model(config_path: Path, checkpoint_path: Path, device: torch.device) -> GALAMotion:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    raw_cfg = dict(config["model"])
    raw_cfg["train_stage"] = "joint"
    allowed = {field.name for field in fields(GALAMotionConfig)}
    cfg = GALAMotionConfig(
        motion_dim=config["dataset"]["motion_dim"],
        num_joints=config["dataset"]["num_joints"],
        max_frames=config["dataset"]["max_frames"],
        **{key: value for key, value in raw_cfg.items() if key in allowed},
    )
    model = GALAMotion(cfg).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state = checkpoint.get("model", checkpoint)
    model.load_state_dict(state, strict=False)
    ema = checkpoint.get("ema") if isinstance(checkpoint, dict) else None
    if ema and ema.get("shadow"):
        for name, value in ema["shadow"].items():
            if name in dict(model.named_parameters()):
                dict(model.named_parameters())[name].data.copy_(value.to(device))
    return model.eval()


def clean_token(token: str) -> str:
    return token.replace("</w>", "").replace("Ġ", "").strip()


@torch.no_grad()
def export(model: GALAMotion, clip: FrozenCLIPTextEncoder, captions: list[str], device: torch.device):
    encoded = clip.tokenizer(
        captions, padding=True, truncation=True, max_length=clip.max_length, return_tensors="pt",
    )
    encoded = {key: value.to(device) for key, value in encoded.items()}
    hidden = clip.model(**encoded).last_hidden_state
    mask = encoded["attention_mask"].bool()
    text = model.encode_text(hidden, mask)
    _, attention = model.part_alignment.part_text(text, mask, return_attention=True)
    records = []
    for index, caption in enumerate(captions):
        valid = int(mask[index].sum())
        ids = encoded["input_ids"][index, :valid].tolist()
        tokens = [clean_token(token) for token in clip.tokenizer.convert_ids_to_tokens(ids)]
        keep = [i for i, token in enumerate(tokens) if token and token not in ("<|startoftext|>", "<|endoftext|>")]
        records.append({
            "caption": caption,
            "part_order": list(PART_NAMES),
            "tokens": [tokens[i] for i in keep],
            "attention": attention[index, :, keep].float().cpu().tolist(),
        })
    return records


def plot(records: list[dict], output: Path) -> None:
    plt.rcParams.update({"font.family": "DejaVu Sans", "pdf.fonttype": 42, "ps.fonttype": 42})
    fig, axes = plt.subplots(len(records), 1, figsize=(7.1, 1.45 * len(records)), squeeze=False)
    for ax, record in zip(axes[:, 0], records):
        matrix = np.asarray(record["attention"])
        image = ax.imshow(matrix, cmap="YlGnBu", aspect="auto", vmin=0, vmax=max(0.22, matrix.max()))
        ax.set_xticks(range(len(record["tokens"])), record["tokens"], rotation=32, ha="right", fontsize=7.5)
        ax.set_yticks(range(5), DISPLAY_NAMES, fontsize=8)
        ax.set_title(record["caption"], loc="left", fontsize=8.5, pad=3)
        ax.tick_params(length=0)
        for row in range(matrix.shape[0]):
            for col in range(matrix.shape[1]):
                if matrix[row, col] >= np.quantile(matrix[row], 0.8):
                    ax.scatter(col, row, s=7, c="white", marker="o", linewidths=0)
    fig.colorbar(image, ax=axes[:, 0].tolist(), fraction=0.015, pad=0.015, label="Attention")
    fig.subplots_adjust(left=0.09, right=0.94, top=0.96, bottom=0.13, hspace=0.85)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/gala_humanml3d_flow_part.yaml"))
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/gala_humanml3d_flow_part/best.pt"))
    parser.add_argument("--caption", action="append", dest="captions")
    parser.add_argument("--output-json", type=Path, default=Path("outputs/attention/part_attention.json"))
    parser.add_argument("--output-figure", type=Path, default=Path("outputs/figures/fig_part_attention.pdf"))
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args()
    device = torch.device(args.device)
    model = load_model(args.config, args.checkpoint, device)
    clip = FrozenCLIPTextEncoder().to(device).eval()
    records = export(model, clip, args.captions or list(DEFAULT_CAPTIONS), device)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(records, indent=2), encoding="utf-8")
    plot(records, args.output_figure)
    print(f"wrote {args.output_json} and {args.output_figure}")


if __name__ == "__main__":
    main()
