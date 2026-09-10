#!/usr/bin/env python3
"""Cache per-sample text embeddings for HumanML3D / KIT-ML.

Uses hashed bag-of-token embeddings so training does not depend on a
pretrained language-model download. Official T2M evaluation still reads
the original captions separately.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from datasets.kit_raw import hashed_text_embedding


def captions_from_text_file(path: Path) -> list[str]:
    captions = []
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        caption = line.split("#", 1)[0].strip()
        if caption:
            captions.append(caption)
    return captions


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--text-dir", default="texts")
    parser.add_argument("--output-dir", default="text_embeddings")
    parser.add_argument("--dim", type=int, default=512)
    parser.add_argument("--max-captions", type=int, default=8)
    args = parser.parse_args()

    text_dir = args.root / args.text_dir
    if not text_dir.exists():
        raise SystemExit(f"Missing caption directory: {text_dir}")
    output_dir = args.root / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    written = 0
    for text_path in sorted(text_dir.glob("*.txt")):
        captions = captions_from_text_file(text_path)[: args.max_captions]
        if not captions:
            continue
        embeddings = np.stack([hashed_text_embedding(caption, args.dim) for caption in captions])[:, None]
        mask = np.ones((len(captions), 1), dtype=bool)
        np.savez_compressed(output_dir / f"{text_path.stem}.npz", embeddings=embeddings, mask=mask)
        written += 1
    print(f"Wrote {written} embedding files to {output_dir}")


if __name__ == "__main__":
    main()
