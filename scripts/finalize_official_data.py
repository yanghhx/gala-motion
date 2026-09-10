#!/usr/bin/env python3
"""Extract finished official archives and validate HumanML3D / KIT-ML layouts."""
from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

import numpy as np

ROOT = Path("/home/qinyang/桌面/project")
SEVEN = Path("/home/qinyang/.local/share/mamba/envs/gala-motion/bin/7z")


def count_files(path: Path, pattern: str) -> int:
    return len(list(path.glob(pattern))) if path.exists() else 0


def extract_rar(archive: Path, dest: Path) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    subprocess.check_call([str(SEVEN), "x", "-y", f"-o{dest}", str(archive)])


def validate_kit(root: Path) -> dict:
    mean = np.load(root / "Mean.npy")
    std = np.load(root / "Std.npy")
    vecs = count_files(root / "new_joint_vecs", "*.npy")
    texts = count_files(root / "texts", "*.txt")
    splits = {
        name: len([line for line in (root / f"{name}.txt").read_text().splitlines() if line.strip()])
        for name in ("all", "train", "val", "test", "train_val")
        if (root / f"{name}.txt").exists()
    }
    sample = None
    vec_files = sorted((root / "new_joint_vecs").glob("*.npy"))
    if vec_files:
        sample = np.load(vec_files[0]).shape
    return {
        "root": str(root),
        "mean": list(mean.shape),
        "std": list(std.shape),
        "vecs": vecs,
        "texts": texts,
        "splits": splits,
        "sample_vec": sample,
        "complete": vecs >= 5000 and texts >= 6000 and list(mean.shape) == [251],
    }


def validate_humanml(root: Path) -> dict:
    mean = np.load(root / "Mean.npy")
    std = np.load(root / "Std.npy")
    vecs = count_files(root / "new_joint_vecs", "*.npy")
    texts = count_files(root / "texts", "*.txt")
    splits = {
        name: len([line for line in (root / f"{name}.txt").read_text().splitlines() if line.strip()])
        for name in ("all", "train", "val", "test", "train_val")
        if (root / f"{name}.txt").exists()
    }
    return {
        "root": str(root),
        "mean": list(mean.shape),
        "std": list(std.shape),
        "vecs": vecs,
        "texts": texts,
        "splits": splits,
        "complete": vecs >= 29000 and texts >= 29000 and list(mean.shape) == [263],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--extract-kit-vecs", action="store_true")
    parser.add_argument("--extract-humanml-vecs", action="store_true")
    args = parser.parse_args()

    kit_archive = ROOT / "data/downloads/KIT-ML-official/new_joint_vecs.rar"
    kit_dest = ROOT / "data/KIT-ML-official"
    if args.extract_kit_vecs and kit_archive.exists():
        extract_rar(kit_archive, kit_dest)

    hml_archive = ROOT / "data/downloads/HumanML3D-official/new_joint_vecs.rar"
    hml_dest = ROOT / "research_sources/HumanML3D/HumanML3D"
    if args.extract_humanml_vecs and hml_archive.exists():
        extract_rar(hml_archive, hml_dest)

    report = {
        "kit_official": validate_kit(kit_dest),
        "kit_from_raw": {
            "root": str(ROOT / "data/KIT-ML"),
            "vecs": count_files(ROOT / "data/KIT-ML/new_joint_vecs", "*.npy"),
            "texts": count_files(ROOT / "data/KIT-ML/texts", "*.txt"),
            "note": "unofficial packed features from raw XML; keep until official vecs arrive",
        },
        "humanml3d": validate_humanml(hml_dest),
    }
    print(report)


if __name__ == "__main__":
    main()
