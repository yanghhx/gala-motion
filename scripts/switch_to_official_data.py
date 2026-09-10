#!/usr/bin/env python3
"""Point training configs at official datasets after archives extract."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import yaml

ROOT = Path("/home/qinyang/桌面/project")
KIT_CFG = ROOT / "configs/gala_kitml.yaml"
HML_CFG = ROOT / "configs/gala_humanml3d.yaml"
KIT_OFFICIAL = ROOT / "data/KIT-ML-official"
HML_ROOT = ROOT / "data/HumanML3D"


def _count(path: Path) -> int:
    return len(list(path.glob("*.npy"))) if path.exists() else 0


def _sample_dim(path: Path) -> int | None:
    files = sorted(path.glob("*.npy"))
    if not files:
        return None
    array = np.load(files[0])
    return int(array.shape[-1])


def kit_ready() -> bool:
    vecs = _count(KIT_OFFICIAL / "new_joint_vecs")
    dim = _sample_dim(KIT_OFFICIAL / "new_joint_vecs")
    mean = np.load(KIT_OFFICIAL / "Mean.npy") if (KIT_OFFICIAL / "Mean.npy").exists() else None
    print(f"KIT official vecs={vecs} dim={dim} mean={None if mean is None else mean.shape}")
    return vecs >= 5000 and dim == 251 and mean is not None and mean.shape == (251,)


def humanml_ready() -> bool:
    vecs = _count(HML_ROOT / "new_joint_vecs")
    dim = _sample_dim(HML_ROOT / "new_joint_vecs")
    print(f"HumanML3D vecs={vecs} dim={dim}")
    return vecs >= 29000 and dim == 263


def rewrite_root(config_path: Path, new_root: Path) -> bool:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    current = Path(config["dataset"]["root"])
    if current.resolve() == new_root.resolve():
        print(f"{config_path.name} already points to {new_root}")
        return False
    text = config_path.read_text(encoding="utf-8")
    config_path.write_text(text.replace(str(current), str(new_root)), encoding="utf-8")
    print(f"updated {config_path.name}: {current} -> {new_root}")
    return True


def smoke_load(root: Path, split: str = "train") -> None:
    import sys

    sys.path.insert(0, str(ROOT))
    from datasets.humanml import HumanMLDataset

    dataset = HumanMLDataset(root, split=split)
    item = dataset[0]
    print(
        f"loaded {root.name} {split} n={len(dataset)} "
        f"id={item['id']} motion={tuple(item['motion'].shape)}"
    )


def main() -> None:
    changed = False
    if kit_ready():
        changed = rewrite_root(KIT_CFG, KIT_OFFICIAL) or changed
        smoke_load(KIT_OFFICIAL)
    else:
        print("KIT official motions are not ready; leave gala_kitml.yaml unchanged")

    if humanml_ready():
        changed = rewrite_root(HML_CFG, HML_ROOT) or changed
        smoke_load(HML_ROOT)
    else:
        print("HumanML3D motions are not ready; keep current humanml config")

    print("changed" if changed else "no config change")


if __name__ == "__main__":
    main()
